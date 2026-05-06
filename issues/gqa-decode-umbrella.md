## Summary

This issue tracks the dense GQA decode release roadmap, interface scope, validation plan, and sub-issue breakdown.

Modern LLM serving decode is not a simple `q_len=1` special case of prefill. Real serving decode is shaped by paged KV cache metadata, batch-slot semantics, heterogeneous per-slot KV lengths, score and position modifiers, long-context split-K, CUDA Graph replay constraints, and manifest-backed benchmark coverage.

This umbrella should stay open until the release-facing dense GQA decode scope is complete and validated. Individual implementation PRs should reference the relevant sub-issue and this umbrella, but should not close this umbrella directly.

## Problem Statement

TileOPs needs a serving-ready dense GQA decode operator family. The public contract should make these distinctions explicit:

- Dense decode only for this release; sparse / block-sparse decode is out of scope.
- Paged KV cache is the serving main path.
- Contiguous KV cache remains a reference / fallback path.
- Decode batch means fixed decode slots, not necessarily active requests.
- KV lengths and page mappings are dynamic tensor contents, not JIT compile keys.
- RoPE / cache position and score modifiers must align with the GQA prefill semantics where applicable.
- Split-K is an internal dispatch strategy, not a public OP.
- CUDA Graph support is a static bucket / buffer contract, not a separate public CUDA Graph OP.

## Surveyed Systems

The feature scope is informed by:

- FlashAttention `flash_attn_with_kvcache`
- FlashInfer decode / paged KV plan-run wrappers
- vLLM PagedAttention metadata
- TensorRT-LLM generation attention
- cuDNN frontend SDPA decode support
- PyTorch SDPA / GQA reference semantics

TileOPs does not need to mirror any single project. The goal is to cover the same serving-relevant dimensions with TileOps-native OP contracts and manifest entries.

## Dense Decode Variant Dimensions

| Dimension | Meaning | Required Scope |
| --- | --- | --- |
| Query unit | Current query tokens per slot | single-token decode |
| Batch shape | Runtime slot layout | fixed `B` decode slots, heterogeneous active lengths |
| KV storage | Where visible KV comes from | paged KV main path, contiguous fallback |
| Page metadata | Logical-to-physical mapping | `block_table [B, max_pages_per_req]` |
| KV length | Per-slot visible prefix | `real_seqlen_kv [B]` |
| Head mapping | MHA/GQA/MQA expression | `heads`, `heads_kv`, `heads % heads_kv == 0` |
| Mask semantics | Who can attend to whom | dense full-prefix first; sliding window tracked separately |
| Position semantics | Decode position alignment | external RoPE reference, fused Q RoPE, `cache_position [B]` |
| Score modifiers | Pre-softmax transforms | `sm_scale`, logits softcap |
| Numeric format | Baseline dtype | fp16, bf16 |
| Dispatch | Long-context performance | no-split fast path, internal split-K |
| CUDA Graph | Replay stability | fixed shape/pointer/launch topology bucket contract |
| Benchmarks | Release readiness | manifest-backed workloads and roofline hooks |

## Release-Facing OP Family

### `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp`

Serving main path:

- `q [B, H, D]`
- `k_pages/v_pages [P_tokens, H_kv, D]`
- `real_seqlen_kv [B]`
- `block_table [B, max_pages_per_req]`
- output `o [B, H, D]`

Dense full-prefix semantics:

```text
visible(q_b, k_j) = 0 <= j < real_seqlen_kv[b]
kv_head = q_head // (H / H_kv)
physical_token = block_table[b, logical_page] * page_size + page_offset
```

### `GroupedQueryAttentionDecodeWithKVCacheFwdOp`

Reference / fallback path:

- `q [B, H, D]`
- `k_cache/v_cache [B, S_kv_cap, H_kv, D]`
- `real_seqlen_kv [B]`
- output `o [B, H, D]`

## KV Layout Decision

Dense decode does not use prefill-style varlen packed KV as the release-facing serving interface.

Do not make this the main decode interface:

```text
k_cache/v_cache [sum(real_seqlen_kv), H_kv, D]
cu_seqlens_kv [B + 1]
```

The serving main path is paged KV:

```text
k_pages/v_pages [P_tokens, H_kv, D]
block_table [B, max_pages_per_req]
real_seqlen_kv [B]
```

Varlen packed KV may be used as a reference, debug helper, or bridge wrapper, but not as the release-facing serving decode path.

## Position / RoPE Contract

For read-only dense decode:

- external RoPE remains the reference path
- fused Q RoPE is part of the serving feature surface
- old cached K is assumed to already be rotated
- old cached K must not be rotated again
- `cache_position [B]` or equivalent per-slot metadata defines the current query position
- append decode is a follow-up; rotating `k_new` and writing it into cache is not part of this dense read-only baseline

## Score Modifier Contract

Required score modifiers:

- `sm_scale: Optional[float] = None`
  - default is `1 / sqrt(head_dim)`
- `softcap: Optional[float] = None`
  - `None` or `0` means disabled
  - `softcap > 0` applies logits soft capping before softmax:
    `softcap * tanh(score / softcap)`

## Spec-Driven Delivery Rule

TileOPs is spec-driven. Every subissue that changes an operator contract or adds a release-facing feature must update the corresponding declarative and validation assets in the same PR.

Each feature PR must include the relevant subset of:

- `tileops/manifest/attention.yaml`
- workload entries under `workloads/`
- manifest `source.kernel`, `source.kernel_map`, `source.op`, `source.test`, and `source.bench`
- roofline function binding or explicit reuse of an existing formula
- unit tests and materialized reference tests
- benchmark path / benchmark metadata for release-facing shapes
- docs or issue-linked contract updates
- manifest and workload validation where applicable

Do not land implementation-only PRs and defer spec / manifest / workload updates to a later cleanup PR.

## Sub-Issue Breakdown

Existing/planned sub-issues:

- {{SUBISSUE_1}}: serving baseline contract and paged heterogeneous correctness
- {{SUBISSUE_2}}: scale, softcap, and fused Q RoPE
- {{SUBISSUE_3}}: internal split-K and CUDA Graph-friendly dispatch contract
- {{SUBISSUE_4}}: sliding-window support for dense paged decode

This list may grow only if a feature needs a smaller review slice. The umbrella remains the release-level tracker.

## Non-Goals

- Prefill / chunked prefill
- sparse decode / block-sparse attention
- full serving scheduler or continuous batching runtime
- page allocation / eviction / prefix cache lifecycle
- sampling or speculative decoding scheduler
- decode context parallelism
- FP8 Tensor Core decode path
- INT8 / NVFP4 KV cache
- fused append decode
- public `return_lse` unless required by a concrete downstream use case

## Release Plan

Suggested staged plan:

1. Serving baseline dense paged decode
   - paged KV main path
   - heterogeneous `real_seqlen_kv`
   - batch slot and inactive slot semantics
   - contiguous fallback / reference alignment

2. Score and position semantics
   - `sm_scale`
   - logits softcap
   - external RoPE reference
   - fused Q RoPE
   - `cache_position [B]`

3. Split-K and CUDA Graph-friendly dispatch
   - internal split-K
   - online softmax merge
   - fixed split policy for graph buckets
   - static vs dynamic metadata contract

4. Sliding window
   - `window_left`
   - logical-position semantics
   - page-boundary and tail-mask correctness

5. Release cleanup under each feature PR
   - manifest / workloads / roofline / source metadata
   - tests
   - benchmark coverage
   - docs

## Validation Expectations

Tests should cover:

- paged dense decode output correctness
- contiguous fallback / reference correctness
- heterogeneous `real_seqlen_kv`
- non-page-aligned lengths
- page tail masking
- unused `block_table` entries are not read
- batch slot and inactive slot semantics
- MHA / GQA / MQA head mapping
- `sm_scale`
- logits softcap
- external RoPE absolute-position correctness
- fused Q RoPE correctness
- old cached K is not re-rotated
- split-K vs no-split reference parity
- fixed graph-bucket metadata constraints
- sliding-window logical-position correctness once the sliding-window subissue lands
- fp16 and bf16 coverage

Implementation details should be discussed in their corresponding PRs. PRs should reference this issue, but should not close it until the overall release scope is complete.
