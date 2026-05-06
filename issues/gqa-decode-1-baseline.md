Parent: {{PARENT}}

Related feature issues: {{RELATED}}

## Scope

Track the serving baseline dense GQA decode contract and paged heterogeneous correctness.

This issue covers the public paged decode OP as the serving main path and the contiguous decode OP as a reference / fallback path. It requires manifest updates because the OP exposes paged physical KV storage, `block_table`, per-slot `real_seqlen_kv`, batch-slot semantics, and dense full-prefix decode behavior.

## Public OPs

### `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp`

Purpose:

- Single-token dense GQA/MQA/MHA decode.
- Paged physical KV cache storage.
- `block_table[b, logical_page] -> physical_page` mapping.
- Per-slot `real_seqlen_kv` visible prefix length.
- Dense full-prefix semantics: attend logical KV positions `0 .. real_seqlen_kv[b] - 1`.
- Supports heterogeneous lengths and page counts in one batch.
- Does not allocate pages and does not update KV cache metadata.
- Returns output only.

### `GroupedQueryAttentionDecodeWithKVCacheFwdOp`

Purpose:

- Contiguous KV-cache decode reference / fallback path.
- `k_cache/v_cache [B, S_kv_cap, H_kv, D]`.
- Explicit `real_seqlen_kv [B]`.
- Same dense full-prefix semantics as paged decode.
- Returns output only.

## Batch Slot Contract

`B` means decode slot count, not necessarily active request count.

```text
B = B_bucket
active_requests <= B_bucket
```

Each slot's visible KV is determined by:

- `real_seqlen_kv[b]`
- `block_table[b, :]` for paged decode

The OP does not own request scheduling, page allocation, page eviction, or slot assignment.

For read-only decode, the first implementation may use `real_seqlen_kv[b] = 0` for inactive slots. Inactive slot output should be deterministic; prefer zero output for testability. If future append decode needs to distinguish a valid request with old length 0 from an inactive slot, add a `valid_mask` or explicit slot mapping in that follow-up.

## Manifest Change Plan

This issue requires manifest changes.

Required manifest updates:

- Update `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp`.
- Update `GroupedQueryAttentionDecodeWithKVCacheFwdOp`.

The entries must declare:

- `family: attention`
- `ref_api: none`
- ordered `signature.inputs`, `signature.outputs`, and `signature.params`
- BHD decode layout
- paged physical cache shape rules
- explicit `real_seqlen_kv [B]` for both paged and contiguous decode
- GQA head grouping shape rule
- `workloads` for serving decode shapes
- `roofline`
- `source.kernel`, `source.kernel_map`, `source.op`, `source.test`, and `source.bench`

## Manifest Field Sketch

This is not copy-paste YAML; it is the field contract the manifest PR should encode using the repo's ordered-dict schema.

### Paged decode

Manifest key: `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp`

Inputs:

- `q`: dtype `float16 | bfloat16`, shape `[B, H, D]`
- `k`: dtype `same_as(q)`, shape `[P_tokens, H_kv, D]`
- `v`: dtype `same_as(q)`, shape `[P_tokens, H_kv, D]`
- `real_seqlen_kv`: dtype `int32`, shape `[B]`
- `block_table`: dtype `int32`, shape `[B, max_pages_per_req]`

Output:

- `o`: dtype `same_as(q)`, shape `[B, H, D]`

Params:

- `page_size`: `{type: int}`

Shape rules / runtime validation:

- `H % H_kv == 0`
- `v.shape == k.shape`
- `P_tokens % page_size == 0`
- `real_seqlen_kv.shape == (B,)`
- `block_table.shape == (B, max_pages_per_req)`
- `real_seqlen_kv[b] <= max_pages_per_req * page_size`
- `block_table` entries used by each slot must be valid physical page ids
- unused `block_table` entries must not be read

Proposed source:

- `source.kernel`: `tileops/kernels/attention/gqa_decode_paged.py`
- `source.kernel_map`: `gqa_decode_paged_kernel: GQADecodePagedKernel` or the final baseline kernel class
- `source.op`: `tileops/ops/attention/gqa.py`
- `source.test`: `tests/ops/attention/test_gqa_decode_paged.py`
- `source.bench`: `benchmarks/ops/attention/bench_gqa_decode_paged.py`

### Contiguous decode

Manifest key: `GroupedQueryAttentionDecodeWithKVCacheFwdOp`

Inputs:

- `q`: dtype `float16 | bfloat16`, shape `[B, H, D]`
- `k`: dtype `same_as(q)`, shape `[B, S_kv_cap, H_kv, D]`
- `v`: dtype `same_as(q)`, shape `[B, S_kv_cap, H_kv, D]`
- `real_seqlen_kv`: dtype `int32`, shape `[B]`

Output:

- `o`: dtype `same_as(q)`, shape `[B, H, D]`

Params:

- none unless the implementation has existing construction-time shape params that must remain represented

Shape rules / runtime validation:

- `H % H_kv == 0`
- `v.shape == k.shape`
- `real_seqlen_kv.shape == (B,)`
- `real_seqlen_kv[b] <= S_kv_cap`

## Workloads / Roofline

Manifest workloads should use `<tensor_name>_shape`, `dtypes`, and `label` according to `docs/design/manifest.md`.

Suggested workload dimensions:

- Llama-3.1-8B style: `H=32`, `H_kv=8`, `D=128`
- Llama-3.1-70B style: `H=64`, `H_kv=8`, `D=128`
- MHA case: `H == H_kv`
- MQA case: `H_kv == 1`
- short / medium / long KV, e.g. 4K, 32K, 128K
- page sizes selected for release, including 16 / 64 / 128 where practical
- fp16 and bf16

Roofline should account for:

- Q read
- old K/V cache reads
- output write
- `real_seqlen_kv` reads
- `block_table` reads for paged decode
- QK and ScoreV/PV FLOPs using visible per-slot lengths

## Validation Expectations

The corresponding PR should show:

- paged output correctness against materialized reference
- contiguous output correctness against PyTorch / materialized reference
- batch with heterogeneous `real_seqlen_kv`
- single-page and multi-page requests in one batch
- non-page-aligned `real_seqlen_kv`
- random page tail values do not affect output
- unused `block_table` entries are not read
- inactive slot behavior
- MHA / GQA / MQA head mapping
- invalid head topology validation
- invalid page id / capacity validation
- fp16 / bf16 coverage

## Non-Goals

- RoPE / softcap / score modifiers beyond existing baseline behavior; tracked separately.
- Sliding window; tracked separately.
- Sparse decode.
- Split-K.
- CUDA Graph replay validation beyond keeping the baseline metadata shape compatible.
- Append decode.
- FP8 KV cache.

## PR Notes

Implementation details should be discussed in the PR that references this issue.
Do not close {{PARENT}} from that PR.
