Parent: {{PARENT}}

Related feature issues: {{RELATED}}

## Scope

Track internal split-K for long-context dense decode and the CUDA Graph-friendly dispatch / metadata contract.

This issue is not a public API expansion. It should keep the public decode OPs output-only and should not introduce a public `SplitK` or `CudaGraph` OP. The work belongs behind OP-layer dispatch and manifest-visible kernel maps only where new kernel classes are added.

## Included Features

### Internal Split-K

Purpose:

- Keep a no-split fast path for short / medium contexts.
- Use split-K for long contexts where one `(slot, q_head)` has too much KV work for one CTA / program.
- Split paged long context by page or page group where practical.
- Merge partial results with online softmax stats.

Internal workspace shape sketch:

```text
partial_m:   [B, H, max_splits]
partial_lse: [B, H, max_splits]
partial_o:   [B, H, max_splits, D]
```

Merge contract:

```text
m = max_i(partial_m_i)
lse = log(sum_i exp(partial_lse_i + partial_m_i - m))
o = sum_i exp(partial_lse_i + partial_m_i - m - lse) * partial_o_i
```

### CUDA Graph-Friendly Contract

Graph-friendly decode means the runtime can capture fixed bucket buffers and replay without changing tensor shape, pointer, dispatch target, or launch topology.

Static bucket / dispatch items:

- `B_bucket`
- `heads`
- `heads_kv`
- `dim`
- dtype
- layout
- `page_size`
- `max_pages_per_req`
- split policy
- `max_splits`
- target arch / kernel dispatch key

Dynamic tensor contents:

- Q contents
- KV cache contents
- `real_seqlen_kv` contents
- `block_table` contents
- `cache_position` contents when present
- active slot mapping when maintained by runtime

Replay must not choose a different kernel based on the runtime contents of `real_seqlen_kv`.

## Manifest Change Plan

Manifest changes are required when this issue adds manifest-visible kernel dispatch targets, workloads, benchmark coverage, or source metadata.

Required manifest work:

- Update `source.kernel_map` for `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp` if split-K kernels are added as manifest-visible dispatch targets.
- Keep split-K as implementation dispatch under the same public OP entry.
- Do not create `GroupedQueryAttentionDecodeSplitKFwdOp`.
- Add long-context workloads that exercise the split-K path.
- Add or update roofline binding if split-K benchmark reporting requires different metadata.
- Update source metadata and benchmark path if new files are introduced.

Manifest changes are not required when:

- only internal thresholds change and public OP contract, workload coverage, roofline binding, and manifest-visible kernel map remain unchanged.

## Manifest / Interface Notes

- Split-K is not a user parameter unless a PR explicitly proposes a debug-only or benchmark-only knob and justifies why it belongs in the public OP contract.
- CUDA Graph is not a public OP variant.
- Runtime graph capture owns buckets; OP/kernel code owns stable shape, workspace, dispatch, and launch topology under a bucket.
- If fixed split policy requires construction-time params such as `max_splits`, encode them only if they are part of the stable OP / manifest contract.

## Workloads / Roofline

Workloads should cover:

- no-split serving shapes
- long-context split-K shapes
- page sizes used by release paged decode
- mixed GQA ratios
- fp16 and bf16

Benchmark metadata should make it possible to tell whether a workload used no-split or split-K dispatch.

Roofline should continue to account for dense visible KV work. If split-K adds extra workspace traffic that is meaningful for release reporting, either include it in the formula or document why it is ignored.

## Validation Expectations

The corresponding PR should show:

- split-K output matches materialized reference
- split-K and no-split outputs match within dtype tolerance
- split boundaries not aligned to page boundaries where applicable
- page tail masking remains correct
- heterogeneous `real_seqlen_kv`
- fixed `max_splits` / invalid split identity stats behavior for graph bucket
- public API remains output-only
- no public SplitK OP added
- long-context benchmark evidence
- static vs dynamic metadata contract documented in code comments, docs, or issue-linked PR description

## Non-Goals

- Public `return_lse` API.
- Public deterministic / bitwise batch-size invariant guarantee.
- Full runtime CUDA Graph capture implementation.
- Serving scheduler, page allocator, or prefix cache runtime.
- Sparse decode.
- Sliding window.
- FP8 KV cache.

## PR Notes

Implementation details should be discussed in the PR that references this issue.
Do not close {{PARENT}} from that PR.
