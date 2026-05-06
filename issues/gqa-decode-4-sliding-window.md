Parent: {{PARENT}}

Related feature issues: {{RELATED}}

## Scope

Track sliding-window support for dense paged GQA decode.

This issue is intentionally separate from scale / softcap / RoPE. Sliding-window decode changes the visible logical KV range and needs dedicated page-boundary, tail-mask, and heterogeneous-length validation.

## Public Feature

Applies to:

- `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp`
- `GroupedQueryAttentionDecodeWithKVCacheFwdOp` if the contiguous fallback is updated in the same PR

Public knob:

- `window_left: int = -1` or the final name that matches existing GQA prefill sliding-window conventions

Semantics:

- `window_left < 0` disables sliding window and preserves dense full-prefix behavior.
- `window_left >= 0` restricts visible keys by logical position:

```text
visible(q_b, k_j) =
    0 <= j < real_seqlen_kv[b]
    and j >= max(0, real_seqlen_kv[b] - window_left)
```

- Semantics are based on logical KV position, not physical page position.
- Page tail remains masked.
- Unused `block_table` entries remain unread.

## Manifest Change Plan

This issue requires manifest changes if the PR adds a public sliding-window parameter to decode OPs.

Required manifest updates:

- Add `window_left` or the chosen param name to affected decode entries.
- Update workloads to include sliding-window decode cases.
- Keep sliding window under the same public decode OP entry unless a separate public variant is explicitly justified.
- Update `source.kernel_map` if sliding-window kernels are separate manifest-visible dispatch targets.
- Update roofline only if benchmark reporting uses effective window length rather than full visible prefix length.

Affected entries:

- `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp`
- `GroupedQueryAttentionDecodeWithKVCacheFwdOp` if included

## Manifest / Interface Notes

- Do not model sliding window as general sparse decode.
- Do not introduce arbitrary block masks in this issue.
- Do not use physical page ids to define visibility.
- Prefer naming aligned with existing GQA prefill sliding-window params unless there is a strong reason to use a decode-specific name.

## Workloads / Roofline

Workloads should cover:

- short and medium context windows
- window crossing page boundaries
- `window_left = 0`
- `window_left` smaller than, equal to, and larger than visible KV length
- heterogeneous `real_seqlen_kv`
- fp16 and bf16

Roofline options:

- use effective window length for FLOPs / bytes if the workload protocol can express it clearly; or
- reuse full-prefix decode roofline and document that sliding-window roofline refinement is deferred.

## Validation Expectations

The corresponding PR should show:

- materialized reference parity for paged decode
- logical-position window semantics
- window across page boundaries
- page tail masking
- unused page entries not read
- heterogeneous `real_seqlen_kv`
- `window_left < 0` preserves dense full-prefix behavior
- `window_left = 0` boundary behavior
- fp16 / bf16 coverage

## Non-Goals

- Sink attention.
- Arbitrary sparse decode.
- Custom block masks.
- Append decode.
- FP8 KV cache.
- CUDA Graph runtime capture.

## PR Notes

Implementation details should be discussed in the PR that references this issue.
Do not close {{PARENT}} from that PR.
