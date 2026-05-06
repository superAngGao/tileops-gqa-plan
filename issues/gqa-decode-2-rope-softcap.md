Parent: {{PARENT}}

Related feature issues: {{RELATED}}

## Scope

Track dense GQA decode score modifiers and fused Q RoPE.

This issue adds the model-level attention semantics needed by serving dense decode and aligns them with GQA prefill where applicable. It covers `sm_scale`, logits softcap, fused Q RoPE, external RoPE reference coverage, and per-slot `cache_position [B]`.

## Included Features

### Score Modifier: `sm_scale`

Public knob:

- `sm_scale: Optional[float] = None`

Semantics:

- `None` uses the default `1 / sqrt(head_dim)`.
- Explicit `sm_scale` overrides the default.
- The scale is applied to QK scores before softcap / softmax, matching the GQA prefill contract.

### Score Modifier: Softcap

Public knob:

- `softcap: Optional[float] = None`

Semantics:

- `softcap=None` or `softcap=0` preserves existing behavior.
- `softcap > 0` applies logits soft capping before softmax:

```text
softcap * tanh(score / softcap)
```

- negative values are rejected by OP runtime validation.

### External RoPE Reference Coverage

For external RoPE, the caller rotates Q before calling the decode attention OP.

The decode OP then treats the supplied tensors as already position-encoded:

- Q is already rotated at the current query position
- old cache K is already rotated
- old cache K must not be rotated again
- V is never RoPE-rotated

### Fused Q RoPE

Applies to read-only dense decode:

- `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp`
- `GroupedQueryAttentionDecodeWithKVCacheFwdOp` if contiguous fallback supports the same feature

Public knobs:

- `fuse_rope: bool = False`
- `max_position: Optional[int] = None`
- `rope_base: float = 10000.0`
- `cache_position [B]` as a tensor input if fused Q RoPE is enabled by the public contract

Contract:

- input Q may be unrotated
- decode OP rotates Q using `cache_position[b]`
- old cache K is assumed to already be rotated
- old cache K must not be rotated again
- V is never RoPE-rotated
- heterogeneous `cache_position [B]` must be supported
- nonzero positions / prefix-hit cases must be covered

## Manifest Change Plan

This issue requires manifest changes if the PR adds `sm_scale`, `softcap`, `fuse_rope`, `rope_base`, `max_position`, `cache_position`, or fused-RoPE kernel map entries to public OP contracts.

Required manifest updates for affected entries:

- Add `sm_scale: {type: float, default: null}`.
- Add `softcap: {type: float, default: null}`.
- Add `fuse_rope: {type: bool, default: false}` if fused Q RoPE is public.
- Add `rope_base: {type: float, default: 10000.0}` when `fuse_rope` is present.
- Add `max_position: {type: "int | None", default: null}` when `fuse_rope` is present.
- Add `cache_position` tensor input if fused Q RoPE requires per-slot positions in the public forward contract.
- Add fused-Q-RoPE kernel classes to `source.kernel_map` for the same public OP entry rather than creating separate public `...Rope...Op` entries.
- Update workloads for scale / softcap / fused-RoPE coverage included by the PR.
- Update roofline only if release benchmark reporting accounts for extra softcap / RoPE math; otherwise document why the existing decode roofline is reused.

Affected entries:

- `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp`
- `GroupedQueryAttentionDecodeWithKVCacheFwdOp` if the contiguous fallback is updated in the same PR

## Manifest / Interface Notes

- Do not create separate user-visible `...Rope...Decode...Op` entries.
- `softcap` is a stable score modifier, not an implementation detail.
- `fuse_rope` is a performance/implementation knob for fused Q position handling; the semantic contract is still RoPE position semantics.
- No optional tensor inputs in manifest. If `cache_position` is required only for fused RoPE, either make it part of the selected variant contract or justify the runtime validation approach in the PR.
- `max_position` conditional required behavior for `fuse_rope=True` is OP runtime validation.

## Workloads / Roofline

Workloads should cover:

- baseline paged decode with explicit `sm_scale`
- baseline paged decode with softcap enabled
- fused Q RoPE with nonzero `cache_position`
- fused Q RoPE + softcap combination if both are supported in the same kernel path
- fp16 and bf16
- at least one Llama-3.1-8B style and one 70B style shape

Roofline should either:

- include the extra RoPE / softcap math if the benchmark report wants exact accounting; or
- explicitly reuse the existing decode roofline and state that RoPE / softcap overhead is ignored for this release metric.

## Validation Expectations

The corresponding PR should show:

- `sm_scale` reference parity
- softcap reference parity
- `softcap=None` and `softcap=0` preserve default behavior
- invalid `softcap < 0` validation
- external RoPE absolute-position regression
- fused Q RoPE correctness against external-RoPE reference
- heterogeneous `cache_position [B]`
- nonzero `cache_position` / prefix-hit case
- old cache K is not re-rotated
- paged dense decode main path coverage
- fp16 / bf16 coverage

## Non-Goals

- Sliding window; tracked separately.
- Append decode.
- Rotating current `k_new`.
- Rotating old cached K.
- Sparse decode.
- FP8 KV cache.
- FP8 Tensor Core decode.

## PR Notes

Implementation details should be discussed in the PR that references this issue.
Do not close {{PARENT}} from that PR.
