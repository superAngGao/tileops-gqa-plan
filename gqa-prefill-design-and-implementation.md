# GQA Prefill 接口设计与当前实现说明

日期：2026-05-06

本文面向参与 TileOPs attention / serving kernel 工作的同事，解释 GQA prefill 这组接口为什么这样拆、各自负责什么，以及当前实现如何落到 OP / kernel / manifest / benchmark 上。

更完整的能力调研见 `prefill.md`，发布收敛路线见 `prefill-plan.md`。本文偏工程讲解，不展开所有长期可能性。

## 1. 我们解决的是什么问题

推理里的 prefill 不只是一次普通 attention。真实 serving 场景里常见的是：

- prompt 一次性 prefill：`q_len == kv_len`
- chunked prefill：当前只处理一段新 token，`q_len < kv_len`
- prefix cache 命中：一部分 KV 已经在 cache 里，只对后续 chunk 做 attention 和 append
- batch 内请求长度不同：需要 packed / varlen 输入
- paged KV cache：serving runtime 管理 page，operator 只消费 `block_table`
- RoPE / softcap / scale 等模型语义要和 cache position 对齐

所以 prefill operator 不能只抽象成 `attention(q, k, v)`。它需要同时表达：

- 当前 Q/K/V 怎么组织
- 历史 KV 在哪里
- 当前调用是否 append KV
- causal mask 如何和 `q_len != kv_len` 对齐
- position / RoPE 是否使用绝对 cache position
- serving runtime 如何把 paged cache metadata 传给 kernel

## 2. 为什么拆成四个公开 OP

当前 release-facing GQA prefill family 保留四个公开入口：

| OP | 主要场景 | 输入布局 | KV cache |
| --- | --- | --- | --- |
| `GroupedQueryAttentionPrefillFwdOp` | dense prefill / 对照路径 | BSHD | none |
| `GroupedQueryAttentionPrefillVarlenFwdOp` | heterogeneous batch，无外部 cache | packed THD + `cu_seqlens_q/kv` | none |
| `GroupedQueryAttentionPrefillWithKVCacheFwdOp` | contiguous cache，单请求或本地推理对照 | BSHD current chunk | contiguous `[B, S_cap, Hkv, D]` |
| `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp` | serving 主路径 | packed THD current chunk | paged flat storage + `block_table` |

这个拆法的核心原则是：按调用边界上的数据契约拆 OP，而不是按 kernel 实现细节拆 OP。

因此我们不会暴露这些用户心智：

- `GroupedQueryAttentionPrefillWgmmaFwdOp`
- `GroupedQueryAttentionPrefillSmallSeqFwdOp`
- `GroupedQueryAttentionPrefillRopeAppendKernelOp`

这些只应该是 kernel class 或 dispatch target。用户和 manifest 关心的是“我传什么 layout、cache 怎么表达、输出是什么”。

## 3. 统一语义：head topology 和 causal alignment

GQA / MQA / MHA 都用同一组参数表达：

```text
heads = Hq
heads_kv = Hkv
groups = heads / heads_kv
```

要求：

```text
heads % heads_kv == 0
```

当 `heads == heads_kv` 时就是 MHA；当 `heads_kv == 1` 时就是 MQA。

对于 `q_len != kv_len` 的 causal prefill，我们使用 bottom-right causal alignment：

```text
visible(q_i, k_j) = j <= i + (kv_len - q_len)
```

cache-aware prefill 中，`kv_len` 是 `old_len + current_chunk_len`，所以：

```text
visible(q_i, kv_j) = j <= old_len + i
```

这保证 chunked prefill 不会把 position reset 到 0，也不会把当前 chunk 的第一个 token 当成整段序列的第一个 token。

## 4. Cache-aware prefill 的 KV append 契约

contiguous cache path 的输入是：

```text
q              [B, Snew, Hq, D]
k_new/v_new    [B, Snew, Hkv, D]
k_cache/v_cache[B, S_cap, Hkv, D]
cache_seqlens  [B]
```

`cache_seqlens[b]` 表示 append 前已有 KV 长度：

```text
old_len = cache_seqlens[b]
new token i writes cache position old_len + i
```

paged cache path 的输入是：

```text
q/k_new/v_new  [Tnew, H, D] / [Tnew, Hkv, D]
k_pages/v_pages[P_tokens, Hkv, D]
cu_seqlens_q   [B + 1]
cache_seqlens  [B]
block_table    [B, max_pages_per_req]
```

logical position 到 physical position 的映射：

```text
logical_pos = old_len + local_i
logical_page = logical_pos // page_size
page_offset = logical_pos % page_size
physical_page = block_table[b, logical_page]
physical_token = physical_page * page_size + page_offset
```

operator 只消费 `block_table`，不负责 page allocation、prefix sharing、eviction 或 cache manager 生命周期。

## 5. RoPE 设计：position 语义比 fuse 方式更重要

RoPE 的核心问题不是“有没有 fuse”，而是 current chunk 是否使用了正确的 absolute position。

cache-aware prefill 中约定：

- old cache K 已经按 logical position 完成 RoPE，当前调用不能重复旋转 old cache。
- current chunk 的 `q` 和 `k_new` 使用 `old_len + local_i` 作为 position。
- append 写入 cache/page 的 K 必须是 rotated K。

当前支持两条路径：

1. 外置 RoPE
   - 调用方先用 standalone RoPE op 或别的实现旋转好 `q/k_new`。
   - GQA prefill OP 只消费已经 rotated 的输入。

2. fused RoPE
   - OP 内部生成 cos/sin table。
   - TileLang path 内完成 current chunk 的 Q/K rotation 和 append。

当前 fused RoPE 首发边界：

```python
fuse_rope=True
rope_base=10000.0
max_position=...
rotary_dim=None  # None means full head_dim
```

`rotary_dim` 规则：

- `None` 等价于 `head_dim`
- 必须是正偶数
- `rotary_dim <= head_dim`
- 只旋转前 `rotary_dim` 维
- `d >= rotary_dim` 的尾部维度保持原样

这个边界覆盖：

- Llama 3.x style full Neox RoPE
- Qwen3.5 full-attention layer 的 partial RoPE，例如 `head_dim=256, rotary_dim=64`

暂不在这轮 fused GQA prefill 里做：

- GPT-J / non-Neox adjacent-pair fused path
- YaRN / MRoPE / Llama scaling
- Llama4 NoPE layer dispatch
- Llama4 local chunk mask
- QK norm

这些应该单独设计，避免把模型级 attention 语义混到一个 RoPE PR 中。

## 6. 为什么 fused RoPE append 拆成两个 kernel

PR #1234 review 后，我们把 fused RoPE cache append 从 attention kernel 内拆出来，由 OP 层编排：

```text
OP forward:
    cos, sin = get_rope_tables(...)
    append_kernel(...)
    output = attention_kernel(...)
```

这样做的原因是 GQA 下 append 和 attention 的天然 dispatch 维度不同：

| 动作 | 天然 dispatch 维度 |
| --- | --- |
| append `k_new/v_new` 到 cache/page | `heads_kv` |
| 计算 attention output | `heads` |

如果把 append 写在 attention kernel 里，attention kernel 的 grid 通常是按 query head `heads` 开的。GQA 下 `heads > heads_kv`，这会迫使 kernel 内出现类似下面的分支：

```text
if query_head maps to this kv_head:
    append kv
```

这种写法容易产生两个问题：

- append 语义被绑到 query-head CTA，代码不自然。
- `heads / heads_kv` 改变时，重复写或漏写 KV 的风险变高。

现在的实现是：

- append kernel 按 `heads_kv` dispatch，负责写 cache/page。
- attention kernel 按 `heads` dispatch，负责算 output。
- attention kernel 不 mutation cache/page tensor。
- attention 当前 chunk 仍然直接从 `k_new/v_new` 读，不依赖刚 append 到 cache 的 current chunk。

所以这仍然是 fused RoPE OP：RoPE 没有退回外部 torch 预处理，只是 OP 内部用了两个 TileLang kernel launch。

## 7. Score modifiers：先支持稳定的一等语义

当前已经支持：

- `sm_scale`
- `softcap`

默认：

```text
sm_scale=None -> 1 / sqrt(head_dim)
softcap=None or 0 -> disabled
softcap>0 -> softcap * tanh(score / softcap)
```

`softcap` 是 score modifier，不是 RoPE 语义。benchmark 中不应该把 softcap 展开成完整矩阵；它更适合做少量 sentinel case，确认路径可编译、可统计、数值正确。

暂不优先做：

- temperature
- arbitrary bias
- arbitrary block mask
- return_lse 公开接口

`lse` 当前更适合作为 kernel/internal stats。公开 OP 默认保持 output-only，避免用户把 stats 当成所有 prefill path 的默认契约。

## 8. Benchmark 选择逻辑

benchmark 的目标不是 feature flag 笛卡尔积，而是代表真实推理场景。

当前建议主轴：

- paged KV cache 是 serving 主路径。
- contiguous KV cache 是单请求、本地推理或对照路径。
- partial RoPE 是现代模型必须覆盖的能力。
- softcap 只做少量 sentinel。
- benchmark id 必须稳定、可统计。

当前关键 benchmark 名称：

| 名称 | 目的 |
| --- | --- |
| `qwen35-9b-prefill-paged-fullattn-b8-prefix32k-chunk1k-p64-partial-rope64-fp16` | Qwen3.5 style paged serving 主路径 |
| `qwen35-9b-prefill-paged-fullattn-mixed-b8-p64-partial-rope64-fp16` | batch 内 prefix/chunk 长度不同 |
| `qwen35-9b-prefill-contig-fullattn-prefix32k-chunk1k-partial-rope64-fp16` | contiguous cache 对照 |
| `llama31-8b-prefill-paged-b8-prefix4k-chunk512-p64-full-rope-fp16` | Llama full RoPE anchor |
| `gqa-prefill-paged-softcap50-b4-prefix4k-chunk512-p64-fp16` | softcap sentinel |

bf16 correctness 放在 tests 里覆盖，benchmark 先以 fp16 控制 nightly 编译矩阵。

## 9. 当前代码落点

当前实现主要分布在：

| 层 | 文件 | 内容 |
| --- | --- | --- |
| OP | `tileops/ops/attention/gqa.py` | 四个 GQA prefill OP，输入校验，kernel dispatch，fused RoPE append + attention 编排 |
| kernel | `tileops/kernels/attention/gqa_fwd.py` | dense prefill、contiguous cache、paged cache、fused RoPE attention、RoPE append kernels |
| varlen kernel | `tileops/kernels/attention/gqa_prefill_varlen_fwd.py` | packed varlen prefill |
| RoPE OP/kernel | `tileops/ops/rope.py`, `tileops/kernels/rope.py` | standalone RoPE，包括 position_ids path |
| manifest | `tileops/manifest/attention.yaml` | release-facing signatures、shape rules、workloads、kernel map |
| workloads | `workloads/attention/gqa_prefill.py` | benchmark/test workload generators |
| tests | `tests/ops/attention/test_gqa.py`, `tests/ops/attention/test_gqa_prefill_paged.py`, `tests/ops/test_rope.py` | correctness regression |
| benchmarks | `benchmarks/ops/attention/bench_gqa.py` | named benchmark cases |

## 10. 当前实现矩阵

| 能力 | 状态 |
| --- | --- |
| dense prefill | 已支持 |
| dense `q_len < kv_len` bottom-right causal | 已支持 |
| packed varlen prefill | 已支持 |
| contiguous cache prefill + append | 已支持 |
| paged cache prefill + append | 已支持 |
| fused RoPE contiguous cache | 已支持 |
| fused RoPE paged cache | 已支持 |
| partial RoPE `rotary_dim < head_dim` | 已支持 |
| external RoPE with position_ids | 已支持 |
| softcap | 已支持 |
| output-only public OP | 已保持 |
| public `return_lse` | 暂不暴露 |
| FP8 KV cache | 下一阶段 |
| Llama4 chunk mask / NoPE / QK norm | 后续单独设计 |

## 11. 后续优先级

当前 #1100 / #1101 / #1234 这轮收敛后，建议优先级是：

1. 确认 PR CI、review 和 nightly benchmark 全部闭环。
2. 设计 FP8 KV cache dequant path 的 manifest-ready 接口。
3. 做 contiguous FP8 KV cache read + append。
4. 做 paged FP8 KV cache read + append。
5. 收集 manifest-backed nightly benchmark 趋势。
6. 再进入 H200 / Hopper dispatch、TMA / WS-friendly 优化。
7. 低优先级再讨论 `return_lse` / stats 公开契约。

一个重要原则：任何新增 release-facing OP 参数或语义，都必须在同一个 PR 里同步更新 manifest、tests、workloads 和 benchmark。不要先合实现，再让 manifest 和统计系统慢慢追。

