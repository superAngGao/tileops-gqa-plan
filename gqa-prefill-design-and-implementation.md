# GQA Prefill 接口设计与当前实现说明

日期：2026-05-06

本文面向参与 TileOPs attention / serving kernel 工作的同事，解释 GQA prefill 这组接口为什么这样拆、各自负责什么，以及当前实现如何落到 OP / kernel / manifest / benchmark 上。

更完整的能力调研见 `prefill.md`，发布收敛路线见 `prefill-plan.md`。本文偏工程讲解，不展开所有长期可能性。

## 1. 我们对 prefill operator family 的期待

我们希望 GQA prefill 不是一两个孤立 kernel，而是一组可以长期承载推理场景的 operator family。它至少要满足几个期待：

- 能覆盖从简单 prompt prefill 到 serving chunked prefill 的主要路径。
- 能表达不同输入布局，而不是只支持 padded dense tensor。
- 能和 KV cache 生命周期对接，包括 contiguous cache 和 paged cache。
- 能在 `q_len != kv_len` 时保持 mask、position 和 cache append 语义一致。
- 能支持现代模型常见的 GQA/MQA、RoPE、partial RoPE、softcap 等语义。
- 能进入 manifest / benchmark / nightly 统计体系，而不是只存在于某个单测或临时代码里。
- 能给后续 FP8 KV cache、Hopper dispatch、TMA/WS 优化留下清楚边界。

换句话说，我们期待的是一套 release-facing prefill contract：用户和 runtime 通过 OP 看到稳定的数据契约，具体 kernel 可以继续演进。

真实 serving 场景里，prefill 至少会遇到这些形态：

- prompt 一次性 prefill：`q_len == kv_len`
- chunked prefill：当前只处理一段新 token，`q_len < kv_len`
- prefix cache 命中：一部分 KV 已经在 cache 里，只对后续 chunk 做 attention 和 append
- batch 内请求长度不同：需要 packed / varlen 输入
- paged KV cache：serving runtime 管理 page，operator 只消费 `block_table`
- RoPE / softcap / scale 等模型语义要和 cache position 对齐

所以 prefill operator 不能只抽象成 `attention(q, k, v)`。它需要同时回答这些问题：

- 当前 Q/K/V 怎么组织
- 历史 KV 在哪里
- 当前调用是否 append KV
- causal mask 如何和 `q_len != kv_len` 对齐
- position / RoPE 是否使用绝对 cache position
- serving runtime 如何把 paged cache metadata 传给 kernel

## 2. 全部变体维度

为了避免后面讨论时把问题混在一起，先把 prefill 的能力空间拆成若干正交维度。这里的目标不是说每个维度都要立刻做满，而是先把“哪些变化会改变接口契约，哪些只是 kernel 实现策略”分清。

| 维度 | 要回答的问题 | 常见取值 / 本轮取舍 |
| --- | --- | --- |
| `head_topology` | Q heads 和 KV heads 如何对应 | MHA: `Hq == Hkv`；GQA: `Hq > Hkv`；MQA: `Hkv == 1` |
| `sequence_layout` | 当前调用的 Q/K/V 如何组织 | dense BSHD；packed THD + `cu_seqlens`；更动态的 ragged layout 暂不做 |
| `kv_layout` | 历史 KV 存在哪里、如何寻址 | no cache；contiguous KV cache；paged KV cache |
| `kv_update_contract` | 当前 prefill 是否写 cache | read-only；append current KV；in-place update 暂不做 |
| `q_vs_kv_length_relation` | Q 长度和可见 KV 长度是什么关系 | `q_len == kv_len`；`q_len < kv_len`；`q_len > kv_len` 当前不是主路径 |
| `mask_semantics` | 谁能看谁 | causal bottom-right；non-causal；sliding/window/block mask 后续单独设计 |
| `position_semantics` | position 如何和 cache 对齐 | no position；external RoPE；fused RoPE；ALiBi / YaRN / MRoPE 后续 |
| `rope_variant` | RoPE 具体形态 | Neox full RoPE；Neox partial RoPE；non-Neox legacy 不进 fused 主路径 |
| `score_modifiers` | QK score 进 softmax 前如何变换 | `sm_scale`；`softcap`；temperature/bias/mask extension 后续 |
| `numeric_format` | 输入、cache、accum、输出分别是什么 dtype | fp16/bf16 baseline；FP8 KV cache 下一阶段 |
| `outputs_and_stats` | 公开返回什么 | 当前 public OP output-only；`lse` 作为 kernel/internal stats |
| `batch_variability` | batch 内请求是否同长 | homogeneous dense；heterogeneous varlen/paged |
| `runtime_boundary` | 哪些属于 operator，哪些属于 serving runtime | OP 消费 metadata；page allocation / prefix sharing / eviction 属于 runtime |
| `benchmark_contract` | 如何让能力进入统计体系 | workload/benchmark 必须有稳定 label，不做 feature flag 笛卡尔积 |

这里有几个容易混淆的地方，需要提前分清：

- `sequence_layout` 和 `kv_layout` 不一样。前者说当前 Q/K/V 怎么传入，后者说历史 KV 怎么存。
- `kv_layout` 和 `kv_update_contract` 不一样。paged cache 可以只读，也可以 append；这是两个维度。
- `mask_semantics` 和 `position_semantics` 不一样。mask 决定可见性，position 决定 RoPE / ALiBi 等编码。
- `fuse_rope` 不是长期语义维度，它只是 `position_semantics=rope` 的一种实现路径。
- FP8 KV cache 不是“把整个 attention 变成 FP8”，首发更合理的是 low-precision cache storage + kernel 内 dequant。

有了这张表之后，接口设计就不应该从“我现在写了哪个 kernel”出发，而应该从“哪些维度组合需要成为稳定调用边界”出发。当前这轮工作的核心不是把上表全部做满，而是把最重要的 release-facing 组合固定下来：

- dense prefill
- packed varlen prefill
- contiguous cache prefill + append
- paged cache prefill + append
- causal bottom-right alignment
- fp16/bf16 baseline
- external RoPE 和 cache-aware fused RoPE
- Neox full/partial RoPE
- `sm_scale` / `softcap`
- manifest-backed benchmark 和可统计 workload 名称

## 3. 如何选择：优先、暂缓、舍弃

列出全部维度之后，下一步不是直接设计 OP，而是先做取舍。否则维度表会变成 feature 清单，最后自然滑向“大而全接口”。

我们的取舍依据主要有三类：

| 依据 | 看什么 | 对接口设计的影响 |
| --- | --- | --- |
| 最新主流模型 | 新模型真实使用的 attention 形态、position 形态、KV cache 压力 | 决定哪些语义必须优先支持，哪些不能只停留在 standalone op |
| 典型开源算子库 / serving runtime | FlashAttention、FlashInfer、cuDNN Frontend、vLLM 等公开接口已经把什么做成一等参数 | 决定哪些维度应该进入 release-facing OP contract，而不是藏在 benchmark 或 kernel 内部 |
| TileOPs 自己的设计逻辑 | manifest-first、OP contract 稳定、kernel dispatch 可替换、benchmark 可统计 | 决定功能如何分层，避免为了一个模型或一个 kernel 把公开接口写乱 |

这三类依据的角色不同。主流模型告诉我们“现实需求在哪里”，开源算子库告诉我们“行业接口已经怎么收敛”，TileOPs 设计逻辑告诉我们“哪些东西适合现在进公开契约”。

参考信号包括：

- Hugging Face Transformers 的 [Qwen3.5 config](https://huggingface.co/docs/transformers/model_doc/qwen3_5) 已经把 `num_attention_heads / num_key_value_heads / head_dim / use_cache / rope_parameters` 作为模型配置的一部分，说明 GQA、cache 和 RoPE 是模型级常规语义。
- [FlashInfer attention API](https://docs.flashinfer.ai/api/attention.html) 把 prefill/append attention、paged KV cache wrapper、ragged KV cache wrapper、`num_qo_heads / num_kv_heads`、`sm_scale`、`logits_soft_cap`、RoPE 参数等放进公开 plan/run 接口。
- [cuDNN Frontend SDPA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html) 暴露了 variable sequence length、bottom-right diagonal alignment、paged attention table、attention scale、softmax stats、bias/block mask/ALiBi/sliding-window 等参数。
- [vLLM 文档](https://docs.vllm.ai/en/v0.6.6/)把 PagedAttention、chunked prefill、prefix caching、FlashAttention/FlashInfer 集成和量化列为 serving 能力，这说明 paged KV 和 chunked prefill 不是边缘用法。

基于这些信号，我们把变体分成三类。

### 3.1 优先支持

这些是当前 prefill family 的主路径，应该进入 OP / manifest / tests / benchmark。

| 功能 | 为什么优先 |
| --- | --- |
| GQA/MQA/MHA 统一 head topology | 主流模型常见，接口上只需要 `heads / heads_kv`，收益大且边界清楚 |
| dense prefill | 基础 correctness 和对照路径 |
| packed varlen prefill | heterogeneous batch 是 serving 基础能力 |
| contiguous KV cache prefill + append | 单请求、本地推理、FP8 contiguous cache 的基线 |
| paged KV cache prefill + append | serving 主路径，和 vLLM / FlashInfer / cuDNN 方向一致 |
| `q_len <= kv_len` bottom-right causal | chunked prefill 和 prefix-hit 场景必需 |
| fp16 / bf16 | 当前 release baseline |
| external RoPE with absolute position ids | 语义最清楚、最容易和模型侧对齐 |
| cache-aware fused Neox RoPE | 现代模型常用，避免额外 torch RoPE 预处理 |
| partial RoPE `rotary_dim < head_dim` | Qwen3.5 这类新模型路径需要，且实现边界清楚 |
| `sm_scale` / `softcap` | 已是主流 attention interface 的一等 score modifier |
| named benchmark workloads | nightly 统计需要稳定名字，否则性能趋势不可比较 |

### 3.2 暂缓实现

这些功能重要，但当前不应该和 #1100 / #1101 / #1234 混在一个阶段里做。

| 功能 | 暂缓原因 |
| --- | --- |
| FP8 KV cache | 需要单独定义 storage dtype、scale 粒度、dequant / quantize 位置和误差边界 |
| per-head / per-token scale | 会扩大 metadata 和 kernel pipeline 复杂度，应在 FP8 baseline 后做 |
| `return_lse` public contract | kernel 已有 stats，但公开返回会影响所有 OP 心智，先保持 output-only |
| arbitrary bias / generic mask extension | 需要更通用的 mask/bias contract，不能只加零散参数 |
| sliding window / local chunk mask | 和 Llama4 等模型相关，但属于 mask 语义，不应塞进 RoPE PR |
| NoPE layer dispatch | 属于模型层 layer routing，不是单个 fused RoPE kernel 的职责 |
| QK norm | 属于 Q/K projection 后处理或模型 attention block 语义，应单独设计 |
| YaRN / MRoPE / Llama scaling | 是 RoPE 频率或多轴 position 语义，应独立设计 |
| TMA / WS / H200 dispatch 优化 | 是 kernel dispatch 和性能路线，不应改变公开 OP signature |

### 3.3 当前舍弃或不进入本轮主路径

这里的“舍弃”不是永久不能做，而是明确不作为当前 GQA prefill release contract 的主线。

| 功能 | 当前处理 |
| --- | --- |
| `q_len > kv_len` causal prefill | 非主流 serving prefill 形态，不作为当前目标 |
| GPT-J / non-Neox fused RoPE | standalone RoPE 可保留兼容，不进入 fused GQA prefill 主 benchmark |
| 完整 prefix cache runtime | page allocation、eviction、prefix sharing 属于 serving runtime，不属于 operator |
| page manager 对象封装 | 当前 OP 消费 `block_table`，不管理 page 生命周期 |
| FP8 Tensor Core attention compute | 当前 FP8 方向先做 KV cache storage + dequant，不承诺完整 FP8 compute |
| 任意模型级 attention block | TileOPs 先做 operator family，不把完整模型 attention block 塞进单个 OP |

这样分层以后，公开 OP 的设计就有了因果关系：优先支持项决定当前 OP family 的稳定契约；暂缓项进入后续 issue；舍弃项不污染当前接口。

## 4. 统一语义：head topology 和 causal alignment

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

## 5. Cache-aware prefill 的 KV append 契约

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

## 6. 从优先级和核心语义推导公开 OP

当前 release-facing GQA prefill family 保留四个公开入口：

| OP | 主要场景 | 输入布局 | KV cache |
| --- | --- | --- | --- |
| `GroupedQueryAttentionPrefillFwdOp` | dense prefill / 对照路径 | BSHD | none |
| `GroupedQueryAttentionPrefillVarlenFwdOp` | heterogeneous batch，无外部 cache | packed THD + `cu_seqlens_q/kv` | none |
| `GroupedQueryAttentionPrefillWithKVCacheFwdOp` | contiguous cache，单请求或本地推理对照 | BSHD current chunk | contiguous `[B, S_cap, Hkv, D]` |
| `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp` | serving 主路径 | packed THD current chunk | paged flat storage + `block_table` |

这个拆法的核心原则是：按调用边界上的数据契约拆 OP，而不是按 kernel 实现细节拆 OP。

从上面的优先级和核心语义看，这四个 OP 本质上固定了最重要的两个轴：

```text
sequence_layout x kv_layout
```

对应关系是：

| `sequence_layout` | `kv_layout` | 对应 OP |
| --- | --- | --- |
| dense BSHD | no cache | `GroupedQueryAttentionPrefillFwdOp` |
| packed THD | no cache | `GroupedQueryAttentionPrefillVarlenFwdOp` |
| dense BSHD current chunk | contiguous cache | `GroupedQueryAttentionPrefillWithKVCacheFwdOp` |
| packed THD current chunk | paged cache | `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp` |

优先支持的其他维度作为参数进入这些 OP：

- `head_topology` 由 `heads / heads_kv` 表达。
- `mask_semantics` 先由 `is_causal` 表达。
- `kv_update_contract` 由 cache-aware OP 的 append 契约表达。
- `score_modifiers` 由 `sm_scale` / `softcap` 表达。
- `position_semantics` 先由 external RoPE 或 `fuse_rope` / `rotary_dim` 表达。
- `numeric_format` 先由 `dtype` 表达，FP8 KV cache 作为暂缓项，后续扩展 cache storage contract。

因此我们不会暴露这些用户心智：

- `GroupedQueryAttentionPrefillWgmmaFwdOp`
- `GroupedQueryAttentionPrefillSmallSeqFwdOp`
- `GroupedQueryAttentionPrefillRopeAppendKernelOp`

这些只应该是 kernel class 或 dispatch target。用户和 manifest 关心的是“我传什么 layout、cache 怎么表达、输出是什么”。

## 7. Position / RoPE：先定语义，再定实现

RoPE 的核心问题不是“有没有 fuse”，而是 position 语义是否和 prefill / cache append 对齐。尤其是 chunked prefill 里，current chunk 的第一个 token 不等于整段序列的第一个 token；如果 position reset，output 和写入 cache 的 K 都会错。

因此我们先固定 cache-aware prefill 的 position 契约：

| 对象 | position 语义 |
| --- | --- |
| old cache K | 已经按 logical position `0..old_len-1` 完成 RoPE，当前调用不能重复旋转 |
| current Q | 使用 `old_len + local_i` |
| current K | 使用 `old_len + local_i` |
| append 到 cache/page 的 K | 必须是按 absolute position 旋转后的 K |

基于这个契约，当前支持两条实现路径。

| 路径 | 谁负责 RoPE | 适用场景 |
| --- | --- | --- |
| external RoPE | 调用方或 standalone RoPE OP 先旋转 `q/k_new` | 最稳定、最显式，适合验证 position 语义 |
| fused RoPE | cache-aware GQA prefill OP 内部生成 cos/sin 并旋转 current chunk | 避免额外 torch 预处理，更接近 serving kernel path |

这两条路径的语义必须一致。区别只是 RoPE 在 OP 外部完成，还是在 TileLang path 内完成。

当前 fused RoPE 的首发参数边界是：

```python
fuse_rope=True
rope_base=10000.0
max_position=...
rotary_dim=None  # None means full head_dim
```

其中 `rotary_dim` 是 partial RoPE 的关键接口：

- `None` 等价于 `head_dim`
- 必须是正偶数
- `rotary_dim <= head_dim`
- 只旋转前 `rotary_dim` 维
- `d >= rotary_dim` 的尾部维度保持原样

这个边界覆盖：

- Llama 3.x style full Neox RoPE
- Qwen3.5 full-attention layer 的 partial RoPE，例如 `head_dim=256, rotary_dim=64`

这些暂不进入本轮 fused GQA prefill 主路径：

- GPT-J / non-Neox adjacent-pair fused path
- YaRN / MRoPE / Llama scaling
- Llama4 NoPE layer dispatch
- Llama4 local chunk mask
- QK norm

这些应该单独设计，避免把模型级 attention 语义混到一个 RoPE PR 中。

## 8. Fused RoPE 的当前实现方法

上一节定义的是语义。这一节讲当前实现：cache-aware fused RoPE 在 OP 层组合两个 TileLang kernel，而不是把 append 分支塞进 attention kernel。

```text
OP forward:
    cos, sin = get_rope_tables(...)
    append_kernel(k_new, v_new, cache/pages, positions, cos, sin)
    output = attention_kernel(q, k_new, v_new, cache/pages, positions, cos, sin)
```

这样做不是把 RoPE “unfuse” 了。`fuse_rope=True` 的含义是 RoPE 不需要外部 torch preprocessing，而是在 TileLang cache-aware path 内完成。它不要求 append 和 attention 必须在同一个物理 kernel launch 里。

拆成两个 kernel 的原因是 append 和 attention 的天然并行维度不同：

| 动作 | 天然 dispatch 维度 |
| --- | --- |
| append `k_new/v_new` 到 cache/page | `heads_kv`，每个 KV head 写一次 |
| 计算 attention output | `heads`，每个 query head 都要算 output |

如果把 append 写进 attention kernel，attention grid 通常按 query head `heads` 开。GQA 下 `heads > heads_kv`，kernel 内就不得不出现类似逻辑：

```text
if query_head maps to this kv_head:
    append kv
```

这种写法容易产生两个问题：

- append 语义被绑到 query-head CTA，代码不自然。
- `heads / heads_kv` 改变时，重复写或漏写 KV 的风险变高。

现在的实现边界更清楚：

- append kernel 按 `heads_kv` dispatch，负责写 cache/page。
- attention kernel 按 `heads` dispatch，负责算 output。
- attention kernel 不 mutation cache/page tensor。
- attention 当前 chunk 仍然直接从 `k_new/v_new` 读，不依赖刚 append 到 cache/page 的 current chunk。
- old cache/page K 视为已经 rotated，只读不重写。

也就是说，OP 层表达 release-facing 语义，kernel 层保持单一职责：append kernel 做 KV materialization，attention kernel 做 attention compute。

## 9. Score modifiers：先支持稳定的一等语义

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

## 10. 当前代码落点

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

## 11. 当前实现矩阵

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

## 12. 后续优先级

当前 #1100 / #1101 / #1234 这轮收敛后，建议优先级是：

1. 确认 PR CI、review 和 nightly benchmark 全部闭环。
2. 设计 FP8 KV cache dequant path 的 manifest-ready 接口。
3. 做 contiguous FP8 KV cache read + append。
4. 做 paged FP8 KV cache read + append。
5. 收集 manifest-backed nightly benchmark 趋势。
6. 再进入 H200 / Hopper dispatch、TMA / WS-friendly 优化。
7. 低优先级再讨论 `return_lse` / stats 公开契约。

一个重要原则：任何新增 release-facing OP 参数或语义，都必须在同一个 PR 里同步更新 manifest、tests、workloads 和 benchmark。不要先合实现，再让 manifest 和统计系统慢慢追。

## 附录 A. Benchmark 选择逻辑

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
