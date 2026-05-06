# GQA Decode 规划

日期：2026-05-04

本文只讨论 `decode`，不讨论 `prefill`，也不讨论完整 serving scheduler、采样器或请求生命周期管理。

本文是 GQA decode 的能力分层与接口调研；具体落地顺序和当前基线收敛计划见 `decode-plan.md`。

这里的 `decode` 包括：
- 单 token decode，通常每个 request 当前只有一个 query token
- 小 `q_len` decode，例如 speculative decode、multi-token decode 或 verification
- 从已有 KV cache 读取历史上下文并计算当前输出
- 可选地在同一调用中把当前 token 的 K/V 写入 cache
- contiguous KV cache 和 paged KV cache 两类主流 cache 组织

不在本文范围内的内容：
- prompt prefill / chunked prefill
- prefix cache 命中、page allocation / eviction / reuse 策略
- continuous batching scheduler
- sampling、beam search、draft model 选择策略
- 某个具体项目的完整差距分析

## 一、decode 支持维度

我们把 inference 场景下 GQA decode 的能力拆成下面这些维度。

| 维度 | 含义 | 典型取值 |
| --- | --- | --- |
| `decode_unit` | 一次 decode 调用里每个 request 的 query token 数 | `single_token`、`multi_token`、`speculative_verify` |
| `query_layout` | Q 输入在调用边界上的组织方式 | `bhd`、`bshd_s1`、`packed_thd_small_q` |
| `kv_layout` | KV cache 的存储和寻址方式 | `contiguous_kv`、`paged_kv`、`prefix_shared_kv`、`compressed_kv` |
| `kv_length_metadata` | 每个 request 当前可见 KV 长度如何表达 | `scalar_len`、`per_batch_seqlen`、`cache_seqlens`、`page_last_len` |
| `page_table_format` | paged KV 的 logical-to-physical 映射格式 | `block_table`、`indptr_indices_last_page_len` |
| `head_topology` | query head 和 KV head 的对应关系 | `mha`、`gqa`、`mqa` |
| `mask_semantics` | decode 时的注意力可见性语义 | `full_prefix`、`sliding_window`、`sink_attention`、`custom_mask` |
| `position_semantics` | 位置编码及跨步位置对齐语义 | `none`、`rope`、`alibi`、`cache_position` |
| `kv_update_contract` | 本次 decode 是否写入 KV cache | `read_only`、`append_current_kv`、`external_append` |
| `split_k_strategy` | 长 KV 上如何拆分和合并 | `no_split`、`split_k`、`fixed_split`、`deterministic_merge` |
| `cuda_graph_contract` | 是否支持被 decode CUDA Graph 捕获 | `eager_only`、`graph_replay_static_batch`、`graph_replay_static_buffers` |
| `score_modifiers` | 点积后、softmax 前对 logits 的额外变换 | `scale`、`softcap`、`temperature`、`sink_bias` |
| `numeric_format` | 计算与 cache 存储使用的数值格式 | `fp16`、`bf16`、`fp8_kv`、`int8_kv`、`nvfp4_kv` |
| `outputs_and_stats` | 除输出张量外还返回什么 | `output_only`、`output_plus_lse`、`internal_lse_for_split` |
| `batch_variability` | batch 内不同请求是否可以有不同 KV 长度和 page 数 | `homogeneous_batch`、`heterogeneous_batch` |

### 补充说明

这些维度尽量保持正交，不建议混在一起讨论。

尤其要注意下面几组不要混淆：
- `decode_unit` 和 `kv_length_metadata` 不是一回事  
  前者说当前有几个 query token，后者说历史 KV 有多长。
- `kv_layout` 和 `page_table_format` 不是一回事  
  前者说 cache 是否 paged，后者说 paged metadata 的具体表示。
- `kv_update_contract` 和 `kv_layout` 不是一回事  
  一个是“这次调用会不会写 cache”，一个是“cache 本身怎么存”。
- `split_k_strategy` 和 `page_table_format` 不是一回事  
  split-K 是计算分工和合并策略，page table 是 cache 寻址策略。
- `cuda_graph_contract` 和 `kernel dispatch` 不是一回事  
  CUDA Graph 关心 replay 时 shape、buffer 和 launch 拓扑是否稳定；kernel dispatch 关心某次调用选择哪个实现。

### decode 与 prefill 的关键差异

prefill 的核心挑战是长 Q 与长 KV 的矩阵 attention，以及 `q_len` / `kv_len` 的对齐语义。

decode 的核心挑战更偏 serving：
- Q 很短，通常是 `[B, Hq, D]`
- KV 很长，通常来自 cache
- batch 内每个 request 的 KV 长度可能不同
- paged KV 是主流 runtime 的自然形态
- decode 常被 CUDA Graph 捕获，要求 replay 形状和 buffer 稳定
- 长上下文下 split-K / split-KV 合并会直接影响性能、确定性和 graph 兼容性
- FP8 / INT8 / NVFP4 KV cache 的收益在 decode 阶段更直接，因为 decode 往往更受 KV 读取带宽限制

因此 decode 规划不能只把 prefill 的 `q_len=1` 特例拿过来；它需要把 cache metadata、split-K、CUDA Graph、quantized KV cache 和 runtime-facing page table 作为一等设计项。

## 二、主流算子库与框架调研：decode 支持能力与接口形态

上面第一章给出的是抽象维度。为了避免维度设计停留在概念层，这一章直接看主流算子库和推理框架的官方接口，反推它们真正认为哪些能力是 `decode` 的一等公民。

这里把对象分成两类：
- 算子库 / 底层接口：`PyTorch SDPA`、`FlashAttention`、`FlashInfer`、`cuDNN Frontend`
- 推理框架 / 运行时：`vLLM`、`TensorRT-LLM`、`SGLang`

### 2.1 算子库与底层接口

| 对象 | 典型 decode 相关接口 | 从接口能直接看出的支持重点 | 对我们设计维度的启发 |
| --- | --- | --- | --- |
| `PyTorch SDPA` | `torch.nn.functional.scaled_dot_product_attention(..., enable_gqa=True)` | 可以作为单 token decode 的数学 reference，支持 GQA 语义，但不管理 KV cache、page table、split-K 或 CUDA Graph replay | PyTorch 更适合做 correctness reference，不足以定义 serving decode 的 cache-aware 契约 |
| `FlashAttention` | `flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None, block_table=None, ...)` | 一个接口同时覆盖 contiguous cache、paged KV cache、可选 K/V append、RoPE、ALiBi、sliding window、MQA/GQA；`block_table` 说明 paged KV 已是 decode 关键路径 | `kv_layout`、`kv_update_contract`、`position_semantics`、`mask_semantics` 和 `head_topology` 都应该进入正式 OP 契约 |
| `FlashInfer` | `single_decode_with_kv_cache`、`BatchDecodeWithPagedKVCacheWrapper.plan/run`、`CUDAGraphBatchDecodeWithPagedKVCacheWrapper` | decode 被设计成 plan/run 两阶段；paged KV 用 `indptr/indices/last_page_len`，并显式支持 CUDA Graph buffer、tensor-core decode、`q_scale/k_scale/v_scale`、`return_lse`、`fixed_split_size` | 对 decode 来说，page metadata、workspace、split-K、quantized KV scale、CUDA Graph 生命周期不是实现细节，而是 release-facing 能力边界 |
| `cuDNN Frontend SDPA` | SDPA decode support matrix、`seq_len_q/kv`、`paged_attention_k_table/v_table`、ragged/padded layout、stats | 官方 support matrix 把 Decode 单列；同时暴露 paged table、sequence length、mask/bias/ALiBI、stats 和 FP8 decode 支持 | 一个通用底层接口如果要覆盖主流 decode，至少要把 per-batch length、paged table、layout、score modifier、stats 和 dtype 写进契约 |

### 2.2 推理框架与运行时

| 对象 | 典型 decode 相关接口或配置 | 从接口能直接看出的支持重点 | 对我们设计维度的启发 |
| --- | --- | --- | --- |
| `vLLM` | PagedAttention、attention backend feature matrix、KV dtype、block size、head size、Decode Context Parallelism | vLLM 把 block/page 管理、KV dtype、backend 选择和 decode-only backend 能力做成系统能力；不同 backend 对 page size、FP8 KV、sink attention、DCP 的支持不同 | TileOps decode plan 需要明确 page size / block table、FP8 KV cache、sink/sliding window、DCP 是否属于当前 release |
| `TensorRT-LLM` | generation phase 的 masked multi-head attention、paged KV cache、INT8/FP8 KV cache、RoPE、QKV bias、quant/dequant | TensorRT-LLM 将 generation phase 作为独立 kernel 路径，并在 kernel 内做 RoPE、量化/反量化等 preprocessing | decode 不应只是通用 attention forward；generation kernel 的 fused preprocessing 和 KV cache dtype 是主线能力 |
| `SGLang` | `--decode-attention-backend`、hybrid prefill/decode backend、attention backend support matrix、speculative decoding backend | SGLang 把 prefill 和 decode backend 分离作为正式配置；decode backend 总是 CUDA Graph captured；spec decode 会选择 decode 或 prefill backend | decode release plan 需要单列 backend 分离、CUDA Graph、spec decode verification 的边界，避免和 prefill plan 混在一起 |

### 2.3 调研结论：接口形态可以不同，但核心契约相同

主流项目不一定都把 read-only decode 和 append decode 拆成两个公开 OP，但都会显式表达 cache metadata、page table、sequence length、append position 和 GQA/MQA head mapping。

- FlashAttention 使用一个 `flash_attn_with_kvcache` 接口覆盖 read-only 与 optional append：`k/v` 可选，`cache_seqlens` 和 `block_table` 显式传入。
- FlashInfer 更偏 plan/run 分层：read-only decode、paged batch decode wrapper、CUDA Graph wrapper 和 paged append helper 分开表达。
- vLLM 把 block table、KV cache block、slot mapping 和 decode metadata 放在 runtime/backend 侧管理，kernel 消费已经准备好的 metadata。
- TensorRT-LLM 把 generation phase 作为独立 attention 路径，并把 KV cache dtype、RoPE、quant/dequant 等 generation preprocessing 纳入 decode 设计。

因此，TileOps 选择把 read-only decode 和 append decode 拆成不同 OP，是为了让 cache 生命周期和 position 语义更稳定，而不是因为其他项目都采用同名拆法。更重要的共同点是：decode 不能只暴露 Q/K/V 张量，还必须暴露或约束 `real_seqlen_kv`、paged metadata、head mapping、score modifier、split-K stats 和 CUDA Graph replay 所需的静态 buffer 契约。

## 三、哪些维度最先决定一个 decode 路径是否真正可用

在真实 inference 系统里，最容易让一个 decode 路径“看起来能算，但其实接不上系统”的，通常是下面几个维度：

1. `kv_layout`
2. `kv_length_metadata`
3. `page_table_format`
4. `head_topology`
5. `split_k_strategy`
6. `cuda_graph_contract`
7. `numeric_format`
8. `position_semantics`

原因：
- 只支持 contiguous KV，很难接主流 paged serving runtime
- 只支持 batch 内同一 KV 长度，很难接 continuous batching
- page table 语义不清楚，就无法和 page manager 对接
- GQA/MQA head mapping 不清楚，会直接算错
- 长上下文没有 split-K，decode 性能容易掉到不可发布
- split-K 合并不稳定，会影响数值确定性、batch-size invariant 和 CUDA Graph replay
- FP8 KV cache 的 scale 粒度不清楚，就无法和主流量化 serving 路线对齐
- RoPE/cache position 不清楚，会在 prefix reuse 或 speculative verification 中出现位置错位

## 四、Release baseline、补充路径、长期支持项

### 4.1 本轮 release baseline

目标：本轮不把 contiguous-only decode 当作发布主目标，而是直接做到可接现代 serving runtime 的 decode baseline。原因是 GQA prefill 已经按进阶功能直接推进，decode 也应避免先发布一个只能做 padded / homogeneous cache 的过渡接口。

本轮 release baseline 应至少包括：

- single-token GQA/MQA/MHA decode
- paged KV cache 主力路径
- contiguous KV cache reference / fallback 路径
- batch 内 heterogeneous `real_seqlen_kv`
- paged `block_table [B, max_pages_per_req]`
- `q [B, Hq, D]`
- `k_pages/v_pages [P_tokens, Hkv, D]`
- fp16 / bf16
- output-only read-only decode
- `sm_scale` / softcap 语义明确
- split-K 作为内部 dispatch policy，不暴露为用户 OP
- CUDA Graph-friendly static buffer / batch bucket contract
- manifest-backed benchmark 和 nightly shape 覆盖

| 维度 | 本轮 release baseline | 后续增强 |
| --- | --- | --- |
| `decode_unit` | `single_token` | `single_token` 稳定，评估 `multi_token` / speculative verification |
| `query_layout` | `bhd` | 小 `packed_thd_small_q` |
| `kv_layout` | `paged_kv` 主力，`contiguous_kv` 作为 reference / fallback | prefix-shared paged cache、compressed KV |
| `kv_length_metadata` | per-batch `real_seqlen_kv [B]` | append variant 下的 `cache_seqlens` |
| `page_table_format` | `block_table [B, max_pages_per_req]` | 可选 wrapper 支持 `indptr/indices/last_page_len` |
| `head_topology` | MHA/GQA/MQA 统一表达 | 常见 GQA ratio 下有稳定高效路径 |
| `mask_semantics` | `full_prefix`，可选 `sliding_window` | `sink_attention`、custom mask |
| `position_semantics` | 外置 RoPE，cache 已保存 rotated K | 明确 `cache_position`，可选 fused RoPE |
| `kv_update_contract` | `read_only` | 新增 append variant |
| `split_k_strategy` | 内部 dispatch：`no_split` + 长上下文 split-K | deterministic merge / batch-size invariant |
| `cuda_graph_contract` | graph replay static batch/buffer 契约明确 | runtime capture bucket 策略细化 |
| `numeric_format` | fp16 / bf16，FP8 KV cache dequant path 设计清楚 | FP8 Tensor Core / INT8 / NVFP4 |
| `outputs_and_stats` | output-only | lse 作为 split-K internal stats，公开暴露低优先级 |

一个合格的本轮 release decode，至少应该是：

- Q 使用 `[B, Hq, D]`
- paged KV 使用 `[P_tokens, Hkv, D]` + `block_table [B, max_pages_per_req]` + `real_seqlen_kv [B]`
- contiguous KV 使用 `[B, Skv_cap, Hkv, D]` + `real_seqlen_kv [B]` 作为 reference / fallback
- 支持 MHA/GQA/MQA
- 支持 batch 内不同 KV 长度
- 支持 fp16 / bf16
- output 与 PyTorch SDPA / materialized paged reference 对齐

### 4.2 KV cache layout 决策

decode 的 KV cache 不采用 prefill-style varlen packed layout 作为 release-facing 主接口。

prefill varlen 常见形态是：

```text
q/k/v: [total_tokens, H, D]
cu_seqlens: [B + 1]
```

这个适合 prompt prefill，因为 Q/K/V 都是本轮输入 token。decode 的 KV 是长期 cache，每步只更新 metadata 和少量 cache 内容；如果把旧 KV materialize 成 `[sum(real_seqlen_kv), Hkv, D]`，会带来额外 copy / compact 成本，也很难表达 prefix sharing、page reuse 和 CUDA Graph static shape。

因此本轮采用：

```text
contiguous decode:
k_cache/v_cache: [B, Skv_cap, Hkv, D]
real_seqlen_kv:  [B]

paged decode:
k_pages/v_pages: [P_tokens, Hkv, D]
block_table:     [B, max_pages_per_req]
real_seqlen_kv:  [B]
```

Varlen packed KV 可以作为 reference、debug 或 bridge wrapper，但不作为 serving decode 主路径。

### 4.3 补充路径

补充路径包括：

- multi-token decode / speculative verification
- append decode variants
- FlashInfer-style `indptr/indices/last_page_len` wrapper
- sink attention / StreamingLLM
- FP8 KV cache implementation

### 4.4 长期支持项

长期支持项包括：
- multi-token decode / speculative verification
- sink attention / StreamingLLM
- INT8 / NVFP4 / 更低比特 KV cache
- per-kv-head、per-token-head 或 block-scale KV cache scale
- DCP / decode context parallelism
- MLA / MQA/GQA 特化 decode 路径
- prefix-shared paged cache 的 copy-on-write 友好路径
- H200 / Blackwell 上更强的 tensor-core decode、TMA / WS / PDL 路线

## 五、OP-facing 与 runtime-heavy 维度拆分

### 5.1 OP-facing 维度

如果我们已经决定要执行一次 decode attention，那么为了把这次计算本身算对、算快，operator 必须知道：

- 当前 Q 的 layout 和 token 数
- KV cache 的物理布局
- 每个 request 当前可见 KV 长度
- page table 如何从 logical token 映射到 physical token
- Q head 到 KV head 的映射
- attention scale、softcap、sliding window 等 score 语义
- 是否需要 split-K 以及如何合并
- cache 中 K 是否已经应用 RoPE
- KV cache dtype 和 scale 如何解释

一句话：

**“这一拍 decode attention 到底读哪些 KV、怎么算、怎么合并。”**

### 5.2 Runtime-heavy 维度

在一个真实 inference 系统里，还有很多事情不是单个 decode OP 应该独立决定的：

- request 如何 continuous batching
- decode step 如何和 prefill step 混排
- page 如何分配、复用、回收
- prefix cache 如何命中和共享
- CUDA Graph capture 采用哪些 batch bucket
- speculative decode 的 draft / verify 如何调度
- 多 GPU decode context parallelism 如何切分上下文

这些属于 runtime-heavy 维度。decode plan 里应该单列它们，但不要把它们硬塞进一个 attention OP 的最小接口。

### 5.3 边界判断

容易误写的说法：
- “decode OP 支持 prefix cache”
- “decode OP 支持 continuous batching”
- “decode OP 支持 CUDA Graph”

更准确的拆分是：
- OP 支持 `paged_kv`、`block_table`、`real_seqlen_kv`
- runtime 支持 prefix cache 命中、page sharing 和 copy-on-write
- OP 的参数和 buffer 契约可以做到 graph replay 友好
- runtime 选择哪些 batch bucket 并负责 capture / replay

这个边界很重要，因为 decode operator 如果只写 kernel 语义，就接不上 serving；但如果把 scheduler 全部放进 OP，又会让接口失真。

## 六、本轮设计决策

本节把 GQA decode 讨论中的关键决策收敛成可执行原则。更具体的落地顺序见 `decode-plan.md`。

### 6.1 Read-only 与 append 分离

现有 decode OP 保持 read-only：

- `GroupedQueryAttentionDecodeWithKVCacheFwdOp`
- `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp`

如果后续需要 fused append，应新增 append variant，而不是改变现有 OP 的语义。append variant 必须明确 `append_before_attention` 还是 `append_after_attention`；LLM 标准自回归 decode 通常应采用 `append_before_attention`，即 current token 的 K/V 参与本次 attention，并写入 cache 的 current position。

这样做的原因是 read-only decode 和 append decode 的 cache 生命周期不同。悄悄改变 read-only OP 会让 runtime 出现重复 append、cache length 错位或 RoPE position 错位。

### 6.2 与 prefill 统一的是数学语义，不是运行时接口

decode 和 prefill 应统一这些 attention 语义：

- `sm_scale`
- softcap / logits soft cap
- sliding window
- causal / full-prefix 可见性
- GQA/MQA head mapping
- `return_lse` 的数学含义
- dtype 和 scale 解释

但不应强行统一这些 runtime-heavy 契约：

- Q/KV layout
- KV cache metadata
- page table
- append 生命周期
- split-K 策略
- CUDA Graph replay bucket

prefill 是长 Q attention，decode 是短 Q 读取长 KV cache。二者数学同源，但 runtime contract 不同。

### 6.3 Continuous serving 下动态的是 metadata 内容，不是 kernel 编译

continuous serving 中，每一步 request、KV 长度和 page 映射都会变化，但这不应导致重复编译 kernel。

应作为编译期或 dispatch key 的静态项：

- dtype
- `heads` / `heads_kv`
- `dim`
- layout
- `page_size`
- batch bucket
- `max_pages_per_req`
- split-K graph bucket 中的 `max_splits`

应作为运行期 tensor 内容变化的动态项：

- Q 内容
- KV cache 内容
- `real_seqlen_kv` 内容
- `block_table` 内容
- slot mapping 或 active request mapping

因此，`real_seqlen_kv[b]`、当前 batch 的最大上下文长度、当前实际 page 数不应成为 TileLang kernel 的 compile-time 参数。OP/runtime 应预分配固定 shape buffer，replay 或连续 decode step 中只更新 tensor 内容。

### 6.4 Split-K 不暴露为用户 OP

split-K 是 dispatch policy 和 kernel implementation，不是用户主要心智模型。公开 OP 默认保持 output-only：

- short / medium context 使用 `no_split` fast path
- long context 由 OP 或 kernel wrapper dispatch 到 split-K
- split-K partial stats 和 merge workspace 为 internal buffer
- 不暴露 `GroupedQueryAttentionDecodeSplitKFwdOp`

split-K merge 必须基于 online softmax stats，而不是直接平均 partial outputs。内部可以使用 partial max / partial lse / partial output。是否公开 `return_lse=True` 是低优先级 follow-up。

### 6.5 FP8 KV cache 首发只承诺 dequant path

FP8 KV cache 首发目标是降低 decode 阶段 KV 读取带宽，而不是一次性进入 FP8 Tensor Core attention compute。

首发约束：

- storage dtype：`fp8_e4m3fn`
- Q/O dtype：`fp16` 或 `bf16`
- `k_scale` / `v_scale`：`float32[1]`
- scale 粒度：per-tensor
- kernel 内部读取 old cache 后 dequant
- 不承诺 FP8 Tensor Core operand layout / swizzle / tile scale pipeline

Hopper / Blackwell 特化版可以后续再讨论 swizzle、TMA、WS、PDL 和更强 tensor-core decode 路径。

### 6.6 CUDA Graph 是 bucket contract，不是 OP 名字

不要暴露 `GroupedQueryAttentionDecodeCudaGraphFwdOp`。CUDA Graph capture/replay 是 runtime 职责；decode OP 只需要保证在指定 bucket contract 下不改变 shape、buffer requirement、dispatch key 和 launch topology。

graph-friendly decode 的核心要求：

- capture 前完成 JIT compile、kernel dispatch、workspace allocation 和 warmup
- replay 中 tensor 指针和 shape 固定
- `block_table`、`real_seqlen_kv`、Q/KV 内容可变
- kernel launch 拓扑固定
- split-K graph bucket 使用 fixed split policy

TileLang 官网中的 `eager` / `lazy` 指 JIT wrapper 的执行模式，不等同于 serving 中的 eager dispatch / CUDA Graph replay。对 TileOps decode，推荐用 TileLang lazy JIT factory 生成可复用 kernel object；CUDA Graph capture 由 PyTorch/runtime/profiler 在固定 buffer 上完成。

## 七、对 release plan 的实际意义

把这些维度拆开，对 release plan 有三个直接好处。

第一，能明确 MVP 的真实边界：
- `single_token`
- contiguous KV
- MHA/GQA/MQA
- fp16 / bf16
- output-only

第二，能明确发布前必须补齐的 serving 能力：
- paged KV
- heterogeneous batch length
- split-K
- page metadata 协议
- score modifiers
- CUDA Graph 约束
- FP8 KV cache 首发策略

第三，能避免团队误解：
- kernel 团队以为“我们已经支持 paged decode 了”
- runtime 团队以为“那 prefix cache、spec decode、CUDA Graph bucket 也自然有了”

更合理的责任边界是：
- kernel / operator 负责人：保证单次 decode 调用语义正确、接口稳定、性能可接受
- runtime 负责人：保证请求调度、page 生命周期、graph capture/replay 和 prefix sharing 正确
- release owner：保证 manifest、benchmark、workload、roofline 和文档一致

## 八、参考资料

- PyTorch SDPA: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention
- FlashAttention README: https://github.com/Dao-AILab/flash-attention
- FlashInfer Attention API: https://docs.flashinfer.ai/api/attention.html
- cuDNN Frontend Attention: https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html
- vLLM Attention Backend Feature Support: https://docs.vllm.ai/en/latest/design/attention_backends/
- vLLM Paged Attention: https://docs.vllm.ai/en/v0.18.0/design/paged_attention/
- TensorRT-LLM GPT Attention: https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html
- SGLang Attention Backend: https://lmsysorg.mintlify.app/docs/advanced_features/attention_backend
- TileLang JIT: https://tilelang.com/autoapi/tilelang/jit/index.html
- TileLang Profiler: https://tilelang.com/autoapi/tilelang/profiler/bench/index.html
- TileLang CUDA capability helpers: https://tilelang.com/autoapi/tilelang/contrib/nvcc/index.html
