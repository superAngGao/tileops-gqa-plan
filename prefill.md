# GQA Prefill 规划

日期：2026-04-24

本文只讨论 `prefill`，不讨论 `decode`，也不讨论具体到某个现有实现或某个仓库当前已经支持什么。

本文是 GQA prefill 的能力分层与接口调研；具体落地顺序和当前基线收敛计划见 `prefill-plan.md`。

这里的 `prefill` 包括：
- 标准 prompt prefill，通常 `q_len == kv_len`
- chunked / incremental prefill，通常 `q_len != kv_len`
- 部分前缀已经命中 KV cache、剩余部分继续 prefill 的场景
- prefill 过程中同时读取和写入 KV cache 的场景

不在本文范围内的内容：
- 单 token decode
- 多 token decode
- prefill / decode 调度策略
- 某个具体项目的差距分析

## 一、prefill 支持维度

我们把 inference 场景下 GQA prefill 的能力拆成下面这些维度。

| 维度 | 含义 | 典型取值 |
| --- | --- | --- |
| `mask_semantics` | 注意力可见性语义 | `none`、`causal`、`arbitrary`、`sliding_window`、`prefix_lm`、`block_mask` |
| `sequence_layout` | Q/K/V 输入在调用边界上的组织方式 | `dense_padded`、`packed_varlen`、`ragged` |
| `kv_layout` | 可见 KV 状态的存储和寻址方式 | `no_cache`、`contiguous_kv`、`paged_kv`、`prefix_shared_kv` |
| `q_vs_kv_length_relation` | 当前 prefill 调用里 Q 和 KV 长度关系 | `q_len_eq_kv_len`、`q_len_lt_kv_len`、`q_len_gt_kv_len` |
| `position_semantics` | 位置编码及跨调用位置对齐语义 | `none`、`rope`、`alibi`，以及 `position_ids`、`offset`、`cache_position` |
| `head_topology` | query head 和 KV head 的对应关系 | `mha`、`gqa`、`mqa` |
| `sparsity_pattern` | attention 是 dense 还是 sparse，以及 sparse 如何表达 | `dense`、`local_sparse`、`block_sparse`、`topk_sparse`、`sink_sparse` |
| `score_modifiers` | 点积后、softmax 前对 logits 的额外变换 | `scale`、`temperature`、`softcap`、`logit_bias`、`sink_bias` |
| `numeric_format` | 计算与存储使用的数值格式，以及量化/混合精度策略 | 原生基线：`fp16`、`bf16`；扩展变体：`fp8`、`int8_kv`、`mixed_precision` |
| `attention_shape` | 除头数关系外的结构性形状特征 | `qk_dim_eq_vo_dim`、`qk_dim_ne_vo_dim`、特殊 head size |
| `kv_update_contract` | 这次 prefill 是只读 KV，还是会更新 KV | `read_only`、`append_kv`、`inplace_update` |
| `outputs_and_stats` | 除输出张量外还返回什么 | `output_only`、`output_plus_lse`、`output_plus_stats` |
| `batch_variability` | batch 内不同请求是否可以有不同长度和状态 | `homogeneous_batch`、`heterogeneous_batch` |
| `modality_prefix_behavior` | 是否存在多模态前缀或特殊区域的可见性规则 | `text_only`、`multimodal_prefix`、`special_prefix_regions` |

### 补充说明

这些维度尽量保持正交，不建议混在一起讨论。

尤其要注意下面几组不要混淆：
- `sequence_layout` 和 `kv_layout` 不是一回事  
  前者说的是输入组织方式，后者说的是 cache 的存储方式。
- `mask_semantics` 和 `position_semantics` 不是一回事  
  前者说谁能看谁，后者说位置信息如何编码与对齐。
- `head_topology` 和 `attention_shape` 不是一回事  
  前者是 MHA/GQA/MQA，后者是 qk/vo 维度是否一致等结构问题。
- `kv_update_contract` 和 `kv_layout` 不是一回事  
  一个是“这次调不调用会写 cache”，一个是“cache 本身怎么存”。

### 关于 `numeric_format` 维度的进一步说明

`numeric_format` 这个维度不能简单写成一串并列取值，因为它内部其实有层次。

更合理的理解方式是：
- `fp16`、`bf16` 是原生基线能力
- `fp8` 是一组扩展变体
- `int8` 也是一组扩展变体
- `mixed precision` 不是单一格式，而是一类组合策略

也就是说，这个维度更像：

| 层次 | 含义 |
| --- | --- |
| `native compute dtype` | 算子天然支持的主要计算输入格式，通常是 `fp16`、`bf16` |
| `quantized storage / transport dtype` | 为了节省带宽或 cache 容量而使用的低精度存储格式，如 `fp8`、`int8` |
| `accumulation / stats dtype` | 中间累加和统计量使用的精度，通常是 `fp32` |
| `mixed precision policy` | 输入、cache、累加、输出分别用什么格式，以及何处做 dequant / cast |

#### 1. `fp16` / `bf16`

这两类通常应被视为 prefill 的原生支持能力，而不是复杂变体。

它们一般表示：
- Q/K/V 的主输入格式
- kernel 的主要 MMA/矩阵计算输入格式
- 输出 `O` 的主要格式

而即使在 `fp16/bf16` 模式下，主流实现里也常常会有：
- `fp32` 累加
- `fp32` 的 `lse` 或其他统计量

所以严格说，很多所谓“fp16/bf16 路径”本身已经是一种轻量 mixed precision，只是这是 today 的默认基线，不必专门当成复杂变体来讨论。

#### 2. `fp8` 不是一个点，而是一组变体

在 prefill 场景下，`fp8` 至少可以分成下面几类：

1. `fp8 compute path`
- Q/K/V 本身以 `fp8` 参与主要计算
- 常见情况仍会配合更高精度累加，例如 `fp16/bf16` 或 `fp32` accum

2. `fp8 kv-cache storage`
- 主要不是把整个 attention 都做成 `fp8`
- 而是把 KV cache 以 `fp8` 存储，读取时再做 dequant 或转换
- 这在 serving 场景尤其重要，因为 KV cache 占用和带宽压力很大

3. `fp8 activation transport`
- 某些路径里不是长期存储为 `fp8`
- 而是为了跨 kernel / 跨阶段传输更省带宽，短暂以 `fp8` 表示

4. `scale granularity` 不同的 `fp8`
- `per-tensor scale`
- `per-head scale`
- `per-channel scale`
- `per-block scale`

因此在 release plan 里，如果写“支持 fp8”，其实还需要进一步明确：
- 是支持 `fp8 compute`
- 还是只支持 `fp8 kv cache`
- scale 粒度是什么
- dequant 发生在 kernel 外还是 kernel 内

#### 3. `int8` 也不是一个点，而是一组变体

在 prefill 里，`int8` 更常见的不是“attention 全流程 int8 compute”，而是以下几类：

1. `int8 kv-cache storage`
- KV cache 以 `int8` 存储
- prefill 读取时做 dequant
- 这是最典型、也最贴近 serving 的 int8 变体

2. `int8 q/k/v input path`
- Q/K/V 输入本身就是量化后的
- kernel 内部或前置阶段完成 dequant
- 这类路径比单纯 `int8 kv cache` 更重

3. `scale / zero-point` 方案不同
- `symmetric` vs `asymmetric`
- `per-tensor`
- `per-channel`
- `per-head`
- `per-block`

所以如果文档只写“支持 int8”，信息量其实不够。更准确的说法通常应该是：
- 支持 `int8 kv cache`
- 是否支持 `int8 qkv input`
- scale/zero-point 的粒度和位置

#### 4. `mixed precision` 不是单一格式，而是组合策略

`mixed precision` 最容易被写得过于含糊。  
它不是一种 dtype，而是一类“不同阶段使用不同 dtype”的策略。

在 prefill 里常见的 mixed precision 组合包括：

1. `fp16/bf16 input + fp32 accumulation`
- 这是最常见、也最基础的一类

2. `fp8 storage + bf16/fp16 compute`
- 例如 KV cache 存成 `fp8`
- 读取后转为 `bf16/fp16` 参与 attention

3. `int8 kv storage + bf16/fp16 compute`
- serving 里非常现实的一类

4. `low-precision input + high-precision stats`
- 例如主路径是 `fp16/bf16`
- 但 `lse`、归一化统计量、部分 reduction 采用 `fp32`

5. `heterogeneous output policy`
- 输出 `O` 保持 `fp16/bf16`
- 统计量保持 `fp32`
- cache 写回可能用另一种压缩格式

因此如果要把这个维度定义得更严谨，`mixed precision` 最好不要当作一个简单枚举值，而应被理解为：

**“compute dtype、storage dtype、accum dtype、stats dtype、output dtype 的组合策略。”**

#### 5. 这对后面的阶段划分意味着什么

因此在后面的 `MVP / 进阶 / 长期` 里，`numeric_format` 更合理的写法应该是：

- `MVP`
  - 原生支持 `fp16`、`bf16`
  - 中间累加和统计量至少 `fp32`

- `进阶支持项`
  - 支持 `fp8 kv cache`
  - 支持 `int8 kv cache`
  - 明确 dequant 位置与 scale 粒度

- `长期支持项`
  - 支持更完整的 `fp8 compute`
  - 支持更广的低比特 cache/storage 组合
  - 支持更成熟的 mixed precision policy 配置

## 二、主流算子库与框架调研：prefill 支持能力与接口形态

上面第一章给出的是抽象维度。为了避免维度设计停留在概念层，这一章直接看主流算子库和推理框架的官方接口，反推它们真正认为哪些能力是 `prefill` 的一等公民。

这里把对象分成两类：
- 算子库 / 底层接口：`PyTorch SDPA`、`FlashAttention`、`FlashInfer`、`cuDNN Frontend`、`xFormers`
- 推理框架 / 运行时：`vLLM`、`TensorRT-LLM`、`SGLang`

### 2.1 算子库与底层接口

| 对象 | 典型 prefill 相关接口 | 从接口能直接看出的支持重点 | 对我们设计维度的启发 |
| --- | --- | --- | --- |
| `PyTorch SDPA` | `torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, is_causal=False, scale=None, enable_gqa=False)` | 明确暴露 `attn_mask`、`is_causal`、`scale`、`enable_gqa`，并天然支持 `L != S` 的非方 attention；但不负责 KV cache 生命周期 | `mask_semantics`、`score_modifiers`、`head_topology`、`q_vs_kv_length_relation` 是基础语义维度；但仅靠框架级 attention API 不足以表达 serving 里的 cache 语义 |
| `FlashAttention` | `flash_attn_varlen_*`、`flash_attn_with_kvcache(...)` | 官方 README 明确支持 `variable sequence lengths`、`arbitrary Q/KV sequence lengths`、`MQA/GQA`、`rotary embeddings`、`ALiBi`、`paged KV cache`、`softcapping`；`flash_attn_with_kvcache` 还显式暴露 `k_cache/v_cache`、`k/v` 追加、`cache_seqlens`、`block_table`、`softmax_scale`、`window_size`、`alibi_slopes` | 一旦进入真实 inference，`q_vs_kv_length_relation`、`kv_layout`、`position_semantics`、`kv_update_contract`、`score_modifiers` 都会立即从“抽象属性”变成函数签名的一部分 |
| `FlashInfer` | `single_prefill_with_kv_cache`、`BatchPrefillWithPagedKVCacheWrapper`、`BatchPrefillWithRaggedKVCacheWrapper`、`append_paged_kv_cache` | prefill 接口不是单一函数，而是围绕 `ragged/paged KV`、`plan/run`、`custom_mask`、`return_lse`、`q_scale/k_scale/v_scale`、`actual_seq_lens_q/kv` 来组织；同时提供独立的 paged KV append API | `sequence_layout` 和 `kv_layout` 必须分开；`outputs_and_stats`、`numeric_format`、`kv_update_contract` 都应该是 prefill 正式维度，不是实现细节 |
| `cuDNN Frontend SDPA` | `scaled_dot_product_attention` graph attributes、`paged_attention_k_table/v_table`、`seq_len_q/seq_len_kv`、`set_diagonal_alignment`、`set_score_mod` | 官方文档明确写出支持 `MHA/MQA/GQA`、任意 `s_q/s_kv`、ragged/padded layout、paged attention、bottom-right 对齐、sliding window、bias、softcap、stats 输出 | 一个底层通用 attention 接口如果想覆盖主流 inference prefill，至少要把 `layout`、`paged table`、`seq_len_q/kv`、`mask alignment`、`score_mod`、`stats` 做成一等配置项 |
| `xFormers` | `xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None, scale=None, ...)` | 核心接口把 `attn_bias` 作为通用 mask/bias 抽象；支持 `Mq != Mkv`；官方文档说明 `MQA/GQA` 是实验性前向功能；但不管理 KV cache | `mask_semantics` 不应只理解为 `causal/noncausal`，还应允许更一般的 `bias` 抽象；但 cache 相关维度不能指望从纯 attention 算子自动长出来 |

### 2.2 推理框架与运行时

| 对象 | 典型 prefill 相关接口或配置 | 从接口能直接看出的支持重点 | 对我们设计维度的启发 |
| --- | --- | --- | --- |
| `vLLM` | `enable_chunked_prefill=True`、`enable_prefix_caching=True`、attention backend 选择 `FLASH_ATTN/FLASHINFER/TRITON/FLEX` | `chunked prefill`、`prefix caching`、paged block 管理、以及后端自动选择都是系统级能力；prefix caching 文档直接围绕 KV block 的 `allocate/append/free/eviction` 展开 | `prefix reuse`、`paged_kv`、`chunked prefill` 更像 runtime 维度，不应混进一个单一 kernel 的最小接口，但必须在 prefill 规划里被单列 |
| `TensorRT-LLM` | `GPT attention operator`、`paged context attention`、`KVCacheConfig`、`use_paged_context_fmha` | 官方明确写出 attention op 会 `populate KV cache`；同时支持 `contiguous/paged KV`、`chunked context`、`INT8/FP8 KV cache`、`sliding window`、`cyclic KV cache`、`sink tokens` | 真实 production 栈里，`kv_update_contract`、`kv_layout`、`numeric_format`、`mask/window/sink` 是被一起设计的；而且 `paged_kv` 不是长期项，而是主流路线 |
| `SGLang` | `chunked_prefill_size`、`prefill_attention_backend`、`disable_radix_cache`、`enable_mixed_chunk` | 官方首页和参数文档把 `RadixAttention`、`chunked prefill`、`paged attention`、`prefill-decode disaggregation`、`prefill/decode backend 分离` 都当作基础能力；attention backend 页面还直接给出 `Native Page Sizes`、`FP8 KV Cache`、`Chunked Prefix Cache` 维度表 | 从 runtime 角度看，prefill 除了算法能力外，还天然带有 `page_size`、`prefix cache`、`prefill backend` 与 `decode backend` 分离这类系统约束 |

### 2.3 从主流接口反推出来的结论

调研后可以看到，第一章列出的维度里，有些只是“理论上合理”，有些则已经被主流生态反复证明是必须做成正式接口的。

#### 2.3.1 已经被主流生态证明必须单列的维度

下面这些维度，在多个主流库和框架里都直接出现在函数签名、wrapper 参数或者运行时配置里：

1. `q_vs_kv_length_relation`
   - `PyTorch SDPA` 直接区分 `L` 和 `S`
   - `FlashAttention` 明确强调 `arbitrary Q/KV sequence lengths`
   - `FlashInfer`、`cuDNN` 都显式传 `actual_seq_lens_q/kv` 或 `seq_len_q/kv`

2. `sequence_layout`
   - `FlashAttention` 用 `varlen` 系列接口处理 packed 变长
   - `FlashInfer` 直接把 `ragged KV cache` 和 `paged KV cache` 分成不同 wrapper
   - `cuDNN` 支持 `Padded`、`Ragged` 等 layout

3. `kv_layout`
   - `FlashAttention` 的 `block_table`
   - `FlashInfer` 的 paged KV wrapper
   - `cuDNN` 的 `paged_attention_k_table/v_table`
   - `TensorRT-LLM` 和 `vLLM` 的 paged KV 设计

4. `position_semantics`
   - `FlashAttention` 直接暴露 `rotary_cos/sin`、`alibi_slopes`
   - `TensorRT-LLM` 的 window/sink/cyclic cache 也和位置语义绑定
   - 这说明位置语义不能只当成模型外部预处理的一句注释

5. `kv_update_contract`
   - `FlashAttention` 的 `flash_attn_with_kvcache` 允许把 `k/v` 追加进已有 cache
   - `FlashInfer` 单独提供 `append_paged_kv_cache`
   - `TensorRT-LLM` 文档明确写 attention op 会填充 KV cache

6. `outputs_and_stats`
   - `FlashInfer` 支持 `return_lse`
   - `cuDNN` 明确支持 stats 输出
   - `xFormers` 也有返回 `output, lse` 的 forward 接口

7. `score_modifiers`
   - `PyTorch SDPA` 暴露 `scale`
   - `FlashAttention` 暴露 `softmax_scale`、`softcap`
   - `xFormers` 暴露 `attn_bias` 和 `scale`
   - `cuDNN` 暴露 `set_score_mod`

#### 2.3.2 已经被主流生态证明不能混为一谈的维度

调研后最值得强调的是下面几组区分：

- `sequence_layout` 和 `kv_layout` 必须分开  
  `FlashInfer` 同时有 ragged KV 和 paged KV 的 wrapper，这已经说明“输入是否 packed/ragged”和“cache 是否 paged”是两个正交问题。

- `mask_semantics` 和 `position_semantics` 必须分开  
  `FlashAttention` 同时支持 `causal/window` 和 `RoPE/ALiBi`，`cuDNN` 同时有 `diagonal_alignment/band` 和 `bias/score_mod`。这说明“谁能看谁”和“位置信息如何编码”不是一回事。

- `kv_update_contract` 和 `kv_layout` 必须分开  
  是否 `append` / `inplace update`，和 cache 是连续还是分页存储，是两个不同问题。

#### 2.3.3 从生态实践看，哪些能力更像 kernel 维度，哪些更像 runtime 维度

更偏 kernel / operator 的维度：
- `mask_semantics`
- `q_vs_kv_length_relation`
- `position_semantics`
- `head_topology`
- `score_modifiers`
- `numeric_format`
- `outputs_and_stats`

更偏 runtime / framework 的维度：
- `kv_layout`
- `prefix_shared_kv`
- `page_size`
- `prefix reuse`
- `chunked prefill`
- `prefill/decode backend 分离`

这一区分很重要，因为一个“prefill 规划”既不能只写 kernel 语义，也不能把所有 runtime 机制都硬塞进同一个 attention 算子接口。

### 2.3.4 什么叫 kernel 维度，什么叫 runtime 维度

这里可以把两者的区别说得更明确一些。

#### 1. kernel 维度是什么

`kernel 维度` 指的是：  
如果我们已经决定要执行一次 attention / prefill 计算，那么为了把这次计算本身算对、算快，kernel 必须知道哪些信息。

这类维度通常直接影响：
- kernel 的输入输出张量形状
- kernel 的数学语义
- kernel 的调度策略和 tile 划分
- kernel 是否需要额外返回中间统计量

它们往往会直接体现在函数签名、kernel attributes、或者编译/dispatch 约束里。

典型例子：
- `mask_semantics`
  kernel 必须知道这次是 `causal`、`sliding_window` 还是更一般的 mask，否则连分数矩阵怎么屏蔽都不知道。
- `q_vs_kv_length_relation`
  kernel 必须知道 `q_len` 和 `kv_len` 的真实关系，尤其在 `q_len != kv_len` 时，causal 对齐方式会变。
- `position_semantics`
  如果 kernel 内部要处理 RoPE、ALiBi、soft bias，或者至少要配合这些语义，那它必须知道位置相关参数。
- `head_topology`
  kernel 要知道这是 `MHA`、`GQA` 还是 `MQA`，因为 head mapping 和访存模式不同。
- `score_modifiers`
  `scale`、`softcap`、`temperature` 这类项直接改变 softmax 前 logits。
- `outputs_and_stats`
  如果要返回 `lse` 或其他统计量，这不是 runtime 后处理能凭空补出来的，必须由 kernel 路径决定。

一句话说，`kernel 维度` 关注的是：

**“这一拍 attention 到底怎么算。”**

#### 2. runtime 维度是什么

`runtime 维度` 指的是：  
在一个真实 inference 系统里，prefill 这次计算如何被组织、如何和 cache 交互、如何放进整个请求生命周期中执行。

这类维度通常直接影响：
- 请求如何分块和调度
- KV cache 如何分配、复用、回收
- batch 内不同请求如何拼接
- prefill 与 decode 是否共用后端或分离后端
- prefix 是否能跳过重算

它们不一定决定某一个 kernel 的数学定义，但会决定：
- kernel 被调用多少次
- 每次调用时传入的张量如何组织
- 计算结果写到哪里
- 后续请求能否复用前一次结果

典型例子：
- `kv_layout`
  `contiguous_kv` 还是 `paged_kv`，往往不只是一个张量布局问题，还牵涉 page table、block manager、回收策略。
- `prefix_shared_kv`
  prefix 是否共享，不是某个单次 attention kernel 能独立决定的，而是 cache manager 和请求系统级决策。
- `chunked prefill`
  把一次长 prompt 拆成几段、如何和 decode 混批，本质是调度问题。
- `page_size`
  page 是 16、32、64 还是 128，不仅影响 kernel，也影响整个 cache 管理系统。
- `prefill/decode backend 分离`
  这是运行时在不同阶段选不同 backend 的策略，不是某个单 kernel 自己能表达的事情。

一句话说，`runtime 维度` 关注的是：

**“这一拍 attention 在整个推理系统里如何被组织和复用。”**

#### 3. 为什么必须把两者分开

如果不把这两类维度分开，规划很容易在两个方向上出问题。

第一类问题是：把 runtime 问题误写成 kernel 能力。

例如：
- “支持 prefix cache”
- “支持 chunked prefill”
- “支持 page manager”

这些说法如果直接挂在一个 attention kernel 上，通常是不准确的。  
kernel 也许能支持 `paged_kv` 输入，也许能支持 `append_kv`，但“prefix cache 是否命中、页怎么分配、旧页什么时候回收”，本质上还是 runtime 的责任。

第二类问题是：把 kernel 问题误当成 runtime 细节。

例如：
- `q_len != kv_len` 时 causal 对齐
- `return_lse`
- `softcap`
- `GQA/MQA` head mapping

这些不是 runtime 靠外围拼一拼就能补上的。  
如果 kernel 语义本身没有定义好，runtime 再强也只是把错误更高效地调度出去。

#### 4. 一个简单判断方法

如果一个能力更像下面这些问题，它通常更偏 kernel 维度：
- 这次 attention 的数学定义是什么？
- 分数矩阵怎么构造和屏蔽？
- 这次调用的输出除了 `O` 还需要 `lse` 吗？
- 这个 feature 会不会直接改变 kernel 的输入签名、输出签名或内部计算路径？

如果一个能力更像下面这些问题，它通常更偏 runtime 维度：
- 请求怎么拆分成多次 prefill？
- KV cache 放在哪里、怎么分配、怎么回收？
- 哪些前缀可以复用、哪些不能？
- prefill 和 decode 是不是要走不同后端？

#### 5. 边界情况：有些维度天然跨两层

也要承认，很多真实能力并不是纯 kernel 或纯 runtime，而是“runtime 机制 + kernel 支持”共同完成的。

最典型的几个例子：

- `paged_kv`
  一半是 runtime 维度，一半是 kernel 维度。  
  runtime 负责 page 分配、block table 维护、回收策略；kernel 负责根据 page table 正确 gather KV 并完成 attention。

- `prefix reuse`
  runtime 负责识别可复用前缀、管理共享块；kernel 负责在已有 KV 基础上继续 prefill 或 append。

- `chunked prefill`
  runtime 负责切 chunk、混 batch、调度先后；kernel 负责在每个 chunk 上正确处理 `q_len/kv_len`、位置偏移和 KV 更新。

- `RoPE offset`
  上层 runtime 或 model executor 可能负责维护全局 position；kernel 或 operator 层则需要一个清晰的消费接口。

因此更准确的说法通常不是：
- “这个能力属于 kernel”
- “这个能力属于 runtime”

而是：
- “这个能力的**主导复杂度**在 kernel 侧”
- “这个能力的**主导复杂度**在 runtime 侧”

#### 6. 对 release plan 的实际意义

把这两类维度分开，对 release plan 有两个直接好处。

第一，能更准确地定义阶段目标。  
例如：
- `q_len != kv_len`
- `rope + offset`
- `append_kv`

这些更像 kernel/operator MVP。

而：
- `prefix_shared_kv`
- `chunked prefill`
- `prefill/decode backend 分离`

这些更像 runtime 集成阶段的目标。

第二，能更准确地分配实现责任。  
通常会变成下面这种分工：
- kernel / operator 负责人：保证单次 prefill 调用语义正确、接口稳定、性能可接受
- runtime / serving 负责人：保证 cache 生命周期、chunk 调度、prefix 复用、跨阶段 backend 选择正确

如果这两个层次不先拆开，后面经常会出现一种情况：
- kernel 团队以为“我们已经支持 paged_kv 了”
- runtime 团队以为“那 prefix reuse 和 chunked prefill 也应该自然有了”

结果就是接口看起来齐全，但系统仍然接不起来。

### 2.4 基于调研，对第二章的直接结论

如果只保留最扎实的结论，可以压成下面几条：

1. `prefill` 的设计不能只参考训练态 dense attention 接口，必须参考 serving 场景下的 cache-aware 接口。
2. `q_len != kv_len` 不是边角 case，而是主流库显式支持的核心能力。
3. `varlen/ragged` 和 `paged_kv` 都已经是主流接口的一等能力，不适合放到很后的“长期支持项”里。
4. `RoPE/ALiBi`、`scale/softcap/bias`、`return_lse/stats` 这些也已经不是附属选项，而是正式接口维度。
5. `prefix reuse`、`chunked prefill`、`page manager` 更像 runtime 层能力，但如果 release plan 不把它们单列出来，规划就会失真。

## 三、哪些维度最先决定一个 prefill 路径是否真正可用

在真实 inference 系统里，最容易让一个 prefill 路径“看起来能算，但其实接不上系统”的，通常是下面几个维度：

1. `q_vs_kv_length_relation`
2. `sequence_layout`
3. `kv_layout`
4. `position_semantics`
5. `kv_update_contract`

原因很简单：
- 只支持 `q_len == kv_len`，就很难接 chunked prefill
- 只支持 dense padded，就很难接真实服务里的 varlen batch
- 只支持连续 KV，不支持 paged KV，就很难接主流 serving runtime
- 位置 offset 语义不清楚，就很难接 prefix reuse 或增量 prefill
- 只会读 KV、不支持 append/update，就不是完整的 prefill 路径

## 四、分阶段目标

下面把这些维度按 `MVP`、`进阶支持项`、`长期支持项` 三个阶段整理。

### 1. MVP

目标：具备一个真实可用的、面向主流 LLM inference 的 dense GQA prefill 路径，并清楚定义从 MVP 走到发布版本的升级路径。

| 维度 | MVP 目标 | 进阶支持项 |
| --- | --- | --- |
| `mask_semantics` | `causal` | `causal`、`sliding_window`、部分 `arbitrary` mask |
| `sequence_layout` | `packed_varlen`，或者其他真正支持变长请求的表示方式 | 稳定支持 `packed_varlen`，并能和 `dense_padded` 互通 |
| `kv_layout` | `contiguous_kv` | 同时支持 `contiguous_kv` 和 `paged_kv` |
| `q_vs_kv_length_relation` | 支持 `q_len_eq_kv_len` 和 `q_len_lt_kv_len` | 在混合 batch 下稳定支持非对称长度 |
| `position_semantics` | `rope`，并且位置 offset 语义明确 | 支持 `rope`、`alibi`，并明确 `position_ids`、`cache_position` |
| `head_topology` | 支持 `mha`、`gqa`、`mqa` | 在常见 GQA ratio 下有稳定高效路径 |
| `sparsity_pattern` | `dense` | 仍以 dense 为主，但接口上为 local sparse 留好位置 |
| `score_modifiers` | 默认 scale，最好允许显式传入 `sm_scale` | 支持 `sm_scale`、`temperature`、`softcap`、部分 bias |
| `numeric_format` | 原生支持 `fp16`、`bf16`，累加和统计量至少 `fp32` | `fp16`、`bf16` 稳定；支持 `fp8 kv cache`、`int8 kv cache` 等量化 KV 路径，并明确 dequant/scale 策略 |
| `attention_shape` | 先支持主流等维 head 路径 | 支持常见非默认 head dim，以及部分不等维情形 |
| `kv_update_contract` | `append_kv` | 支持 `append_kv`，并在部分路径支持 `inplace_update` |
| `outputs_and_stats` | 至少 `output_only`，最好支持 `output_plus_lse` | `output_plus_lse` 成为一等能力 |
| `batch_variability` | 支持 `heterogeneous_batch` | 对长短混合请求保持稳定 |
| `modality_prefix_behavior` | `text_only` | 明确支持带特殊前缀区域的场景 |

MVP 的核心结论可以压成一句话：

一个合格的 MVP prefill，至少应该是：
- dense causal
- 支持 varlen
- 支持 `q_len != kv_len`
- 支持 RoPE 位置对齐
- 支持 MHA/GQA/MQA
- 能把新 K/V 正确写入 cache

### 2. 发布门槛与对标判断

目标：让 prefill 不只是“能用”，而是能较好地接入现代 serving 系统。我们计划在这一阶段完成后进入发布。

对这一阶段可以做一个更直接的总结：
- 如果 `MVP` 解决的是“prefill 语义是否成立”
- 那么 `进阶支持项` 解决的是“prefill 是否已经具备现代 serving 可发布性”

从发布视角看，进阶支持项最关键的几件事是：
- `paged_kv`
- 混合 batch 下稳定的 `q_len != kv_len`
- 更清晰的位置接口
- 更完整的 cache 更新协议
- 更现实的量化 KV 路径
- `output_plus_lse` 这类更完整的接口契约

因此，第四章表格的第三列可以直接理解为：

**“如果准备发布，这个维度相比 MVP 还要补到什么程度。”**

### 2.1 完成发布门槛后的定量对标

如果我们在“发布门槛”这一阶段全部完成，再去和主流算子库、推理框架做对比，可以给出一个更直观的定量判断。

这里先说明口径：
- 这不是性能评分
- 这不是生态成熟度评分
- 这不是代码质量评分
- 这只是一个**prefill 功能面对齐度**的估计

为了避免把不同类型对象混在一起，这里分成两类看：

1. `算子库对齐度`
- 比较的是 prefill 相关的 operator-facing 能力面
- 重点看：mask、varlen、paged/contiguous kv、`q_len != kv_len`、RoPE/ALiBi、GQA/MQA、score modifiers、quantized KV、KV update、`return_lse`

2. `推理框架对齐度`
- 比较的是完整 prefill 栈
- 除了 operator-facing 能力，还包括：prefix reuse、chunked prefill 调度、page manager、backend 分离、cache 生命周期

因此下面的分数不能混着解读：
- 对算子库的 8 分，通常已经很强
- 对推理框架的 8 分，则意味着 runtime 能力也比较完整

#### A. 对主流算子库的能力面对齐度

这里的“欠缺”只按前面定义过的维度来写，不讨论性能和工程成熟度。

| 对象 | 类型 | 发布门槛完成后的对齐度估计 | 仍欠缺的关键维度 |
| --- | --- | --- | --- |
| `PyTorch SDPA` | 框架级 attention API | `8/10` | `kv_layout`、`kv_update_contract`、`outputs_and_stats`、低比特 `numeric_format` |
| `FlashAttention` | 通用高性能 attention 算子库 | `8.5/10` | 更完整的 `kv_update_contract`、更宽的 `numeric_format`、更全的 `outputs_and_stats` |
| `FlashInfer` | serving-oriented attention 算子库 | `9/10` | 更一般的 `mask_semantics`、更宽的 `numeric_format`、更完整的 `outputs_and_stats` |
| `cuDNN Frontend SDPA` | 底层通用 attention 接口 | `8.5/10` | 更广的 `mask_semantics` 组合、`score_modifiers` 组合性、`numeric_format` 广度 |
| `xFormers` | memory-efficient attention 算子库 | `7/10` | 更通用的 `mask_semantics` 抽象、`outputs_and_stats`、更宽的 `position_semantics` |

对算子库这条线，可以压缩成一句话：

**发布门槛完成后，我们在 prefill operator 层面的能力，大致会进入主流第一梯队，和 FlashAttention / FlashInfer / cuDNN Frontend 处于同一档次讨论，但还不等于它们的整体成熟度。**

如果一定要给一个更粗的区间判断：
- 相对 `PyTorch SDPA / xFormers`：大致会达到 `100%+` 的 prefill 能力覆盖
- 相对 `FlashAttention / FlashInfer / cuDNN Frontend`：大致会达到 `80%~90%` 的 prefill 能力覆盖

这里的 `100%+` 不是说全面超越，而是说：  
在“cache-aware prefill”这个特定问题上，我们的功能面会超过这些更偏通用 attention API 的对象。

#### B. 对主流推理框架的 prefill 完整度对齐

这里的“欠缺”主要按 runtime 侧维度来写，不再泛泛讨论“系统能力”。

| 对象 | 类型 | 发布门槛完成后的对齐度估计 | 仍欠缺的关键 runtime 维度 |
| --- | --- | --- | --- |
| `vLLM` | 推理框架 / runtime | `6.5/10` | `prefix_shared_kv`、`chunked_prefill` 调度、page/block manager、backend selection policy |
| `TensorRT-LLM` | 推理框架 + attention operator 栈 | `7/10` | `page manager`、更完整的 cache lifecycle、prefill/decode 协同、runtime 配置传播 |
| `SGLang` | 推理框架 / runtime | `6/10` | `prefix_shared_kv`、`chunked_prefill` 调度、`page_size` 策略、prefill/decode backend 分离 |

对推理框架这条线，可以压缩成一句话：

**发布门槛完成后，我们会在“prefill 算子与接口”上接近一线 runtime 的底层能力，但离它们的完整 prefill 系统能力通常还有一段距离。**

如果一定要给一个更粗的区间判断：
- 相对主流推理框架的**算子层 prefill 能力**：大致达到 `75%~85%`
- 相对主流推理框架的**完整 prefill 系统能力**：大致达到 `60%~70%`

#### C. 为什么算子库对齐度高，但框架对齐度没有那么高

这是因为“发布门槛”的目标，本质上还是：
- 做成一个足够强的 prefill operator / operator-family
- 而不是做完整的 serving runtime

因此在“发布门槛”完成后，我们通常已经具备：
- 主流 prefill 数学语义
- 主流 cache-aware 输入输出契约
- 主流量化 KV 路径
- 面向发布的 operator 接口完整性

但通常还不完全具备：
- prefix reuse 的系统级命中与共享策略
- chunked prefill 的调度器行为
- page allocation / recycle / eviction 管理
- prefill backend 与 decode backend 的分离和协同
- 像 vLLM / SGLang / TensorRT-LLM 那样完整的 runtime 约束传播

所以从 release plan 的角度，一个更准确的说法是：

- “发布门槛”完成后，我们的定位会更接近  
  **“主流一线的 prefill 算子能力”**
- 但还没有完全到  
  **“主流一线的 prefill runtime 完整性”**

### 3. 长期支持项

目标：覆盖更广的模型族、更复杂的 runtime 语义，以及更长期的演化需求。

| 维度 | 长期目标 |
| --- | --- |
| `mask_semantics` | 通用 `arbitrary` mask、`block_mask`、`prefix_lm`、更复杂区域可见性 |
| `sequence_layout` | dense、packed、ragged、多种执行管线之间灵活互通 |
| `kv_layout` | `prefix_shared_kv`、更复杂的页共享和 cache indirection |
| `q_vs_kv_length_relation` | 对各种非对称关系提供完整支持 |
| `position_semantics` | 覆盖更广的位置编码家族及组合式 offset 策略 |
| `head_topology` | 如未来模型需要，支持更复杂的 query/KV 分组方式 |
| `sparsity_pattern` | `block_sparse`、`topk_sparse`、`sink_sparse`、dense-sparse hybrid |
| `score_modifiers` | 可组合 bias pipeline 和模型特定 logits 变换 |
| `numeric_format` | 成熟的 `fp8`、低比特 KV cache 支持，并有稳定精度边界 |
| `attention_shape` | 覆盖更广的特殊维度和架构特定布局 |
| `kv_update_contract` | 更丰富的 cache 更新与复用协议 |
| `outputs_and_stats` | 除 `lse` 外，按需支持调试、标定、分析类统计信息 |
| `batch_variability` | 在高度动态、混合工作负载下仍能无缝工作 |
| `modality_prefix_behavior` | 多模态和特殊前缀区域成为一等公民 |

## 五、算子设计草案

这一节的目标不是给出某一种实现，而是把 `prefill` operator family 的命名、layout 和接口契约定义清楚，作为后续实现、评审和测试的共同基线。

### 5.1 设计原则

1. 命名要反映语义和数据组织方式，不反映具体实现细节  
   例如名字里可以体现 `prefill`、`paged_kv`，但不应体现具体 kernel 策略。

2. `MHA / GQA / MQA` 统一走同一套 operator family  
   不单独做 `mha_prefill` 或 `mqa_prefill`。  
   统一通过 `heads` 和 `heads_kv` 来表达。

3. `sequence_layout` 和 `kv_layout` 分开体现在接口上  
   不把 packed、contiguous cache、paged cache 强行塞进一个全可选大接口。

4. 接口层级尽量贴近 `TileOPs` 现有风格  
   `__init__` 固定结构参数和大多数语义配置，`forward()` 主要接收动态 tensor 输入。

5. 发布阶段优先保证接口稳定和语义清楚  
   比“只用一个超灵活大函数”更重要。

### 5.2 命名设计

推荐采用一套 operator family，而不是一个参数极其臃肿的单接口。

从现有 `TileOPs` 风格看，attention 命名大致遵循下面几条：
- 使用完整语义类名，而不是短函数名
- 采用 `PascalCase`
- 前向统一以 `FwdOp` 结尾
- cache 相关语义直接进入类名，如 `DecodeWithKVCacheFwdOp`
- `Varlen`、`SlidingWindow`、`Paged` 这类数据组织或语义修饰词直接进入类名

因此这一节的推荐命名，尽量与现有风格对齐。

#### 1. 用户可见的稳定命名

- `GroupedQueryAttentionPrefillVarlenFwdOp`
- `GroupedQueryAttentionPrefillWithKVCacheFwdOp`
- `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp`

这三个名字分别对应：
- packed/varlen 的普通 prefill
- 带连续 KV cache 的 prefill
- 带 paged KV cache 的 prefill

从 release plan 的设计目标看，推荐先把 `GroupedQueryAttention*` 这一组作为主接口族，
把 `MHA` 与 `MQA` 视为它的特例，而不是平行维护多套接口。

#### 2. 为什么不单独命名 `mha_prefill` / `mqa_prefill`

因为在接口层，三者本质上只是 `head_topology` 不同：
- `MHA`: `heads == heads_kv`
- `GQA`: `heads > heads_kv` 且整除
- `MQA`: `heads_kv == 1`

如果接口设计得当，调用方不需要关心名字差异，只需要正确设置：
- `heads`
- `heads_kv`

#### 3. 与现有 TileOPs 风格的对齐建议

如果未来真要落到 `TileOPs` 代码里，命名和组织上建议进一步保持这些约定：

- 文件继续放在：
  - `tileops/ops/attention/gqa.py`

- `__all__` 暴露类名，而不是函数名

- kernel map key 使用 snake_case，并与类名语义对应，例如：
  - `gqa_prefill_varlen_kernel`
  - `gqa_prefill_with_kv_cache_kernel`
  - `gqa_prefill_paged_with_kv_cache_kernel`

- 如果存在更专门的语义修饰词，也沿用现有顺序风格：
  - 主体名词在前：`GroupedQueryAttention`
  - 语义修饰在中间：`Prefill` / `Varlen` / `Paged` / `WithKVCache`
  - 生命周期后缀在最后：`FwdOp`

- 参数命名也尽量对齐现有风格：
  - 用 `heads`、`heads_kv`，不再另起 `num_q_heads`、`num_kv_heads`
  - 用 `dim` 作为首发版本的主 head dim 命名
  - 用 `seqlen_kv` 表示 cache 容量上界或逻辑长度上界
  - paged 路径沿用 `block_table`、`real_seqlen_kv`

按这个规则，像下面这样的命名会比函数式短名更贴近现有风格：
- `GroupedQueryAttentionSlidingWindowVarlenFwdOp`
- `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp`
- `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp`

### 5.3 推荐 layout 设计

下面定义一套推荐的 canonical layout，供 operator contract 使用。

#### 1. Packed / Varlen 激活布局

这是推荐的主 layout。

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `q` | `[Tq, Hq, Dqk]` | packed query |
| `k` | `[Tk, Hkv, Dqk]` | packed key |
| `v` | `[Tk, Hkv, Dv]` | packed value |
| `o` | `[Tq, Hq, Dv]` | output |
| `lse` | `[Hq, Tq]` | 可选返回统计量 |
| `cu_seqlens_q` | `[B+1]` | packed q 边界 |
| `cu_seqlens_k` | `[B+1]` | packed kv 边界 |

推荐把这种 layout 记作：
- `THD` for activations

其中：
- `T` = packed token 数
- `Hq` = query heads
- `Hkv` = KV heads
- `Dqk` = q/k head dim
- `Dv` = value/output head dim

#### 2. 连续 KV cache 布局

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `q` | `[Tq, Hq, Dqk]` | 当前 prefill 的 packed query |
| `k_new` | `[Tnew, Hkv, Dqk]` | 本次新增 key |
| `v_new` | `[Tnew, Hkv, Dv]` | 本次新增 value |
| `k_cache` | `[B, Skv_cap, Hkv, Dqk]` | 连续 KV cache |
| `v_cache` | `[B, Skv_cap, Hkv, Dv]` | 连续 KV cache |
| `real_seqlen_kv` | `[B]` | 每个请求当前可见 KV 长度 |
| `cu_seqlens_q` | `[B+1]` | 当前 query 边界 |
| `cu_seqlens_new` | `[B+1]` | 本次新增 KV 边界 |

推荐把 cache layout 记作：
- `BSHD` for contiguous cache

其中：
- `B` = batch
- `S` = cache capacity 或逻辑序列维

#### 3. Paged KV cache 布局

推荐使用 page-major 物理布局：

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `q` | `[Tq, Hq, Dqk]` | 当前 prefill 的 packed query |
| `k_new` | `[Tnew, Hkv, Dqk]` | 本次新增 key |
| `v_new` | `[Tnew, Hkv, Dv]` | 本次新增 value |
| `k_pages` | `[P, page_size, Hkv, Dqk]` | paged KV cache |
| `v_pages` | `[P, page_size, Hkv, Dv]` | paged KV cache |
| `block_table` | `[B, max_pages_per_req]` | 每个请求映射到哪些物理 block/page |
| `real_seqlen_kv` | `[B]` | 每个请求当前可见 KV 长度 |
| `cu_seqlens_q` | `[B+1]` | 当前 query 边界 |
| `cu_seqlens_new` | `[B+1]` | 本次新增 KV 边界 |

这里推荐：
- page 内 token 维在前，即 `[page_size, H, D]`
- `block_table` 只表达逻辑顺序，不掺杂生命周期策略

### 5.4 接口设计

不建议把所有路径塞进一个超大接口。  
建议发布时对外暴露三类接口。

#### 1. `GroupedQueryAttentionPrefillVarlenFwdOp`

用于：
- 普通 packed varlen prefill
- 不直接操作外部 KV cache 的场景

```python
class GroupedQueryAttentionPrefillVarlenFwdOp(Op):
    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool = True,
        window_size_left: int = -1,
        window_size_right: int = -1,
        position_mode: str = "rope",
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        return_lse: bool = False,
        dtype: torch.dtype = torch.float16,
        accum_dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        ...

    def forward(
        self,
        q: torch.Tensor,                 # [Tq, Hq, Dqk]
        k: torch.Tensor,                 # [Tk, Hkv, Dqk]
        v: torch.Tensor,                 # [Tk, Hkv, Dv]
        cu_seqlens_q: torch.Tensor,      # [B+1]
        cu_seqlens_k: torch.Tensor,      # [B+1]
        max_seqlen_q: int,
        max_seqlen_k: int,
        custom_mask: Optional[torch.Tensor] = None,
        position_ids_q: Optional[torch.Tensor] = None,
        position_ids_k: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ...
```

这个接口覆盖的核心维度：
- `mask_semantics`
- `q_vs_kv_length_relation`
- `position_semantics`
- `head_topology`
- `score_modifiers`
- `outputs_and_stats`

#### 2. `GroupedQueryAttentionPrefillWithKVCacheFwdOp`

用于：
- 连续 KV cache 的 prefill
- 本次 prefill 同时需要 append 新 KV

```python
class GroupedQueryAttentionPrefillWithKVCacheFwdOp(Op):
    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seqlen_kv: int,
        dim: int,
        is_causal: bool = True,
        window_size_left: int = -1,
        window_size_right: int = -1,
        position_mode: str = "rope",
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        return_lse: bool = False,
        append_kv: bool = True,
        dtype: torch.dtype = torch.float16,
        accum_dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        ...

    def forward(
        self,
        q: torch.Tensor,                 # [Tq, Hq, Dqk]
        k_new: torch.Tensor,             # [Tnew, Hkv, Dqk]
        v_new: torch.Tensor,             # [Tnew, Hkv, Dv]
        cu_seqlens_q: torch.Tensor,      # [B+1]
        cu_seqlens_new: torch.Tensor,    # [B+1]
        k_cache: torch.Tensor,           # [B, Skv_cap, Hkv, Dqk]
        v_cache: torch.Tensor,           # [B, Skv_cap, Hkv, Dv]
        real_seqlen_kv: torch.Tensor,    # [B]
        max_seqlen_q: int,
        custom_mask: Optional[torch.Tensor] = None,
        position_ids_q: Optional[torch.Tensor] = None,
        cache_positions_new: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ...
```

这个接口相对 `varlen` 版新增的核心契约：
- `kv_layout = contiguous_kv`
- `kv_update_contract = append_kv`
- `position_semantics` 需要和 `cache_positions_new`、`real_seqlen_kv` 对齐

#### 3. `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp`

用于：
- paged KV cache 的 prefill
- 主流 serving runtime 对接

```python
class GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(Op):
    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seqlen_kv: int,
        dim: int,
        page_size: int,
        is_causal: bool = True,
        window_size_left: int = -1,
        window_size_right: int = -1,
        position_mode: str = "rope",
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        return_lse: bool = False,
        append_kv: bool = True,
        dtype: torch.dtype = torch.float16,
        accum_dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        ...

    def forward(
        self,
        q: torch.Tensor,                 # [Tq, Hq, Dqk]
        k_new: torch.Tensor,             # [Tnew, Hkv, Dqk]
        v_new: torch.Tensor,             # [Tnew, Hkv, Dv]
        cu_seqlens_q: torch.Tensor,      # [B+1]
        cu_seqlens_new: torch.Tensor,    # [B+1]
        k_pages: torch.Tensor,           # [P, page_size, Hkv, Dqk]
        v_pages: torch.Tensor,           # [P, page_size, Hkv, Dv]
        block_table: torch.Tensor,       # [B, max_pages_per_req]
        real_seqlen_kv: torch.Tensor,    # [B]
        max_seqlen_q: int,
        custom_mask: Optional[torch.Tensor] = None,
        position_ids_q: Optional[torch.Tensor] = None,
        cache_positions_new: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ...
```

这个接口相对连续 cache 版新增的核心契约：
- `kv_layout = paged_kv`
- `block_table`
- `page_size`

#### 4. 与当前 TileOPs 风格进一步对齐后的接口取舍

为了让这份设计更像现有 `TileOPs` op，而不是通用框架函数，这里明确几条取舍：

- 首发版本优先使用单一 `dim`
  也就是接口层先假定主流 `Dqk == Dv == dim` 路径。  
  `qk_dim != vo_dim` 属于后续扩展能力，可以在将来再拆成更细接口或新增结构参数。

- `window_size_left/right`、`position_mode`、`sm_scale`、`softcap`、`return_lse` 放在 `__init__`
  这些更像一次 op 实例的固定语义配置，和 `TileOPs` 现有把 `is_causal`、`page_size` 放在构造阶段的习惯一致。

- `forward()` 尽量只接动态张量和少量动态长度信息
  例如 `q`、`k`、`v`、`cu_seqlens_*`、`real_seqlen_kv`、`block_table`。  
  这样也更方便后续做 kernel dispatch 和 autotune cache。

### 5.5 与 TileOPs 现有风格的差异和取舍

即使尽量对齐 `TileOPs` 现有风格，这里还是有几处是有意和当前实现拉开的。

#### 1. 为什么这里写 `Prefill`，而不是沿用 `Fwd`

`TileOPs` 现有 attention 命名里：
- `MultiHeadAttentionFwdOp`
- `GroupedQueryAttentionFwdOp`
- `GroupedQueryAttentionDecodeWithKVCacheFwdOp`
- `GroupedQueryAttentionSlidingWindowVarlenFwdOp`

这里的 `Fwd` 更接近“训练态或通用前向”。  
而本文讨论的是明确的 inference `prefill` 语义，所以建议新 operator 直接把 `Prefill` 写进名字里，避免和现有普通 dense `FwdOp` 混淆。

#### 2. 为什么这里把 `Varlen` 放在 `Prefill` 后面

按现有 TileOPs 风格，修饰词通常放在主体名词后、生命周期后缀前。  
因此更推荐：
- `GroupedQueryAttentionPrefillVarlenFwdOp`

而不是：
- `GroupedQueryAttentionVarlenPrefillFwdOp`

前者和现有：
- `GroupedQueryAttentionSlidingWindowVarlenFwdOp`
- `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp`

在阅读习惯上更一致。

#### 3. 为什么这里仍然新增了 `Prefill` 这一层语义词

`TileOPs` 当前已有：
- `GroupedQueryAttentionFwdOp`
- `GroupedQueryAttentionDecodeWithKVCacheFwdOp`
- `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp`

但还没有一组专门表达 inference prefill 的 operator。  
因此这里新增 `Prefill` 这个语义词，不是为了偏离现有风格，而是为了把：
- 普通 dense forward
- cache-aware decode
- cache-aware prefill

这三类路径在命名上彻底区分开。

#### 4. 为什么这里仍然保留清晰的参数签名

虽然 `TileOPs` 现有接口是类 + `forward()`，但在设计文档里保留清晰的参数签名仍然有价值，
因为它能更直接体现 operator contract。  
真正落地实现时，仍应遵循 `TileOPs` 风格：
- `__init__` 固定结构参数和大多数语义配置
- `forward()` 接收动态 tensor 输入

### 5.6 参数语义约定

为了避免接口语义漂移，建议把下面这些规则固定下来。

#### 1. 命名和形状规则

- 统一使用 `heads`、`heads_kv`
- 必须满足 `heads % heads_kv == 0`
- `heads == heads_kv` 视为 `MHA`
- `heads_kv == 1` 视为 `MQA`
- 发布版本接口层优先采用单一 `dim`
- 如果后续需要 `qk_dim != vo_dim`，建议新增显式结构参数，而不是提前把首发接口做得过重

#### 2. mask 规则

- `is_causal=False` 且没有其他限制时：不加可见性约束
- `is_causal=True`：采用 causal 语义
- 提供 `custom_mask` 时：走显式 mask 语义
- `window_size_left/right >= 0` 时：表示 sliding-window 约束

建议约定：
- `custom_mask` 的优先级高于普通 causal/window 推导
- `custom_mask` 的 shape 和 layout 必须在文档中单独固定

#### 3. position 规则

- `position_mode="none"`：不使用额外位置语义
- `position_mode="rope"`：必须有清晰的 q/k 对应位置输入
- `position_mode="alibi"`：必须提供 `alibi_slopes`

对于 cache-aware prefill，建议固定：
- 新写入 KV 的位置通过 `cache_positions_new` 或同等语义字段给出
- 不能依赖“默认从 0 开始”这种隐式规则

#### 4. score modifier 规则

- `sm_scale is None` 时，默认使用 `1 / sqrt(Dqk)`
- `softcap is None` 表示不开启 softcap
- `temperature` 若未来支持，应和 `sm_scale` 的组合关系单独定义

### 5.7 返回值设计

推荐返回契约尽量简单：

- 默认返回：
  - `output`

- 当 `return_lse=True` 时返回：
  - `(output, lse)`

不建议在发布阶段把 cache 写入结果也作为复杂对象返回。  
更清晰的契约是：
- cache tensor 作为输入传入
- operator 对其执行约定好的 append/update
- 调用方通过 `real_seqlen_kv` 和外部状态管理结果

### 5.8 为什么不用一个总入口 `gqa_prefill(...)`

理论上可以做一个统一入口，但不建议把它作为第一阶段稳定接口。

原因是：
- packed varlen / contiguous cache / paged cache 的参数集合差异太大
- 一个总入口会产生大量 mutually-exclusive optional args
- 对调用方和测试都不友好

更稳妥的做法是：
- 对外稳定暴露三类接口
- 如有需要，在更上层提供一个 convenience wrapper 做 dispatch

这样可以同时兼顾：
- operator contract 清晰
- runtime 对接自然
- 后续 kernel 演化自由度更大

## 六、压缩版结论

如果把这份规划压缩成最重要的几句话：

- 本文讨论的是 `prefill`，不是 `decode`
- prefill 不能只被理解为“长序列 attention forward”
- 一个真正可用的 prefill 设计，至少要同时考虑：
  - mask
  - 变长输入组织
  - KV cache 组织
  - `q_len` 和 `kv_len` 关系
  - 位置语义
  - 头数拓扑
  - KV 写入协议
  - 数值格式
- `MVP` 的重点不是把所有 feature 都做全，而是先把真实 runtime 最离不开的几件事做对：
  - `causal`
  - `varlen`
  - `q_len != kv_len`
  - `rope + offset`
  - `mha/gqa/mqa`
  - `append_kv`

## 七、设计优先级建议

如果不同维度之间有冲突，建议优先级如下：

1. 真实 prefill 语义正确
2. cache 读写协议清楚
3. 接口在不同模型之间稳定
4. 再追求极限性能

这通常是更稳妥的路线，因为如果语义和 cache 契约都没有先定义清楚，后面即使 kernel 很快，也很难真正接进 inference 系统。

## 参考资料

- PyTorch SDPA: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention
- FlashAttention README: https://github.com/Dao-AILab/flash-attention
- FlashInfer Attention API: https://docs.flashinfer.ai/api/attention.html
- FlashInfer KV Layout: https://docs.flashinfer.ai/tutorials/kv_layout.html
- FlashInfer page API: https://docs.flashinfer.ai/api/page.html
- cuDNN Frontend Attention: https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html
- xFormers ops: https://facebookresearch.github.io/xformers/components/ops.html
- vLLM Prefix Caching: https://docs.vllm.ai/en/latest/design/prefix_caching/
- vLLM Chunked Prefill: https://docs.vllm.ai/en/v0.4.2/models/performance.html
- vLLM Attention Backend Feature Support: https://docs.vllm.ai/en/latest/design/attention_backends/
- TensorRT-LLM GPT Attention: https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html
- TensorRT-LLM KV Cache Reuse: https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html
- SGLang overview: https://docs.sglang.ai/
- SGLang Attention Backend: https://docs.sglang.io/docs/advanced_features/attention_backend
