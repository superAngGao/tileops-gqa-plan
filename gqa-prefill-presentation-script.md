# GQA Prefill Operator Family：语义、场景与接口设计

日期：2026-04-24

本文只讨论 `prefill`，不讨论 `decode`，也不讨论完整 serving scheduler、采样器或请求生命周期管理。

本文是 GQA prefill 的能力分层与接口调研；具体落地顺序和当前基线收敛计划见 `prefill-plan.md`。

这份报告面向同事评审，组织顺序按设计主线展开：

1. 定义 `prefill` 的算子语义和 cache population 责任。
2. 从 serving 场景推导 full prefill、chunked prefill、prefix-hit continuation、paged cache-aware prefill 等填充方式。
3. 将 prefill 填充方式映射到 kernel / OP 契约维度。
4. 对照主流生态接口，校准哪些能力已经是一等契约。
5. 基于生态依据和 TileOps 设计逻辑确定能力取舍、发布边界和 OP family。
6. 对 RoPE、FP8、paged KV 这类复杂变体单独展开子变体、行业做法、TileOps 决策和首发边界。

不在本文范围内的内容：
- 单 token decode
- 多 token decode
- prefill / decode 全局调度策略
- 某个具体项目的差距分析

## 一、Prefill 算子语义

在自回归 LLM inference 中，`prefill` 负责处理一段已经存在的输入 token，并建立后续 decode 可以复用的注意力状态。

从单层 attention 的角度看，prefill 这一拍会同时涉及三类 KV/Q 来源：

| 对象 | 来源 | 生命周期 | 在 attention 中的角色 |
| --- | --- | --- | --- |
| `prefix K/V` | 已经处理过的 prefix token 经过 K/V projection 后写入 KV cache | 调用前已经存在，跨 prefill/decode step 持久保存 | 作为当前 chunk 可见的历史上下文 |
| `current chunk Q` | 当前输入 chunk 的 hidden states 经过 Q projection 得到 | 只服务于本次 attention 输出 | 作为本次要计算输出的 query |
| `current chunk K/V` | 当前输入 chunk 的 hidden states 经过 K/V projection 得到 | 本次 attention 立即使用，并在调用后写入 KV cache | 既是 current chunk 内部 attention 的 KV，也是未来 token 的 prefix KV |

这里的 `prefix K/V` 可能来自几种路径：

- 同一个 request 前面 chunked prefill 已经写入的 cache。
- prefix cache / radix cache 命中的共享前缀。
- 对于首个 prompt chunk，`prefix K/V` 为空，即 `old_len = 0`。

`current chunk Q/K/V` 则来自当前模型层对本次输入 hidden states 的 projection。对 GQA/MQA 来说，`Q` 的 head 数是 `Hq`，`K/V` 的 head 数是 `Hkv`，二者通过 `groups = Hq / Hkv` 映射。

因此，一次 cache-aware prefill 的 attention 语义可以写成：

```text
visible KV = prefix K/V from cache + current chunk K/V
O_current  = attention(current chunk Q, visible KV)
```

调用结束后，还要完成 cache population：

```text
new prefix K/V for future steps = old prefix K/V + current chunk K/V
```

对一个 cache-aware prefill 来说，一次调用通常同时包含两个结果：

```text
1. output:
   为当前输入 token 计算 attention 输出 O

2. cache population:
   把当前 token 对应的 K/V 写入 caller-owned KV cache
```

因此 prefill 不应该只被理解成“长序列 attention forward”。在 serving 里，它更准确的语义是：

```text
current Q attends to old prefix KV + current chunk KV,
then current chunk K/V becomes part of the persistent KV cache.
```

注意这里有一个实现边界：attention 计算可以直接读取 `current chunk K/V`，不要求先把它们写入 cache 再从 cache 读回来。只要调用结束后 cache 中包含这些 current K/V，就满足 cache-aware prefill 的外部语义。

这和 decode 的区别是：decode 通常每个 request 当前只处理一个或少量 query token，而 prefill 通常处理一个 prompt 或 prompt chunk。二者数学上都是 attention，但 prefill 的主要问题是如何处理当前 chunk、已有 prefix、位置偏移和 cache 写入。

## 二、Serving 场景与 Prefill 填充方式

不同 serving 场景会导出不同 prefill 方式。这些是 **prefill 填充逻辑**，不是 `softcap`、`rotary_dim` 这种小 feature。它们来自真实用户请求形态：有人提交短 prompt，有人提交超长文档，有人连续多轮对话复用历史，有人请求共享同一系统提示词或检索前缀。runtime 为了降低延迟、提升吞吐和复用 KV，会把这些请求转化成不同 prefill 调用形态。

因此第二章的判断顺序是：

```text
用户行为 / 请求形态
    -> runtime 需要的 cache 与调度策略
        -> prefill OP 必须支持的读写语义
```

| 用户行为 / 请求形态 | Runtime 需求 | Prefill 方式 | 这次调用在做什么 | 典型 `q_len / kv_len` | KV cache 行为 | Serving 定位 |
| --- | --- | --- | --- | --- | --- | --- |
| 离线评测、小 batch 本地推理、固定长度 benchmark | 简单 correctness / baseline，不需要长期 cache 生命周期 | dense full prefill | 对一整段 dense padded prompt 算 attention | `q_len == kv_len` | 不读写外部 cache | reference、对照路径 |
| 多个不同长度 prompt 同时进入服务 | 避免 padding 浪费，把不同长度请求 packed 到同一批 | varlen full prefill | 多条 prompt packed 后一起算 attention | 每个 request 内通常 `q_len == kv_len` | 不读写外部 cache | 部分 serving 可用，但不是 cache-aware 主路径 |
| 用户提交一个普通长度新请求，后面要继续 decode | 一次性处理 prompt，并为 decode 建立 KV cache | cache-aware full prefill | 一次处理当前 prompt，并写入 caller-owned cache | `old_len=0`，`kv_len=q_len` | append 当前整段 K/V | 简单 serving / 单 chunk 路径 |
| 用户提交长文档、长上下文检索结果或超长对话历史 | 不能或不应一次处理完整 prompt，需要切 chunk 控制延迟和显存峰值 | chunked prefill | 长 prompt 被 runtime 切成多个 chunk 逐段处理 | 首 chunk `old_len=0`；后续 `q_len < kv_len` | 每个 chunk append 当前 K/V | 长上下文 serving 主路径 |
| 多轮对话、共享系统提示词、相同文档前缀或 radix/prefix cache 命中 | 复用已有 prefix KV，只计算新增 suffix | prefix-hit continuation | 只对新增 suffix 做 prefill，并读取 old prefix cache | `q_len` 是 suffix，`kv_len=prefix_len+q_len` | 读 old prefix cache，append suffix K/V | prefix cache / radix cache 主路径 |
| 服务同时承载大量请求，cache 需要动态分配、复用、回收 | 用 page/block 管理 KV，避免连续大块分配和搬移 | paged cache-aware prefill | old prefix 和 current chunk 通过 page table 读写 | batch 内各 request 可不同 | 读写 caller-owned paged cache | 现代 serving 主路径 |
| continuous batching 下，不同请求处在不同 prefill 阶段 | 一个 batch 内混合短 prompt、长 prompt chunk、prefix-hit suffix | mixed prefill batch | 每个 request 按自己的长度和 cache 状态计算 | 每个 request 不同 | per-request cache length / page table 不同 | continuous batching 的自然形态 |

这张表里最重要的分界是：

- dense/varlen no-cache prefill 主要解决 **当前 attention 算对**。
- cache-aware/chunked/prefix-hit/paged prefill 同时解决 **attention 算对 + KV cache 填充正确**。
- prefix 命中、chunk 切分、page 分配是 runtime 逻辑；但 kernel/OP 必须消费这些逻辑产生的 `q_lens`、`cache_lens`、`block_table`、position metadata。

换句话说，serving 定位不是因为某个 kernel 名字叫 `paged` 或 `chunked`，而是因为真实请求会不断制造这些需求：

- 为了首 token latency，长 prompt 需要 chunked prefill，而不是一次性阻塞整个 batch。
- 为了 prefix reuse，多轮对话和共享系统提示词需要 prefix-hit continuation。
- 为了显存利用率和并发，KV cache 需要 paged layout。
- 为了吞吐，runtime 会把不同阶段的请求混在一个 prefill batch 中。

### 2.1 Cache Append Contract

`append_kv` 表示这次 prefill 调用结束后，当前 chunk 的 K/V 已经进入 caller-owned KV cache。它不是简单的“多写一个输出 tensor”，而是 cache-aware prefill 的核心语义。

用户并不会直接说“请 append KV”；用户行为通常是“提交一个 prompt 并希望模型继续生成”。对 serving runtime 来说，这意味着 prompt 处理结束后必须留下可供 decode 使用的 KV cache。因此 append KV 是从生成需求自然导出的 OP contract。

更稳定的语义是：

```text
attention reads:
    old cache + current k_new/v_new

cache after call:
    old cache + appended current k_new/v_new
```

这意味着 current chunk 的 attention 不应该依赖“先把 current K/V append 到 cache，再从 cache 读回来”这个实现细节。实现上可以把 append kernel 和 attention kernel 分开：

- append kernel 按 `heads_kv` 维度写 cache/page。
- attention kernel 按 `heads` 维度计算 output。
- OP 负责把两者编排成一个稳定的 cache-aware prefill 契约。

这个拆分对 GQA 尤其重要，因为 append 的天然并行维度是 KV head，而 attention 的天然并行维度是 query head。

### 2.2 Chunked Prefill Contract

`chunked prefill` 的 chunk 边界由 runtime 决定，TileOps OP 不负责把 prompt 切 chunk。但是一旦 runtime 把某个 chunk 交给 OP，OP 必须支持：

- 当前 chunk 的 `q_len`
- 已有 prefix 的 `cache_len`
- `q_len < kv_len` 的 causal 对齐
- current chunk 的 absolute position
- 把 current K/V append 到正确 cache 位置

因此 chunked prefill 是 runtime 主导能力，但不是 kernel 可以忽略的外部细节。没有 `q_len != kv_len`、position offset 和 append contract，chunked prefill 接不上。

它对应的用户需求通常是长上下文输入：上传文档、长对话历史、检索增强上下文、代码库片段等。runtime 切 chunk 是为了控制单次 prefill 的显存峰值、调度粒度和 batch 内阻塞时间；OP 侧则必须保证每个 chunk 与已有 prefix 的 attention 语义一致。

### 2.3 Prefix-Hit Continuation Contract

prefix cache / radix cache 的命中判断、共享块管理、eviction 和引用计数都属于 runtime。TileOps OP 只需要在一次调用里正确消费 runtime 准备好的旧 KV 状态：

```text
old prefix:
    already materialized in contiguous or paged cache

new suffix:
    current q/k_new/v_new

OP result:
    output for suffix tokens
    suffix K/V appended into cache
```

因此不能说“prefill OP 支持 prefix cache”就等于 runtime 已经支持 prefix cache。更准确的说法是：prefill OP 支持 prefix-hit continuation 所需的 cache-aware attention 和 append 语义。

它对应的用户行为包括多轮对话继续、多个请求共享同一系统 prompt、同一文档被多次提问、或者 retrieval pipeline 产生相同前缀。用户看到的是更低延迟；runtime 看到的是已有 prefix KV 可复用；OP 看到的是 `old prefix cache + current suffix`。

### 2.4 Serving-Critical Path 与 Reference Path

从填充方式看，真实 serving 主路径不是单一的 “full prefill”。它通常由下面几类 prefill 调用组合出来：

- 新请求进入时，需要 cache-aware full prefill 建立第一段 KV cache。
- 长 prompt 或长上下文请求，需要 chunked prefill 把一次长输入拆成多次填充。
- 多轮对话、共享系统提示词或文档前缀命中时，需要 prefix-hit continuation 只处理新增 suffix。
- 高并发 serving 下，需要 paged cache-aware prefill 让 cache 可以动态分配、复用和回收。
- continuous batching 下，一个 batch 内会自然混合短 prompt、长 prompt chunk、prefix-hit suffix 等不同状态。

因此这里的 serving-critical 不是指某个单独 kernel 形态，而是指 runtime 真实需要 OP 承接这些填充语义：

- 当前调用可能只有 current prompt，也可能有已经存在的 old prefix。
- 当前调用可能一次处理完整 prompt，也可能只处理其中一个 chunk。
- 当前调用结束后，当前 K/V 可能需要成为后续 decode 或后续 chunk 的 cache 状态。
- batch 内不同请求可能处在不同 prefill 阶段。

dense no-cache 和连续 cache 路径仍然重要，但在这一章的语义层面，它们更像：

- correctness reference
- 单请求 / 本地推理路径
- 简单 cache-aware prefill 路径
- 更复杂 cache 管理方式的对照路径

它们可以帮助建立正确性和基线性能，但不能代表完整 serving prefill 需求。

## 三、Kernel / OP 功能变体

上面两章讲的是 prefill 方式，也就是“这次调用承担什么填充逻辑”。在具体落到 kernel/OP 时，还需要把一次 prefill 调用拆成更细的设计维度。

下面这张表是 kernel/operator 直接感知的功能变体表。它不等价于 serving 场景表：例如 `chunked prefill` 是一种 serving 方式，而 `q_vs_kv_length_relation`、`position_semantics`、`kv_update_contract` 是一次 kernel/operator 调用可以直接感知的语义维度。

这里关注的是会改变一次 prefill kernel / OP 的输入 tensor、metadata、mask 语义、cache 读写路径、数值路径或返回值的维度。prefix sharing、page allocation、chunk 切分、request lifecycle 这类 runtime ownership 机制属于另一层系统语义。

| 维度 | 含义 | 典型取值 |
| --- | --- | --- |
| `sequence_layout` | Q/K/V 输入在调用边界上的组织方式 | `dense_padded`、`packed_varlen`、`ragged` |
| `kv_layout` | OP 如何读取和寻址可见 KV storage | `no_cache`、`contiguous_kv`、`paged_kv` |
| `q_vs_kv_length_relation` | 当前 prefill 调用里 Q 和 KV 长度关系 | `q_len_eq_kv_len`、`q_len_lt_kv_len`、`q_len_gt_kv_len` |
| `mask_semantics` | 注意力可见性语义 | `none`、`causal`、`arbitrary`、`sliding_window`、`prefix_lm`、`block_mask` |
| `position_semantics` | 位置编码及跨调用位置对齐语义 | `none`、`rope`、`alibi`，以及 `position_ids`、`offset`、`cache_position` |
| `head_topology` | query head 和 KV head 的对应关系 | `mha`、`gqa`、`mqa` |
| `sparsity_pattern` | attention 是 dense 还是 sparse，以及 sparse 如何表达 | `dense`、`local_sparse`、`block_sparse`、`topk_sparse`、`sink_sparse` |
| `score_modifiers` | 点积后、softmax 前对 logits 的额外变换 | `scale`、`temperature`、`softcap`、`logit_bias`、`sink_bias` |
| `numeric_format` | 计算与存储使用的数值格式，以及量化/混合精度策略 | compute dtype：`fp16`、`bf16`；cache / mixed precision：`fp8`、`int8_kv`、`mixed_precision` |
| `attention_shape` | 除头数关系外的结构性形状特征 | `qk_dim_eq_vo_dim`、`qk_dim_ne_vo_dim`、特殊 head size |
| `kv_update_contract` | 这次 prefill 是只读 KV，还是会更新 KV | `read_only`、`append_kv`、`inplace_update` |
| `outputs_and_stats` | 除输出张量外还返回什么 | `output_only`、`output_plus_lse`、`output_plus_stats` |
| `batch_variability` | batch 内不同请求是否可以有不同长度和状态 | `homogeneous_batch`、`heterogeneous_batch` |
| `modality_prefix_behavior` | 是否存在多模态前缀或特殊区域的可见性规则 | `text_only`、`multimodal_prefix`、`special_prefix_regions` |

这张表只负责列出顶层功能轴，不表示每个取值的复杂度相同。有些取值接近简单开关，例如 `softcap=None/50`；有些取值本身包含多个子维度，例如 `position_semantics=rope`、`numeric_format=fp8`、`kv_layout=paged_kv`。

### 3.1 维度正交性说明

这些维度描述的是不同层面的事实，彼此容易混淆。

尤其要注意下面几组不要混淆：
- `sequence_layout` 和 `kv_layout` 不是一回事  
  前者说的是输入组织方式，后者说的是 cache 的存储方式。
- `mask_semantics` 和 `position_semantics` 不是一回事  
  前者说谁能看谁，后者说位置信息如何编码与对齐。
- `head_topology` 和 `attention_shape` 不是一回事  
  前者是 MHA/GQA/MQA，后者是 qk/vo 维度是否一致等结构问题。
- `kv_update_contract` 和 `kv_layout` 不是一回事  
  一个是“这次调不调用会写 cache”，一个是“cache 本身怎么存”。

这里尤其要区分 `kv_layout` 和 prefix reuse：

- `kv_layout=contiguous_kv` 表示 kernel 通过连续 cache storage 和 per-request length 寻址 KV。
- `kv_layout=paged_kv` 表示 kernel 通过 page table / block table 寻址 physical KV page。
- prefix sharing / prefix reuse 描述 runtime cache ownership。它涉及多个 request 是否共享同一段已经存在的 KV、page 引用如何管理、cache 命中如何判断。
- 在一次 kernel/operator 调用边界上，已存在的 prefix KV 通常仍然表现为 contiguous 或 paged 的旧 KV storage，以及对应的 sequence length / block table metadata。

### 3.2 数值格式维度的层次

`numeric_format` 不只是 `fp16/bf16/fp8/int8` 的并列枚举，它包含多个层次：

| 层次 | 含义 | 典型内容 |
| --- | --- | --- |
| native compute dtype | Q/K/V 和 O 的主要计算格式 | `fp16`、`bf16` |
| accumulation / stats dtype | 中间累加、online softmax stats、可选 lse | 至少 `fp32` 语义 |
| quantized cache storage | KV cache 为节省容量和带宽使用低精度存储 | `fp8`、`int8_kv` 等 |
| mixed precision policy | 输入、cache、accum、stats、output 分别用什么 dtype，以及何处 cast/dequant | storage dtype、compute dtype、scale granularity、cast/dequant path |

## 四、生态接口依据

前面先从 serving 语义推导了 prefill 方式和 OP 维度。为了避免设计停留在概念层，这一章直接看主流算子库和推理框架的官方接口，反推它们真正认为哪些能力是 `prefill` 的一等公民。

这里把对象分成两类：
- 算子库 / 底层接口：`PyTorch SDPA`、`FlashAttention`、`FlashInfer`、`cuDNN Frontend`、`xFormers`
- 推理框架 / 运行时：`vLLM`、`TensorRT-LLM`、`SGLang`

### 4.1 算子库与底层接口形态

| 对象 | 典型 prefill 相关接口 | 从接口能直接看出的支持重点 | 对我们设计维度的启发 |
| --- | --- | --- | --- |
| `PyTorch SDPA` | `torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, is_causal=False, scale=None, enable_gqa=False)` | 明确暴露 `attn_mask`、`is_causal`、`scale`、`enable_gqa`，并天然支持 `L != S` 的非方 attention；但不负责 KV cache 生命周期 | `mask_semantics`、`score_modifiers`、`head_topology`、`q_vs_kv_length_relation` 是基础语义维度；但仅靠框架级 attention API 不足以表达 serving 里的 cache 语义 |
| `FlashAttention` | `flash_attn_varlen_*`、`flash_attn_with_kvcache(...)` | 官方 README 明确支持 `variable sequence lengths`、`arbitrary Q/KV sequence lengths`、`MQA/GQA`、`rotary embeddings`、`ALiBi`、`paged KV cache`、`softcapping`；`flash_attn_with_kvcache` 还显式暴露 `k_cache/v_cache`、`k/v` 追加、`cache_seqlens`、`block_table`、`softmax_scale`、`window_size`、`alibi_slopes` | 一旦进入真实 inference，`q_vs_kv_length_relation`、`kv_layout`、`position_semantics`、`kv_update_contract`、`score_modifiers` 都会立即从“抽象属性”变成函数签名的一部分 |
| `FlashInfer` | `single_prefill_with_kv_cache`、`BatchPrefillWithPagedKVCacheWrapper`、`BatchPrefillWithRaggedKVCacheWrapper`、`append_paged_kv_cache` | prefill 接口不是单一函数，而是围绕 `ragged/paged KV`、`plan/run`、`custom_mask`、`return_lse`、`q_scale/k_scale/v_scale`、`actual_seq_lens_q/kv` 来组织；同时提供独立的 paged KV append API | `sequence_layout` 和 `kv_layout` 必须分开；`outputs_and_stats`、`numeric_format`、`kv_update_contract` 都应该是 prefill 正式维度，不是实现细节 |
| `cuDNN Frontend SDPA` | `scaled_dot_product_attention` graph attributes、`paged_attention_k_table/v_table`、`seq_len_q/seq_len_kv`、`set_diagonal_alignment`、`set_score_mod` | 官方文档明确写出支持 `MHA/MQA/GQA`、任意 `s_q/s_kv`、ragged/padded layout、paged attention、bottom-right 对齐、sliding window、bias、softcap、stats 输出 | 一个底层通用 attention 接口如果想覆盖主流 inference prefill，至少要把 `layout`、`paged table`、`seq_len_q/kv`、`mask alignment`、`score_mod`、`stats` 做成一等配置项 |
| `xFormers` | `xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None, scale=None, ...)` | 核心接口把 `attn_bias` 作为通用 mask/bias 抽象；支持 `Mq != Mkv`；官方文档说明 `MQA/GQA` 是实验性前向功能；但不管理 KV cache | `mask_semantics` 不应只理解为 `causal/noncausal`，还应允许更一般的 `bias` 抽象；但 cache 相关维度不能指望从纯 attention 算子自动长出来 |

### 4.2 推理框架与 Runtime 形态

| 对象 | 典型 prefill 相关接口或配置 | 从接口能直接看出的支持重点 | 对我们设计维度的启发 |
| --- | --- | --- | --- |
| `vLLM` | `enable_chunked_prefill=True`、`enable_prefix_caching=True`、attention backend 选择 `FLASH_ATTN/FLASHINFER/TRITON/FLEX` | `chunked prefill`、`prefix caching`、paged block 管理、以及后端自动选择都是系统级能力；prefix caching 文档直接围绕 KV block 的 `allocate/append/free/eviction` 展开 | `prefix reuse`、`paged_kv`、`chunked prefill` 更像 runtime 维度，不应混进一个单一 kernel 的最小接口，但必须在 prefill 规划里被单列 |
| `TensorRT-LLM` | `GPT attention operator`、`paged context attention`、`KVCacheConfig`、`use_paged_context_fmha` | 官方明确写出 attention op 会 `populate KV cache`；同时支持 `contiguous/paged KV`、`chunked context`、`INT8/FP8 KV cache`、`sliding window`、`cyclic KV cache`、`sink tokens` | 真实 production 栈里，`kv_update_contract`、`kv_layout`、`numeric_format`、`mask/window/sink` 是被一起设计的；而且 `paged_kv` 不是长期项，而是主流路线 |
| `SGLang` | `chunked_prefill_size`、`prefill_attention_backend`、`disable_radix_cache`、`enable_mixed_chunk` | 官方首页和参数文档把 `RadixAttention`、`chunked prefill`、`paged attention`、`prefill-decode disaggregation`、`prefill/decode backend 分离` 都当作基础能力；attention backend 页面还直接给出 `Native Page Sizes`、`FP8 KV Cache`、`Chunked Prefix Cache` 维度表 | 从 runtime 角度看，prefill 除了算法能力外，还天然带有 `page_size`、`prefix cache`、`prefill backend` 与 `decode backend` 分离这类系统约束 |

### 4.3 接口反推结论

基于上面的生态接口调研，我们把 prefill 相关接口定位为三类：

1. 必须单列的契约维度：这些维度已经反复出现在主流库和框架的函数签名、wrapper 参数或 runtime 配置里。
2. 必须保持正交的契约维度：这些维度经常同时出现，但描述的问题不同，不能在接口设计里相互替代。
3. 需要明确责任边界的跨层维度：这些能力同时牵涉 kernel/operator 和 runtime/cache manager，必须说明哪一部分由 OP 承接，哪一部分由 runtime 承接。

#### 4.3.1 必须单列的契约维度

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

#### 4.3.2 必须保持正交的契约维度

调研后最值得强调的是下面几组区分：

- `sequence_layout` 和 `kv_layout` 必须分开  
  `FlashInfer` 同时有 ragged KV 和 paged KV 的 wrapper，这已经说明“输入是否 packed/ragged”和“cache 是否 paged”是两个正交问题。

- `mask_semantics` 和 `position_semantics` 必须分开  
  `FlashAttention` 同时支持 `causal/window` 和 `RoPE/ALiBi`，`cuDNN` 同时有 `diagonal_alignment/band` 和 `bias/score_mod`。这说明“谁能看谁”和“位置信息如何编码”不是一回事。

- `kv_update_contract` 和 `kv_layout` 必须分开  
  是否 `append` / `inplace update`，和 cache 是连续还是分页存储，是两个不同问题。

#### 4.3.3 Kernel 维度与 Runtime 维度的边界

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

这里把 `kv_layout` 归到 runtime-heavy 维度，并不表示 paged KV 不应该成为独立 OP。更准确的拆分是：page allocation、page reuse、eviction、prefix sharing 和 `cache_seqlens` 生命周期是 runtime/cache manager 的责任；但一次 paged attention 计算本身需要消费 `block_table`、`cache_seqlens`、physical page storage，并执行 gather/read/append，这部分必须成为 kernel/OP 的正式契约。因此 `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp` 暴露的是 paged attention 的计算接口，不是 page manager。

这一区分很重要，因为一个“prefill 规划”既不能只写 kernel 语义，也不能把所有 runtime 机制都硬塞进同一个 attention 算子接口。

### 4.4 Kernel / Runtime 责任边界

这一节只解决一个问题：哪些信息必须进入 kernel 计算的函数签名，哪些信息应该在 OP 层、model executor 或 serving runtime 里提前准备好。

这里的边界不是“这个能力重不重要”，而是“kernel 在这一拍计算时是否必须直接消费它”。

#### 4.4.1 Kernel 计算必须输入的签名

如果一个信息直接决定这一拍 attention 如何计算，它就应该进入 kernel/operator 的显式输入、attribute 或 dispatch 约束。

典型判断标准是：

- 不给这个信息，kernel 无法构造正确的 score matrix。
- 不给这个信息，kernel 无法正确读取 Q/K/V 或 KV cache。
- 不给这个信息，kernel 无法决定 causal 对齐、position 对齐或 head mapping。
- 不给这个信息，kernel 无法决定输出是否包含额外 stats。

典型例子：
- `q_lens / kv_lens / cache_lens`  
  决定本次 attention 的有效范围，以及 `q_len != kv_len` 时的 causal 对齐。
- `cu_seqlens` 或 packed layout metadata  
  决定 kernel 如何在 packed batch 中找到每个 request 的 Q/K/V。
- `block_table` / paged cache metadata  
  决定 kernel 如何从 logical token/page 找到 physical KV cache。
- `position_ids` / `cache_position` / RoPE offset  
  决定当前 token 的位置编码和 cache 中已有 token 的位置关系。
- `heads / heads_kv`  
  决定 MHA/GQA/MQA 的 head mapping。
- `sm_scale` / `softcap` / bias 或 mask 参数  
  决定 softmax 前 logits 如何变换。
- `return_lse` / stats flag  
  决定 kernel 是否需要产生额外统计输出。

这些信息可以由上层准备，但一旦进入计算，它们必须以明确形式被 kernel/operator 消费。

#### 4.4.2 OP 层或 Runtime 层负责准备的数据

另一类信息不应该直接变成 kernel 的“能力宣称”。它们描述的是请求如何被组织、cache 如何被管理、哪些数据可以复用。

典型例子：

- prefix cache 是否命中
- 哪些 request 共享同一段 prefix
- page 如何分配、回收、evict
- prompt 如何切成 chunk
- continuous batching 如何把不同请求拼成一个 batch
- prefill 和 decode 是否使用不同 backend
- cache scale、block table、sequence metadata 从哪里生成

这些事情会影响 kernel 收到什么输入，但它们本身通常不是 kernel 在单次计算里能独立决定的。它们更适合由 OP wrapper、model executor、cache manager 或 serving scheduler 负责。

#### 4.4.3 跨层能力的正确表达方式

很多 serving 能力都横跨两层，写文档时需要拆开表达：

| 能力 | OP / Runtime 准备什么 | Kernel / Operator 消费什么 |
| --- | --- | --- |
| chunked prefill | runtime 决定 chunk 边界和调度顺序 | 当前 chunk 的 Q/K/V、`q_len`、`cache_len`、position offset、append 位置 |
| prefix reuse | cache manager 判断命中、管理共享 prefix | 已存在的 KV cache、有效 prefix length、current suffix Q/K/V |
| paged KV | cache manager 分配 physical page 并维护 block table | paged KV storage、`block_table`、`cache_lens`、append offset |
| mixed prefill batch | scheduler 把不同状态请求拼成 batch | per-request `q_lens/cache_lens`、packed offsets、page metadata |
| FP8 KV cache | model/runtime 提供 scale 来源和 cache storage policy | FP8 cache tensor、K/V scale、dequant/quantize 语义 |

因此更准确的表述不是“kernel 支持 prefix cache”或“kernel 支持 chunked prefill”，而是：

- OP/runtime 准备 prefix-hit 或 chunked prefill 所需的数据状态。
- kernel/operator 消费这些状态，并完成一次语义明确的 attention + cache update。

这样写可以避免两个误解：一是把 page allocation、prefix hit 判断这类系统问题塞进 kernel；二是把 `q_len != kv_len`、paged gather、append offset 这类计算必需信息当成 runtime 外围细节。

### 4.5 生态调研结论

如果只保留最扎实的结论，可以压成下面几条：

1. `prefill` 的设计不能只参考训练态 dense attention 接口，必须参考 serving 场景下的 cache-aware 接口。
2. `q_len != kv_len` 不是边角 case，而是主流库显式支持的核心能力。
3. `varlen/ragged` 和 `paged_kv` 都已经是主流接口的一等能力，不适合放到很后的“长期支持项”里。
4. `RoPE/ALiBi`、`scale/softcap/bias`、`return_lse/stats` 这些也已经不是附属选项，而是正式接口维度。
5. `prefix reuse`、`chunked prefill`、`page manager` 更像 runtime 层能力，但如果 release plan 不把它们单列出来，规划就会失真。

## 五、能力取舍与首发边界

第三章列出的维度可以继续展开成大量排列组合：不同 layout、cache contract、position 语义、score modifier、数值格式和 runtime 状态彼此相乘后，不可能在一次发布里全部实现。这里的目标不是做一个覆盖所有组合的“大而全 attention 接口”，而是挑出最重要、最能接入 serving 主路径的功能，尽力覆盖主流算子库和推理框架已经证明有价值的能力面。

在这个前提下，真实 inference 系统里最容易让一个 prefill 路径“看起来能算，但其实接不上系统”的，通常是下面几个维度：

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

### 5.1 取舍原则

列出全部维度之后，下一步不是把所有能力都塞进首发。否则维度表会自然滑向“大而全接口”，最后用户、runtime 和 kernel 都很难形成稳定心智。

本轮取舍主要看三类依据：

| 依据 | 看什么 | 对接口设计的影响 |
| --- | --- | --- |
| 主流模型需求 | 新模型真实使用的 head topology、position、KV cache 和 score modifier | 决定哪些模型语义必须进入 release-facing contract |
| 主流算子库 / serving runtime | FlashAttention、FlashInfer、cuDNN、vLLM、TensorRT-LLM、SGLang 已经把什么做成一等接口 | 决定哪些维度不能藏在 benchmark 或 kernel 内部 |
| TileOps 设计逻辑 | spec-driven manifest、OP contract 稳定、kernel dispatch 可替换、benchmark 可统计 | 决定哪些东西适合作为公开 OP，哪些只应作为 dispatch target |

基于这个取舍，当前 prefill family 的能力可以分三类。

**优先支持**

| 功能 | 原因 |
| --- | --- |
| GQA/MQA/MHA 统一 head topology | 主流模型常见，`heads / heads_kv` 足以表达 |
| dense prefill | correctness reference 和固定长度 baseline |
| packed varlen prefill | heterogeneous batch 是 serving 基础能力 |
| contiguous KV cache prefill + append | 单请求、本地推理、FP8 contiguous cache 的基线 |
| paged KV cache prefill + append | serving 主路径，和 vLLM / FlashInfer / cuDNN 方向一致 |
| `q_len <= kv_len` bottom-right causal | chunked prefill 和 prefix-hit continuation 必需 |
| external RoPE 和 cache-aware fused RoPE | 位置语义必须和 cache position 对齐 |
| Neox full / partial RoPE | Llama/Qwen 类现代 GQA 模型需要 |
| `sm_scale` / `softcap` | 已是主流 attention interface 的一等 score modifier |
| fp16 / bf16 baseline | 当前 release 的原生计算格式 |
| FP8 KV cache storage | serving 场景下重要的 cache bandwidth / capacity 路径；本次定义 storage + dequant / quantize |
| FP8 Tensor Core attention compute | FP8 cache storage 之后的关键性能路径；需要独立命名和 kernel dispatch，避免和 storage-only 语义混淆 |
| manifest-backed benchmark | spec-driven repo 必须让能力进入 workload / benchmark / nightly 统计 |

**暂缓，但要保留清楚边界**

| 功能 | 暂缓原因 |
| --- | --- |
| per-head / per-token scale | 会扩大 metadata 和 kernel pipeline 复杂度，应在 FP8 baseline 后做 |
| public `return_lse` | kernel 可内部使用 stats，但公开返回会影响 OP 心智，先保持 output-only |
| arbitrary bias / generic mask extension | 需要通用 mask/bias contract，不能只加零散参数 |
| sliding window / local chunk mask | 属于 mask 语义，应单独 issue，不塞进 RoPE 或 FP8 |
| QK norm / NoPE layer dispatch | 属于模型 attention block 或 layer routing，不属于单个 prefill OP 的首发核心 |
| YaRN / MRoPE / Llama scaling | 是 RoPE 频率或多轴 position 语义，应独立设计 |
| H200 / WS / TMA dispatch 优化 | 是性能路线，不改变公开 OP signature |

**不进入本轮主路径**

| 功能 | 当前处理 |
| --- | --- |
| `q_len > kv_len` causal prefill | 非主流 serving prefill 形态，不作为当前目标 |
| GPT-J / non-Neox fused RoPE | standalone RoPE 可保留兼容，不进入 fused GQA prefill 主 benchmark |
| 完整 prefix cache runtime | page allocation、eviction、prefix sharing 属于 serving runtime |
| page manager 对象封装 | 当前 OP 消费 `block_table`，不管理 page 生命周期 |
| 任意模型级 attention block | TileOps 先做 operator family，不把完整模型 block 塞进单个 OP |

这样分层以后，公开 OP 的设计就有了因果关系：优先支持项决定当前 OP family 的稳定契约；暂缓项进入后续 issue；不进入本轮的内容不污染首发接口。

## 六、本次发布范围

前面的章节已经把 `MVP` 和“进阶支持项”合并成同一个目标：本次发布不只做 correctness baseline，而是直接形成一组可以接入 serving 主路径的 GQA prefill operator family。

这一章只列发布清单，不展开接口细节。具体参数、layout contract 和命名理由见第七章。

### 6.1 发布的公开 OP

本次发布面向用户暴露下面几类 OP：

| OP | 定位 | 主要覆盖的 prefill 方式 |
| --- | --- | --- |
| `GroupedQueryAttentionPrefillFwdOp` | dense no-cache prefill | 离线 correctness、固定长度 benchmark、本地基础路径 |
| `GroupedQueryAttentionPrefillVarlenFwdOp` | packed varlen no-cache prefill | 多请求变长 full prefill、serving 对照路径 |
| `GroupedQueryAttentionPrefillWithKVCacheFwdOp` | contiguous KV cache-aware prefill | 单请求 cache-aware full prefill、chunked continuation、prefix-hit continuation 的连续 cache 形态 |
| `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp` | paged KV cache-aware prefill | serving 主路径、paged cache、heterogeneous batch、chunked / prefix-hit continuation |
| `GroupedQueryAttentionPrefillWithFP8KVCacheFwdOp` | contiguous FP8 KV cache storage prefill | FP8 KV cache storage + kernel 内 dequant / append quantize |
| `GroupedQueryAttentionPrefillPagedWithFP8KVCacheFwdOp` | paged FP8 KV cache storage prefill | paged serving 路径上的 FP8 KV cache storage |
| `GroupedQueryAttentionPrefillWithFP8KVCacheTensorCoreFwdOp` | contiguous FP8 KV cache Tensor Core prefill | FP8 KV cache storage + FP8 Tensor Core attention compute |
| `GroupedQueryAttentionPrefillPagedWithFP8KVCacheTensorCoreFwdOp` | paged FP8 KV cache Tensor Core prefill | paged serving 路径上的 FP8 Tensor Core attention compute |

这里把 FP8 分成两个公开层次：`FP8KVCache` 表示 **KV cache storage 是 FP8，但 attention 主计算仍按 fp16/bf16 语义执行**；`FP8KVCacheTensorCore` 表示 **KV cache storage 是 FP8，attention compute 也进入 FP8 Tensor Core / WGMMA 路线**。两者都在本次发布目标内，但命名必须区分，避免用户把 storage-only 版本误解成完整 FP8 attention。

### 6.2 OP 到内部 Kernel 的 Dispatch 关系

公开 OP 不暴露 split-K、warp specialization、TMA、persistent scheduling 等 kernel 策略。OP 层根据 shape、dtype、layout、cache contract 和硬件能力选择内部 kernel。

本次发布按公开 OP 组织内部 dispatch target：

**`GroupedQueryAttentionPrefillFwdOp`**

- dense prefill kernel：处理 dense padded Q/K/V，完成 causal full prefill。
- 不读写外部 KV cache，主要作为 correctness reference、固定长度 benchmark 和本地基础路径。

**`GroupedQueryAttentionPrefillVarlenFwdOp`**

- packed varlen prefill kernel：消费 packed Q/K/V、`cu_seqlens` 和 per-request length metadata。
- 在一个 batch 内处理不同 prompt 长度，但不承担外部 KV cache update。

**`GroupedQueryAttentionPrefillWithKVCacheFwdOp`**

- contiguous cache attention kernel：读取 old contiguous KV cache 与 current K/V，完成 cache-aware attention。
- contiguous cache append kernel：把 current K/V 写回 caller-owned contiguous cache。
- 实现上 attention 和 append 可以分离，也可以后续融合；OP contract 始终是 attention + append。

**`GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp`**

- paged cache attention kernel：通过 `block_table`、`cache_lens` 等 metadata 读取 old paged KV cache。
- paged cache append kernel：按 page/block metadata 写入 current K/V。
- page allocation、eviction 和 prefix sharing 仍由 runtime 负责。

**`GroupedQueryAttentionPrefillWithFP8KVCacheFwdOp`**

- contiguous FP8 cache attention kernel：读取 FP8 old cache，并按 `k_scale/v_scale` dequant 后参与 fp16/bf16 attention。
- contiguous FP8 cache append kernel：将 current K/V 按同一 scale 量化写入 FP8 contiguous cache。
- 这个 OP 是 FP8 KV cache storage-only 路径，不表示 attention compute 已经走 FP8 Tensor Core。

**`GroupedQueryAttentionPrefillPagedWithFP8KVCacheFwdOp`**

- paged FP8 cache attention kernel：通过 `block_table` gather paged FP8 cache，并按 `k_scale/v_scale` dequant。
- paged FP8 cache append kernel：将 current K/V quantize 后写入 paged FP8 cache。
- 这个 OP 同时消费 paged metadata 和 FP8 scale metadata。

**`GroupedQueryAttentionPrefillWithFP8KVCacheTensorCoreFwdOp`**

- contiguous FP8 Tensor Core attention kernel：读取 FP8 cache，并让 attention MMA operand 走 FP8 Tensor Core / WGMMA。
- contiguous FP8 cache append kernel：复用 storage-only FP8 append 语义，把 current K/V quantize 后写回 caller-owned contiguous cache。
- 与 `FP8KVCache` storage-only 版本的差异在 compute policy，而不是 cache ownership。

**`GroupedQueryAttentionPrefillPagedWithFP8KVCacheTensorCoreFwdOp`**

- paged FP8 Tensor Core attention kernel：通过 `block_table` 读取 paged FP8 cache，并让 attention compute 进入 FP8 Tensor Core / WGMMA 路线。
- paged FP8 cache append kernel：复用 paged FP8 append 语义，将 current K/V quantize 后写回 paged cache。
- 这是 paged serving 场景下的 FP8 compute 主力路径。

如果某些路径在实现上做成 fused kernel，也不改变公开 OP 契约。文档和 manifest 仍按 OP 能力描述，kernel fusion 只是 dispatch implementation。

同时，split-K、split-Q、persistent scheduling、warp specialization、TMA 等属于本次发布的内部性能目标。它们应该作为具体 kernel variant 或 dispatch policy 出现在实现和 benchmark/manifest 里，但不作为用户可见的 OP 参数暴露。

### 6.3 本次发布实现的功能

本次发布的功能面按第三章的维度列如下：

| 维度 | 本次发布范围 |
| --- | --- |
| `sequence_layout` | `dense_padded`、`packed_varlen` |
| `kv_layout` | `no_cache`、`contiguous_kv`、`paged_kv` |
| `q_vs_kv_length_relation` | `q_len == kv_len`、`q_len < kv_len` |
| `mask_semantics` | causal prefill，采用 cache-aware / bottom-right 对齐语义 |
| `position_semantics` | external RoPE；cache-aware fused RoPE；显式 position offset / cache position |
| `head_topology` | MHA、GQA、MQA，统一由 `heads` 和 `heads_kv` 表达 |
| `sparsity_pattern` | dense attention |
| `score_modifiers` | `sm_scale`、`softcap` |
| `numeric_format` | fp16/bf16 compute；FP8 KV cache storage + dequant / quantize；FP8 Tensor Core attention compute |
| `attention_shape` | 主流 `qk_dim == vo_dim == dim` 路径 |
| `kv_update_contract` | `append_kv`，caller-owned cache |
| `outputs_and_stats` | public `output_only`；kernel 内部可保留 online softmax stats |
| `batch_variability` | heterogeneous batch，包括不同 `q_lens`、`cache_lens`、page table |
| `modality_prefix_behavior` | text-only；多模态特殊前缀不进入本次发布 |

这一版发布后，prefill operator family 应能覆盖：

- dense full prefill
- packed varlen full prefill
- contiguous KV cache-aware full prefill
- contiguous chunked / prefix-hit continuation
- paged KV cache-aware full prefill
- paged chunked / prefix-hit continuation
- fp16/bf16 baseline compute
- FP8 KV cache storage 路径
- FP8 Tensor Core attention compute 路径

### 6.4 本次发布不包含的内容

下面这些能力要明确不混入本次发布的公开 contract：

| 能力 | 处理方式 |
| --- | --- |
| sliding window / local chunk mask | 单独 issue，放在 dense serving 主路径稳定后 |
| arbitrary mask / generic bias pipeline | 单独设计通用 mask/bias contract |
| sparse prefill | 本轮只做 dense prefill，sparse 另行规划 |
| public `return_lse` | 暂不作为公开返回；需要时另开接口增强 |
| prefix cache runtime | prefix hit 判断、共享块管理、eviction、ref count 属于 runtime |
| page manager | OP 消费 `block_table`，不管理 page 生命周期 |
| YaRN / MRoPE / NoPE layer / QK norm | 属于更复杂 position 或模型 block 语义，单独设计 |

### 6.5 与主流生态相比仍欠缺的能力

本次发布完成后，TileOps 的 GQA prefill 在 operator-facing 能力上已经覆盖主流 serving attention 的核心路径：varlen、paged KV、`q_len < kv_len`、GQA/MQA、RoPE offset、softcap、FP8 KV cache 和 FP8 Tensor Core compute 都进入发布目标。本节只讨论算子级功能 gap，不讨论 wrapper、graph build、scheduler、page manager、prefix cache manager 这类框架级能力。

**相对 FlashAttention**

TileOps 本次发布覆盖 varlen、GQA/MQA、cache-aware prefill、paged KV、RoPE 和 softcap 这些主路径能力。主要 gap 是：

- FlashAttention 已有 window / local attention 语义，本次发布不包含 sliding window / local chunk mask。
- FlashAttention 暴露 ALiBi 等 position/bias 相关能力，本次发布只把 RoPE 作为主 position path。
- FlashAttention 的成熟实现覆盖大量 head dim、GPU 架构和 kernel 特化组合；TileOps 本次发布只把这些作为内部 dispatch / benchmark 目标逐步补齐。

**相对 FlashInfer**

TileOps 本次发布覆盖 paged KV、varlen、append、FP8 KV cache 和 FP8 Tensor Core compute 的核心算子能力。主要 gap 是：

- FlashInfer 支持 `return_lse` 和 custom mask；TileOps 本次发布 public contract 仍保持 `output_only`，也不做 generic mask。
- FlashInfer 的 prefill 侧对 ragged/paged、mask、lse、量化 scale 的组合更丰富；TileOps 本次发布只覆盖 serving 主路径组合。
- FlashInfer 的量化接口包含更丰富的 scale 入口；TileOps 本次发布先固定 per-layer per-K/V tensor scalar scale，不覆盖 per-head、per-token、per-page scale。

**相对 cuDNN Frontend**

cuDNN Frontend 的 attention operation 能表达 layout、mask、bias、stats、paged table、diagonal alignment 等大量算子属性。TileOps 本次发布主要 gap 是：

- 不覆盖通用 bias pipeline、arbitrary mask、prefix-lm、block mask 等组合式 mask 语义。
- 不把 `return_lse` / stats 做成公开稳定输出。
- 对 layout 和 numeric format 的组合做了发布裁剪，不追求一次性覆盖所有合法组合。

**相对 vLLM**

只看 vLLM attention backend 的算子能力，TileOps 本次发布能覆盖 paged/cache-aware prefill 主路径。主要 gap 是：

- vLLM 后端能力表里常见的 sliding window、ALiBi、部分 attention bias / mask 组合，本次发布不覆盖。
- vLLM 的 FP8 KV cache scale 支持可以覆盖 per-tensor 和 per-head 等不同粒度；TileOps 本次发布先固定 per-layer per-K/V tensor scalar scale。
- vLLM runtime 能把 paged KV 与 prefix/chunk 状态组织成统一后端输入；TileOps OP 只消费已经准备好的 `block_table`、length metadata 和 cache tensor，不定义更高层输入格式。

**相对 TensorRT-LLM**

只看 TensorRT-LLM GPT attention operator / context attention 的算子能力，TileOps 本次发布的 gap 是：

- TensorRT-LLM 覆盖 sliding window、sink token、cyclic KV cache 等和 mask/cache 语义绑定的 attention 变体；TileOps 本次发布不覆盖。
- TensorRT-LLM 同时覆盖 INT8/FP8 KV cache 等更多量化 cache 路径；TileOps 本次发布只把 FP8 KV cache 作为量化主线。
- TensorRT-LLM 的 paged context attention 已在多种 runtime 配置下打磨成熟；TileOps 本次发布只定义 caller-owned `block_table` / cache tensor 的 OP contract。
- FP8 进入本次发布，但校准、scale 来源、checkpoint metadata 管理仍由上层传入；OP 只消费本层 scale。

**相对 SGLang**

只看 SGLang attention backend 暴露的算子能力，TileOps 本次发布的 gap 是：

- SGLang 后端能力表里和 chunked prefix cache、page size、FP8 KV cache 组合相关的输入约束更多；TileOps 本次发布只定义当前 OP 所需的 page table、length 和 scale 输入。
- SGLang 支持的 attention backend 组合会覆盖更多硬件和模型配置；TileOps 本次发布需要通过 manifest / benchmark 逐步补齐这些组合。
- SGLang 的 FP8 KV cache 当前依赖 runtime 提供 scale 来源；TileOps OP 同样消费 scale，但不定义 runtime scale 文件、checkpoint metadata 或 backend policy。

因此这一节的结论不是“TileOps 缺完整 runtime”，而是：本次发布已经覆盖 serving prefill 的主算子路径，但在更广的 mask/bias、position 语义、stats 返回、量化 scale 粒度、head dim / 硬件组合覆盖上，仍需要后续继续补齐。

## 七、TileOps OP Family 设计

这一节的目标不是给出某一种实现，而是把 `prefill` operator family 的命名、layout 和接口契约定义清楚，作为后续实现、评审和测试的共同基线。

### 7.1 设计原则

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

### 7.2 OP 命名

推荐采用一套 operator family，而不是一个参数极其臃肿的单接口。

从现有 `TileOPs` 风格看，attention 命名大致遵循下面几条：
- 使用完整语义类名，而不是短函数名
- 采用 `PascalCase`
- 前向统一以 `FwdOp` 结尾
- cache 相关语义直接进入类名，如 `DecodeWithKVCacheFwdOp`
- `Varlen`、`SlidingWindow`、`Paged` 这类数据组织或语义修饰词直接进入类名

因此这一节的推荐命名，尽量与现有风格对齐。

#### 7.2.1 用户可见的稳定命名

- `GroupedQueryAttentionPrefillFwdOp`
- `GroupedQueryAttentionPrefillVarlenFwdOp`
- `GroupedQueryAttentionPrefillWithKVCacheFwdOp`
- `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp`
- `GroupedQueryAttentionPrefillWithFP8KVCacheFwdOp`
- `GroupedQueryAttentionPrefillPagedWithFP8KVCacheFwdOp`
- `GroupedQueryAttentionPrefillWithFP8KVCacheTensorCoreFwdOp`
- `GroupedQueryAttentionPrefillPagedWithFP8KVCacheTensorCoreFwdOp`

这八个名字分别对应：
- dense BSHD 的基础 prefill
- packed/varlen 的普通 prefill
- 带连续 KV cache 的 prefill
- 带 paged KV cache 的 prefill
- 带连续 FP8 KV cache storage 的 prefill
- 带 paged FP8 KV cache storage 的 prefill
- 带连续 FP8 KV cache Tensor Core compute 的 prefill
- 带 paged FP8 KV cache Tensor Core compute 的 prefill

从 release plan 的设计目标看，推荐先把 `GroupedQueryAttention*` 这一组作为主接口族，
把 `MHA` 与 `MQA` 视为它的特例，而不是平行维护多套接口。

#### 7.2.2 不单独命名 `mha_prefill` / `mqa_prefill` 的原因

因为在接口层，三者本质上只是 `head_topology` 不同：
- `MHA`: `heads == heads_kv`
- `GQA`: `heads > heads_kv` 且整除
- `MQA`: `heads_kv == 1`

如果接口设计得当，调用方不需要关心名字差异，只需要正确设置：
- `heads`
- `heads_kv`

#### 7.2.3 与现有 TileOPs 风格的对齐

如果未来真要落到 `TileOPs` 代码里，命名和组织上建议进一步保持这些约定：

- 文件继续放在：
  - `tileops/ops/attention/gqa.py`

- `__all__` 暴露类名，而不是函数名

- kernel map key 使用 snake_case，并与类名语义对应，例如：
  - `gqa_prefill_varlen_kernel`
  - `gqa_prefill_with_kv_cache_kernel`
  - `gqa_prefill_paged_with_kv_cache_kernel`
  - `gqa_prefill_with_fp8_kv_cache_kernel`
  - `gqa_prefill_paged_with_fp8_kv_cache_kernel`
  - `gqa_prefill_with_fp8_kv_cache_tensor_core_kernel`
  - `gqa_prefill_paged_with_fp8_kv_cache_tensor_core_kernel`

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

### 7.3 Layout 契约

下面定义一套推荐的 canonical layout，供 operator contract 使用。

#### 7.3.1 Packed / Varlen 激活布局

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

#### 7.3.2 Contiguous KV Cache 布局

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

#### 7.3.3 Paged KV Cache 布局

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

### 7.4 OP 接口族

不建议把所有路径塞进一个超大接口。  
建议发布时对外暴露四类稳定入口。

#### 7.4.1 `GroupedQueryAttentionPrefillFwdOp`

用于：
- dense BSHD prefill
- q 和 kv 都已经 materialized，不直接操作外部 KV cache 的场景
- 固定长度 batch 或 benchmark / reference-friendly path

```python
class GroupedQueryAttentionPrefillFwdOp(Op):
    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool = True,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        dtype: torch.dtype = torch.float16,
        accum_dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        ...

    def forward(
        self,
        q: torch.Tensor,                 # [B, S_q, Hq, D]
        k: torch.Tensor,                 # [B, S_kv, Hkv, D]
        v: torch.Tensor,                 # [B, S_kv, Hkv, D]
    ) -> torch.Tensor:
        ...
```

这个接口覆盖 dense baseline：
- `q_len == kv_len` 的标准 prompt prefill
- `q_len < kv_len` 的 chunk/reference path
- bottom-right causal alignment
- MHA/GQA/MQA head topology

#### 7.4.2 `GroupedQueryAttentionPrefillVarlenFwdOp`

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

#### 7.4.3 `GroupedQueryAttentionPrefillWithKVCacheFwdOp`

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

#### 7.4.4 `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp`

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

#### 7.4.5 与当前 TileOPs 风格进一步对齐后的接口取舍

为了让这份设计更像现有 `TileOPs` op，而不是通用框架函数，这里明确几条取舍：

- 首发版本优先使用单一 `dim`
  也就是接口层先假定主流 `Dqk == Dv == dim` 路径。  
  `qk_dim != vo_dim` 属于后续扩展能力，可以在将来再拆成更细接口或新增结构参数。

- `window_size_left/right`、`position_mode`、`sm_scale`、`softcap`、`return_lse` 放在 `__init__`
  这些更像一次 op 实例的固定语义配置，和 `TileOPs` 现有把 `is_causal`、`page_size` 放在构造阶段的习惯一致。

- `forward()` 尽量只接动态张量和少量动态长度信息
  例如 `q`、`k`、`v`、`cu_seqlens_*`、`real_seqlen_kv`、`block_table`。  
  这样也更方便后续做 kernel dispatch 和 autotune cache。

### 7.5 与 TileOPs 现有风格的差异和取舍

即使尽量对齐 `TileOPs` 现有风格，这里还是有几处是有意和当前实现拉开的。

#### 7.5.1 使用 `Prefill` 语义层的原因

`TileOPs` 现有 attention 命名里：
- `MultiHeadAttentionFwdOp`
- `GroupedQueryAttentionFwdOp`
- `GroupedQueryAttentionDecodeWithKVCacheFwdOp`
- `GroupedQueryAttentionSlidingWindowVarlenFwdOp`

这里的 `Fwd` 更接近“训练态或通用前向”。  
而本文讨论的是明确的 inference `prefill` 语义，所以建议新 operator 直接把 `Prefill` 写进名字里，避免和现有普通 dense `FwdOp` 混淆。

#### 7.5.2 `Varlen` 在命名中的位置

按现有 TileOPs 风格，修饰词通常放在主体名词后、生命周期后缀前。  
因此更推荐：
- `GroupedQueryAttentionPrefillVarlenFwdOp`

而不是：
- `GroupedQueryAttentionVarlenPrefillFwdOp`

前者和现有：
- `GroupedQueryAttentionSlidingWindowVarlenFwdOp`
- `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp`

在阅读习惯上更一致。

#### 7.5.3 新增 `Prefill` 语义词的必要性

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

#### 7.5.4 保留清晰参数签名的原因

虽然 `TileOPs` 现有接口是类 + `forward()`，但在设计文档里保留清晰的参数签名仍然有价值，
因为它能更直接体现 operator contract。  
真正落地实现时，仍应遵循 `TileOPs` 风格：
- `__init__` 固定结构参数和大多数语义配置
- `forward()` 接收动态 tensor 输入

### 7.6 参数语义

为了避免接口语义漂移，建议把下面这些规则固定下来。

#### 7.6.1 命名和形状规则

- 统一使用 `heads`、`heads_kv`
- 必须满足 `heads % heads_kv == 0`
- `heads == heads_kv` 视为 `MHA`
- `heads_kv == 1` 视为 `MQA`
- 发布版本接口层优先采用单一 `dim`
- 如果后续需要 `qk_dim != vo_dim`，建议新增显式结构参数，而不是提前把首发接口做得过重

#### 7.6.2 Mask 规则

- `is_causal=False` 且没有其他限制时：不加可见性约束
- `is_causal=True`：采用 causal 语义
- 提供 `custom_mask` 时：走显式 mask 语义
- `window_size_left/right >= 0` 时：表示 sliding-window 约束

建议约定：
- `custom_mask` 的优先级高于普通 causal/window 推导
- `custom_mask` 的 shape 和 layout 必须在文档中单独固定

#### 7.6.3 Position 规则

- `position_mode="none"`：不使用额外位置语义
- `position_mode="rope"`：必须有清晰的 q/k 对应位置输入
- `position_mode="alibi"`：必须提供 `alibi_slopes`

对于 cache-aware prefill，建议固定：
- 新写入 KV 的位置通过 `cache_positions_new` 或同等语义字段给出
- 不能依赖“默认从 0 开始”这种隐式规则
- old cache 中的 K 被视为已经按对应 logical position 完成位置编码，不能在当前 prefill 调用中重复编码

RoPE 不应只表达为 `rope=True/False`。现代模型至少需要区分：

| 参数 | 含义 |
| --- | --- |
| `rope_style` | rotation layout，例如 `neox` / `non_neox`，首发 fused GQA prefill 以 `neox` 为主 |
| `rotary_dim` | 参与 RoPE 的前缀维度，`None` 表示整个 head dim |
| `rope_base` / `rope_theta` | 频率基底 |
| `rope_scaling` | Llama scaling / YaRN / MRoPE 等频率或位置扩展策略，建议后续独立设计 |

首发 release-facing 语义建议：

- 外置 RoPE 路径允许调用方预先旋转 `q/k_new`，OP 只消费已经编码好的张量。
- fused RoPE 路径只作为 cache-aware OP 内部实现路径。
- fused RoPE 的实现不必等同于单个物理 kernel launch；OP 层可以先运行 append kernel，再运行 attention kernel。
- 对 GQA 来说，append 的天然 dispatch 维度是 `Hkv`，attention 的天然 dispatch 维度是 `Hq`，不应为了“单 kernel”把 KV append 放进 query-head CTA 分支。
- fused RoPE 首发支持 Neox-style full RoPE 和 partial RoPE：
  - `rotary_dim is None` 等价于 `rotary_dim = head_dim`
  - `rotary_dim < head_dim` 时，仅前 `rotary_dim` 维旋转，尾部维度保持原样
  - 这覆盖 Qwen3.5 full-attention layer 这类 partial RoPE 场景
- GPT-J / non-Neox legacy RoPE 可继续由 standalone RoPE op 覆盖，不应成为 fused GQA prefill 的主 benchmark 路径。
- Llama4 的 NoPE layer、local chunk mask、QK norm、attention temperature tuning 属于模型级 attention 语义，应拆成后续 issue。

#### 7.6.4 Score Modifier 规则

- `sm_scale is None` 时，默认使用 `1 / sqrt(Dqk)`
- `softcap is None` 表示不开启 softcap
- `temperature` 若未来支持，应和 `sm_scale` 的组合关系单独定义

### 7.7 返回值契约

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

### 7.8 不采用单一总入口的原因

理论上可以做一个统一入口，但不建议把它作为第一阶段稳定接口。

原因是：
- packed varlen / contiguous cache / paged cache 的参数集合差异太大
- 一个总入口会产生大量 mutually-exclusive optional args
- 对调用方和测试都不友好

更稳妥的做法是：
- 对外稳定暴露四类接口：dense baseline、packed varlen、contiguous cache、paged cache
- 如有需要，在更上层提供一个 convenience wrapper 做 dispatch

这样可以同时兼顾：
- operator contract 清晰
- runtime 对接自然
- 后续 kernel 演化自由度更大

### 7.9 接口决策汇总

前面的调研可以进一步压缩成一个用户心智问题：

**用户在调用 prefill 时，首先关心的不是 kernel 变体，而是这批数据和 KV cache 是怎么组织的。**

因此 TileOps 的公开接口应该围绕数据组织和 cache 契约拆分，而不是围绕 Hopper/WGMMA/WS/TMA、是否 fused、是否走某个特殊 kernel 拆分。

#### 7.9.1 接口形态归纳

不同项目的接口风格不一样，但它们大体落在几类模式里：

- `PyTorch SDPA` / `xFormers`
  更像通用 attention API。它们把 `mask`、`causal`、`scale`、`GQA` 作为语义参数，但基本不接管 serving KV cache 生命周期。

- `FlashAttention`
  把 dense、varlen、KV cache 路径拆成不同函数。`flash_attn_with_kvcache` 一类接口会直接暴露 `k_cache/v_cache`、`cache_seqlens`、`block_table`、RoPE、softcap 等参数，说明 cache-aware prefill 需要独立接口，而不是普通 dense attention 的一个小开关。

- `FlashInfer`
  更偏 serving operator。它把 ragged/paged KV、plan/run、paged append、`return_lse`、量化 scale 等拆成明确 wrapper 或辅助 API，说明 `sequence_layout` 和 `kv_layout` 应该被分开建模。

- `TensorRT-LLM` / `vLLM` / `SGLang`
  更偏 runtime。它们把 paged KV、chunked prefill、prefix cache、FP8 KV cache、backend selection 放在系统配置或运行时策略里，而不是让用户直接选择某个底层 kernel。

- `cuDNN Frontend`
  更像完整 graph API。它能表达非常多 attention 维度，包括 ragged/paged、score modifier、stats、mask alignment 等，但这类接口对 TileOps 当前公开 OP 来说过重，更适合作为能力边界参考，而不是直接照搬。

#### 7.9.2 公开入口按 Layout / Cache Contract 拆分

基于上面的调研，TileOps 首发发布接口建议保持八个用户可见入口：

- `GroupedQueryAttentionPrefillFwdOp`
  用于 dense BSHD baseline prefill，覆盖固定长度 batch、reference-friendly path 和 q/kv 已经 materialized 的场景。

- `GroupedQueryAttentionPrefillVarlenFwdOp`
  用于 packed varlen prefill，不直接管理外部 KV cache。

- `GroupedQueryAttentionPrefillWithKVCacheFwdOp`
  用于 contiguous KV cache prefill，读取 old cache，并把 current chunk 的 K/V append 回 cache。

- `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp`
  用于 paged KV cache prefill，通过 `block_table` 寻址物理页，是后续 serving runtime 的主力对接入口。

- `GroupedQueryAttentionPrefillWithFP8KVCacheFwdOp`
  用于 contiguous FP8 KV cache storage prefill，读取旧 FP8 cache 时 dequant，append 当前 K/V 时 quantize 写回。

- `GroupedQueryAttentionPrefillPagedWithFP8KVCacheFwdOp`
  用于 paged FP8 KV cache storage prefill，是 paged serving 路径上的量化 KV cache 入口。

- `GroupedQueryAttentionPrefillWithFP8KVCacheTensorCoreFwdOp`
  用于 contiguous FP8 KV cache Tensor Core prefill，storage 和 attention compute 都进入 FP8 路径。

- `GroupedQueryAttentionPrefillPagedWithFP8KVCacheTensorCoreFwdOp`
  用于 paged FP8 KV cache Tensor Core prefill，是 paged serving 路径上的高性能 FP8 attention 入口。

这个拆分和用户心智更一致：

- 输入是否 packed / varlen，是 `sequence_layout` 问题。
- KV 是 contiguous 还是 paged，是 `kv_layout` 问题。
- 这次调用是否 append 新 K/V，是 `kv_update_contract` 问题。
- KV cache storage 是否量化，是 `numeric_format` / cache storage policy 问题。
- attention compute 是否走 FP8 Tensor Core，是 `numeric_format` / compute policy 问题。
- 这些问题比“内部走哪个 kernel”更适合作为公开 OP 边界。

#### 7.9.3 Cache Ownership：Caller-Owned Cache

cache-aware prefill 的底层 OP 应该采用 caller-owned cache 契约：

- 调用方或 runtime 预先分配 `k_cache/v_cache` 或 `k_pages/v_pages`。
- 调用方或 runtime 准备 `cache_seqlens`、`block_table`、page allocation 等 metadata。
- OP 在调用前校验 shape、dtype、capacity、page table 边界。
- kernel 在已有 storage 上原地 append current chunk。
- OP / kernel 不负责分配 cache 空间，也不负责更新 `cache_seqlens`。
- 调用成功后，runtime 再把对应 request 的 `cache_seqlens` 增加 current chunk 长度。

这个设计符合底层算子库心智，也和 FlashInfer / FlashAttention 这类 operator API 接近。原因是 page allocation、prefix sharing、eviction、cyclic cache、request ownership 都属于 runtime/cache manager 的职责，不应放进单个 attention kernel。

但这并不完全等于 serving 用户心智。serving 用户通常想表达的是：

```text
把当前 chunk prefill 进这个 request/session 的 cache
```

而不是手动管理：

```text
k_pages / v_pages / block_table / cache_seqlens / page_size / max_pages_per_req
```

因此文档里需要明确分层：

- **底层 TileOps OP**：保持 caller-owned cache，便于接入已有 runtime。
- **可选高层 wrapper**：未来可以提供 `PagedKVCache` / `KVCacheConfig` 这类对象，封装 page 分配、`cache_seqlens` 更新和 metadata 管理，底层仍调用相同 OP。

对 FP8 KV cache 也采用同样边界：

- FP8 cache tensor 和 scale / scale metadata 由外部 runtime 预分配并传入。
- `k_new/v_new` 仍是新算出来的 `fp16/bf16`。
- OP 负责 attention 计算中读取 old FP8 cache 并按 scale 使用。
- OP 负责把 current `k_new/v_new` 量化后写入 caller-owned FP8 cache。
- 如果未来支持 per-token-head / per-page-head scale，相关 scale storage 也应由 runtime 传入，OP 只负责写对应位置。

当前已有实现已经符合这个方向：contiguous 和 paged append 都在 kernel 内原地写 cache；current chunk 的 attention 直接读 `k_new/v_new`，old prefix 才从 cache 读。后续 FP8 实现应在这个结构上增加 old-cache dequant 和 append-time quantize，而不是改成 OP 自动分配 cache。

#### 7.9.4 Kernel 形态不进入公开接口

下面这些应该是内部 dispatch 目标，而不是用户需要直接选择的 OP：

- WGMMA kernel
- WS / warp-specialized kernel
- TMA kernel
- fast path / gather path
- fused RoPE kernel
- H200-specialized kernel

原因是这些名字描述的是实现策略，不是用户正在表达的 prefill 语义。

对用户来说，更稳定的问题是：

- 我的输入是 dense、packed varlen，还是 cache-aware？
- 我的 KV cache 是 contiguous 还是 paged？
- 我是否要在这次调用里 append K/V？
- 我是否需要 RoPE、softcap、FP8 KV cache、`return_lse`？

内部则可以在相同 OP 契约下根据 dtype、head dim、page size、sequence length、硬件和 feature flag 选择不同 kernel。

#### 7.9.5 RoPE 决策摘要

RoPE 不应作为 `fuse_rope=True/False` 这种小开关来理解。接口层更稳定的语义是 `position_mode="rope"`，fused 只是实现路径。

在 cache-aware prefill 中，RoPE 的核心契约是：

- old cache 中的 K 已经在正确位置编码空间，不能重复旋转。
- current chunk 的 Q 和 `k_new` 使用绝对位置旋转。
- append 到 cache 的 `k_new` 必须是已经旋转后的 K。
- partial RoPE 是首发能力；NoPE layer、QK norm、YaRN/MRoPE 等模型级扩展单独设计。

详细子变体、行业接口和首发边界见第八章 `RoPE / Position Semantics`。

#### 7.9.6 Score Modifier 决策摘要

从 PyTorch、FlashAttention、xFormers、cuDNN 的接口看，`scale`、`softcap`、bias/mask 都是主流 attention 接口的一等语义。

TileOps 当前阶段建议：

- `sm_scale` 保持基础参数。
- `softcap` 作为明确的一等参数。
- `temperature` 暂不单独暴露，除非后面明确它和 `sm_scale` 的组合语义。
- 通用 `score_mod` callable / expression 不进入首发范围。
- `attn_bias` / `custom_mask` 以后可以作为更一般 score path 扩展，但需要单独定义 shape、layout 和广播规则。

这个决策让接口足够覆盖现阶段模型需求，同时避免一开始就引入过大的通用 score modifier 系统。

benchmark 层面不应把 softcap 展开成完整矩阵。softcap 更适合作为少量 sentinel case；主 benchmark 应围绕现代 serving 场景，例如 Qwen3.5 full-attention layer 的 paged KV + partial RoPE。

#### 7.9.7 FP8 / 量化 KV 决策摘要

FP8 KV cache 不是普通 dtype 小开关，而是 cache storage policy 和 compute policy 的组合。TileOps 首发同时承诺两条路径：FP8 KV cache storage + kernel-internal dequant，以及 FP8 KV cache + Tensor Core attention compute。

storage-only 版本的接口心智是：

- `k_cache/v_cache` 用 `fp8_e4m3fn` 存储。
- `q/k_new/v_new/o` 仍为 `fp16` 或 `bf16`。
- `k_scale/v_scale` 为必传 `fp32[1]`，scope 是 per layer、per K/V cache tensor。
- append 时把 current K/V 量化写回 caller-owned FP8 cache。

Tensor Core 版本沿用同一组 cache storage 和 scale 语义，但 attention MMA operand 走 FP8 Tensor Core / WGMMA 路线，并用 `FP8KVCacheTensorCore` 在公开 OP 名里明确区分。

详细 scale 粒度、其他项目依据和 TC 命名见第八章 `FP8 / Quantized KV Cache`。

#### 7.9.8 `return_lse` 决策摘要

FlashInfer、xFormers、cuDNN 都说明 `lse` / stats 是真实存在的接口需求，尤其在 backward、debug、组合 attention、数值分析或部分高级 runtime 中有价值。

但 TileOps 当前 OP 习惯是 kernel 可内部返回 `(output, lse)`，公开 OP 默认只给 `output`。因此发布计划里建议：

- 默认保持 `output_only`，符合当前 TileOps 调用习惯。
- 把 `return_lse=True` 作为低优先级 open question 或后续增强项。
- 如果未来暴露，返回契约统一为 `(output, lse)`，不要为每个 prefill 变体设计不同返回对象。

#### 7.9.9 Paged KV 决策摘要

Paged KV 首发采用 `block_table[b, logical_page] -> physical_page` 的 caller-owned cache 契约。OP 只消费 runtime 已经准备好的 page table，并按 page-major physical storage 读写 KV；page allocation、eviction、prefix sharing 和 request 生命周期不属于 OP。

详细 page ownership、layout 和 CUDA Graph 约束见第八章 `Paged KV / Cache Layout`。

#### 7.9.10 H200 / WS / TMA 决策摘要

H200 上针对 prefill 的 WS/TMA-friendly 优化应该在 release scope 里，因为这是发布性能的一部分。

但它不改变用户 API：

- 用户仍然调用同一个 prefill OP。
- OP 或 kernel map 根据硬件、shape、dtype、page layout 选择实现。
- benchmark 需要覆盖这些路径，帮助判断 dispatch 是否应该切到 H200-specialized kernel。

因此 H200/WS/TMA 是 release plan 的性能目标，不是额外公开 OP。

## 八、复杂变体展开

顶层维度表解决的是“有哪些轴”，但 release 设计真正容易出错的，往往是大变体内部的分层。RoPE、FP8、paged KV 都不是一个普通布尔开关；它们分别有自己的子变体、行业接口形态、runtime/kernel 边界和发布阶段。

因此后续设计和 issue 拆分建议采用下面这个模板：

| 项 | 要回答的问题 |
| --- | --- |
| 子变体 | 这个大变体内部还能拆成哪些互斥或可组合能力 |
| 行业做法 | FlashAttention、FlashInfer、cuDNN、vLLM、TensorRT-LLM、SGLang 等项目如何暴露它 |
| TileOps 决策 | 哪些进入公开 OP contract，哪些留给 kernel dispatch，哪些交给 runtime |
| 首发边界 | 第一阶段支持什么，明确不支持什么 |
| follow-up | 后续增强如何命名和拆 issue，避免污染首发接口 |

### 8.1 RoPE / Position Semantics

RoPE 属于 `position_semantics` 的复杂变体，不是简单的 `fuse_rope=True/False`。它至少包含这些子维度：

| 子维度 | 典型取值 | 说明 |
| --- | --- | --- |
| RoPE 执行位置 | external、operator-fused | external 表示 Q/K 已由模型层旋转；operator-fused 表示 OP 接收原始 Q/K 并在 cache-aware path 中旋转 current chunk |
| RoPE style | Neox、GPT-J、模型特化扩展 | 首发以 Neox 为主；GPT-J 保留兼容，不作为主 benchmark |
| rotary dim | full、partial | full 表示 `rotary_dim == head_dim`；partial 支持 Qwen 类 `rotary_dim < head_dim` |
| position source | `position_ids`、`cache_positions_new`、offset | cache-aware prefill 必须使用绝对位置，不能在 chunk 内 reset |
| old/current KV 处理 | old cache already rotated、current K rotate before append | old cache 不能重复旋转；current K 写入 cache 前必须处在正确 RoPE 空间 |
| NoPE / mixed layer | RoPE layer、NoPE layer | NoPE 是模型层 routing，不应变成 fused RoPE kernel 的隐藏分支 |

主流项目给出的接口信号是：FlashAttention 的 cache-aware 接口把 RoPE、cache seqlens、block table 放在同一类调用里；FlashInfer 把 `pos_encoding_mode`、`rope_scale`、`rope_theta` 放进 plan/run；cuDNN Frontend 把 position/mask/bias 等作为 attention graph attributes；模型侧则通过 config 暴露 RoPE 类型和维度。共同点不是“所有项目都叫 fused RoPE”，而是它们都承认位置语义必须和 cache position 对齐。

TileOps 的决策是：

- 公开语义优先叫 `position_mode="rope"`，`fuse_rope` 只是实现或过渡开关。
- fused RoPE 只处理 current chunk 的 Q 和 `k_new`；old cache 视为已经旋转。
- cache append 写入的是 rotated K。
- partial RoPE 是首发能力，因为现代 GQA 模型会用到。
- Llama4 NoPE layer、QK norm、local chunk mask、YaRN/MRoPE 等不混进首发 fused RoPE PR。

首发边界：

```text
rope_style       = neox
rotary_dim       = None or <= head_dim
position_source  = cache-aware absolute positions
old_cache_k      = already rotated
current_k_append = rotate then append
```

后续如果扩展 GPT-J、YaRN、MRoPE 或 NoPE dispatch，应各自单独成 issue。这样 RoPE 章节服务的是位置语义，不承担所有模型 attention block 的复杂度。

### 8.2 FP8 / Quantized KV Cache

FP8 属于 `numeric_format` 的复杂变体。它内部至少要拆成 storage、compute、scale 和 append 四组问题，不能只写“支持 FP8”。

| 子变体 | 含义 | 首发处理 |
| --- | --- | --- |
| FP8 KV cache storage + dequant | K/V cache 用 FP8 存，attention 计算仍按 fp16/bf16 语义；old cache 在 kernel 内 dequant | 首发支持 |
| FP8 KV cache + Tensor Core compute | cache 是 FP8，attention MMA operand 也走 FP8 Tensor Core / WGMMA | 首发支持 |
| FP8 activation-only compute | Q/K/V activation 本身以 FP8 参与 attention，但不以 FP8 KV cache 为接口中心 | 独立路线 |
| FP8 activation transport | FP8 只作为中间传输格式 | 暂不进入 GQA prefill 首发 |
| scale granularity | per-tensor、per-kv-head、per-token-head、per-page、per-block | 首发只做 per-tensor |

公开项目给出的依据是：

| 项目 | 做法 | 对我们的结论 |
| --- | --- | --- |
| TensorRT-LLM | `FP8_KV_CACHE` 使用 shape `[1]` 的 `kv_cache_scaling_factor`，generation 中 on-the-fly dequant | per-tensor scalar 是成熟 baseline |
| SGLang | `--kv-cache-dtype fp8_e4m3/fp8_e5m2`，scale 从 checkpoint 或 JSON 来，当前只支持 scalar scale | scale 来源交给 runtime，OP 消费当前 layer scale |
| vLLM | 支持 per-tensor `[1]`，也支持 per-attention-head scale | per-head 有价值，但应作为增强项 |
| FlashInfer | `q_scale/k_scale/v_scale` 暴露在 attention run 接口，另有更复杂 `kv_cache_sf` | 首发显式 scale 参数即可，复杂 scale metadata 后续再设计 |

TileOps 的首发 contract 固定为：

```text
storage dtype         = fp8_e4m3fn
q/k_new/v_new/o dtype = fp16 or bf16
k_scale/v_scale dtype = fp32[1]
scale scope           = per layer, per K/V cache tensor
scale required        = true
dequant semantic      = fp_value = fp8_value * scale
append semantic       = fp8_value = cast_fp8(fp_value / scale)
```

这里的 `[1]` 不是 model-global scale。每个 attention layer 的 K cache 和 V cache 各有一个 scalar scale；runtime 可以保存 `[num_layers, 2]` 的 scale 表，但调用 OP 时只传当前 layer 的 `k_scale[1]` 和 `v_scale[1]` view。

Tensor Core 版本不改变上面的 cache storage contract。它额外定义 compute policy：attention MMA operand 走 FP8 Tensor Core / WGMMA 路线，公开 OP 名用 `FP8KVCacheTensorCore` 和 storage-only 的 `FP8KVCache` 区分。这里不把 activation-only FP8 attention 合并进同一个接口心智。

TileOps 的命名决策是：

| 能力 | 公开 OP 名 | manifest tag |
| --- | --- | --- |
| FP8 KV cache storage + dequant | `GroupedQueryAttentionPrefillWithFP8KVCacheFwdOp`、`GroupedQueryAttentionPrefillPagedWithFP8KVCacheFwdOp` | `fp8-kv-cache-dequant` |
| FP8 KV cache + Tensor Core compute | `GroupedQueryAttentionPrefillWithFP8KVCacheTensorCoreFwdOp`、`GroupedQueryAttentionPrefillPagedWithFP8KVCacheTensorCoreFwdOp` | `fp8-kv-cache-tc` |

首发明确不支持：

- per-kv-head / per-token-head scale
- dynamic scale 计算
- e5m2 多格式 dispatch
- per-page 或 per-block FP8 scale

本次发布通过 `FP8KVCache` 和 `FP8KVCacheTensorCore` 两组公开 OP 名区分 storage-only 路径和 Tensor Core compute 路径，避免 `FP8KVCache` 被误读成已经覆盖完整 FP8 attention。

### 8.3 Paged KV / Cache Layout

Paged KV 属于 `kv_layout` 的复杂变体。它不是“把 K/V 多传一个 block table”这么简单，而是 runtime cache manager 和 kernel gather 语义的交界面。

| 子维度 | TileOps 首发决策 |
| --- | --- |
| page ownership | caller/runtime owner，OP 不分配、不回收 page |
| page table | `block_table[b, logical_page] -> physical_page` |
| physical layout | page-major flat storage |
| prefix sharing | runtime 负责共享 page 和引用生命周期 |
| cache append | OP 可以写 caller-owned cache，但不改变 page ownership |
| CUDA Graph | page table buffer shape 固定，内容可变 |

这个变体的关键边界是：TileOps OP 只负责“根据这次调用给定的 page table 正确读写 KV 并计算 attention”，不负责 page allocation、eviction、prefix cache 命中判断或 request 生命周期。

### 8.4 简单变体的处理原则

不是所有变体都需要单独成章。下面这些目前可以留在接口章节和测试矩阵里：

| 变体 | 原因 |
| --- | --- |
| `sm_scale` | 数学语义稳定，输入是 scalar |
| `softcap` | 是明确 score modifier，首发只需 sentinel benchmark |
| `head_topology` | `heads / heads_kv` 已能表达 MHA/GQA/MQA |
| `output_only` vs `return_lse` | 会影响返回值，但子变体不多；先保持 output-only |

如果后续某个简单变体开始引入 layout、metadata、runtime 交互或多种 scale/mask 子策略，就应该从接口章节提升到专题章。

## 九、结论摘要

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
- 本次发布的重点不是把所有 feature 都做全，而是先把真实 runtime 最离不开的几件事做对：
  - `causal`
  - `varlen`
  - `q_len != kv_len`
  - `rope + offset`
  - `mha/gqa/mqa`
  - `append_kv`
  - `paged_kv`
  - `fp8_kv_cache_storage`

## 十、设计优先级

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
