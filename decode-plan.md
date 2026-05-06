# GQA Decode Plan：发布收敛计划

日期：2026-05-04

目标：把 GQA decode 从当前已有的 contiguous-cache / paged-cache 功能面，直接收敛到可接现代 serving runtime 的 release-facing operator family。本轮不以 contiguous-only MVP 作为发布目标，而是把 paged KV cache、heterogeneous batch、score modifier、split-K、CUDA Graph-friendly metadata 和 benchmark 作为 release baseline。

本文只讨论 GQA decode operator family，不讨论完整 serving runtime、调度器、prefix cache 命中策略、page manager 生命周期或 sampling。

## 一、当前基线

从当前 TileOps GQA attention 代码和测试形态看，已有或接近已有的 release-facing decode 能力包括：

1. `GroupedQueryAttentionDecodeWithKVCacheFwdOp`
   - Q layout：`[B, Hq, D]`
   - contiguous KV cache：`k/v [B, Skv, Hkv, D]`
   - 构造参数包含 `batch`、`heads`、`heads_kv`、`seqlen_kv`、`dim`、`dtype`
   - 支持 GQA/MHA/MQA 统一表达：`heads` / `heads_kv`
   - 当前 forward 根据 `k.shape[1]` 推导 `real_seqlen_kv`
   - batch 内默认同一个 `real_seqlen_kv`
   - 当前语义是 read-only decode，不在 OP 内 append current K/V

2. `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp`
   - Q layout：`[B, Hq, D]`
   - physical KV cache：`k/v [P_tokens, Hkv, D]`
   - `real_seqlen_kv [B]` 表示每个 request 当前可见 KV 长度
   - `block_table [B, num_pages]` 表示 logical page 到 physical page 的映射
   - 构造参数包含 `page_size`
   - 当前测试覆盖多个 batch、多个 head ratio、多个 page size
   - 当前语义是 read-only paged decode，不在 OP 内 append current K/V

当前测试覆盖大致包括：
- contiguous decode output correctness
- paged decode output correctness
- fp16 / bf16 的 contiguous decode smoke
- paged decode 的不同 batch / head / page size 组合
- PyTorch SDPA 作为 reference

当前需要补齐或明确的点：
- contiguous decode batch 内 heterogeneous `real_seqlen_kv`
- paged decode 的非 page 对齐 `real_seqlen_kv`
- `heads == heads_kv`、`heads_kv == 1`、`heads / heads_kv in {2, 4, 8}` 的系统覆盖
- OP 层 shape / dtype / contiguity / capacity 校验
- `sm_scale` 和 softcap
- sliding window / sink attention 是否进入本轮 release
- split-K / split-KV 策略和内部 `lse` 合并契约
- FP8 KV cache read path
- CUDA Graph replay 所需的静态 buffer / batch bucket 契约
- manifest、workload、roofline、source metadata 和 benchmark 对齐

## 二、本轮 release baseline 定义

本轮 release baseline 的核心目标不是“feature 全部做满”，而是直接具备现代 LLM serving 的可发布性。由于 GQA prefill 已按进阶能力推进，decode 也应直接面向 serving 主力路径，而不是先发布只覆盖 homogeneous contiguous cache 的过渡能力。

baseline 完成时，至少应具备：

- 稳定的 paged KV cache decode 主力路径
- 稳定的 contiguous KV cache reference / fallback 路径
- batch 内不同 `real_seqlen_kv` 稳定
- `heads/heads_kv` 统一覆盖 MHA/GQA/MQA
- `sm_scale` / softcap 契约明确
- paged KV metadata 协议明确
- 长上下文 split-K / merge 策略明确
- fp16 / bf16 稳定覆盖
- FP8 KV cache 首发 dequant path 设计清楚，并完成 manifest-ready issue / PR 切分
- CUDA Graph replay 约束明确，至少不阻碍 runtime 做固定 batch bucket capture
- benchmark / H200 dispatch 纳入发布验证，而不是 release 之后才补

本轮明确不采用 prefill-style varlen packed KV 作为 serving decode 主接口。decode 的主路径是：

```text
paged decode:
k_pages/v_pages: [P_tokens, Hkv, D]
block_table:     [B, max_pages_per_req]
real_seqlen_kv:  [B]

contiguous decode:
k_cache/v_cache: [B, Skv_cap, Hkv, D]
real_seqlen_kv:  [B]
```

Varlen packed KV 可用于 reference、debug 或 bridge wrapper，但不作为本轮 release-facing serving decode path。

不属于本阶段的目标：

- 完整 continuous batching scheduler
- 完整 prefix cache runtime
- page allocation / eviction / reuse 策略
- speculative decoding scheduler
- multi-token verification kernel 全量发布
- DCP / decode context parallelism
- NVFP4 / INT8 / 低比特 KV cache
- 完整 sink attention / StreamingLLM 发布
- MLA decode 发布收敛

## 三、接口分层原则

用户应直接调用 OP；kernel 不作为用户主要心智模型。

公开 OP 按稳定数据契约拆分：

- `GroupedQueryAttentionDecodeWithKVCacheFwdOp`
  - single-token Q
  - contiguous KV cache
  - read-only attention

- `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp`
  - single-token Q
  - paged KV cache
  - `block_table`
  - `real_seqlen_kv`
  - serving runtime 主力接口

如果后续支持 fused append，应新增语义清楚的 cache-update variant，而不是悄悄改变现有 read-only OP：

- `GroupedQueryAttentionDecodeAppendWithKVCacheFwdOp`
  - contiguous KV cache
  - current `k_new/v_new` append
  - `cache_seqlens` 表示 append 前已有长度

- `GroupedQueryAttentionDecodePagedAppendWithKVCacheFwdOp`
  - paged KV cache
  - current `k_new/v_new` append
  - `block_table` 和 `cache_seqlens` 由 runtime 准备

OP 层负责 dispatch，kernel 层负责固定契约的实现。

不要按实现细节暴露 OP。例如不暴露：

- `GroupedQueryAttentionDecodeSplitKFwdOp`
- `GroupedQueryAttentionDecodeWgmmaFwdOp`
- `GroupedQueryAttentionDecodeCudaGraphFwdOp`
- `GroupedQueryAttentionDecodeSmallBatchFwdOp`

这些应作为 kernel class、dispatch target 或 runtime capture policy 存在。

### 3.1 OP Dispatch 与 TileLang JIT 使用约定

TileOps decode OP 应把 TileLang kernel 视为可复用的 compiled kernel object，而不是每个 decode step 动态生成的新 kernel。

推荐形态：

- 使用 TileLang lazy JIT factory 按静态 config 生成 kernel object
- OP 层缓存 kernel object 或 builder result
- dispatch key 只包含 shape / dtype / layout / architecture / split policy 等静态项
- `real_seqlen_kv`、`block_table`、slot mapping、KV cache 内容作为 tensor 内容动态变化

不应放入 JIT compile key 的项：

- `real_seqlen_kv[b]`
- 当前 batch 内实际最大 KV 长度
- 当前 request 实际 page 数
- `block_table` 的具体 page id
- KV cache 的当前内容

TileLang 文档中的 `eager` / `lazy` 是 JIT wrapper 模式，不等同于 serving 的 eager dispatch / CUDA Graph replay。GQA decode operator family 推荐优先使用 lazy JIT factory，因为它更适合 inspect、cache 和 reuse kernel object。

## 四、数据布局约定

### 4.1 Contiguous KV Cache Decode

首发 contiguous decode layout：

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `q` | `[B, Hq, D]` | current query |
| `k_cache` | `[B, Skv_cap, Hkv, D]` | contiguous K cache |
| `v_cache` | `[B, Skv_cap, Hkv, D]` | contiguous V cache |
| `real_seqlen_kv` | `[B]` 或 scalar | 每个 request 当前可见 KV 长度 |
| `o` | `[B, Hq, D]` | output |

首发建议把 `real_seqlen_kv` 正式作为 forward 参数，而不是从 `k.shape[1]` 隐式推导。

语义：

```text
visible(q_b, k_j) = j < real_seqlen_kv[b]
kv_head = q_head // (Hq / Hkv)
```

decode 的 causal 在单 token full-prefix 情况下等价于“当前 query 能看见所有已有 KV”。因此 OP 不需要暴露 bottom-right causal 作为用户主要心智模型；需要暴露的是 `real_seqlen_kv` 和 optional window/sink 约束。

### 4.2 Paged KV Cache Decode

首发 paged decode layout，也是本轮 serving 主力 layout：

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `q` | `[B, Hq, D]` | current query |
| `k_pages` | `[P_tokens, Hkv, D]` | flattened page-major physical K cache |
| `v_pages` | `[P_tokens, Hkv, D]` | flattened page-major physical V cache |
| `real_seqlen_kv` | `[B]` | 每个 request 当前可见 KV 长度 |
| `block_table` | `[B, max_pages_per_req]` | logical page 到 physical page |
| `o` | `[B, Hq, D]` | output |

必须满足：

- `P_tokens % page_size == 0`
- `physical_token = physical_page * page_size + page_offset`
- `block_table[b, logical_page]` 存 physical page id
- `num_logical_pages_b = ceil(real_seqlen_kv[b] / page_size)`
- attention mask 由 logical token position 和 `real_seqlen_kv` 决定，不由 physical page position 决定

首版继续使用 FlashAttention-like `block_table`，不引入 FlashInfer-style `indptr/indices/last_page_len` 作为公开 OP 参数。

原因：

- 与当前 TileOps decode paged 风格一致
- shape 固定，适合现有 OP 风格和 CUDA Graph bucket
- runtime 负责 page allocation / eviction / prefix sharing，OP 只消费已经准备好的 `block_table`

后续如需支持更动态 metadata，可以新增 wrapper 或新 OP，而不是让同一个 forward 同时接受两套 page table 格式。

### 4.2.1 Batch Slot 语义

decode OP 中的 `B` 表示 decode slot count，不等同于 active request count。serving runtime 可以把请求映射到固定 batch bucket 的 slot 中：

```text
B = B_bucket
active requests <= B_bucket
```

每个 slot 的有效 KV 由 metadata 决定：

- `real_seqlen_kv[b]`
- `block_table[b, :]`
- optional slot mapping / active request mapping

OP 不负责 continuous batching、request 生命周期或 slot assignment；OP 只按这一拍传入的 slot metadata 计算。

CUDA Graph replay 下 `B_bucket` 固定，inactive slot 由 runtime 填充。首版可用 `real_seqlen_kv[b] = 0` 表示 read-only decode 的 inactive slot；如果后续 append decode 需要区分“有效请求 old_len=0”和 inactive slot，应新增 `valid_mask` 或显式 slot mapping。

### 4.3 Optional Append Decode

如果需要 fused decode + append，建议单独定义 append variant。

contiguous append layout：

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `q` | `[B, Hq, D]` | current query |
| `k_new` | `[B, Hkv, D]` | current key |
| `v_new` | `[B, Hkv, D]` | current value |
| `k_cache` | `[B, Skv_cap, Hkv, D]` | caller-owned cache |
| `v_cache` | `[B, Skv_cap, Hkv, D]` | caller-owned cache |
| `cache_seqlens` | `[B]` | append 前已有 KV 长度 |

语义：

```text
old_len_b = cache_seqlens[b]
attention reads positions 0..old_len_b inclusive if current K/V participates
new token writes cache position old_len_b
```

这里有一个必须明确的选择：

1. `append_before_attention`
   - current token 的 K/V 参与本次 attention
   - 更接近 FlashAttention `flash_attn_with_kvcache(k/v != None)` 的语义

2. `append_after_attention`
   - 本次 attention 只读 old cache
   - current token 的 K/V 写给下一步使用

LLM 标准自回归 decode 通常需要当前 token 的 K/V 能参与当前 token attention，尤其当 Q/K/V 来自同一 token projection 时，应采用 `append_before_attention` 语义。若模型或 runtime 已经在外部 append，则 read-only decode OP 应保持只读，不重复写 cache。

## 五、阶段路线

### 阶段 0：当前基线收敛

目标：把现有 contiguous decode 和 paged decode 变成后续 split-K / FP8 / CUDA Graph 工作可以依赖的稳定基线。

必做：

- 确认 `GroupedQueryAttentionDecodeWithKVCacheFwdOp` 的 read-only 语义
- 确认 `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp` 的 read-only 语义
- 为 contiguous decode 增加显式 `real_seqlen_kv` 参数，或在文档中声明当前只支持 batch-homogeneous length
- 增加输入 shape / dtype / contiguity 校验
- 增加 head topology 校验：`heads % heads_kv == 0`
- 统一 output-only 返回契约
- 保留 kernel dispatch 在 OP 层

验收：

- `tests/ops/attention/test_gqa_decode.py` 通过
- `tests/ops/attention/test_gqa_decode_paged.py` 通过
- fp16 / bf16 contiguous decode 通过
- paged decode fp16 通过
- MHA/GQA/MQA head mapping correctness 通过

### 阶段 1：OP 公共契约与 dispatch 整理

目标：避免 GQA decode family 扩张时重复校验和 dispatch 逻辑，但不提前引入大一统 OP 继承层级。

公共 helper 职责：

- `heads` / `heads_kv` / `dim` / `dtype` 通用校验
- `groups = heads // heads_kv`
- MHA/GQA/MQA 通过 `heads/heads_kv` 统一表达
- decode Q layout 校验
- KV cache layout 校验
- output-only 解包规则
- `sm_scale=None` 默认 `1 / sqrt(dim)`

公共 helper 不负责：

- 统一 contiguous 和 paged 的 `forward()` 参数
- 引入 optional 大一统接口
- 决定 runtime 的 page allocation 或 CUDA Graph bucket

验收：

- 现有公开 OP 行为不变，或有明确 migration note
- 现有测试全量通过
- 新增 OP 时只需声明 layout-specific forward 和 kernel key / wrapper

### 阶段 2：Contiguous Decode 完善

目标：让 contiguous decode 成为可发布的基础路径，也作为 FP8 KV cache contiguous variant 的基线。

必做：

- 正式支持 `real_seqlen_kv [B]`
- batch 内不同 KV 长度 correctness
- 增加更多 GQA ratio：
  - `heads == heads_kv`
  - `heads_kv == 1`
  - `heads / heads_kv in {2, 4, 8}`
- 支持 `sm_scale`
- 支持 `softcap`
- 增加 short / medium / long cache benchmark

建议接口：

```python
forward(
    q,                # [B, Hq, D]
    k_cache,          # [B, Skv_cap, Hkv, D]
    v_cache,          # [B, Skv_cap, Hkv, D]
    real_seqlen_kv,   # [B]
)
```

验收：

- homogeneous length 与当前 reference 对齐
- heterogeneous length 与 per-request materialized SDPA reference 对齐
- `real_seqlen_kv` 非 block 对齐
- fp16 / bf16 覆盖
- `sm_scale` / softcap 单独测试

### 阶段 3：Paged Decode 完善

目标：具备 serving runtime 对接的主力接口。

当前公开 OP：

```python
GroupedQueryAttentionDecodePagedWithKVCacheFwdOp
```

建议接口保持：

```python
forward(
    q,                # [B, Hq, D]
    k_pages,          # [P * page_size, Hkv, D]
    v_pages,          # [P * page_size, Hkv, D]
    real_seqlen_kv,   # [B]
    block_table,      # [B, max_pages_per_req]
)
```

必做语义：

- `real_seqlen_kv` 表示每个 request 可见 KV 长度
- old KV 根据 `block_table` gather
- page tail 不要求有效数据
- attention mask 由 logical position 决定，不由 physical page position 决定
- block_table 中未使用 page 不应被读取

必补测试：

- 单 batch single-page
- 单 batch multi-page
- batch 内不同 page table
- `real_seqlen_kv` 非 page 对齐
- `real_seqlen_kv` 非 block 对齐
- batch 内不同 `real_seqlen_kv`
- page tail 填随机值不影响输出
- output 与 materialized reference 对齐

当前 page size 建议：

- smoke：`16 / 32 / 64 / 128`
- full：加入更长 context 和更大 batch
- 不在同一 release 中同时引入 CSR-style metadata

### 阶段 4：Score Modifiers 与 Mask

目标：补齐发布阶段常见 decode 接口契约。

优先级：

1. `sm_scale`
2. `softcap`
3. `window_left`
4. `sink_attention`
5. simple bias / custom mask

建议：

- `sm_scale=None` 时默认 `1 / sqrt(dim)`
- `softcap=None` 或 `0` 表示 disabled
- `softcap>0` 时在 QK score 进入 online softmax 前执行 softcap
- sliding window 首发只支持 left window：`kv_pos >= real_seqlen_kv[b] - window_left`
- sink attention 首发先作为 open question，除非 serving 目标明确需要 StreamingLLM

验收：

- `sm_scale` 与 reference 对齐
- softcap 单独测试
- sliding window materialized reference 对齐
- paged 下 window 根据 logical position 生效

### 阶段 5：Split-K / Split-KV

目标：让长上下文 decode 具备可发布性能，并明确 stats / merge 契约。

decode 的 split-K 与 prefill 不同：
- Q 很短，KV 很长
- 对一个 request/head 的 KV 维度拆分后，需要合并 partial output
- 合并必须使用稳定的 online softmax stats，例如 partial max / partial lse
- split 数量影响性能、确定性和 CUDA Graph launch 拓扑

首发策略：

- 保留 `no_split` fast path
- 对长 `real_seqlen_kv` 启用 split-K
- kernel 内部可以返回或使用 partial `lse`
- 公开 OP 默认仍然 output-only
- dispatch 规则放在 OP 或 kernel wrapper，不暴露成用户主接口

建议首发 policy：

- short / medium context：`no_split`
- contiguous long context：优先按 token chunk split
- paged long context：优先按 page 或 page group split
- eager dispatch：允许根据 bucket / shape 选择 no-split 或 split-K kernel
- CUDA Graph replay：使用 fixed split policy，固定 `split_size` 和 `max_splits`

split-K 内部 workspace 建议：

```text
partial_m:   [B, Hq, max_splits]
partial_lse: [B, Hq, max_splits]
partial_o:   [B, Hq, max_splits, D]
```

merge 必须使用 online softmax stats 合并：

```text
m = max_i(partial_m_i)
lse = log(sum_i exp(partial_lse_i + partial_m_i - m))
o = sum_i exp(partial_lse_i + partial_m_i - m - lse) * partial_o_i
```

对于 CUDA Graph bucket，无效 split 不应改变 launch topology。无效 split 可以在 kernel 内写 identity stats，merge kernel 仍按固定 shape 执行。

待决策：

- split size 的默认值和 threshold 需要通过 benchmark 确定
- eager dispatch 下是否允许按当前 shape 动态选择 split 数
- 是否需要 fixed split size 以获得 batch-size invariant output
- 是否支持 deterministic merge

验收：

- split/no-split 与 reference 对齐
- split 边界非 page 对齐
- split 边界非 block 对齐
- long context benchmark 有基本吞吐记录
- 公开返回仍保持 output-only

### 阶段 6：位置语义

目标：把 decode 的 cache position 从隐式约定推进到正式接口。

首版建议：

- read-only decode 默认假设 cache 中的 K 已经按 logical position rotated
- Q 由调用方外置 RoPE，或由单独 RoPE op 使用 `cache_position` 处理
- decode OP 不默认重复旋转 cache 中的 old K

如果支持 fused RoPE：

- contiguous decode append variant 使用 `cache_seqlens[b]` 作为 current token position
- paged decode append variant 使用 `cache_seqlens[b]` 作为 current token position
- append 写入 cache/page 的是 rotated `k_new`
- old cache K 视为已经 rotated

验收：

- prefix-hit 后 decode 不出现 position reset
- `cache_position` 与 `real_seqlen_kv` 一致
- 外置 RoPE reference 测试

### 阶段 7：Numeric Format

目标：`fp16/bf16` 作为 release baseline 保持稳定；下一步进入 FP8 KV cache 的 serving storage path。

优先级：

1. fp16 contiguous / paged decode
2. bf16 contiguous / paged decode
3. contiguous `fp8_e4m3fn` KV cache, per-tensor scale, kernel-internal dequant
4. paged `fp8_e4m3fn` KV cache, per-tensor scale, kernel-internal dequant
5. per-kv-head scale
6. per-token-head / dynamic scale
7. int8 / nvfp4 / lower-bit KV cache

`fp8 kv cache` 首版需要明确：

- storage dtype：`float8_e4m3fn`
- Q/O dtype：`float16` 或 `bfloat16`
- old cache 读取时在 kernel 内按 scale dequant
- 首发 scale 粒度：per-tensor
- scale tensor：`k_scale` / `v_scale`，dtype `float32`，shape `[1]`
- 首发不承诺 FP8 Tensor Core attention compute
- append variant 如存在，append 时量化写入 caller-owned FP8 cache

验收：

- dtype matrix 测试
- 精度误差边界文档化
- old cache dequant correctness
- contiguous cache 和 paged cache 各有最小回归

当前建议：

- Decode FP8 KV cache 优先级高于 prefill FP8 compute，因为 decode 更受 KV cache bandwidth 影响。
- 首发采用 dequant path，避免在同一阶段处理 FP8 Tensor Core operand layout / swizzle、tile 级量化和 scale pipeline。
- per-page scale 暂不作为首发目标；per-block 更接近 NVFP4 / 低比特 block scaling，不应混进普通 FP8 KV cache 首发。

### 阶段 8：CUDA Graph Contract

目标：让 decode OP 的参数和 buffer 契约不阻碍 runtime 做固定 batch bucket capture。

OP 层需要明确：

- 哪些参数是构造期静态参数
- 哪些 tensor shape 在 replay 中必须固定
- 哪些 tensor 内容可以变
- paged metadata buffer 是否固定 shape
- split-K launch 拓扑是否可能随 `real_seqlen_kv` 改变

建议首发约束：

- `batch` 固定
- `max_pages_per_req` 固定
- `page_size` 固定
- `block_table` shape 固定，内容可变
- `real_seqlen_kv` shape 固定，内容可变
- no-split path graph-friendly
- split-K path 是否 graph-friendly 由 fixed split policy 决定

注意：

- CUDA Graph capture/replay 是 runtime 职责，不是 OP 单独完成的功能。
- OP 只需要保证它的 shape、buffer、dispatch 和 launch 拓扑有清楚契约。

### 8.1 Continuous Serving 中的动态 KV Cache

continuous serving 中，KV cache 每步都会变化，但这不应导致 TileLang kernel 重新编译。

OP / runtime 应区分：

```text
静态编译或 dispatch 项：
batch bucket, heads, heads_kv, dim, dtype, page_size,
max_pages_per_req, layout, split policy, target arch

动态 tensor 内容：
q, k_cache/v_cache content, real_seqlen_kv, block_table,
slot mapping, active request mapping
```

对于 paged decode，推荐 runtime 预分配固定 shape buffer：

```text
q_static:              [B_bucket, Hq, D]
k_pages/v_pages:       [P_tokens, Hkv, D]
real_seqlen_kv_static: [B_bucket]
block_table_static:    [B_bucket, max_pages_per_req]
o_static:              [B_bucket, Hq, D]
workspace_static:      fixed shape if split-K is enabled
```

每个 decode step 只更新这些 buffer 的内容，然后调用同一个 kernel object 或 replay 同一个 graph。

### 8.2 TileLang Kernel 的 CUDA Graph 使用方式

TileLang kernel 不需要写成特殊的 CUDA Graph kernel；需要写成 capture-friendly 的普通 kernel。

推荐使用 lazy JIT factory：

```python
@tilelang.jit(out_idx=[-1])
def build_gqa_decode_paged(
    batch: int,
    heads: int,
    heads_kv: int,
    dim: int,
    page_size: int,
    max_pages_per_req: int,
    total_page_tokens: int,
    max_splits: int = 1,
    dtype: str = "float16",
):
    @T.prim_func
    def kernel(
        q: T.Tensor((batch, heads, dim), dtype),
        k_pages: T.Tensor((total_page_tokens, heads_kv, dim), dtype),
        v_pages: T.Tensor((total_page_tokens, heads_kv, dim), dtype),
        real_seqlen_kv: T.Tensor((batch,), "int32"),
        block_table: T.Tensor((batch, max_pages_per_req), "int32"),
        o: T.Tensor((batch, heads, dim), dtype),
    ):
        ...
    return kernel
```

capture 前必须完成：

- JIT compile
- OP dispatch
- workspace allocation
- static buffer allocation
- warmup launch

capture / replay 中必须保持：

- tensor shape 固定
- tensor pointer 固定
- dispatch target 固定
- launch topology 固定
- split-K 的 `max_splits` 固定

replay 前允许变化：

- `q_static` 内容
- `real_seqlen_kv_static` 内容
- `block_table_static` 内容
- `k_pages/v_pages` 内容

TileLang profiler 可以用 `backend="cudagraph"` 做最小 capture-friendly 检查，但 release-facing CUDA Graph capture/replay 仍应由 runtime bucket 负责。

## 六、优先级建议

从当前状态继续推进的推荐顺序：

1. 当前 read-only decode OP 契约和校验收敛
2. paged decode 非 page 对齐和 heterogeneous batch 完善
3. batch slot / inactive slot / fixed bucket 语义文档化和测试覆盖
4. CUDA Graph-friendly static metadata shape
5. `sm_scale` / softcap
6. split-K / internal lse merge
7. manifest/workload/roofline/source metadata 对齐
8. benchmark matrix 和 nightly 形状覆盖
9. FP8 KV cache dequant path 设计和 manifest issue 收敛
10. contiguous decode 显式 `real_seqlen_kv [B]`，作为 reference / fallback 完善
11. paged FP8 KV cache read
12. contiguous FP8 KV cache read
13. CUDA Graph replay 契约验证
14. H200/Hopper dispatch 与更强 decode kernel 优化

注意：这里的顺序表达的是实现关注点，不表示 manifest 可以滞后。任何新增或改变 release-facing OP contract 的 PR，都必须在同一个 PR 中同步更新 manifest、workloads、roofline、source metadata 和对应 tests；不能先合实现、再用后续 PR 补 manifest。

## 七、风险与待决策项

### 1. Contiguous decode 是否显式传 `real_seqlen_kv`

当前倾向：

- 应显式传 `real_seqlen_kv [B]`。

原因：

- batch 内 heterogeneous decode 是 serving 基础能力
- 从 `k.shape[1]` 推导只适合 homogeneous 或 padded reference
- 与 paged decode 的 `real_seqlen_kv` 语义统一

### 2. 是否支持 fused append decode

当前倾向：

- 现有 read-only OP 保持只读。
- 如需 fused append，新增 append variant。

原因：

- read-only decode 和 append decode 的 cache 生命周期不同
- 悄悄改变现有 OP 会让 runtime 重复 append 或位置错位
- append variant 可以单独定义 `append_before_attention` 语义

### 3. Paged metadata 格式

当前决策：

- 首版固定 `block_table: [B, max_pages_per_req]`。
- 首版 physical cache 使用 flattened page-major layout：`[P_tokens, Hkv, D]`。
- `P_tokens % page_size == 0`。
- 不同时支持 `indptr/indices/last_page_len`。

待决策：

- 是否需要面向 FlashInfer-style metadata 新增 wrapper
- 是否需要高层 `PagedKVCache` 对象封装 page allocation 和 `real_seqlen_kv` 更新

### 4. 是否暴露 `lse` / stats

当前倾向：

- 公开 OP 主契约保持 output-only。
- split-K merge 所需 lse / stats 先作为 kernel internal。

待决策：

- 是否需要 `return_lse=True`
- 如果公开支持，是否只在 split-K / debug / partial attention 场景支持
- `lse` shape 是 `[B, Hq]` 还是带 split 维的 internal tensor

### 5. Split-K 与确定性

当前倾向：

- 不把 split-K 暴露为公开 OP。
- 公开 OP 保持 output-only。
- short / medium context 使用 no-split fast path。
- long context 由 OP 层 dispatch 到 split-K。
- paged long context 优先考虑 page 或 page group split。
- CUDA Graph bucket 中使用 fixed split policy，固定 launch topology。

待决策：

- split size 是否固定
- 是否要求 batch-size invariant output
- eager dispatch 下是否允许动态 split-K
- split-K merge 是否有 deterministic mode

建议：

- release 阶段先保证 correctness 和 benchmark 形状覆盖。
- deterministic merge 作为明确 open question，不混进首发必要条件，除非 serving 目标需要严格复现。

### 6. FP8 KV Cache 首发颗粒度

当前倾向：

- 首发只做 `fp8_e4m3fn` KV cache storage。
- 首发 scale 粒度为 per-tensor：`k_scale/v_scale` 各一个 `float32[1]`。
- 首发走 kernel-internal dequant path，不走 FP8 Tensor Core attention compute。

待决策：

- per-kv-head scale 是否作为第二阶段增强
- dynamic/per-token-head scale 是否需要与 paged metadata 一起设计
- FP8 Tensor Core decode 进入 H200/Blackwell 优化路线的具体触发条件

### 7. CUDA Graph Contract

当前倾向：

- 不在 OP 名字中暴露 CUDA Graph。
- `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp` 的参数和 buffer contract 应做到 graph-friendly。
- CUDA Graph capture/replay 是 runtime 职责，OP 负责不破坏 bucket contract。
- split-K 在 graph bucket 中按 fixed split size / fixed max splits 保持 launch 拓扑稳定。
- replay 中不允许根据 `real_seqlen_kv` 内容改变 kernel dispatch。

待决策：

- graph bucket 的默认 batch / max pages / max splits 如何划分
- inactive slot 使用 `real_seqlen_kv=0` 还是额外 `valid_mask`
- split-K graph bucket 是否和 no-split bucket 分开 capture

## 八、进阶支持项完成标准

进阶支持项完成时，应满足：

- `GroupedQueryAttentionDecodeWithKVCacheFwdOp` 稳定
- `GroupedQueryAttentionDecodePagedWithKVCacheFwdOp` 稳定
- read-only decode 语义明确
- 如新增 append variant，append-before-attention 语义明确
- `heads/heads_kv` 统一覆盖 MHA/GQA/MQA
- fp16/bf16 基础路径稳定
- contiguous decode 支持 batch 内不同 `real_seqlen_kv`
- paged decode 支持 batch 内不同 `real_seqlen_kv`
- paged KV metadata 协议明确
- page tail 和非 page 对齐边界稳定
- `sm_scale` 契约明确
- `softcap=None` / `softcap=0` / `softcap>0` 语义明确
- split-K / internal lse merge 契约明确
- 公开 OP 默认 output-only，`return_lse` 保持低优先级 follow-up 或明确暴露契约
- FP8 KV cache 首发 dequant path 契约明确
- CUDA Graph replay 约束明确
- 有 manifest-backed workloads、roofline、source metadata 和最小 benchmark
- H200/Hopper dispatch 不改变公开 OP 契约

完成后，operator-facing 能力应接近 FlashAttention / FlashInfer / cuDNN Frontend 的主流 decode 功能面，但仍不等于完整 serving runtime。
