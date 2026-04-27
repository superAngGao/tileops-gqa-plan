# GQA Prefill Plan：进阶支持项开发计划

日期：2026-04-27

目标：把 GQA prefill 从当前的 dense / contiguous-cache 基础能力，推进到 `prefill.md` 中定义的“进阶支持项”，也就是具备现代 serving 系统可发布性的 operator-facing 能力。

本文只讨论 GQA prefill operator family，不讨论完整 serving runtime、调度器、prefix cache 命中策略或 page manager 生命周期。

## 一、当前基线

当前已经完成或正在验证的能力：

1. `GroupedQueryAttentionPrefillFwdOp`
   - dense BSHD layout
   - 支持 `seq_len_q != seq_len_kv`
   - causal 使用 bottom-right alignment
   - 支持 GQA/MHA/MQA 统一表达：`heads` / `heads_kv`
   - kernel 层已有普通 TileLang 路径和 Hopper WGMMA 路径

2. `GroupedQueryAttentionPrefillWithKVCacheFwdOp`
   - dense BSHD current chunk
   - contiguous KV cache：`[B, Skv_cap, Hkv, D]`
   - `cache_seqlens` 表示 append 前已有 KV 长度
   - fused kernel 同时完成：
     - old KV 从 cache 读取
     - current chunk KV 从 `k_new/v_new` 读取
     - append `k_new/v_new` 到 `k_cache/v_cache`
   - `cache_seqlens` 不要求 block 对齐
   - current fused kernel 使用 `T.Pipelined`，但禁用 TMA lowering 和 warp-specialized lowering，以避开动态 cache/new 分流 load 的 mbarrier lowering 问题

当前测试覆盖：

- dense prefill `q_len == kv_len`
- dense prefill `q_len < kv_len`
- bottom-right causal mask 定向测试
- contiguous cache prefill output correctness
- contiguous cache append correctness
- old cache length 非 block 对齐
- batch 内不同 `cache_seqlens`

## 二、进阶支持项定义

进阶支持项的核心目标不是“feature 全部做满”，而是让 GQA prefill operator family 具备现代 LLM serving 的可发布性。

进阶完成时，至少应具备：

- 稳定的 dense prefill 路径
- 稳定的 contiguous KV cache prefill 路径
- packed / varlen prefill 路径
- paged KV cache prefill 路径
- 混合 batch 下稳定支持 `q_len != kv_len`
- 清楚的位置接口，包括 RoPE offset / cache position
- 明确的 `output` / `lse` 返回契约
- 明确的 `sm_scale` / softcap 等 score modifier 契约
- fp16 / bf16 稳定覆盖
- 至少一种低精度 KV cache 扩展路径设计清楚，优先 `fp8 kv cache`

不属于本阶段的目标：

- 完整 prefix cache runtime
- page allocation / eviction / reuse 策略
- prefill/decode scheduler
- 任意通用 mask / block mask
- 完整低比特 attention compute
- 完整多模态 prefix 区域语义

## 三、接口分层原则

用户应直接调用 OP；kernel 不作为用户主要心智模型。

公开 OP 按稳定数据契约拆分：

- `GroupedQueryAttentionPrefillFwdOp`
  - dense BSHD
  - 不直接操作外部 cache

- `GroupedQueryAttentionPrefillWithKVCacheFwdOp`
  - dense BSHD current chunk
  - contiguous KV cache
  - fused append

- `GroupedQueryAttentionPrefillVarlenFwdOp`
  - packed THD activations
  - `cu_seqlens_q` / `cu_seqlens_kv`
  - heterogeneous batch

- `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp`
  - packed current chunk
  - paged KV cache
  - `block_table`
  - serving runtime 主力接口

OP 层负责 dispatch，kernel 层负责固定契约的实现。

不要按实现细节暴露 OP。例如不暴露：

- `GroupedQueryAttentionPrefillWgmmaFwdOp`
- `GroupedQueryAttentionPrefillWsFwdOp`
- `GroupedQueryAttentionPrefillSmallSeqFwdOp`

这些应作为 kernel class 或 dispatch target 存在。

## 四、数据布局约定

### 4.1 Dense BSHD

当前 dense prefill 使用：

| 张量 | 形状 |
| --- | --- |
| `q` | `[B, Sq, Hq, D]` |
| `k` | `[B, Skv, Hkv, D]` |
| `v` | `[B, Skv, Hkv, D]` |
| `o` | `[B, Sq, Hq, D]` |
| `lse` | `[B, Hq, Sq]` |

causal 语义：

```text
visible(q_i, k_j) = j <= i + (Skv - Sq)
```

### 4.2 Contiguous KV Cache

首发 contiguous cache layout：

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `q` | `[B, Snew, Hq, D]` | current chunk query |
| `k_new` | `[B, Snew, Hkv, D]` | current chunk key |
| `v_new` | `[B, Snew, Hkv, D]` | current chunk value |
| `k_cache` | `[B, Skv_cap, Hkv, D]` | contiguous cache，in-place append |
| `v_cache` | `[B, Skv_cap, Hkv, D]` | contiguous cache，in-place append |
| `cache_seqlens` | `[B]` | append 前已有 KV 长度 |

语义：

```text
old_len_b = cache_seqlens[b]
total_len_b = old_len_b + Snew
new token i 写入 cache position old_len_b + i
visible(q_i, k_j) = j <= old_len_b + i
```

attention 读取规则：

```text
if kv_pos < old_len_b:
    read k_cache/v_cache
elif kv_pos < total_len_b:
    read k_new/v_new
else:
    mask as invalid
```

注意：attention 不依赖本 kernel 内刚写入 cache 的 current chunk，因此不需要 kernel 内全局同步。

### 4.3 Packed / Varlen THD

进阶阶段需要新增：

| 张量 | 形状 |
| --- | --- |
| `q` | `[Tq, Hq, D]` |
| `k` | `[Tkv, Hkv, D]` |
| `v` | `[Tkv, Hkv, D]` |
| `o` | `[Tq, Hq, D]` |
| `cu_seqlens_q` | `[B + 1]` |
| `cu_seqlens_kv` | `[B + 1]` |

per-request causal offset：

```text
offset_b = kv_len_b - q_len_b
visible(q_i, k_j) = j <= i + offset_b
```

### 4.4 Paged KV Cache

进阶阶段推荐 page-major physical layout：

| 张量 | 形状 |
| --- | --- |
| `q` | `[Tq, Hq, D]` 或 `[B, Snew, Hq, D]` |
| `k_new` | `[Tnew, Hkv, D]` 或 `[B, Snew, Hkv, D]` |
| `v_new` | `[Tnew, Hkv, D]` 或 `[B, Snew, Hkv, D]` |
| `k_pages` | `[P, page_size, Hkv, D]` |
| `v_pages` | `[P, page_size, Hkv, D]` |
| `block_table` | `[B, max_pages_per_req]` |
| `cache_seqlens` | `[B]` |

首版 paged 建议使用 FlashAttention-like `block_table`，不使用 CSR-style `kv_indptr/kv_indices`。

原因：

- 与当前 decode paged 风格更接近
- shape 固定，适合 TileOPs 当前 OP 风格
- runtime 可以先负责 padding block table

## 五、阶段路线

### 阶段 0：当前基线收敛

目标：把当前 dense prefill 和 contiguous-cache prefill 变成稳定基线。

任务：

- 确认 `GroupedQueryAttentionPrefillFwdOp` 命名保留为 dense 默认入口
- 确认 `GroupedQueryAttentionPrefillWithKVCacheFwdOp` 的 `cache_seqlens` 语义
- 增加输入 shape / dtype / capacity 校验
- 明确 `return_lse` 行为：当前 kernel 返回 `(output, lse)`，OP 需要稳定契约
- 为当前两个 OP 增加文档注释和最小示例
- 保留 kernel dispatch 在 OP 层

验收：

- `tests/ops/attention/test_gqa.py` 全量通过
- `fp16` 基础 shape 通过
- `bf16` 至少 dense prefill 通过
- cache append correctness 通过
- old length 非 block 对齐通过

### 阶段 1：OP 基类与 dispatch 整理

目标：避免 GQA prefill family 继续扩张时重复校验和 dispatch 逻辑。

建议新增内部基类：

```python
class _GroupedQueryAttentionBaseOp(Op):
    ...

class _GroupedQueryAttentionPrefillBaseOp(_GroupedQueryAttentionBaseOp):
    ...
```

基类职责：

- `heads` / `heads_kv` / `dim` / `dtype` 通用校验
- `groups = heads // heads_kv`
- MHA/GQA/MQA 通过 `heads/heads_kv` 统一表达
- prefill causal length 约束
- common kernel dispatch helper

基类不负责：

- 统一不同 layout 的 `forward()` 参数
- 引入 optional 大一统接口

验收：

- 现有公开 OP 行为不变
- 现有测试全量通过
- 新增 OP 时只需声明 layout-specific forward 和 kernel key

### 阶段 2：Packed / Varlen Prefill

目标：支持 heterogeneous batch 的非 cache dense prefill。

新增公开 OP：

```python
GroupedQueryAttentionPrefillVarlenFwdOp
```

建议接口：

```python
forward(
    q,                # [Tq, Hq, D]
    k,                # [Tkv, Hkv, D]
    v,                # [Tkv, Hkv, D]
    cu_seqlens_q,     # [B + 1]
    cu_seqlens_kv,    # [B + 1]
    max_seqlen_q,
    max_seqlen_kv,
)
```

必做语义：

- batch 内每个 request 独立计算 `q_len_b` / `kv_len_b`
- per-request bottom-right causal
- `q_len_b <= kv_len_b`
- padding / tail 不产生 NaN

验收：

- heterogeneous batch correctness
- `q_len_b == kv_len_b`
- `q_len_b < kv_len_b`
- 每个 batch 不同 offset
- 与 PyTorch 手写 reference 对齐

### 阶段 3：Contiguous Cache Prefill 完善

目标：把 contiguous cache prefill 从基础 fused kernel 提升到可发布质量。

任务：

- 增加 `return_lse` 显式契约
- 增加 `sm_scale` 参数，默认 `1 / sqrt(dim)`
- 增加 `bf16` 覆盖
- 增加更多 GQA ratio：
  - `heads == heads_kv`
  - `heads_kv == 1`
  - `heads / heads_kv in {2, 4, 8}`
- 增加 capacity 校验：
  - `cache_seqlens[b] + Snew <= Skv_cap`
- 增加 fast path：
  - `old_len` block 对齐
  - `Snew` block 对齐
  - `seq_len_new == block_m`
- 评估是否重新启用 TMA lowering 的局部路径
- 增加 benchmark

当前注意事项：

- 动态 cache/new 分流 load 使用 `T.Pipelined` 可以成立
- 当前需要禁用 TMA lowering / warp-specialized lowering
- 后续如果拆成 old-cache tiles 和 new-chunk tiles 两段，也许可以恢复更强的 pipeline/TMA 路径

验收：

- correctness 全覆盖
- cache append in-place 正确
- 不依赖 kernel 内全局同步
- 对非 block 对齐 old length 稳定
- benchmark 有基本吞吐记录

### 阶段 4：Paged KV Cache Prefill

目标：具备 serving runtime 对接的主力接口。

新增公开 OP：

```python
GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp
```

建议首版接口：

```python
forward(
    q,
    k_new,
    v_new,
    k_pages,
    v_pages,
    block_table,
    cache_seqlens,
)
```

构造参数包含：

- `batch`
- `heads`
- `heads_kv`
- `seq_len_new`
- `seqlen_kv`
- `page_size`
- `dim`
- `is_causal`
- `dtype`

必做语义：

- `cache_seqlens` 表示 append 前长度
- old KV 根据 `block_table` gather
- current chunk KV 从 `k_new/v_new` 读
- append 写入对应 page position
- page tail 不要求有效数据
- attention mask 由 logical position 决定，不由 physical page position 决定

验收：

- 单 batch single-page
- 单 batch multi-page
- batch 内不同 page table
- old length 非 page 对齐
- append 跨 page 边界
- output 与 materialized reference 对齐
- cache page 内容 append 正确

### 阶段 5：位置语义

目标：把位置对齐从隐式 offset 推进到正式接口。

需要支持：

- `position_mode="none"`
- `position_mode="rope"`
- 后续 `position_mode="alibi"`

首版建议先明确 consume contract：

- dense prefill 可选 `position_ids_q` / `position_ids_kv`
- cache prefill 可选 `cache_positions_new`
- 如果未提供 position ids，则使用默认连续位置：
  - dense：`0..Skv-1`
  - cache：old cache `0..old_len-1`，new chunk `old_len..old_len+Snew-1`

RoPE 可以先在 OP 外部完成；但接口文档必须说明 offset 如何对齐。

验收：

- bottom-right causal 与 position offset 一致
- prefix-hit / chunked prefill 不出现 position reset
- RoPE 外置路径有测试

### 阶段 6：Score Modifiers 与 Stats

目标：补齐发布阶段常见接口契约。

优先级：

1. `sm_scale`
2. `return_lse`
3. `softcap`
4. `temperature`
5. simple bias / mask extension

建议：

- `sm_scale=None` 时默认 `1 / sqrt(dim)`
- `return_lse=False` 时 OP 只返回 `output`
- `return_lse=True` 时 OP 返回 `(output, lse)`
- kernel 可以始终计算 lse，OP 层先稳定返回契约

验收：

- `sm_scale` 与 reference 对齐
- `return_lse` shape 和 dtype 固定
- softcap 单独测试

### 阶段 7：Numeric Format

目标：先把 `fp16/bf16` 做稳，再进入低精度 KV cache。

优先级：

1. `fp16` dense / cache / varlen / paged
2. `bf16` dense / cache / varlen / paged
3. `fp8 kv cache`
4. `int8 kv cache`

`fp8 kv cache` 首版需要明确：

- cache storage dtype
- dequant 在 kernel 内还是 kernel 外
- scale 粒度：
  - per-tensor
  - per-head
  - per-block
- scale tensor shape

验收：

- dtype matrix 测试
- 精度误差边界文档化
- 与 reference dequant 路径对齐

## 六、优先级建议

推荐顺序：

1. 当前基线收敛
2. 内部基类和 dispatch 整理
3. `return_lse` / `sm_scale` 契约
4. contiguous cache prefill 完善
5. packed / varlen prefill
6. paged KV cache prefill
7. position / RoPE offset 契约
8. fp8 KV cache

如果目标是尽快接 serving runtime，则优先级可调整为：

1. contiguous cache prefill 完善
2. paged KV cache prefill
3. packed current chunk / heterogeneous batch
4. position / RoPE offset

## 七、风险与待决策项

### 1. Paged layout 选择

待决策：

- 是否首版固定 `block_table: [B, max_pages_per_req]`
- 是否需要同时支持 CSR-style `kv_indptr/kv_indices`

建议：

- 首版只做 `block_table`
- 后续如需更动态的 metadata，再新增 wrapper 或新 OP

### 2. `return_lse` 契约

待决策：

- 当前是否立即让 OP 支持 `return_lse`
- 如果 `return_lse=False`，是否仍在 kernel 内计算 lse

建议：

- OP 层先暴露 `return_lse`
- kernel 可暂时仍返回 lse

### 3. Position 处理位置

待决策：

- RoPE 在 OP 内做，还是 OP 外做
- 是否需要在首个进阶发布里支持 kernel 内 RoPE

建议：

- 先文档化位置 offset
- RoPE 首版保持外置
- 未来如性能需要再融合

### 4. TMA / Pipeline 优化

当前 contiguous cache fused kernel 为了动态 load 分流，禁用了 TMA lowering 和 warp-specialized lowering。

待优化方向：

- 分 old-cache tile path 和 new-chunk tile path
- 对完全落在 old cache 的 tile 使用更强的 `T.copy` / pipeline
- 对完全落在 current chunk 的 tile 使用常规 contiguous load
- 只有跨 old/new 边界的 tile 使用 guarded elementwise load

### 5. Base Op 抽象时机

待决策：

- 现在立刻抽内部基类
- 还是等 varlen / paged 新 OP 开始前抽

建议：

- 在新增 varlen OP 前抽内部基类
- 不把 forward 参数统一到基类

## 八、进阶支持项完成标准

进阶支持项完成时，应满足：

- `GroupedQueryAttentionPrefillFwdOp` 稳定
- `GroupedQueryAttentionPrefillWithKVCacheFwdOp` 稳定
- `GroupedQueryAttentionPrefillVarlenFwdOp` 稳定
- `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp` 稳定
- causal bottom-right 在 dense / varlen / cache / paged 下语义一致
- `heads/heads_kv` 统一覆盖 MHA/GQA/MQA
- fp16/bf16 基础路径稳定
- `return_lse` 契约明确
- `sm_scale` 契约明确
- cache append 协议明确
- paged KV metadata 协议明确
- 非 block 对齐和非 page 对齐边界稳定
- 有最小 benchmark 和对齐 reference

完成后，operator-facing 能力应接近 FlashAttention / FlashInfer / cuDNN Frontend 的主流 prefill 功能面，但仍不等于完整 serving runtime。
