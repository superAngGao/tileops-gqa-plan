# GQA Prefill Plan：发布收敛计划

日期：2026-04-27

目标：把 GQA prefill 从当前已经完成的 dense / varlen / contiguous-cache / paged-cache 功能面，收敛到 `prefill.md` 中定义的 release-facing operator family，并明确剩余的 FP8 KV cache、benchmark、H200/Hopper dispatch 和 manifest 工作。

本文只讨论 GQA prefill operator family，不讨论完整 serving runtime、调度器、prefix cache 命中策略或 page manager 生命周期。

## 一、当前基线

当前已经完成或正在验证的 release-facing 能力：

1. `GroupedQueryAttentionPrefillFwdOp`
   - dense BSHD layout
   - 支持 `seq_len_q != seq_len_kv`
   - causal 使用 bottom-right alignment
   - 支持 GQA/MHA/MQA 统一表达：`heads` / `heads_kv`
   - kernel 层已有普通 TileLang 路径和 Hopper WGMMA 路径

2. `GroupedQueryAttentionPrefillVarlenFwdOp`
   - packed THD layout
   - 使用 `cu_seqlens_q` / `cu_seqlens_kv`
   - 支持 heterogeneous batch
   - 支持 per-request `q_len == kv_len` 和 `q_len < kv_len`
   - 支持 fp16 / bf16、MHA / GQA / MQA

3. `GroupedQueryAttentionPrefillWithKVCacheFwdOp`
   - dense BSHD current chunk
   - contiguous KV cache：`[B, Skv_cap, Hkv, D]`
   - `cache_seqlens` 表示 append 前已有 KV 长度
   - fused kernel 同时完成：
     - old KV 从 cache 读取
     - current chunk KV 从 `k_new/v_new` 读取
     - append `k_new/v_new` 到 `k_cache/v_cache`
   - `cache_seqlens` 不要求 block 对齐
   - current fused kernel 使用 `T.Pipelined`，但禁用 TMA lowering 和 warp-specialized lowering，以避开动态 cache/new 分流 load 的 mbarrier lowering 问题

4. `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp`
   - packed THD current chunk
   - flattened physical page storage：`[P_tokens, Hkv, D]`
   - `block_table[b, logical_page] -> physical_page`
   - `cache_seqlens` 表示 append 前已有 KV 长度
   - kernel 同时完成 old paged KV gather、current chunk attention、current KV append
   - 已覆盖 page size `16 / 32 / 64 / 128`

5. RoPE / score modifier
   - 外置 RoPE regression 覆盖 contiguous cache 和 paged cache
   - contiguous fused RoPE 已支持
   - paged fused RoPE 已支持
   - `sm_scale` 已支持
   - `softcap` 已支持，`None` / `0` 表示 disabled，`>0` 表示启用
   - 公开 OP 默认保持 output-only；`return_lse` 仍是低优先级 open question

当前测试覆盖：

- dense prefill `q_len == kv_len`
- dense prefill `q_len < kv_len`
- bottom-right causal mask 定向测试
- packed varlen heterogeneous batch
- contiguous cache prefill output correctness
- contiguous cache append correctness
- paged cache output correctness
- paged cache append correctness
- old cache length 非 block 对齐
- batch 内不同 `cache_seqlens`
- 外置 RoPE / fused RoPE
- `sm_scale` / `softcap`

## 二、进阶支持项定义

进阶支持项的核心目标不是“feature 全部做满”，而是让 GQA prefill operator family 具备现代 LLM serving 的可发布性。

进阶完成时，至少应具备：

- 稳定的 dense prefill 路径
- 稳定的 contiguous KV cache prefill 路径
- packed / varlen prefill 路径
- paged KV cache prefill 路径
- 混合 batch 下稳定支持 `q_len != kv_len`
- 清楚的位置接口，包括 RoPE offset / cache position
- 明确的 `output` 返回契约，`lse` 暴露作为低优先级 open question
- 明确的 `sm_scale` / softcap 等 score modifier 契约
- fp16 / bf16 稳定覆盖
- FP8 KV cache 首发 dequant path 设计清楚，并完成 manifest-ready issue / PR 切分
- benchmark / H200 dispatch 纳入发布验证，而不是 release 之后才补

不属于本阶段的目标：

- 完整 prefix cache runtime
- page allocation / eviction / reuse 策略
- prefill/decode scheduler
- 任意通用 mask / block mask
- FP8 Tensor Core attention compute
- 完整多模态 prefix 区域语义

## 三、接口分层原则

用户应直接调用 OP；kernel 不作为用户主要心智模型。

公开 OP 按稳定数据契约拆分：

- `GroupedQueryAttentionPrefillFwdOp`
  - dense BSHD
  - 不直接操作外部 cache

- `GroupedQueryAttentionPrefillVarlenFwdOp`
  - packed THD activations
  - `cu_seqlens_q` / `cu_seqlens_kv`
  - heterogeneous batch

- `GroupedQueryAttentionPrefillWithKVCacheFwdOp`
  - dense BSHD current chunk
  - contiguous KV cache
  - fused append

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

当前 varlen prefill 使用：

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

首发 paged cache 使用 flattened physical page layout。概念上它仍然是 page-major；实现和 manifest 中把 page 维与 page 内 token 维展平成 `P_tokens = P * page_size`。

| 张量 | 形状 |
| --- | --- |
| `q` | `[Tnew, Hq, D]` |
| `k_new` | `[Tnew, Hkv, D]` |
| `v_new` | `[Tnew, Hkv, D]` |
| `k_pages` | `[P_tokens, Hkv, D]` |
| `v_pages` | `[P_tokens, Hkv, D]` |
| `cu_seqlens_q` | `[B + 1]` |
| `block_table` | `[B, max_pages_per_req]` |
| `cache_seqlens` | `[B]` |

必须满足：

- `P_tokens % page_size == 0`
- `physical_token = physical_page * page_size + page_offset`
- `block_table[b, logical_page]` 存 physical page id
- current chunk 使用 packed THD，与 varlen prefill 的 batch 边界一致

首版 paged 使用 FlashAttention-like `block_table`，不使用 CSR-style `kv_indptr/kv_indices`。

原因：

- 与当前 decode paged 风格更接近
- shape 固定，适合 TileOPs 当前 OP 风格
- runtime 负责 page allocation / eviction / prefix sharing，OP 只消费已经准备好的 `block_table`

## 五、阶段路线

### 阶段 0：当前基线收敛

目标：把 dense prefill 和 contiguous-cache prefill 变成后续 varlen / paged / FP8 工作可以依赖的稳定基线。

已完成 / 应保持：

- 确认 `GroupedQueryAttentionPrefillFwdOp` 命名保留为 dense 默认入口
- 确认 `GroupedQueryAttentionPrefillWithKVCacheFwdOp` 的 `cache_seqlens` 语义
- 增加输入 shape / dtype / capacity 校验
- 明确公开 OP 默认返回 `output`：当前 kernel 可以内部返回 `(output, lse)`，但 OP 层应保持 TileOPs 的 output-only 主契约
- 为当前两个 OP 增加文档注释和最小示例
- 保留 kernel dispatch 在 OP 层

验收：

- `tests/ops/attention/test_gqa.py` 全量通过
- `fp16` 基础 shape 通过
- `bf16` 至少 dense prefill 通过
- cache append correctness 通过
- old length 非 block 对齐通过

### 阶段 1：OP 公共契约与 dispatch 整理

目标：避免 GQA prefill family 继续扩张时重复校验和 dispatch 逻辑，但不提前引入大一统 OP 继承层级。

公共 helper 职责：

- `heads` / `heads_kv` / `dim` / `dtype` 通用校验
- `groups = heads // heads_kv`
- MHA/GQA/MQA 通过 `heads/heads_kv` 统一表达
- prefill causal length 约束
- output-only 解包规则：kernel 内部可返回 `(output, lse)`，公开 OP 默认只返回 `output`

公共 helper 不负责：

- 统一不同 layout 的 `forward()` 参数
- 引入 optional 大一统接口
- 制造暂时没有稳定多态边界的内部基类

验收：

- 现有公开 OP 行为不变
- 现有测试全量通过
- 新增 OP 时只需声明 layout-specific forward 和 kernel key / wrapper

### 阶段 2：Packed / Varlen Prefill

目标：支持 heterogeneous batch 的非 cache dense prefill。

已新增公开 OP：

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

当前实现状态：

- 已新增 `GroupedQueryAttentionPrefillVarlenFwdOp`
- 公开接口使用 packed THD + `cu_seqlens_q` / `cu_seqlens_kv`
- 当前实现复用 unlimited `GroupedQueryAttentionSlidingWindowVarlenFwdOp` kernel
- OP 层额外校验 `q_len_b <= kv_len_b`、packed total length、`max_seqlen_q` / `max_seqlen_kv`
- 已覆盖 `q_len == kv_len`、`q_len < kv_len`、heterogeneous batch、MHA/MQA/GQA、fp16/bf16

### 阶段 3：Contiguous Cache Prefill 完善

目标：把 contiguous cache prefill 从基础 fused kernel 提升到可发布质量，并保持它作为 FP8 KV cache contiguous variant 的基线。

已完成 / 已覆盖：

- 增加 `sm_scale` 参数，默认 `1 / sqrt(dim)`
- 增加 `bf16` 覆盖
- 增加更多 GQA ratio：
  - `heads == heads_kv`
  - `heads_kv == 1`
  - `heads / heads_kv in {2, 4, 8}`
- 增加 capacity 校验：
  - `cache_seqlens[b] + Snew <= Skv_cap`

剩余优化：

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

已新增公开 OP：

```python
GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp
```

当前首版接口：

```python
forward(
    q,                # [Tnew, Hq, D]
    k_new,            # [Tnew, Hkv, D]
    v_new,            # [Tnew, Hkv, D]
    k_pages,          # [P * page_size, Hkv, D]
    v_pages,          # [P * page_size, Hkv, D]
    cu_seqlens_q,     # [B + 1]
    cache_seqlens,    # [B]
    block_table,      # [B, max_pages_per_req]
    max_seqlen_q,
)
```

构造参数包含：

- `batch`
- `heads`
- `heads_kv`
- `max_pages_per_req`
- `page_size`
- `dim`
- `is_causal`
- `dtype`

必做语义：

- current chunk 使用 packed THD，与 varlen prefill 的 batch 边界语义一致
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

当前实现状态：

- 已新增 `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp`
- 首版使用 packed current chunk：`q/k_new/v_new [Tnew, H, D]`
- 复用 varlen 的 `cu_seqlens_q` 组织 batch 内 current chunk
- 使用 flat page-major physical cache：`k_pages/v_pages [P * page_size, Hkv, D]`
- `block_table[b, logical_page]` 映射到 physical page id
- kernel 同时完成 old paged KV 读取、current chunk attention、current KV in-place append
- 首版约束 `page_size` 为 2 的幂，当前测试覆盖 `16 / 32 / 64 / 128`
- 已新增非 TMA old-cache load fast path：
  - `page_size % block_n == 0`：一个 KV tile 是 page 内子块，整块规则 copy
  - `block_n % page_size == 0`：一个 KV tile 覆盖多个完整 page，按 page segment copy
- append 写回已有单页整块 fast path；跨页 append 暂时走 generic path
- 当前未做 split-k / page manager / TMA / benchmark

### 阶段 5：位置语义

目标：把位置对齐从隐式 offset 推进到正式接口。

需要支持 / 已开始支持：

- `position_mode="none"`
- `position_mode="rope"`
- 后续 `position_mode="alibi"`

首版建议先明确 consume contract：

- dense prefill 可选 `position_ids_q` / `position_ids_kv`
- cache prefill 可选 `cache_positions_new`
- 如果未提供 position ids，则使用默认连续位置：
  - dense：`0..Skv-1`
  - cache：old cache `0..old_len-1`，new chunk `old_len..old_len+Snew-1`

RoPE 首版支持两条路径：

- 外置 RoPE：GQA prefill family 消费已经旋转好的 `q` / `k`
- contiguous fused RoPE：contiguous cache prefill OP 内部旋转 current chunk 的 `q/k_new`

cache 中保存的 K 统一按已旋转后的 K 处理。

外置 RoPE 首版约定：

- 使用 packed THD RoPE position_ids 路径表达 chunked prefill / paged prefill 的绝对位置
- 对 contiguous / paged cache prefill：
  - old cache K 已按 logical position `0..old_len_b-1` 旋转并写入 cache
  - current chunk 的 `q` 和 `k_new` 使用 `old_len_b + local_i` 作为 position id
  - append 写入 cache 的是已经旋转后的 `k_new`
- 对 dense prefill：
  - KV 使用 `0..Skv-1`
  - 当 `Sq < Skv` 时，Q 默认使用 bottom-right 对齐位置 `Skv - Sq .. Skv - 1`

验收：

- bottom-right causal 与 position offset 一致
- prefix-hit / chunked prefill 不出现 position reset
- RoPE 外置路径有测试

当前实现状态：

- 已新增 `RopeNeoxPositionIdsOp`
  - 输入 `x [T, H, D]`
  - 输入 `position_ids [T]`
  - 内部生成 `max_position` 长度的 cos/sin table
  - 支持 packed current chunk / paged prefill 的 absolute cache position
- 已新增 RoPE position_ids correctness 测试
- 已新增 contiguous cache GQA prefill 外置 RoPE 回归测试，验证 old cache 不被重写，current chunk 使用 absolute cache position
- 已新增 paged GQA prefill 外置 RoPE 回归测试，验证 current chunk 使用 `cache_seqlens[b] + local_i`
- 已新增 contiguous cache fused RoPE 路径：
  - `GroupedQueryAttentionPrefillWithKVCacheFwdOp(..., fuse_rope=True, max_position=...)`
  - old cache K 视为已经 rotated，kernel 不重复旋转
  - current chunk 的 `q/k_new` 使用 `old_len_b + local_i` 在 kernel 内旋转
  - append 写入 cache 的是 rotated `k_new`
- 已新增 paged fused RoPE 路径：
  - `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(..., fuse_rope=True, max_position=...)`
  - old page cache K 视为已经 rotated，kernel 只 gather 读取
  - current packed chunk 的 `q/k_new` 使用 `cache_seqlens[b] + local_i` 在 kernel 内旋转
  - append 写入 physical page 的是 rotated `k_new`

### 阶段 6：Score Modifiers 与 Stats

目标：补齐发布阶段常见接口契约。

优先级：

1. `sm_scale`
2. `softcap`
3. `temperature`
4. simple bias / mask extension
5. `return_lse` / stats 暴露

建议：

- `sm_scale=None` 时默认 `1 / sqrt(dim)`
- 公开 OP 默认只返回 `output`
- kernel 内部可以继续计算 / 返回 `lse`，OP 层先 unwrap，避免把 kernel stats 变成用户默认心智模型
- `return_lse=True` 只有在训练 backward、partial/split attention 合并、或对齐 FlashInfer/xFormers/cuDNN stats 接口时再考虑暴露

验收：

- `sm_scale` 与 reference 对齐
- softcap 单独测试

当前实现状态：

- 已支持 `sm_scale`
- 已新增 `softcap`
  - 公开 OP 参数：`softcap: Optional[float] = None`
  - `softcap=None` 或 `0` 保持原行为
  - `softcap>0` 时在 QK score 进入 online softmax 前执行 `softcap * tanh(score / softcap)`
  - 覆盖 dense / contiguous cache / paged cache / varlen packed
  - 覆盖 contiguous fused RoPE / paged fused RoPE 组合路径
- 尚未支持 `temperature` / bias / mask extension
- `return_lse` 仍保持低优先级 open question

### 阶段 7：Numeric Format

目标：`fp16/bf16` 作为 release baseline 保持稳定；下一步进入 FP8 KV cache 的 serving storage path。

优先级：

1. `fp16` dense / cache / varlen / paged
2. `bf16` dense / cache / varlen / paged
3. contiguous `fp8_e4m3fn` KV cache, per-tensor scale, kernel-internal dequant
4. paged `fp8_e4m3fn` KV cache, per-tensor scale, kernel-internal dequant
5. per-kv-head scale
6. per-token-head / dynamic scale
7. int8 / lower-bit KV cache

`fp8 kv cache` 首版需要明确：

- storage dtype：`float8_e4m3fn`
- Q/O dtype：`float16` 或 `bfloat16`
- `k_new/v_new` 输入仍为 `float16` / `bfloat16`
- old cache 读取时在 kernel 内按 scale dequant
- current chunk 本次 attention 直接使用 `k_new/v_new`
- append 时把 current `k_new/v_new` 量化写入 caller-owned FP8 cache
- 首发 scale 粒度：per-tensor
- scale tensor：`k_scale` / `v_scale`，dtype `float32`，shape `[1]`
- fused RoPE 顺序：RoPE -> quantize -> append
- 首发不承诺 FP8 Tensor Core attention compute

验收：

- dtype matrix 测试
- 精度误差边界文档化
- 与 reference dequant 路径对齐
- old cache dequant correctness
- append-time quantize correctness
- contiguous cache 和 paged cache 各有最小回归

当前决策：

- FP8 KV cache 是 serving cache policy，不是完整 FP8 attention compute。
- 首发采用 dequant path，避免在同一阶段处理 FP8 Tensor Core operand layout / swizzle、tile 级量化和 scale pipeline。
- per-page scale 暂不作为首发目标；per-block 更接近 NVFP4 / 低比特 block scaling，不应混进普通 FP8 KV cache 首发。
- 后续如果做 FA3-style FP8 attention compute，应放到 H200 / WS / TMA 优化路线中单独设计。

## 六、优先级建议

从当前状态继续推进的推荐顺序：

1. FP8 KV cache dequant path 设计和 manifest issue 收敛
2. contiguous FP8 KV cache read + append
3. paged FP8 KV cache read + append
4. manifest/workload/roofline/source metadata 与 #1097-#1103 对齐
5. benchmark matrix 和 nightly 形状覆盖
6. H200/Hopper dispatch 与 WS/TMA-friendly 优化
7. 低优先级 `return_lse` / stats 暴露决策

注意：这里的顺序表达的是实现关注点，不表示 manifest 可以滞后。任何新增或改变 release-facing OP contract 的 FP8 PR，都必须在同一个 PR 中同步更新 manifest、workloads、roofline、source metadata 和对应 tests；不能先合实现、再用后续 PR 补 manifest。

如果目标是尽快接 serving runtime，则优先级可调整为：

1. paged FP8 KV cache
2. contiguous FP8 KV cache
3. manifest-backed benchmark / roofline 完整度
4. H200 dispatch

## 七、风险与待决策项

### 1. Paged layout 选择

当前决策：

- 首版固定 `block_table: [B, max_pages_per_req]`。
- 首版 physical cache 使用 flattened page-major layout：`[P_tokens, Hkv, D]`。
- `P_tokens % page_size == 0`。
- 不同时支持 CSR-style `kv_indptr/kv_indices`。

待决策：

- 后续如需更动态的 metadata，是新增 wrapper，还是新增 OP。
- 是否需要面向 runtime 的高层 `PagedKVCache` / `KVCacheConfig` 对象封装 page allocation 和 `cache_seqlens` 更新。

### 2. 是否暴露 `lse` / stats

待决策：

- 是否需要在公开 OP 支持 `return_lse=True`
- 如果公开支持，是否所有 prefill OP 都支持，还是只在 varlen / paged / partial attention 场景支持
- 如果公开 OP 默认只返回 `output`，kernel 内部是否仍计算并返回 `lse` 给 wrapper unwrap

建议：

- 低优先级。TileOPs 公开 OP 主契约保持 output-only。
- 先把 `lse` 视为 kernel/internal stats；当前 kernel 返回 `(output, lse)` 时由 OP 层 unwrap。
- 只有在明确需要兼容以下场景时再暴露：
  - FlashInfer 的 `return_lse=True`
  - xFormers 的 `memory_efficient_attention_forward_requires_grad` / partial attention 合并
  - cuDNN SDPA 的 `generate_stats=True` 训练统计量

### 3. Position 处理位置

当前决策：

- 外置 RoPE 是稳定语义路径，调用方可传入已经旋转好的 `q/k_new`。
- fused RoPE 是 cache-aware prefill 的内部实现路径，不单独成为公开 OP。
- old cache K 视为已经 rotated，不能重复旋转。
- current chunk 的 `q/k_new` 使用 absolute cache position 旋转。
- append 写入 cache/page 的是 rotated `k_new`。

待决策：

- 长期公开接口是否从 `fuse_rope` 过渡到更语义化的 `position_mode="rope"`。
- 是否在 FP8 KV cache 首发中同时支持 fused RoPE，还是先要求外置 RoPE。

### 4. TMA / Pipeline 优化

当前 contiguous cache fused kernel 为了动态 load 分流，禁用了 TMA lowering 和 warp-specialized lowering。

重新启用 TMA / WS 的判断标准：

- old-cache tile 和 new-chunk tile 能在 kernel 内被静态或 tile-level predicate 清晰分离；
- 完全落在 old cache 的 tile 能表达成规则 contiguous copy，满足 TMA / bulk async copy 的对齐和粒度要求；
- 完全落在 current chunk 的 tile 能走常规 contiguous load；
- 只有跨 old/new 边界的少数 tile 继续走 guarded elementwise load；
- 对 paged KV，只有当 page segment 足够规则，且 `page_size` / `block_n` 关系能让一个 tile 分解成少量整页或整块 segment 时，才考虑 TMA-friendly gather path；
- dispatch 必须留在 #1102 的 H200/Hopper 路线中，不能改变公开 OP signature。

待优化方向：

- 分 old-cache tile path 和 new-chunk tile path
- 对完全落在 old cache 的 tile 使用更强的 `T.copy` / pipeline
- 对完全落在 current chunk 的 tile 使用常规 contiguous load
- 只有跨 old/new 边界的 tile 使用 guarded elementwise load

### 5. Base Op 抽象时机

当前决策：

- 不急着抽象成大一统内部基类。
- 先沉淀公共 helper：head/group 校验、dtype 校验、capacity 校验、output-only unwrap、dispatch key 选择。
- 不统一不同 layout 的 `forward()` 参数。

待决策：

- 当 FP8 variants 加入后，如果重复校验明显增加，再评估是否需要更正式的内部 base class。

### 6. FP8 KV Cache 首发颗粒度

当前决策：

- 首发只做 `fp8_e4m3fn` KV cache storage。
- 首发 scale 粒度为 per-tensor：`k_scale/v_scale` 各一个 `float32[1]`。
- 首发走 kernel-internal dequant path，不走 FP8 Tensor Core attention compute。
- current `k_new/v_new` 输入保持 fp16/bf16，append 时量化写入 FP8 cache。

待决策：

- per-kv-head scale 是否作为第二阶段增强。
- dynamic/per-token-head scale 是否需要与 paged metadata 一起设计。
- FA3-style FP8 attention compute 进入 H200/WS/TMA 路线的具体触发条件。

## 八、进阶支持项完成标准

进阶支持项完成时，应满足：

- `GroupedQueryAttentionPrefillFwdOp` 稳定
- `GroupedQueryAttentionPrefillWithKVCacheFwdOp` 稳定
- `GroupedQueryAttentionPrefillVarlenFwdOp` 稳定
- `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp` 稳定
- causal bottom-right 在 dense / varlen / cache / paged 下语义一致
- `heads/heads_kv` 统一覆盖 MHA/GQA/MQA
- fp16/bf16 基础路径稳定
- 公开 OP 默认 output-only，`return_lse` 保持低优先级 follow-up 或明确暴露契约
- `sm_scale` 契约明确
- `softcap=None` / `softcap=0` / `softcap>0` 语义明确
- cache append 协议明确
- paged KV metadata 协议明确
- FP8 KV cache 首发 dequant path 契约明确
- 非 block 对齐和非 page 对齐边界稳定
- 有 manifest-backed workloads、roofline、source metadata 和最小 benchmark
- H200/Hopper dispatch 不改变公开 OP 契约

完成后，operator-facing 能力应接近 FlashAttention / FlashInfer / cuDNN Frontend 的主流 prefill 功能面，但仍不等于完整 serving runtime。
