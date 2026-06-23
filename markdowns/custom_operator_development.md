# vLLM Ascend 算子架构与自定义算子开发指南

本文整理 vLLM Ascend 仓库中和算子相关的代码位置、主要调用链，以及如何开发一个新的自定义算子。最后以
`vllm_ascend/ops/triton/reject_sample.py` 中的 `rejection_random_sample_kernel` 为例，说明从需求拆解到测试验证的完整流程。

## 1. 项目架构速览

vLLM Ascend 是 vLLM 的 Ascend NPU 硬件插件，整体上遵循“少改上游、插件内适配”的结构：

- `vllm_ascend/platform.py`：平台能力、配置校验和 Ascend 后端注册入口。
- `vllm_ascend/worker/`、`vllm_ascend/worker/v2/`、`vllm_ascend/_310p/`：模型执行器、输入 batch、采样、310P 专用适配。
- `vllm_ascend/attention/`：attention 后端和 NPU attention 相关算子封装。
- `vllm_ascend/ops/`：通用算子封装、Triton-Ascend kernel、MoE/Linear/Norm/Rope 等模块级算子实现。
- `vllm_ascend/sample/`、`vllm_ascend/spec_decode/`：采样和 speculative decoding 逻辑，包含多个 Triton kernel 调用点。
- `vllm_ascend/compilation/`：ACL Graph、Inductor pass、图融合和算子替换。
- `vllm_ascend/patch/`：对上游 vLLM 的最小化 monkey patch。
- `csrc/`：CANN/ACLNN 自定义算子的 C++/AscendC 源码、CMake 构建、PyTorch binding 和 meta registration。
- `tests/ut/`、`tests/e2e/nightly/single_node/ops/`：Python wrapper 单测和 NPU 真机算子验证。

## 2. 算子相关目录地图

### 2.1 Python 算子封装

主要目录：

- `vllm_ascend/ops/`
- `vllm_ascend/ops/fused_moe/`
- `vllm_ascend/_310p/ops/`
- `vllm_ascend/lora/lora_ops.py`
- `vllm_ascend/device/device_op.py`

这些文件通常做三件事：

- 对外提供和 vLLM layer/worker 对接的 Python 类或函数，例如 `AscendRMSNorm`、`AscendSiluAndMul`、MoE dispatcher。
- 根据配置、硬件类型、shape 和 dtype 选择 `torch_npu`、`torch.ops._C_ascend`、Triton kernel 或 PyTorch fallback。
- 处理张量布局、NZ 格式、TP/EP/DP 通信、in-place 输出和错误检查。

代表文件：

- `vllm_ascend/ops/layernorm.py`
- `vllm_ascend/ops/rotary_embedding.py`
- `vllm_ascend/ops/fused_moe/fused_moe.py`
- `vllm_ascend/ops/fused_moe/moe_mlp.py`
- `vllm_ascend/ops/gdn.py`
- `vllm_ascend/_310p/ops/layernorm.py`
- `vllm_ascend/_310p/ops/fla/gdn_310.py`

### 2.2 Triton-Ascend kernel

主要目录：

- `vllm_ascend/ops/triton/`
- `vllm_ascend/ops/triton/activation/`
- `vllm_ascend/ops/triton/batch_invariant/`
- `vllm_ascend/ops/triton/fla/`
- `vllm_ascend/ops/triton/linearnorm/`
- `vllm_ascend/ops/triton/mamba/`
- `vllm_ascend/ops/triton/spec_decode/`
- `vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py`
- `vllm_ascend/worker/v2/input_batch.py`

Triton 算子一般由两层组成：

- `@triton.jit` kernel：只写设备侧逻辑，参数是 pointer、标量和 `tl.constexpr`。
- Python launcher/wrapper：检查 shape/dtype/contiguous，申请输出 tensor，计算 grid/block，发射 kernel。

代表文件：

- `vllm_ascend/ops/triton/reject_sample.py`
- `vllm_ascend/ops/triton/muls_add.py`
- `vllm_ascend/ops/triton/penalty.py`
- `vllm_ascend/ops/triton/rms_norm.py`
- `vllm_ascend/ops/triton/spec_decode/utils.py`
- `vllm_ascend/ops/triton/triton_utils.py`

`triton_utils.py` 提供 Ascend Triton 的关键辅助能力：

- `init_device_properties_triton()`：初始化 AICore/VectorCore 数量。
- `get_aicore_num()`、`get_vectorcore_num()`：供 launcher 计算并行度。
- `get_element`、`insert_slice`、`extract_slice`：优先从 `triton.language.extra.cann.extension` 获取 Ascend 扩展 op。

### 2.3 CANN/ACLNN 自定义算子

主要目录：

- `csrc/`
- `csrc/attention/`
- `csrc/moe/`
- `csrc/gmm/`
- `csrc/mc2/`
- `csrc/kernels/`
- `csrc/torch_binding.cpp`
- `csrc/torch_binding_meta.cpp`
- `vllm_ascend/_cann_ops_custom/`

CANN/ACLNN 自定义算子最终会绑定到 `torch.ops._C_ascend`。Python 侧通过：

```python
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
torch.ops._C_ascend.some_custom_op(...)
```

触发注册和调用。`enable_custom_op()` 会尝试导入 `vllm_ascend.vllm_ascend_C` 和 `vllm_ascend.meta_registration`，如果扩展不可用则禁用 custom op 并回退。已有文档
`docs/source/developer_guide/Design_Documents/add_custom_aclnn_op.md` 说明了 ACLNN 算子的底层接入步骤。

### 2.4 torch.library / vLLM custom op 注册层

主要文件：

- `vllm_ascend/ops/register_custom_ops.py`
- `vllm_ascend/ops/__init__.py`

`register_custom_ops.py` 通过 vLLM 的 `direct_register_custom_op` 注册 `torch.ops.vllm.*` 风格的 Python custom op，例如：

- `torch.ops.vllm.maybe_chunk_residual`
- `torch.ops.vllm.maybe_all_gather_and_maybe_unpad`
- `torch.ops.vllm.quantize`
- `torch.ops.vllm.npu_rotary_embedding`
- `torch.ops.vllm.muls_add`

这类 op 适合封装 Python/Triton/torch_npu 组合逻辑，并为编译图提供 fake implementation。

### 2.5 Attention、采样和量化中的算子调用点

这些目录不是纯算子目录，但密集调用 NPU 算子：

- `vllm_ascend/attention/`：例如 sparse attention、DSA、SFA、MLA、KV cache reshape/cache。
- `vllm_ascend/sample/`：top-k/top-p、penalty、rejection sampling。
- `vllm_ascend/spec_decode/`：EAGLE/DFlash proposer 的输入准备和投机解码辅助 kernel。
- `vllm_ascend/quantization/`：量化方法里大量调用 `torch_npu` 和 `_C_ascend`。
- `vllm_ascend/compilation/passes/`：图融合 pass 会把通用 pattern 替换成 NPU fused/custom op。

## 3. 自定义算子开发路径怎么选

### 3.1 优先选 Triton-Ascend 的场景

适合：

- 逻辑是 elementwise、reduction、scatter/gather、sampling、metadata 构造等中小粒度 kernel。
- 需要快速迭代，Python 层可直接测试。
- 输入输出形状相对明确，可以通过 `tl.constexpr` 控制编译特化。
- 不需要复杂 CANN tiling、host/kernel 分离或厂商 custom package。

建议落点：

- kernel 文件放在 `vllm_ascend/ops/triton/<domain>.py`。
- Python wrapper 放在同文件或相邻业务文件中。
- 使用方优先通过 wrapper 调用，不要在业务层散落 grid/block 计算。

### 3.2 选 CANN/ACLNN 自定义算子的场景

适合：

- 对性能、内存布局、通信融合要求很高。
- 需要调用 AscendC、ACLNN 或厂商底层能力。
- 算子需要被 `torch.ops._C_ascend.*` 统一暴露，或需要 ACL Graph/meta registration 支持。
- 需要服务于多处 Python 调用，且 Triton 版本性能或能力不足。

建议流程：

1. 在 `csrc/<domain>/<op_name>/` 下创建算子目录。
2. 按现有算子结构补齐 `op_host/`、`op_kernel/` 或 torch adapter header。
3. 在对应 `CMakeLists.txt` 和 `csrc/build_aclnn.sh` 增加构建配置。
4. 在 `csrc/torch_binding.cpp` 绑定到 `torch.ops._C_ascend`。
5. 在 `csrc/torch_binding_meta.cpp` 添加 meta/fake 实现，保证编译图和 ACL Graph 可识别。
6. Python 侧调用前通过 `enable_custom_op()` 检查扩展是否可用。
7. 增加 `tests/e2e/nightly/single_node/ops/` 下的真机精度和边界测试。

### 3.3 选 Python custom op 注册的场景

适合：

- 想给 `torch.compile`/vLLM 图捕获一个稳定 op 边界。
- 实际实现可以是 Python、Triton、torch_npu 或 `_C_ascend` 调用组合。
- 需要 fake implementation 描述输出 shape/dtype。

建议流程：

1. 在 `vllm_ascend/ops/register_custom_ops.py` 写 impl 和 fake_impl。
2. 用 `direct_register_custom_op` 注册。
3. 确认 `vllm_ascend/ops/__init__.py` 会导入注册文件。
4. 使用方通过 `torch.ops.vllm.<op_name>` 调用。
5. 增加单测覆盖 eager/fake/compile 场景。

## 4. 通用开发流程

1. 明确接口契约：输入输出 shape、dtype、device、contiguous、是否 in-place、是否支持 310P、是否支持 TP/EP/DP。
2. 写 PyTorch reference：先用纯 PyTorch 写正确性版本，作为 UT/e2e 的 golden。
3. 选实现路径：Triton-Ascend、CANN/ACLNN、`torch_npu` 组合，或 Python custom op。
4. 实现 kernel：控制 block size、mask、越界、dtype cast、`tl.constexpr` 特化参数。
5. 写 wrapper：负责断言、输出分配、grid/block 计算、fallback、配置开关。
6. 接入业务：只在必要的 module 或 patch 中接入，避免硬编码环境变量和分散调用。
7. 添加测试：单测覆盖 Python 逻辑；NPU e2e 覆盖 kernel 正确性、边界 shape、dtype 和性能敏感路径。
8. 评估性能：重点检查 CPU-NPU 同步、`tensor.item()`、非 contiguous 拷贝、过度 all-gather、随机数一致性。
9. 更新文档：说明算子用途、接口、限制、测试命令和回退逻辑。

## 5. 示例：`rejection_random_sample_kernel`

### 5.1 业务背景

`rejection_random_sample_kernel` 位于：

```text
vllm_ascend/ops/triton/reject_sample.py
```

它服务于 speculative decoding 的 rejection sampling。目标是在 NPU 上把每个请求的 draft tokens 和 target probabilities 做接受/拒绝判断：

- 如果 draft token 被接受，输出 draft token。
- 如果某个 draft token 被拒绝，输出提前采样好的 recovered token，并停止继续接受后续 draft token。
- 如果所有 draft tokens 都接受，则追加 bonus token。
- 对 greedy request 跳过 random rejection 逻辑。
- 支持普通全 vocab 概率，也支持 reduce sampling 下的 selected vocab。

### 5.2 调用链

主调用链如下：

```text
worker/model_runner_v1.py
  -> AscendRejectionSampler.forward()
     -> apply_sampling_constraints()
     -> rejection_sample()
        -> sample_recovered_tokens()
        -> rejection_random_sample_kernel[(grid,)](...)
```

相关文件：

- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/sample/rejection_sampler.py`
- `vllm_ascend/ops/triton/reject_sample.py`
- `vllm_ascend/patch/worker/patch_rejection_sampler.py`

`patch_rejection_sampler.py` 会把上游 `vllm.v1.sample.rejection_sampler` 中的部分函数替换为 Ascend 版本。注释里也说明了动机：上游函数在 NPU 上不支持或性能较慢，因此这里增加了 custom Triton kernel。

### 5.3 输入输出契约

`rejection_random_sample_kernel` 的核心参数：

- `output_token_ids_ptr`：输出 `[batch_size, max_spec_len + 1]`。
- `cu_num_draft_tokens_ptr`：每个 request 的累计 draft token 数，形状 `[batch_size]`。
- `draft_token_ids_ptr`：flatten 后的 draft token ids，形状 `[num_tokens]`。
- `draft_probs_ptr`：draft 概率，形状 `[num_tokens, global_vocab_size]`，ngram 场景可为 `None`。
- `target_probs_ptr`：target 概率，普通模式是 `[num_tokens, vocab_size]`，reduce sampling 是 `[num_tokens, selected_vocab_size]`。
- `target_indices_ptr`：reduce sampling 时 selected vocab 的全局 token id。
- `bonus_token_ids_ptr`：每个 request 的 bonus token。
- `recovered_token_ids_ptr`：每个 draft position 的 recovered token。
- `uniform_probs_ptr`：接受/拒绝判定用随机数。
- `is_greedy_ptr`：请求是否 greedy。
- `max_spec_len`、`vocab_size`、`global_vocab_size`、`vec_len`：shape 和寻址参数。
- `NO_DRAFT_PROBS`、`ENABLE_REDUCE_SAMPLING`、`BLOCK_SIZE`、`VOCAB_BLOCK_SIZE`：编译期特化参数。

### 5.4 grid 和 block 设计

`cal_grid_and_block_size(batch_size)` 根据 VectorCore 数量决定并行度：

```python
vectorcore_num = get_vectorcore_num()
if batch_size <= vectorcore_num:
    grid = batch_size
    block_size = 1
else:
    grid = vectorcore_num
    block_size = triton.next_power_of_2(triton.cdiv(batch_size, grid))
```

也就是说，一个 Triton program 处理 `BLOCK_SIZE` 个 request。kernel 内部先构造：

```python
offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < vec_len
```

再通过 `cu_num_draft_tokens` 计算每个 request 在 flatten token 数组中的 `[start_idx, end_idx)`。

### 5.5 kernel 主逻辑

逻辑可以拆成五步：

1. 读取 `is_greedy`，只处理 non-greedy request。
2. 对每个 request，读取该 request 的 draft token 范围。
3. 按 draft token 顺序做接受/拒绝判断。
4. 第一次拒绝时写入 `recovered_token_ids`，并停止继续处理后续 draft token。
5. 如果没有任何拒绝，则在 `num_draft_tokens` 位置写入 bonus token。

普通全 vocab 模式下，核心判断是：

```text
target_prob = target_probs[token_idx, draft_token_id]
draft_prob = draft_probs[token_idx, draft_token_id] 或 1
accept if draft_prob > 0 and target_prob / draft_prob >= uniform_prob
```

reduce sampling 模式下，`target_probs` 只包含 selected vocab，需要先在 `target_indices` 中查找当前 `draft_token_id` 对应的位置。找不到时 `target_prob` 保持 0，相当于拒绝。

### 5.6 Python wrapper 接入

`rejection_sample()` 在 Python 层负责：

- 创建并初始化 `output_token_ids`。
- 判断全 greedy、全 random 或混合请求。
- 计算 target probabilities 和 uniform probabilities。
- 调用 `sample_recovered_tokens()` 预先生成 recovered token。
- 根据 `HAS_TRITON` 选择 Triton kernel 或 PyTorch fallback。
- 根据 `enable_reduce_sample` 和 `target_indices` 选择普通模式或 reduce sampling 模式。
- 根据 `max_spec_len >= 3 and draft_probs is not None` 选择普通 random kernel 或 block verify kernel。

调用普通 random kernel 的形态是：

```python
grid, block_size = cal_grid_and_block_size(batch_size)
rejection_random_sample_kernel[(grid,)](
    output_token_ids,
    cu_num_draft_tokens,
    draft_token_ids,
    draft_probs,
    target_probs,
    target_indices,
    bonus_token_ids,
    recovered_token_ids,
    uniform_probs.to(torch.float32),
    is_greedy,
    max_spec_len,
    vocab_size,
    global_vocab_size,
    batch_size,
    NO_DRAFT_PROBS=draft_probs is None,
    ENABLE_REDUCE_SAMPLING=target_indices is not None,
    BLOCK_SIZE=block_size,
)
```

### 5.7 写这个算子的完整流程

如果从零开发一个类似 `rejection_random_sample_kernel` 的 Triton-Ascend 算子，推荐流程如下：

1. 在业务层确认瓶颈：rejection sampling 在 NPU 上不能高效使用上游 GPU Triton kernel 或 PyTorch 循环。
2. 在 `vllm_ascend/sample/rejection_sampler.py` 保留 PyTorch reference，例如 `rejection_random_sample_pytorch()`。
3. 在 `vllm_ascend/ops/triton/reject_sample.py` 新增 `@triton.jit` kernel。
4. 把运行时变量和编译期变量分清楚：shape、模式开关、block size 适合作为 `tl.constexpr`。
5. 用 `mask` 包住所有可能越界的 `tl.load`/`tl.store`。
6. 对 request 级循环和 token 级循环设定上界，避免动态循环不可控。
7. 对 `None` 指针和模式分支使用 `NO_DRAFT_PROBS`、`ENABLE_REDUCE_SAMPLING` 这样的 constexpr 开关。
8. 在 Python wrapper 中统一计算 grid/block，保持业务调用简洁。
9. 保留 `HAS_TRITON` fallback，便于无 Triton 环境、UT 或问题定位。
10. 增加 e2e 测试，把 Triton 输出和 PyTorch reference 对齐。
11. 用不同 `batch_size`、`max_spec_len`、`vocab_size`、`draft_probs is None`、`target_indices is not None` 覆盖边界。
12. 真机上执行测试并同步接口变化，避免测试还停留在旧 kernel 签名。

### 5.8 测试位置和建议命令

已有相关测试：

- `tests/ut/sample/test_rejection_sampler.py`
- `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample.py`
- `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_prepare_inputs_padded.py`

建议命令：

```bash
pytest -sv tests/ut/sample/test_rejection_sampler.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample.py
```

注意：Triton/NPU e2e 需要 Ascend NPU、Triton-Ascend 和 torch_npu 环境。当前 `test_rejection_sample.py` 中部分测试仍保留旧接口形态或被 skip，新增/修改 kernel 时应同步更新测试签名。

## 6. 开发注意事项

- 不要在热路径调用 NPU tensor 的 `.item()`，它会触发 CPU-NPU 同步。
- 新环境变量必须集中定义在 `vllm_ascend/envs.py`，并按项目规范写注释、默认值和范围。
- 对 `_C_ascend` custom op 必须考虑 `enable_custom_op()` 失败时的 fallback 或 skip。
- 对 compile/ACL Graph 路径要提供 fake/meta 实现，否则可能只能 eager 跑通。
- 对 Triton kernel，要明确 `tl.constexpr` 参数，否则会造成过度重编译或动态 shape 不受控。
- 对 in-place 输出，要写清楚 `mutates_args` 或 wrapper 语义，避免编译图错误优化。
- 对 sampling/random 相关算子，要固定随机输入做 deterministic correctness test。
- 对 310P/A2/A3/A5 等差异，要按目录和硬件能力隔离，不要把 310P 专用逻辑混进通用路径。

