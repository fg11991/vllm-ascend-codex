# Triton-Ascend 算子开发教程

本文面向第一次在 vLLM Ascend 里写 Triton 算子的开发者。示例以
`vllm_ascend/ops/triton/reject_sample.py` 中的 `rejection_random_sample_kernel` 为主，说明一个 Triton-Ascend 算子从接口设计、kernel 编写、Python wrapper 接入到测试验证的完整流程。

## 1. 先理解 vLLM Ascend 的 Triton 算子形态

在这个仓库里，一个 Triton 算子通常分成两层：

1. 设备侧 kernel：用 `@triton.jit` 修饰，参数多为 pointer、标量和 `tl.constexpr`。
2. Python 调用层：检查输入、申请输出、计算 grid/block、选择 fallback，并发射 kernel。

常见文件位置：

- `vllm_ascend/ops/triton/*.py`
- `vllm_ascend/ops/triton/spec_decode/*.py`
- `vllm_ascend/ops/triton/linearnorm/*.py`
- `vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py`

常见导入方式：

```python
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import (
    get_element,
    get_vectorcore_num,
)
```

`vllm.triton_utils` 会屏蔽没有 Triton 时的导入差异。`vllm_ascend.ops.triton.triton_utils` 则提供 Ascend Triton 相关辅助，例如获取 VectorCore 数量和解析 `get_element` 等 CANN 扩展算子。

## 2. 写算子前先定接口

写 Triton kernel 前，不要急着写 `tl.load`。先把接口契约定清楚：

- 输入 tensor 的 shape 是什么？
- flatten 后的索引关系是什么？
- dtype 是什么？输出 dtype 是否和输入一致？
- 输入是否必须 contiguous？
- 哪些参数会影响编译特化，应该设成 `tl.constexpr`？
- 是否需要支持 `None` 输入？
- 是否有 greedy/random、reduce sampling、block verify 等模式分支？
- 是否需要 PyTorch fallback？

以 `rejection_random_sample_kernel` 为例，它的业务目标是 speculative decoding 的 random rejection sampling。输入包括：

- `draft_token_ids`：flatten 后的 draft token id。
- `cu_num_draft_tokens`：每个 request 的累计 draft token 数，用于从 flatten 数组还原 request 范围。
- `draft_probs`：draft 模型概率，ngram 场景可为空。
- `target_probs`：target 模型概率。
- `target_indices`：reduce sampling 时 selected vocab 对应的全局 token id。
- `recovered_token_ids`：拒绝后要使用的 recovered token。
- `uniform_probs`：接受/拒绝采样的随机数。
- `is_greedy`：标记哪些 request 是 greedy，random kernel 会跳过这些 request。
- `output_token_ids`：最终输出 `[batch_size, max_spec_len + 1]`。

## 3. 先写 PyTorch reference

Triton 算子必须有一个容易读、容易调试的 reference。这个仓库里 `rejection_sample` 相关 reference 在：

```text
vllm_ascend/sample/rejection_sampler.py
```

相关函数包括：

- `rejection_random_sample_pytorch`
- `rejection_random_sample_block_verify_pytorch`
- `rejection_greedy_sample_pytorch`

写 reference 的价值：

- 给 Triton e2e 测试提供 golden output。
- 方便先确认业务语义，而不是把业务 bug 和 kernel bug 混在一起。
- 在 `HAS_TRITON` 为 false 时可以 fallback。

新增算子时建议先补 reference，再写 Triton kernel。

## 4. 设计 grid 和 block

Triton kernel 需要明确每个 program 处理多少数据。`reject_sample.py` 里用 request 维度切分并行：

```python
def cal_grid_and_block_size(batch_size: int):
    vectorcore_num = get_vectorcore_num()
    if batch_size <= vectorcore_num:
        grid = batch_size
        block_size = 1
    else:
        grid = vectorcore_num
        block_size = triton.next_power_of_2(triton.cdiv(batch_size, grid))
    return grid, block_size
```

含义：

- batch 小于等于 VectorCore 数量时，一个 program 处理一个 request。
- batch 更大时，grid 固定为 VectorCore 数，一个 program 处理多个 request。
- `BLOCK_SIZE` 取 2 的幂，便于向量化和 mask。

kernel 内通常这样拿到当前 program 的 request 范围：

```python
block_idx = tl.program_id(0)
offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < vec_len
```

这里 `vec_len` 就是 `batch_size`。所有对 request 维度的 load/store 都要带上 `mask`，避免越界。

## 5. 区分运行时参数和编译期参数

Triton 里有些参数应该用 `tl.constexpr`，因为它们会决定编译出来的 kernel 形态：

```python
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    ...
    NO_DRAFT_PROBS: tl.constexpr,
    ENABLE_REDUCE_SAMPLING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VOCAB_BLOCK_SIZE: tl.constexpr = 512,
):
    ...
```

这个例子里：

- `NO_DRAFT_PROBS`：是否没有 draft_probs。用 constexpr 后可以让编译器消掉对应分支。
- `ENABLE_REDUCE_SAMPLING`：是否启用 selected vocab 路径。
- `BLOCK_SIZE`：一个 program 处理多少 request。
- `VOCAB_BLOCK_SIZE`：reduce sampling 查找 vocab 时每次扫描的块大小。

`do_not_specialize=["max_spec_len"]` 表示不要因为 `max_spec_len` 的不同反复特化编译，降低重编译成本。什么时候用 `do_not_specialize` 要看参数是否影响控制流、shape 和性能。如果参数只是寻址 stride，通常可以考虑不特化。

## 6. 写 kernel 主体

`rejection_random_sample_kernel` 的主干可以理解成：

```text
for request in current_program_requests:
    if request is greedy:
        skip

    rejected = False
    for pos in request_draft_tokens:
        if rejected:
            skip

        draft_token_id = draft_token_ids[token_idx]
        target_prob = P_target(draft_token_id)
        draft_prob = P_draft(draft_token_id) or 1
        uniform_prob = uniform_probs[token_idx]

        if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
            output = draft_token_id
        else:
            rejected = True
            output = recovered_token_ids[token_idx]

    if not rejected:
        append bonus_token
```

关键代码结构如下：

```python
block_idx = tl.program_id(0)
offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < vec_len

is_greedy = tl.load(is_greedy_ptr + offsets, mask, other=1)
not_greedy_mask = is_greedy == 0

start_idxs = tl.where(
    offsets == 0,
    0,
    tl.load(cu_num_draft_tokens_ptr + offsets - 1, not_greedy_mask),
)
end_idxs = tl.load(cu_num_draft_tokens_ptr + offsets, not_greedy_mask)
n_num_draft_tokens = end_idxs - start_idxs
```

因为每个 request 的 draft token 数不同，kernel 内会有 request 级循环：

```python
for req_i in range(BLOCK_SIZE):
    not_greedy = get_element(not_greedy_mask, (req_i,))
    if not_greedy:
        ...
```

`get_element` 是 Ascend Triton 扩展能力，用于从向量变量里取出单个标量元素。

## 7. 处理普通 vocab 和 reduce sampling

普通模式下，`target_probs` 是全 vocab：

```python
target_prob = tl.load(
    target_probs_ptr + (start_idx + pos) * global_vocab_size + draft_token_id
)
```

reduce sampling 模式下，`target_probs` 只保存 selected vocab，不能直接用 `draft_token_id` 当列下标。需要在 `target_indices` 里查找全局 token id：

```python
target_prob = 0.0
found = False

for v_offset in range(0, vocab_size, VOCAB_BLOCK_SIZE):
    if not found:
        vocab_offsets = v_offset + tl.arange(0, VOCAB_BLOCK_SIZE)
        vocab_mask = vocab_offsets < vocab_size

        candidate_indices = tl.load(
            target_indices_ptr + token_idx * vocab_size + vocab_offsets,
            mask=vocab_mask,
            other=-1,
        )
        match_mask = candidate_indices == draft_token_id

        candidate_probs = tl.load(
            target_probs_ptr + token_idx * vocab_size + vocab_offsets,
            mask=vocab_mask,
            other=0.0,
        )
        current_match_prob = tl.sum(candidate_probs * match_mask, axis=0)
        if current_match_prob > 0.0:
            target_prob = current_match_prob
            found = True
```

这个分支说明了写 Triton 算子时很常见的模式：用块扫描替代 Python 侧循环，把查找逻辑留在设备上。

## 8. 处理 `None` 输入

Triton kernel 不能像普通 Python 一样自由处理 `None` 指针。这个仓库常用 constexpr 开关表达是否存在某个输入：

```python
if NO_DRAFT_PROBS:
    draft_prob = 1
else:
    draft_prob = tl.load(draft_probs_ptr + token_idx * global_vocab_size + draft_token_id)
```

Python 调用层负责传：

```python
NO_DRAFT_PROBS=draft_probs is None
```

这种写法的好处是：

- 编译后只保留当前模式需要的分支。
- 避免在 kernel 里对无效指针做 load。
- 让调用层的模式选择更清楚。

## 9. 写 Python launcher

不要在业务代码里到处写 kernel 发射参数。推荐写一个 wrapper 或集中在业务函数中统一发射。

`rejection_sample()` 中的发射逻辑大致是：

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

调用前要做的检查：

- `draft_token_ids.ndim == 1`
- `draft_probs is None or draft_probs.ndim == 2`
- `cu_num_draft_tokens.ndim == 1`
- `target_probs.ndim == 2`
- 关键 tensor 必须 contiguous
- 输出 tensor 的 dtype 和 shape 明确

输出一般在 Python 层申请：

```python
output_token_ids = torch.empty(
    (batch_size, max_spec_len + 1),
    dtype=torch.int32,
    device=device,
)
output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)
```

## 10. 接入业务路径

Triton kernel 写完后，需要接进实际业务。以 rejection sampling 为例：

```text
vllm_ascend/worker/model_runner_v1.py
  -> AscendRejectionSampler
     -> vllm_ascend/sample/rejection_sampler.py:rejection_sample
        -> vllm_ascend/ops/triton/reject_sample.py:rejection_random_sample_kernel
```

此外，`vllm_ascend/patch/worker/patch_rejection_sampler.py` 会把上游 vLLM 的部分函数替换成 Ascend 版本，用于解决上游 GPU 路径在 NPU 上不支持或性能不佳的问题。

新增算子时，优先把调用收敛在 Ascend 自己的 wrapper 里。如果必须 patch 上游函数，patch 要小而明确，并在注释里写清：

- 为什么需要 patch？
- patch 了哪个上游对象？
- 后续是否有 upstream 或移除计划？

## 11. 写测试

Triton 算子的测试至少应有两类：

1. Python reference 单测：覆盖纯 PyTorch 逻辑，不依赖 NPU kernel。
2. NPU e2e kernel 测试：在真机上对比 Triton 输出和 reference 输出。

rejection sampling 相关测试位置：

```text
tests/ut/sample/test_rejection_sampler.py
tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample.py
```

建议覆盖的 case：

- `batch_size` 小于、等于、大于 VectorCore 数。
- `max_spec_len` 为 1、2、3 及更大值。
- `draft_probs is None` 和非 None。
- 普通全 vocab 和 reduce sampling。
- 全 greedy、全 random、greedy/random 混合。
- 第一个 token 就 reject、中间 reject、全部 accept 并追加 bonus token。
- `draft_prob == 0`，确认不会产生 NaN。
- selected vocab 中找不到 draft token 的情况。

测试命令示例：

```bash
pytest -sv tests/ut/sample/test_rejection_sampler.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample.py
```

注意：NPU e2e 需要 Ascend NPU、`torch_npu` 和 Triton-Ascend 环境。当前仓库里的部分 rejection sample e2e 测试存在 skip 或旧签名残留，改 kernel 接口时要同步维护测试。

## 12. 性能和稳定性检查清单

写完 Triton 算子后，至少检查这些点：

- 所有可能越界的 `tl.load`/`tl.store` 都有 mask。
- 没有在热路径使用 NPU tensor 的 `.item()`。
- Python wrapper 没有引入不必要的 `.cpu()`、`.tolist()` 或同步。
- `tl.constexpr` 参数没有导致过度重编译。
- 输出 tensor 初始化和未写入位置的语义明确。
- 对 `None` 输入使用 constexpr 分支保护。
- 随机数输入由 Python 层生成并固定，便于 deterministic test。
- dtype cast 明确，尤其是概率计算一般用 `float32`。
- grid/block 对小 batch 和大 batch 都合理。
- 如果要被 compile/graph 捕获，确认 fake/meta 或 fallback 路径可用。

## 13. 一个最小 Triton-Ascend 算子模板

可以参考 `muls_add.py` 的风格写一个最小模板：

```python
import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def my_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def my_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape
    assert x.is_contiguous()
    assert y.is_contiguous()

    out = torch.empty_like(x)
    n_elements = x.numel()
    block_size = 1024
    grid = (min(get_vectorcore_num(), triton.cdiv(n_elements, block_size)),)

    my_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)
    return out
```

真实业务算子会比这个复杂，但基本骨架是一样的：定接口、算 grid、mask load/store、写 wrapper、对齐 reference。

