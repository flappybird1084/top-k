import torch
import triton
import triton.language as tl


def _get_tl_dtype(dtype: torch.dtype) -> tl.dtype:
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    raise ValueError(f"Unsupported dtype {dtype}")


autotune_configs = [
    triton.Config({"BLOCK_V": 1024}, num_warps=32, num_stages=1),
]


@triton.jit
def _ce_fwd_kernel(
    logits_ptr,
    targets_ptr,
    loss_ptr,
    count_ptr,
    logsoftmax_ptr,
    N,
    V,
    stride_ln,
    stride_lv,
    stride_tn,
    stride_out,
    ignore_index: tl.int32,
    BLOCK_V: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid
    target = tl.load(targets_ptr + row * stride_tn)

    row_logits = logits_ptr + row * stride_ln
    row_out = logsoftmax_ptr + row * stride_out

    max_acc = tl.full((1,), -float("inf"), tl.float32)
    sum_acc = tl.full((1,), 0.0, tl.float32)

    for off in tl.range(0, V, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V
        x = tl.load(row_logits + cols * stride_lv, mask=mask, other=-float("inf")).to(tl.float32)

        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(max_acc, block_max)

        exp_x = tl.exp(x - new_max)
        block_sum = tl.sum(exp_x, axis=0)
        sum_acc = sum_acc * tl.exp(max_acc - new_max) + block_sum
        max_acc = new_max

    lse = max_acc + tl.log(sum_acc)

    valid = target != ignore_index
    row_loss = tl.full((1,), 0.0, tl.float32)

    for off in tl.range(0, V, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V
        x = tl.load(row_logits + cols * stride_lv, mask=mask, other=0.0).to(tl.float32)

        lsoft = x - lse
        tl.store(row_out + cols * stride_lv, lsoft.to(DTYPE), mask=mask)

        one_hot = (cols == target) & valid
        row_loss = row_loss + tl.sum(-lsoft * one_hot.to(tl.float32), axis=0)

    zero_off = tl.zeros((1,), tl.int32)
    tl.atomic_add(loss_ptr + zero_off, row_loss)
    tl.atomic_add(count_ptr + zero_off, tl.full((1,), 1, tl.int64), mask=valid)


@triton.jit
def _ce_bwd_kernel(
    logsoftmax_ptr,
    targets_ptr,
    dlogits_ptr,
    grad_output_ptr,
    count_ptr,
    N,
    V,
    stride_ln,
    stride_lv,
    stride_tn,
    stride_out,
    ignore_index: tl.int32,
    BLOCK_V: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid
    target = tl.load(targets_ptr + row * stride_tn)

    g = tl.load(grad_output_ptr).to(tl.float32)
    cnt = tl.load(count_ptr).to(tl.float32)
    scale = g / cnt

    row_soft = logsoftmax_ptr + row * stride_out
    row_grad = dlogits_ptr + row * stride_ln
    valid = target != ignore_index

    for off in tl.range(0, V, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V
        lsoft = tl.load(row_soft + cols * stride_lv, mask=mask, other=0.0).to(tl.float32)

        p = tl.exp(lsoft)
        one_hot = (cols == target).to(tl.float32)

        grad = tl.where(valid, (p - one_hot) * scale, 0.0)
        tl.store(row_grad + cols * stride_lv, grad.to(DTYPE), mask=mask)


class _CrossEntropyMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ignore_index):
        N, V = logits.shape
        logsoftmax = torch.empty_like(logits, memory_format=torch.preserve_format).requires_grad_(False)

        loss = torch.zeros((), dtype=torch.float32, device=logits.device)
        count = torch.zeros((), dtype=torch.int64, device=logits.device)
        dtype = _get_tl_dtype(logits.dtype)

        grid = (N,)
        _ce_fwd_kernel[grid](
            logits,
            targets,
            loss,
            count,
            logsoftmax,
            N,
            V,
            logits.stride(0),
            logits.stride(1),
            targets.stride(0),
            logsoftmax.stride(0),
            ignore_index=ignore_index,
            BLOCK_V=1024,
            DTYPE=dtype,
            num_warps=32,
            num_stages=1,
        )

        out = (loss / count).to(logits.dtype)

        ctx.save_for_backward(logsoftmax, targets, count)
        ctx.dtype = dtype
        ctx.ignore_index = ignore_index
        ctx.strides = (logits.stride(0), logits.stride(1), targets.stride(0), logsoftmax.stride(0))

        return out

    @staticmethod
    def backward(ctx, grad_output):
        logsoftmax, targets, count = ctx.saved_tensors
        N, V = logsoftmax.shape
        dlogits = torch.empty_like(logsoftmax, memory_format=torch.preserve_format)

        stride_ln, stride_lv, stride_tn, stride_out = ctx.strides

        grid = (N,)
        _ce_bwd_kernel[grid](
            logsoftmax,
            targets,
            dlogits,
            grad_output,
            count,
            N,
            V,
            stride_ln,
            stride_lv,
            stride_tn,
            stride_out,
            ignore_index=ctx.ignore_index,
            BLOCK_V=1024,
            DTYPE=ctx.dtype,
            num_warps=32,
            num_stages=1,
        )

        return dlogits, None, None


def kernel(logits, targets, ignore_index):
    return _CrossEntropyMean.apply(logits, targets, ignore_index)
