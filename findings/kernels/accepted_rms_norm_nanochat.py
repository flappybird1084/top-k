import torch
import triton
import triton.language as tl

_TORCH_TO_TL = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float64: tl.float64,
}

autotune_configs = []

_DEFAULT_EPS = 1e-6


@triton.jit
def rms_norm_fwd_kernel(
    x_ptr, y_ptr, w_ptr, rscale_ptr, M, D, eps, HAS_WEIGHT: tl.constexpr, DTYPE: tl.constexpr
):
    pid_m = tl.program_id(0)
    base_x = x_ptr + pid_m * D
    base_y = y_ptr + pid_m * D

    ar_tid = tl.arange(0, 32)[:, None]
    ar_vec = tl.arange(0, 8)[None, :]

    sum_sq = tl.full([1], 0.0, tl.float32)
    for off in range(0, D, 256):
        cols = off + ar_tid * 8 + ar_vec
        mask = cols < D
        x = tl.load(base_x + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x)

    rscale = tl.rsqrt(tl.sum(sum_sq) / D + eps)
    tl.store(rscale_ptr + pid_m, rscale)

    for off in range(0, D, 256):
        cols = off + ar_tid * 8 + ar_vec
        mask = cols < D
        x = tl.load(base_x + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        else:
            w = 1.0
        y = x * rscale * w
        tl.store(base_y + cols, y.to(DTYPE), mask=mask)


@triton.jit
def rms_norm_bwd_dx_kernel(
    x_ptr,
    dout_ptr,
    w_ptr,
    rscale_ptr,
    dx_ptr,
    scratch_ptr,
    M,
    D,
    HAS_WEIGHT: tl.constexpr,
    DTYPE: tl.constexpr,
    COMPUTE_DWEIGHT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    base_x = x_ptr + pid_m * D
    base_d = dout_ptr + pid_m * D
    base_dx = dx_ptr + pid_m * D
    base_scr = scratch_ptr + pid_m * D

    s = tl.load(rscale_ptr + pid_m).to(tl.float32)

    ar_tid = tl.arange(0, 32)[:, None]
    ar_vec = tl.arange(0, 8)[None, :]

    sum_xd = tl.full([1], 0.0, tl.float32)
    for off in range(0, D, 256):
        cols = off + ar_tid * 8 + ar_vec
        mask = cols < D
        x = tl.load(base_x + cols, mask=mask, other=0.0).to(tl.float32)
        d = tl.load(base_d + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            sum_xd += tl.sum(x * d * w)
        else:
            sum_xd += tl.sum(x * d)

    rdoutx_over_D = s * s * tl.sum(sum_xd) / D

    for off in range(0, D, 256):
        cols = off + ar_tid * 8 + ar_vec
        mask = cols < D
        x = tl.load(base_x + cols, mask=mask, other=0.0).to(tl.float32)
        d = tl.load(base_d + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            dx = s * (d * w - x * rdoutx_over_D)
        else:
            dx = s * (d - x * rdoutx_over_D)
        tl.store(base_dx + cols, dx.to(DTYPE), mask=mask)
        if COMPUTE_DWEIGHT:
            scratch = x * d * s
            tl.store(base_scr + cols, scratch, mask=mask)


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        if eps is None:
            eps = _DEFAULT_EPS
        orig_shape = x.shape
        D = orig_shape[-1]
        M = x.numel() // D

        x2 = x.reshape(M, D)
        y = torch.empty(orig_shape, dtype=x.dtype, device=x.device)
        y2 = y.view(M, D)

        rscale = torch.empty((M,), dtype=torch.float32, device=x.device)

        has_weight = weight is not None
        w_arg = weight.contiguous() if has_weight else torch.empty(0, dtype=x.dtype, device=x.device)

        tl_dtype = _TORCH_TO_TL.get(x.dtype)
        if tl_dtype is None:
            raise RuntimeError(f"rms_norm: unsupported dtype {x.dtype}")

        rms_norm_fwd_kernel[(M,)](
            x2,
            y2,
            w_arg,
            rscale,
            M,
            D,
            eps,
            HAS_WEIGHT=has_weight,
            DTYPE=tl_dtype,
            num_warps=1,
        )

        ctx.save_for_backward(x2, w_arg, rscale)
        ctx.orig_shape = orig_shape
        ctx.tl_dtype = tl_dtype
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x2, w_arg, rscale = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        D = orig_shape[-1]
        M = x2.shape[0]

        dout2 = grad_output.reshape(M, D)
        dx = torch.empty((M, D), dtype=x2.dtype, device=x2.device)

        has_weight = w_arg.numel() > 0
        compute_dweight = has_weight and ctx.needs_input_grad[1]

        if compute_dweight:
            scratch = torch.empty((M, D), dtype=torch.float32, device=x2.device)
        else:
            scratch = torch.empty(0, dtype=torch.float32, device=x2.device)

        rms_norm_bwd_dx_kernel[(M,)](
            x2,
            dout2,
            w_arg,
            rscale,
            dx,
            scratch,
            M,
            D,
            HAS_WEIGHT=has_weight,
            DTYPE=ctx.tl_dtype,
            COMPUTE_DWEIGHT=compute_dweight,
            num_warps=1,
        )

        grad_x = dx.view(orig_shape)
        grad_w = torch.sum(scratch, dim=0).to(w_arg.dtype) if compute_dweight else None

        return grad_x, grad_w, None


def kernel(x, weight=None, eps=None):
    return RMSNormFunction.apply(x, weight, eps)
