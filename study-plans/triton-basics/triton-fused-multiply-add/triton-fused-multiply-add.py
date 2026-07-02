import torch
import triton
import triton.language as tl


@triton.jit
def fma_kernel(x_ptr, y_ptr, out_ptr, n, a, BLOCK_SIZE: tl.constexpr):
    # calculate address
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # load
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask = mask)

    # calc
    z = a * x + y

    # save
    tl.store(out_ptr + offsets, z, mask=mask)


def solve(a: float, x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """Launch fma_kernel: out = a * x + y."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fma_kernel[grid](x, y, out, n, a, BLOCK_SIZE=BLOCK_SIZE)