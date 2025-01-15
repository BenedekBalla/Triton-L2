import torch
import triton
import triton.language as tl
import argparse 


parser = argparse.ArgumentParser()
parser.add_argument('--l2_tile_size', type=int, default=8)
parser.add_argument("--tiled", action="store_true")
parser.add_argument("--row_maj", action="store_true")
parser.add_argument("--col_maj", action="store_true")
args = parser.parse_args()


def get_cuda_autotune_config():
    return [         
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': args.l2_tile_size}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': args.l2_tile_size}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': args.l2_tile_size}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': args.l2_tile_size}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': args.l2_tile_size}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': args.l2_tile_size}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': args.l2_tile_size}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': args.l2_tile_size}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': args.l2_tile_size}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': args.l2_tile_size}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': args.l2_tile_size}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': args.l2_tile_size}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': args.l2_tile_size}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': args.l2_tile_size}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': args.l2_tile_size}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': args.l2_tile_size}, num_stages=4,
                      num_warps=4)
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def row_l2_matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,  
        stride_bk, stride_bn,  
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, 
        GROUP_SIZE_M: tl.constexpr,   
    ):    
    # Determine CTA's position in its group. (NOTE: Row major order.)
    pid = tl.program_id(axis=0)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # Compute offsets for data pointers. 
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Main loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        accumulator = tl.dot(a, b, accumulator)

        # Advance pointers to next tile along K-dimension. 
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Compute offsets for result.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    accumulator = accumulator.to(tl.float16)
    tl.store(c_ptrs, accumulator)


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def col_l2_matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,  
        stride_bk, stride_bn,  
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  
        GROUP_SIZE_M: tl.constexpr,   
    ):    
    # Determine CTA's position in its group. (NOTE: Column major order.)
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % grid_m
    pid_n = pid // grid_m

    # Compute offsets for data pointers. 
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Main loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        accumulator = tl.dot(a, b, accumulator)

        # Advance pointers to next tile along K-dimension. 
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Compute offsets for result.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    accumulator = accumulator.to(tl.float16)
    tl.store(c_ptrs, accumulator)


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def tiled_l2_matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,  
        stride_bk, stride_bn,  
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  
        GROUP_SIZE_M: tl.constexpr,  
    ):    
    # Determine CTA's position in its group. 
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute offsets for data pointers. 
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Main loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        accumulator = tl.dot(a, b, accumulator)

        # Advance pointers to next tile along K-dimension. 
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Compute offsets for result.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    accumulator = accumulator.to(tl.float16)
    tl.store(c_ptrs, accumulator)


def row_l2_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    # Allocate output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)     # Accumulating into Int32 avoids overflow. 
    # 1D launch grid.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    row_l2_matmul_kernel[grid](
        a, b, c,  
        M, N, K,  
        a.stride(0), a.stride(1),  
        b.stride(0), b.stride(1),  
        c.stride(0), c.stride(1),  
    )
    return c


def col_l2_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    # Allocate output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)     # Accumulating into Int32 avoids overflow. 
    # 1D launch grid.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    col_l2_matmul_kernel[grid](
        a, b, c,  
        M, N, K,  
        a.stride(0), a.stride(1),  
        b.stride(0), b.stride(1),  
        c.stride(0), c.stride(1),  
    )
    return c


def tiled_l2_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    # Allocate output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)     # Accumulating into Int32 avoids overflow. 
    # 1D launch grid.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    tiled_l2_matmul_kernel[grid](
        a, b, c,  
        M, N, K,  
        a.stride(0), a.stride(1),  
        b.stride(0), b.stride(1),  
        c.stride(0), c.stride(1),  
    )
    return c


# Correctness test
# torch.manual_seed(0)
# a = torch.rand(4096, 1024, device='cuda', dtype=torch.float16)
# b = torch.rand(1024, 1024, device='cuda', dtype=torch.float16)
# triton_output1 = row_l2_matmul(a, b)
# triton_output2 = col_l2_matmul(a, b)
# triton_output3 = tiled_l2_matmul(a, b)
# print((triton_output1 == triton_output2).all().item()) 
# print((triton_output2 == triton_output3).all().item()) 

# Performance test
ref_lib = 'cuBLAS'
configs = []
configs.append(
    triton.testing.Benchmark(
            x_names=["K"],  # Argument names to use as an x-axis for the plot
            x_vals=[512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192], 
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            line_vals=[ref_lib.lower(), "triton"],  # Label name for the lines
            line_names=[ref_lib, "Triton"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="ms",  # Label name for the y-axis
            plot_name="matmul-perf",
            args={"M": 8192, "N": 8192},
    )
)


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.rand(M, K, device='cuda', dtype=torch.float16)
    b = torch.rand(K, N, device='cuda', dtype=torch.float16)
    if provider == ref_lib.lower():
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
    if provider == 'triton' and args.row_maj:
        ms = triton.testing.do_bench(lambda: row_l2_matmul(a, b))
    if provider == 'triton' and args.col_maj:
        ms = triton.testing.do_bench(lambda: col_l2_matmul(a, b))
    if provider == 'triton' and args.tiled:
        ms = triton.testing.do_bench(lambda: tiled_l2_matmul(a, b))
    return ms

benchmark.run(show_plots=True, print_data=True)
print("Done.")
