import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
from harness import run_benchmark, emit_verify_tag

STUDENT_ID = 230110056
P_PHYS = 8
N = 384

def gemm_v1_worker(sa: str, sb: str, sc: str, r0: int, r1: int) -> None:
    ma = shared_memory.SharedMemory(name=sa)
    mb = shared_memory.SharedMemory(name=sb)
    mc = shared_memory.SharedMemory(name=sc)
    A = (ctypes.c_double * (N * N)).from_buffer(ma.buf)
    B = (ctypes.c_double * (N * N)).from_buffer(mb.buf)
    C = (ctypes.c_double * (N * N)).from_buffer(mc.buf)
    for i in range(r0, r1):
        i_off = i * N
        for j in range(N):
            dot = 0.0
            for k in range(N):
                dot += A[i_off + k] * B[k * N + j]
            C[i_off + j] = dot
    del A, B, C
    ma.close(); mb.close(); mc.close()

def gemm_v2_worker(sa: str, sb: str, sc: str, r0: int, r1: int) -> None:
    ma = shared_memory.SharedMemory(name=sa)
    mb = shared_memory.SharedMemory(name=sb)
    mc = shared_memory.SharedMemory(name=sc)
    A = (ctypes.c_double * (N * N)).from_buffer(ma.buf)
    B = (ctypes.c_double * (N * N)).from_buffer(mb.buf)
    C = (ctypes.c_double * (N * N)).from_buffer(mc.buf)
    for i in range(r0, r1):
        row_c = [0.0] * N
        i_off = i * N
        for k in range(N):
            a_ik = A[i_off + k]
            k_off = k * N
            b_row = B[k_off : k_off + N]
            for j in range(N):
                row_c[j] += a_ik * b_row[j]
        C[i_off : i_off + N] = row_c
    del A, B, C
    ma.close(); mb.close(); mc.close()

def gemm_v3_worker(sa: str, sb: str, sc: str, r0: int, r1: int, bs: int) -> None:
    ma = shared_memory.SharedMemory(name=sa)
    mb = shared_memory.SharedMemory(name=sb)
    mc = shared_memory.SharedMemory(name=sc)
    A = (ctypes.c_double * (N * N)).from_buffer(ma.buf)
    B = (ctypes.c_double * (N * N)).from_buffer(mb.buf)
    C = (ctypes.c_double * (N * N)).from_buffer(mc.buf)
    zeros = [0.0] * N
    for i in range(r0, r1):
        C[i * N : (i + 1) * N] = zeros

    for ii in range(r0, r1, bs):
        i_max = min(ii + bs, r1)
        for kk in range(0, N, bs):
            k_max = min(kk + bs, N)
            for jj in range(0, N, bs):
                j_max = min(jj + bs, N)
                for i in range(ii, i_max):
                    i_off = i * N
                    for k in range(kk, k_max):
                        a_ik = A[i_off + k]
                        k_off = k * N
                        for j in range(jj, j_max):
                            C[i_off + j] += a_ik * B[k_off + j]
    del A, B, C
    ma.close(); mb.close(); mc.close()

if __name__ == "__main__":
    bytes_sz = N * N * 8
    sa = shared_memory.SharedMemory(create=True, size=bytes_sz)
    sb = shared_memory.SharedMemory(create=True, size=bytes_sz)
    sc = shared_memory.SharedMemory(create=True, size=bytes_sz)

    A = (ctypes.c_double * (N * N)).from_buffer(sa.buf)
    B = (ctypes.c_double * (N * N)).from_buffer(sb.buf)
    C = (ctypes.c_double * (N * N)).from_buffer(sc.buf)
    for idx in range(N * N):
        A[idx] = ((idx % 17) - 8) * 0.1
        B[idx] = ((idx % 13) - 6) * 0.1

    rows_per_p = N // P_PHYS
    ranges = [(t * rows_per_p, (t + 1) * rows_per_p if t < P_PHYS - 1 else N) for t in range(P_PHYS)]
    total_flops = 2.0 * (N ** 3)

    def launch(fn, extra_args=()):
        procs = [mp.Process(target=fn, args=(sa.name, sb.name, sc.name, r0, r1) + extra_args) for r0, r1 in ranges]
        for p in procs: p.start()
        for p in procs: p.join()

    try:
        st_v1 = run_benchmark(lambda: launch(gemm_v1_worker), warmup_runs=0, timed_runs=1)
        gflops_v1 = total_flops / ((st_v1.median_ms / 1000.0) * 1e9)
        print(f"Variant 1 (Naive i-j-k):    {st_v1.median_ms:8.2f} ms | {gflops_v1:.4f} GFLOP/s", flush=True)

        st_v2 = run_benchmark(lambda: launch(gemm_v2_worker), warmup_runs=1, timed_runs=3)
        gflops_v2 = total_flops / ((st_v2.median_ms / 1000.0) * 1e9)
        ref_c = list(C)
        print(f"Variant 2 (Reordered i-k-j):{st_v2.median_ms:8.2f} ms | {gflops_v2:.4f} GFLOP/s", flush=True)

        print("\n=== Variant 3: 10-Point Tile Size (BS) Sweep ===", flush=True)
        print(f"{'BS':>4} | {'Tile Foot (B)':>14} | {'Median (ms)':>12} | {'GFLOP/s':>10} | {'Max Diff':>10}", flush=True)
        best_gflops_v3 = 0.0
        for bs in [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]:
            st_v3 = run_benchmark(lambda: launch(gemm_v3_worker, (bs,)), warmup_runs=1, timed_runs=3)
            gf = total_flops / ((st_v3.median_ms / 1000.0) * 1e9)
            max_diff = max(abs(C[i] - ref_c[i]) for i in range(N * N))
            foot = 3 * (bs ** 2) * 8
            print(f"{bs:>4} | {foot:>14} | {st_v3.median_ms:>12.2f} | {gf:>10.4f} | {max_diff:>10.2e}", flush=True)
            if gf > best_gflops_v3:
                best_gflops_v3 = gf

        print("\n" + emit_verify_tag("PHASE3", STUDENT_ID, gflops_v1, best_gflops_v3))
    finally:
        del A, B, C
        sa.close(); sa.unlink()
        sb.close(); sb.unlink()
        sc.close(); sc.unlink()