import multiprocessing as mp
from multiprocessing import shared_memory
from harness import run_benchmark

P_PHYS = 8

def bw_worker(shm_a_name: str, shm_b_name: str, start: int, end: int) -> None:
    shm_a = shared_memory.SharedMemory(name=shm_a_name)
    shm_b = shared_memory.SharedMemory(name=shm_b_name)
    mv_a = memoryview(shm_a.buf)
    mv_b = memoryview(shm_b.buf)
    mv_a[start:end] = mv_b[start:end]
    mv_a.release()
    mv_b.release()
    shm_a.close()
    shm_b.close()

def fma_worker(flops: int) -> float:
    iters = flops // 32
    a, b = 1.0000001, 0.9999999
    x0=x1=x2=x3=x4=x5=x6=x7=x8=x9=x10=x11=x12=x13=x14=x15 = 1.0
    for _ in range(iters):
        x0 = x0 * a + b; x1 = x1 * a + b; x2 = x2 * a + b; x3 = x3 * a + b
        x4 = x4 * a + b; x5 = x5 * a + b; x6 = x6 * a + b; x7 = x7 * a + b
        x8 = x8 * a + b; x9 = x9 * a + b; x10 = x10 * a + b; x11 = x11 * a + b
        x12 = x12 * a + b; x13 = x13 * a + b; x14 = x14 * a + b; x15 = x15 * a + b
    return x0 + x15

if __name__ == "__main__":
    buf_bytes = 96 * 1024 * 1024
    shm_a = shared_memory.SharedMemory(create=True, size=buf_bytes)
    shm_b = shared_memory.SharedMemory(create=True, size=buf_bytes)
    chunk = buf_bytes // P_PHYS

    def run_bw():
        procs = [
            mp.Process(target=bw_worker, args=(shm_a.name, shm_b.name, t * chunk, (t + 1) * chunk if t < P_PHYS - 1 else buf_bytes))
            for t in range(P_PHYS)
        ]
        for p in procs: p.start()
        for p in procs: p.join()

    try:
        st_bw = run_benchmark(run_bw, warmup_runs=1, timed_runs=5)
    finally:
        shm_a.close(); shm_a.unlink()
        shm_b.close(); shm_b.unlink()

    bw_sustained = (2.0 * 96e6) / ((st_bw.median_ms / 1000.0) * 1e9)

    flops_per_worker = 80_000_000
    def run_fma():
        procs = [mp.Process(target=fma_worker, args=(flops_per_worker,)) for _ in range(P_PHYS)]
        for p in procs: p.start()
        for p in procs: p.join()

    st_fma = run_benchmark(run_fma, warmup_runs=1, timed_runs=5)
    total_flops = flops_per_worker * P_PHYS
    p_sustained = total_flops / ((st_fma.median_ms / 1000.0) * 1e9)
    ridge_point = p_sustained / bw_sustained

    print(f"Sustained DRAM Bandwidth (BW_sustained): {bw_sustained:.4f} GB/s (median {st_bw.median_ms:.2f} ms)")
    print(f"Sustained Compute Ceiling (P_sustained): {p_sustained:.4f} GFLOP/s (median {st_fma.median_ms:.2f} ms)")
    print(f"Empirical Ridge Point (I*):              {ridge_point:.4f} FLOPs/Byte")