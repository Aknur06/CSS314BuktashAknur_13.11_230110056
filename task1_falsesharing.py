import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
from harness import run_benchmark, emit_verify_tag

STUDENT_ID = 230110056
P_PHYS = 8

def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005
    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc
    del arr
    shm.close()

def run_fs_experiment(p: int, stride: int, total_ops: int = 24_000_000):
    shm = shared_memory.SharedMemory(create=True, size=4096)
    iters_per_worker = total_ops // p

    def workload():
        procs = [
            mp.Process(target=fs_worker, args=(shm.name, t * stride, iters_per_worker))
            for t in range(p)
        ]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()

    try:
        stats = run_benchmark(workload, warmup_runs=2, timed_runs=7)
    finally:
        shm.close()
        shm.unlink()
    return stats

if __name__ == "__main__":
    p_log = mp.cpu_count()
    total_ops = 24_000_000

    print("=== TABLE 1A: Stride Sweep at p = P_phys ===")
    print(f"{'Stride':>6} | {'Bytes':>6} | {'Median (ms)':>12} | {'IQR (ms)':>10} | {'Throughput (Mops/s)':>20}")
    med_s1_phys = 0.0
    med_s16_phys = 0.0
    for stride in [1, 2, 4, 8, 16, 32]:
        st = run_fs_experiment(P_PHYS, stride, total_ops)
        mops = (total_ops / 1e6) / (st.median_ms / 1000.0)
        print(f"{stride:>6} | {stride*8:>6} | {st.median_ms:>12.2f} | {st.iqr_ms:>10.2f} | {mops:>20.2f}")
        if stride == 1:
            med_s1_phys = st.median_ms
        if stride == 16:
            med_s16_phys = st.median_ms

    p_list = []
    curr = 1
    while curr <= p_log:
        p_list.append(curr)
        curr *= 2
    if p_list[-1] != p_log:
        p_list.append(p_log)

    print("\n=== TABLE 1B: Process Scaling Sweep (Stride=1 vs Stride=16) ===")
    for stride in [1, 16]:
        print(f"\n--- Stride = {stride} ({stride*8} bytes) ---")
        print(f"{'p':>4} | {'Median (ms)':>12} | {'Speedup S_p':>12} | {'Eff E_p':>10} | {'Karp-Flatt e(p)':>16}")
        t1 = 0.0
        for p in p_list:
            st = run_fs_experiment(p, stride, total_ops)
            if p == 1:
                t1 = st.median_ms
                sp, ep, kf = 1.0, 1.0, 0.0
            else:
                sp = t1 / st.median_ms
                ep = sp / p
                kf = ((1.0 / sp) - (1.0 / p)) / (1.0 - (1.0 / p))
            print(f"{p:>4} | {st.median_ms:>12.2f} | {sp:>12.4f} | {ep:>10.4f} | {kf:>16.4f}")

    print("\n" + emit_verify_tag("PHASE1", STUDENT_ID, med_s1_phys, med_s16_phys))