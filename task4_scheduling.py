import math
import multiprocessing as mp
import time
from harness import emit_verify_tag

STUDENT_ID = 230110056
P_PHYS = 8
N_ITEMS = 12_000

def heavy_item(i: int) -> float:
    k = 1 + (i // 60) ** 2
    x = (i + 1) * 0.001
    sin, cos, sqrt = math.sin, math.cos, math.sqrt
    for _ in range(k):
        x = sin(x) + cos(x) + sqrt(abs(x) + 1.0)
    return x

def worker_static_block(tid: int, p: int, times_arr) -> None:
    t0 = time.perf_counter_ns()
    chunk = N_ITEMS // p
    start = tid * chunk
    end = (tid + 1) * chunk if tid < p - 1 else N_ITEMS
    for i in range(start, end):
        heavy_item(i)
    times_arr[tid] = (time.perf_counter_ns() - t0) / 1e6

def worker_static_cyclic(tid: int, p: int, times_arr) -> None:
    t0 = time.perf_counter_ns()
    for i in range(tid, N_ITEMS, p):
        heavy_item(i)
    times_arr[tid] = (time.perf_counter_ns() - t0) / 1e6

def worker_dynamic(tid: int, chunk_size: int, next_idx, lock, times_arr) -> None:
    t0 = time.perf_counter_ns()
    while True:
        with lock:
            start = next_idx.value
            if start >= N_ITEMS:
                break
            next_idx.value = start + chunk_size
        end = min(start + chunk_size, N_ITEMS)
        for i in range(start, end):
            heavy_item(i)
    times_arr[tid] = (time.perf_counter_ns() - t0) / 1e6

def run_sched(mode: str, chunk_size: int = 1, trials: int = 3):
    wall_samples = []
    lam_samples = []
    for _ in range(trials):
        times_arr = mp.Array('d', P_PHYS)
        next_idx = mp.Value('i', 0)
        lock = mp.Lock()
        t0 = time.perf_counter_ns()
        if mode == "block":
            procs = [mp.Process(target=worker_static_block, args=(t, P_PHYS, times_arr)) for t in range(P_PHYS)]
        elif mode == "cyclic":
            procs = [mp.Process(target=worker_static_cyclic, args=(t, P_PHYS, times_arr)) for t in range(P_PHYS)]
        else:
            procs = [mp.Process(target=worker_dynamic, args=(t, chunk_size, next_idx, lock, times_arr)) for t in range(P_PHYS)]
        for p in procs: p.start()
        for p in procs: p.join()
        t1 = time.perf_counter_ns()

        w_times = list(times_arr)
        lam = max(w_times) / (sum(w_times) / P_PHYS)
        wall_samples.append((t1 - t0) / 1e6)
        lam_samples.append(lam)

    wall_samples.sort()
    lam_samples.sort()
    mid = trials // 2
    return wall_samples[mid], lam_samples[mid]

if __name__ == "__main__":
    print(f"{'Strategy':<24} | {'T_wall (ms)':>12} | {'Imbalance lambda':>16}", flush=True)
    t_block, lam_block = run_sched("block")
    print(f"{'1. Static Block':<24} | {t_block:>12.2f} | {lam_block:>16.4f}", flush=True)

    t_cyclic, lam_cyclic = run_sched("cyclic")
    print(f"{'2. Static Cyclic':<24} | {t_cyclic:>12.2f} | {lam_cyclic:>16.4f}", flush=True)

    best_t_dyn = float("inf")
    for c in [1, 4, 16, 64, 256, 1024]:
        t_dyn, lam_dyn = run_sched("dynamic", chunk_size=c)
        print(f"{f'3. Dynamic (C={c})':<24} | {t_dyn:>12.2f} | {lam_dyn:>16.4f}", flush=True)
        if t_dyn < best_t_dyn:
            best_t_dyn = t_dyn

    print("\n" + emit_verify_tag("PHASE4", STUDENT_ID, t_block, best_t_dyn))