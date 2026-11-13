import math
import multiprocessing as mp
import time
from harness import emit_verify_tag

STUDENT_ID = 230110056

def stage_a_serial(n: int) -> float:
    limit = int(0.05 * n)
    acc = 0.0
    for i in range(1, limit + 1):
        x = i * 0.001
        for _ in range(12):
            x = math.sin(x) + math.sqrt(abs(x) + 1.0)
        acc += x
    return acc

def stage_b_worker(args) -> float:
    start, end = args
    acc = 0.0
    for i in range(start, end):
        x = (i + 1) * 0.0001
        for _ in range(35):
            x = x * 1.00001 + math.sin(x)
        acc += x
    return acc

def measure_pipeline(pool, p: int, n_total: int, warmup: int = 2, trials: int = 5):
    t_total_list = []
    t_a_list = []
    chunk = n_total // p
    ranges = [(t * chunk, (t + 1) * chunk if t < p - 1 else n_total) for t in range(p)]

    for w in range(warmup + trials):
        t0 = time.perf_counter_ns()
        stage_a_serial(n_total)
        t1 = time.perf_counter_ns()
        pool.map(stage_b_worker, ranges)
        t2 = time.perf_counter_ns()
        if w >= warmup:
            t_a_list.append((t1 - t0) / 1e6)
            t_total_list.append((t2 - t0) / 1e6)

    t_a_list.sort()
    t_total_list.sort()
    mid = trials // 2
    return t_a_list[mid], t_total_list[mid]

if __name__ == "__main__":
    p_log = mp.cpu_count()
    n_strong = 1_600_000
    n0_weak = 250_000

    print("=== EXPERIMENT 2A: Strong Scaling (Fixed N = 1,600,000) ===")
    print(f"{'p':>3} | {'T_A (ms)':>10} | {'T_strong (ms)':>14} | {'S_strong':>10} | {'E_strong':>10}")
    t1_strong = 0.0
    t_a1_strong = 0.0
    s_plog_strong = 0.0

    for p in range(1, p_log + 1):
        with mp.Pool(processes=p) as pool:
            ta, ttot = measure_pipeline(pool, p, n_strong)
        if p == 1:
            t1_strong = ttot
            t_a1_strong = ta
        sp = t1_strong / ttot
        ep = sp / p
        if p == p_log:
            s_plog_strong = sp
        print(f"{p:>3} | {ta:>10.2f} | {ttot:>14.2f} | {sp:>10.4f} | {ep:>10.4f}")

    s_frac = t_a1_strong / t1_strong
    s_inf = 1.0 / s_frac
    s_amdahl_plog = 1.0 / (s_frac + (1.0 - s_frac) / p_log)
    print(f"\nSerial fraction s = {s_frac:.4f} | Amdahl S_inf = {s_inf:.2f} | Amdahl S(P_log) = {s_amdahl_plog:.2f}")

    print("\n=== EXPERIMENT 2B: Weak Scaling (N(p) = p * 250,000) ===")
    print(f"{'p':>3} | {'N(p)':>10} | {'T_weak (ms)':>12} | {'E_weak':>10} | {'alpha':>8} | {'S_Gustafson':>12}")
    t1_weak = 0.0
    s_plog_gustafson = 0.0

    for p in range(1, p_log + 1):
        n_p = p * n0_weak
        with mp.Pool(processes=p) as pool:
            ta, ttot = measure_pipeline(pool, p, n_p)
        if p == 1:
            t1_weak = ttot
        ew = t1_weak / ttot
        alpha = ta / ttot
        sg = p - alpha * (p - 1)
        if p == p_log:
            s_plog_gustafson = sg
        print(f"{p:>3} | {n_p:>10} | {ttot:>12.2f} | {ew:>10.4f} | {alpha:>8.4f} | {sg:>12.4f}")

    print("\n" + emit_verify_tag("PHASE2", STUDENT_ID, s_plog_strong, s_plog_gustafson))