import multiprocessing as mp
import time
from harness import emit_verify_tag

STUDENT_ID = 230110056

if __name__ == "__main__":
    p_log = mp.cpu_count()
    iters = 2_000_000
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        time.perf_counter_ns()
    t1 = time.perf_counter_ns()
    avg_ns = (t1 - t0) / iters

    print(f"Logical CPUs (P_log): {p_log}")
    print(f"Average timer call latency: {avg_ns:.2f} ns")
    print(emit_verify_tag("PHASE0", STUDENT_ID, float(p_log), avg_ns))