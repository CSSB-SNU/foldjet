"""Host speed probe.

Run it in the SAME allocation / cgroup / affinity as the kit run, right before and right
after it. Prints one JSON line: fixed CPU microbenchmarks + the CPU clocks and affinity
seen by this process.
"""

import json
import os
import socket
import sys
import time

import numpy as np

t0 = time.perf_counter()
s = 0
# pure-python loop: single-thread interpreter speed (the kit's host-side launch path)
for i in range(20_000_000):
    s += i
loop_s = time.perf_counter() - t0


def matmul(threads: int) -> float:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    a = np.random.default_rng(0).standard_normal((4096, 4096), dtype=np.float32)
    t = time.perf_counter()
    a @ a
    return time.perf_counter() - t


# BLAS threads as the process finds them
mm_all = min(matmul(os.cpu_count() or 1) for _ in range(2))
x = np.ones(512 * 1024 * 1024 // 4 * 1, dtype=np.float32)  # 2 GiB source
y = np.empty_like(x)
t = time.perf_counter()
np.copyto(y, x)
copy_s = time.perf_counter() - t
aff = sorted(os.sched_getaffinity(0))
mhz = []
try:
    # stdlib-only probe: plain open() kept (no pathlib import)
    with open("/proc/cpuinfo") as fh:  # noqa: PTH123 - stdlib-only probe
        mhz.extend(
            float(line.split(":")[1]) for line in fh if line.startswith("cpu MHz")
        )
except OSError:
    pass
print(
    json.dumps(
        {
            "host": socket.gethostname(),
            "when": sys.argv[1] if len(sys.argv) > 1 else "",
            "ts": time.strftime("%H:%M:%S"),
            "loop_2e7_s": round(loop_s, 3),
            "matmul4096_f32_s": round(mm_all, 4),
            "copy_2GiB_s": round(copy_s, 3),
            "copy_GBps": round(2.147 / copy_s, 2),
            "n_cpus_allowed": len(aff),
            "cpu_mhz_mean_allowed": round(
                sum(mhz[i] for i in aff if i < len(mhz))
                / max(1, len([i for i in aff if i < len(mhz)])),
                0,
            )
            if mhz
            else None,
            "load1": os.getloadavg()[0],
        }
    )
)
