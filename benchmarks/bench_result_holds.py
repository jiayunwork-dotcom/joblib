"""Scale probe for explicit protection of cached results.

Run with::

    python benchmarks/bench_result_holds.py            # both sizes
    python benchmarks/bench_result_holds.py --size 16777216

The probe caches one array of the requested size, pins it, and runs 32
``loky`` tasks (2 workers) that all read the same result through the hold.
It records:

* how many bytes were actually copied on disk (cache and hold bookkeeping),
* how many bytes were read by the parent and by the workers (``rchar``),
* the extra resident memory of the parent and of the workers,
* the duration of cache maintenance and the number of protection records.

Array pages are memory-mapped, so the parent's resident memory only grows by
the pages it touches, and the workers' reads are page-cache hits, not copies.
The values reported below distinguish those from protocol overhead.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joblib import Memory, Parallel, delayed  # noqa: E402
from joblib._result_holds import holds_supported  # noqa: E402

TASKS = 32
WORKERS = 2


def _proc_rchar(pid="self"):
    try:
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                if line.startswith("rchar:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def _proc_work_units(pid="self"):
    try:
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def _rss_kib(pid="self"):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def _dir_size(path):
    total = 0
    if not os.path.exists(path):
        return 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def _count_lease_records(memory):
    directory = os.path.abspath(memory.store_backend.location) + ".joblib_holds/leases"
    try:
        return len(os.listdir(directory))
    except OSError:
        return 0


def _worker_pids(parallel):
    workers = getattr(parallel._backend, "_workers", None)
    if workers is None:
        return []
    return [proc.pid for proc in workers._processes.values()]


def consume(hold, chunk):
    array = hold.get()
    # Touch a chunk of pages so mapping is exercised without copying.
    return float(array[chunk : chunk + 1024].sum())


def run(size, tmp_root):
    from tempfile import mkdtemp

    location = mkdtemp(dir=tmp_root)
    memory = Memory(location=location, mmap_mode="r", verbose=0)
    calls = []

    @memory.cache
    def make(n):
        calls.append(n)
        return np.arange(n, dtype=np.uint8)

    n = size
    result = make.call_and_shelve(n)
    assert len(calls) == 1

    # Warm the worker pool up before measuring: the first tasks pay for
    # interpreter/numpy imports, which are not part of the hold protocol.
    Parallel(n_jobs=WORKERS, backend="loky")(delayed(int)(i) for i in range(WORKERS))

    cache_dir = memory.store_backend.location
    result_path = os.path.join(cache_dir, *result._call_id)
    cache_bytes_before = _dir_size(cache_dir)
    holds_bytes_before = _dir_size(cache_dir + ".joblib_holds")
    parent_rss_before = _rss_kib()
    rchar_before = _proc_rchar()

    hold = result.pin()

    parallel = Parallel(
        n_jobs=WORKERS,
        backend="loky",
        return_as="generator",
        pre_dispatch="all",
        batch_size=1,
    )
    generator = parallel(
        delayed(consume)(hold, (i * 4096) % (n - 4096)) for i in range(TASKS)
    )

    result_file_bytes = os.path.getsize(os.path.join(result_path, "output.pkl"))
    worker_rchar = {}
    worker_rss = {}
    for pid in _worker_pids(parallel):
        worker_rchar[pid] = _proc_rchar(pid)
        worker_rss[pid] = _rss_kib(pid)

    outputs = list(generator)
    assert len(outputs) == TASKS
    # The workers are reused by loky: subtract the baseline captured before
    # consuming so the reported values only cover the tasks themselves.
    worker_rchar_after = {}
    worker_rss_after = {}
    for pid in worker_rchar:
        worker_rchar_after[pid] = _proc_rchar(pid)
        worker_rss_after[pid] = _rss_kib(pid)

    parent_rss_after = _rss_kib()
    rchar_after = _proc_rchar()
    cache_bytes_after = _dir_size(cache_dir)
    holds_bytes_after = _dir_size(cache_dir + ".joblib_holds")
    records = _count_lease_records(memory)

    maintenance_start = time.perf_counter()
    reached = memory.reduce_size(items_limit=0)
    maintenance_duration = time.perf_counter() - maintenance_start
    # The held result must survive, so items_limit=0 cannot be reached.
    assert reached is False

    hold.close()
    maintenance_start = time.perf_counter()
    reached_after_close = memory.reduce_size(items_limit=0)
    maintenance_after_close = time.perf_counter() - maintenance_start
    assert reached_after_close is True

    return {
        "size_bytes": size,
        "size_mib": round(size / 1024**2, 2),
        "function_calls": len(calls),
        "tasks": TASKS,
        "workers": WORKERS,
        "result_file_bytes": result_file_bytes,
        "cache_bytes_before": cache_bytes_before,
        "cache_bytes_during_hold": cache_bytes_after,
        "cache_bytes_copied_by_hold": max(0, cache_bytes_after - cache_bytes_before),
        "hold_bookkeeping_bytes": max(0, holds_bytes_after - holds_bytes_before),
        "parent_rss_delta_kib": (parent_rss_after or 0) - (parent_rss_before or 0),
        "parent_read_bytes": (
            (rchar_after - rchar_before)
            if (rchar_after is not None and rchar_before is not None)
            else None
        ),
        "worker_rss_delta_kib": {
            pid: (worker_rss_after.get(pid) or 0) - (worker_rss.get(pid) or 0)
            for pid in worker_rss
        },
        "worker_read_bytes": {
            pid: (
                (worker_rchar_after.get(pid) or 0) - (worker_rchar.get(pid) or 0)
                if worker_rchar.get(pid) is not None
                and worker_rchar_after.get(pid) is not None
                else None
            )
            for pid in worker_rchar
        },
        "protection_records_during_run": records,
        "maintenance_seconds_held": round(maintenance_duration, 4),
        "maintenance_seconds_after_close": round(maintenance_after_close, 4),
        "limits_reached_while_held": reached,
        "limits_reached_after_close": reached_after_close,
    }


def gc_objects():
    import gc

    return gc.get_objects()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=None, help="array size in bytes")
    parser.add_argument("--tmp-root", default="/tmp")
    args = parser.parse_args()

    if not holds_supported():
        raise SystemExit("result holds require POSIX file locking")

    if args.size is not None:
        sizes = [args.size]
    else:
        sizes = [16 * 1024**2, 256 * 1024**2]

    reports = []
    for size in sizes:
        report = run(size, args.tmp_root)
        reports.append(report)
        print(json.dumps(report, indent=2))
    return reports


if __name__ == "__main__":
    main()
