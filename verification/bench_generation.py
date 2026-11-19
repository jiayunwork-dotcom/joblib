"""Memory/file/lock cost of generation bookkeeping, 16 MiB and 256 MiB.

Reported separately:
  * mapping pages: RSS growth of the holder's process while reading a memmap
    (only the touched pages), plus RssFile;
  * the benchmark's own memory: parent RSS growth across the whole run,
    ru_maxrss, and the Python-side bookkeeping object counts.

Everything runs in a temp dir created by this script; every measurement is
taken with events/completion, never with a timing sleep.
"""
import gc
import json
import os
import resource
import statistics
import tempfile
import time

import numpy as np

from joblib import Memory, Parallel, delayed

MIB = 1024**2
TASKS = 8
WORKERS = 2


def status_kib(field, pid="self"):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def dir_bytes(path):
    total = 0
    if not os.path.exists(path):
        return 0
    for dirpath, _, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def count_files(path):
    n = 0
    for _, _, names in os.walk(path):
        n += len(names)
    return n


def count_generation_markers(store):
    n = 0
    for dirpath, _, names in os.walk(store):
        if "generation" in names:
            n += 1
    return n


def worker_read(hold, idx):
    return int(hold.get()[idx])


def run(size):
    with tempfile.TemporaryDirectory() as tmp:
        mem_rss0 = status_kib("VmRSS")
        file_rss0 = status_kib("RssFile")
        mem = Memory(location=tmp + "/cache", mmap_mode="r", verbose=0)
        store = mem.store_backend.location

        state = {"value": 1}

        @mem.cache
        def make(n):
            return np.full(n, state["value"], dtype=np.uint8)

        t0 = time.perf_counter()
        shelf = make.call_and_shelve(size)
        publish_s = time.perf_counter() - t0

        cache_bytes_1 = dir_bytes(store)
        gc.collect()
        rss_before_hold = status_kib("VmRSS")
        file_before_hold = status_kib("RssFile")

        hold = shelf.pin()
        t0 = time.perf_counter()
        arr = hold.get()             # memmap, no copy
        t_map = time.perf_counter() - t0
        rss_after_map = status_kib("VmRSS")
        file_after_map = status_kib("RssFile")
        # Touch exactly 16 pages spread across the array.
        touched = 0
        for i in range(0, size, max(1, size // 16)):
            touched += int(arr[i])
        rss_after_touch = status_kib("VmRSS")
        file_after_touch = status_kib("RssFile")

        holds_dir = os.path.abspath(store) + ".joblib_holds"
        bookkeeping_files = count_files(holds_dir)
        bookkeeping_bytes = dir_bytes(holds_dir)
        markers_before = count_generation_markers(store)

        # Publish a second generation at the *same* key while the first is
        # held.  The old directory must be renamed (no byte copied).
        state["value"] = 2
        t0 = time.perf_counter()
        make.call(size)
        replace_s = time.perf_counter() - t0
        cache_bytes_2 = dir_bytes(store)
        retired_bytes = dir_bytes(os.path.join(holds_dir, "retired"))

        # Two generations through all three Parallel modes; parent closes.
        outcomes = {}
        for backend, n_jobs in (("sequential", 1), ("threading", 2), ("loky", 2)):
            # A fresh shelf for generation 2 (value 2); ``shelf`` is still 1.
            hold_old = shelf.pin()
            new_shelf = make.call_and_shelve(size)
            hold_new = new_shelf.pin()
            assert int(hold_old.get()[0]) == 1
            assert int(hold_new.get()[0]) == 2
            gen = Parallel(
                n_jobs=n_jobs,
                backend=backend,
                return_as="generator",
                pre_dispatch="all",
                batch_size=1,
            )(
                delayed(worker_read)(h, i)
                for i in range(TASKS // 2)
                for h in (hold_old, hold_new)
            )
            if backend != "sequential":
                hold_old.close()
                hold_new.close()
            values = list(gen)
            if backend == "sequential":
                hold_old.close()
                hold_new.close()
            assert values == [1, 2] * (TASKS // 2), (backend, values)
            outcomes[backend] = {"values": sorted(set(values)), "tasks": len(values)}
            del new_shelf

        # Lock costs measured on the store directory itself.
        from joblib import _result_holds

        lock_times = []
        for _ in range(20):
            t0 = time.perf_counter()
            with _result_holds.store_lock(store, exclusive=True):
                pass
            lock_times.append(time.perf_counter() - t0)
        pin_times = []
        for _ in range(20):
            t0 = time.perf_counter()
            h = shelf.pin()
            pin_times.append(time.perf_counter() - t0)
            h.close()

        markers_after = count_generation_markers(store)
        arr = None
        gc.collect()
        # A live hold still protects generation 1: maintenance cannot reach
        # the target and must say so without deleting the held value.
        t0 = time.perf_counter()
        reached_while_held = mem.reduce_size(items_limit=0)
        maintenance_while_held_s = time.perf_counter() - t0
        assert reached_while_held is False
        hold.close()
        gc.collect()
        t0 = time.perf_counter()
        reached = mem.reduce_size(items_limit=0)
        maintenance_s = time.perf_counter() - t0
        assert reached is True
        gc.collect()
        mem_rss1 = status_kib("VmRSS")
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        return {
            "size_mib": round(size / MIB, 2),
            "publish_seconds": round(publish_s, 3),
            "cache_bytes_one_generation": cache_bytes_1,
            "cache_overhead_bytes": cache_bytes_1 - size,
            "cache_bytes_after_second_generation": cache_bytes_2,
            "bytes_added_by_second_generation": cache_bytes_2 - cache_bytes_1,
            "retired_bytes_after_replace": retired_bytes,
            "bookkeeping_files_with_two_generations": bookkeeping_files,
            "bookkeeping_bytes_with_two_generations": bookkeeping_bytes,
            "generation_markers_before_after_replace": [markers_before, markers_after],
            "hold_get_map_seconds": round(t_map, 5),
            "mapping_rss_delta_kib_after_get": (rss_after_map or 0)
            - (rss_before_hold or 0),
            "mapping_rss_delta_kib_after_16_touched_pages": (
                (rss_after_touch or 0) - (rss_before_hold or 0)
            ),
            "mapping_file_rss_delta_kib_after_16_touched_pages": (
                (file_after_touch or 0) - (file_before_hold or 0)
            ),
            "bench_process_rss_delta_kib": (mem_rss1 or 0) - (mem_rss0 or 0),
            "bench_max_rss_kib": max_rss,
            "lock_seconds_median": round(statistics.median(lock_times), 6),
            "lock_seconds_max": round(max(lock_times), 6),
            "pin_seconds_median": round(statistics.median(pin_times), 6),
            "maintenance_seconds_while_held": round(maintenance_while_held_s, 4),
            "maintenance_seconds_after_release": round(maintenance_s, 4),
            "parallel_two_generations": outcomes,
        }


def main():
    print(json.dumps([run(16 * MIB), run(256 * MIB)], indent=2))


if __name__ == "__main__":
    main()
