"""Tests for explicit protection ("holds") of cached results.

These tests exercise ``MemorizedResult.pin()``, its interaction with cache
maintenance, invalidation and :class:`joblib.Parallel`, including
multi-process scheduling.
"""

import datetime
import gc
import os
import pickle
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from joblib import Memory, Parallel, delayed
from joblib._result_holds import (
    CacheInUseError,
    ResultReleasedError,
    ResultUnavailableError,
    holds_supported,
)
from joblib.memory import MemorizedResult
from joblib.test.common import np, with_numpy
from joblib.testing import raises

pytestmark = pytest.mark.skipif(
    not holds_supported(), reason="result holds require POSIX file locking"
)

HAVE_LOKY = True
try:  # pragma: no cover - environment dependent
    from joblib.externals.loky.process_executor import TerminatedWorkerError  # noqa
except ImportError:  # pragma: no cover
    HAVE_LOKY = False


def item_path(result):
    return os.path.join(result.store_backend.location, *result._call_id)


def output_filename(result):
    return os.path.join(item_path(result), "output.pkl")


def holds_dir(memory):
    return os.path.abspath(memory.store_backend.location) + ".joblib_holds"


def lease_files(memory):
    directory = os.path.join(holds_dir(memory), "leases")
    try:
        return sorted(os.listdir(directory))
    except OSError:
        return []


def directory_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, filename))
            except OSError:  # pragma: no cover - race with deletion
                pass
    return total


def _make_counted(memory, mmap_mode=None, n_outputs=16):
    calls = []
    mem = Memory(location=memory, mmap_mode=mmap_mode, verbose=0)

    @mem.cache
    def make(x):
        calls.append(x)
        return np.arange(n_outputs, dtype=np.float64) + x

    return mem, make, calls


###############################################################################
# Small arrays: exact values, file identity, mmap and evaluation count


@with_numpy
def test_pin_returns_handle_with_same_value_and_file(tmp_path):
    memory, make, calls = _make_counted(tmp_path, mmap_mode="r")

    result = make.call_and_shelve(0)
    assert len(calls) == 1
    expected = np.arange(16, dtype=np.float64)

    original_file = output_filename(result)
    assert os.path.exists(original_file)

    hold = result.pin()
    try:
        value = hold.get()
        # Exact full value, not just a few elements.
        np.testing.assert_array_equal(value, expected)
        # It is a memmap of the original cache file, read-only.
        assert isinstance(value, np.memmap)
        assert value.mode == "r"
        assert os.path.realpath(value.filename) == os.path.realpath(original_file)
        # No recomputation and no additional copy on disk.
        assert len(calls) == 1
        assert os.path.exists(original_file)
    finally:
        hold.close()


@with_numpy
def test_pin_without_mmap_mode_returns_plain_array(tmp_path):
    memory, make, calls = _make_counted(tmp_path, mmap_mode=None)

    result = make.call_and_shelve(0)
    with result.pin() as hold:
        value = hold.get()
        assert not isinstance(value, np.memmap)
        np.testing.assert_array_equal(value, np.arange(16, dtype=np.float64))
    assert len(calls) == 1


@with_numpy
def test_pin_does_not_copy_result_bytes(tmp_path):
    calls = []
    memory = Memory(location=tmp_path, verbose=0)

    @memory.cache
    def make(n):
        calls.append(n)
        return np.full(n, 3.5)

    size = 1 << 20  # 8 MiB
    result = make.call_and_shelve(size)
    assert len(calls) == 1
    hold = result.pin()
    try:
        original_file = output_filename(result)
        original_stat = os.stat(original_file)
        before_cache = directory_size(memory.store_backend.location)
        before_holds = directory_size(holds_dir(memory))

        value = hold.get()
        np.testing.assert_array_equal(value, np.full(size, 3.5))

        after_stat = os.stat(original_file)
        # The cache file was neither rewritten nor duplicated.
        assert (original_stat.st_ino, original_stat.st_size) == (
            after_stat.st_ino,
            after_stat.st_size,
        )
        assert directory_size(memory.store_backend.location) == before_cache
        # Bookkeeping grows with the number of holds, not with array bytes.
        assert directory_size(holds_dir(memory)) - before_holds < 4096
        assert len(calls) == 1
    finally:
        hold.close()


@with_numpy
def test_pin_handle_context_manager_and_idempotent_close(tmp_path):
    memory, make, _ = _make_counted(tmp_path)

    result = make.call_and_shelve(1)
    with result.pin() as hold:
        np.testing.assert_array_equal(hold.get(), np.arange(16) + 1)
    # close() can be called repeatedly.
    hold.close()
    hold.close()
    with raises(ResultReleasedError):
        hold.get()
    # The regular result is not affected by the hold being closed.
    np.testing.assert_array_equal(result.get(), np.arange(16) + 1)


@with_numpy
def test_pin_raises_when_result_already_deleted(tmp_path):
    memory, make, calls = _make_counted(tmp_path)

    result = make.call_and_shelve(2)
    result.clear()
    with raises(KeyError):
        result.pin()
    assert len(calls) == 1

    # A later recomputation at the same path produces a new value, which an
    # old (impossible to create) hold must never take over.  Here we simply
    # check that a fresh pin sees the new computation.
    make.call(2)
    assert len(calls) == 2


@with_numpy
def test_multiple_holds_on_same_result(tmp_path):
    memory, make, calls = _make_counted(tmp_path)

    result = make.call_and_shelve(3)
    hold_a = result.pin()
    hold_b = result.pin()
    assert len(lease_files(memory)) == 2

    hold_a.close()
    # Releasing one holder does not release the other.
    with raises(CacheInUseError):
        result.clear()
    np.testing.assert_array_equal(hold_b.get(), np.arange(16) + 3)

    hold_b.close()
    assert lease_files(memory) == []
    result.clear()
    assert not os.path.exists(output_filename(result))
    assert len(calls) == 1


@with_numpy
def test_holds_on_different_results_are_independent(tmp_path):
    memory, make, _ = _make_counted(tmp_path)

    result_a = make.call_and_shelve(0)
    result_b = make.call_and_shelve(1)
    hold_a = result_a.pin()
    try:
        memory.reduce_size(items_limit=1)
        # The held result survives, the other one is reclaimed.
        assert os.path.exists(output_filename(result_a))
        assert not os.path.exists(output_filename(result_b))
        np.testing.assert_array_equal(hold_a.get(), np.arange(16))
        with raises(ResultReleasedError):
            hold_a.close()
            hold_a.get()
    finally:
        hold_a.close()


###############################################################################
# Maintenance and invalidation


@with_numpy
def test_reduce_size_keeps_held_and_returns_not_reached(tmp_path):
    memory, make, _ = _make_counted(tmp_path)

    result = make.call_and_shelve(0)
    held = result.pin()
    make(1)
    make(2)
    try:
        assert len(memory.store_backend.get_items()) == 3
        # Cannot reach the target while the held result must stay.
        assert memory.reduce_size(items_limit=0) is False
        remaining = memory.store_backend.get_items()
        assert len(remaining) == 1
        assert os.path.realpath(remaining[0].path) == os.path.realpath(
            item_path(result)
        )
        np.testing.assert_array_equal(held.get(), np.arange(16))
    finally:
        held.close()

    # Once released, the next maintenance can reclaim it.
    assert memory.reduce_size(items_limit=0) is True
    assert memory.store_backend.get_items() == []


@with_numpy
def test_reduce_size_bytes_and_age_limits_with_hold(tmp_path):
    memory, make, _ = _make_counted(tmp_path)

    held = make.call_and_shelve(0).pin()
    make(1)
    make(2)
    try:
        # Byte limit smaller than the cache: it cannot be reached because the
        # held item must be kept, but other items are reclaimed.
        assert memory.reduce_size(bytes_limit=1) is False
        assert len(memory.store_backend.get_items()) == 1

        # Same for the age limit.
        time.sleep(0.01)
        assert memory.reduce_size(age_limit=datetime.timedelta(seconds=0)) is False
        assert len(memory.store_backend.get_items()) == 1
    finally:
        held.close()


@with_numpy
def test_clear_scope_with_hold_raises_before_deleting_anything(tmp_path):
    memory, make, _ = _make_counted(tmp_path)

    held_result = make.call_and_shelve(0)
    other_result = make.call_and_shelve(1)
    hold = held_result.pin()
    try:
        snapshot = sorted(
            os.path.relpath(item.path, memory.store_backend.location)
            for item in memory.store_backend.get_items()
        )

        with raises(CacheInUseError):
            memory.clear()
        with raises(CacheInUseError):
            make.clear()
        with raises(CacheInUseError):
            held_result.clear()

        # Nothing was deleted by the failed clear operations.
        after = sorted(
            os.path.relpath(item.path, memory.store_backend.location)
            for item in memory.store_backend.get_items()
        )
        assert snapshot == after
        np.testing.assert_array_equal(hold.get(), np.arange(16))
    finally:
        hold.close()

    # Once the scope is free, clearing works normally.
    held_result.clear()
    assert not os.path.exists(output_filename(held_result))
    # An unrelated result is untouched by a per-item clear.
    assert os.path.exists(output_filename(other_result))
    other_result.clear()
    assert not os.path.exists(output_filename(other_result))


@with_numpy
def test_clear_scope_without_hold_still_deletes(tmp_path):
    memory, make, _ = _make_counted(tmp_path)
    result = make.call_and_shelve(0)
    hold = result.pin()
    hold.close()
    memory.clear()
    assert memory.store_backend.get_items() == []


@with_numpy
def test_force_recompute_does_not_replace_held_value(tmp_path):
    memory = Memory(location=tmp_path, verbose=0)
    calls = []

    def make_impl(x):
        calls.append(x)
        return np.arange(x, dtype=np.float64) + 100 * len(calls)

    make = memory.cache(make_impl)
    result = make.call_and_shelve(4)
    hold = result.pin()
    try:
        old_value = hold.get().copy()
        # Force a recomputation at the same cache path.
        new_value = make.call(4)
        np.testing.assert_array_equal(hold.get(), old_value)
        assert not np.array_equal(new_value, old_value)
        assert len(calls) == 2
    finally:
        hold.close()


@with_numpy
def test_function_code_change_keeps_held_old_value(tmp_path):
    memory = Memory(location=tmp_path, verbose=0)

    def make_sum(a, b):
        return np.array([a + b], dtype=np.float64)

    def make_product(a, b):
        return np.array([a * b], dtype=np.float64)

    make_sum.__module__ = make_product.__module__ = "hold_test_module"
    make_sum.__name__ = make_product.__name__ = "changing"

    cached = memory.cache(make_sum)
    result = cached.call_and_shelve(3, 4)
    hold = result.pin()
    try:
        np.testing.assert_array_equal(hold.get(), [7.0])
        # Simulate a code change in place.
        cached.func.__code__ = make_product.__code__
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_value = cached(3, 4)
        np.testing.assert_array_equal(new_value, [12.0])
        # The held old value remains readable and is not returned as a hit.
        np.testing.assert_array_equal(hold.get(), [7.0])
        assert memory.store_backend.get_items(), "new result must be cached"
    finally:
        hold.close()


@with_numpy
def test_invalid_cache_validation_callback_does_not_return_old_value(tmp_path):
    memory = Memory(location=tmp_path, verbose=0)
    valid = {"value": True}
    calls = []

    @memory.cache(cache_validation_callback=lambda metadata: valid["value"])
    def make(x):
        calls.append(x)
        return np.arange(x, dtype=np.float64) + len(calls)

    result = make.call_and_shelve(5)
    hold = result.pin()
    try:
        old_value = hold.get().copy()
        valid["value"] = False
        # The invalid value must not be returned as a hit, and the held copy
        # must stay readable.
        new_value = make(5)
        assert len(calls) == 2
        assert not np.array_equal(new_value, old_value)
        np.testing.assert_array_equal(hold.get(), old_value)
    finally:
        hold.close()


###############################################################################
# Parallel integration


def _consume(hold, offset=0):
    array = hold.get()
    return float(array[offset])


def _consume_nested(payload, expected_offset=0):
    if isinstance(payload, dict):
        hold = payload["hold"]
    elif isinstance(payload, (list, tuple)):
        hold = payload[0]
    else:  # pragma: no cover - defensive
        hold = payload
    array = hold.get()
    return float(array[expected_offset])


@with_numpy
@pytest.mark.parametrize(
    "backend,n_jobs",
    [
        ("sequential", 1),
        ("threading", 2),
        pytest.param(
            "loky", 2, marks=pytest.mark.skipif(not HAVE_LOKY, reason="loky missing")
        ),
    ],
)
def test_parallel_accepts_hold_as_argument(tmp_path, backend, n_jobs):
    memory, make, calls = _make_counted(tmp_path, mmap_mode="r")
    result = make.call_and_shelve(0)
    hold = result.pin()
    try:
        outputs = Parallel(n_jobs=n_jobs, backend=backend, batch_size=1)(
            delayed(_consume)(hold, i) for i in range(4)
        )
        assert outputs == [0.0, 1.0, 2.0, 3.0]
        assert len(calls) == 1
    finally:
        hold.close()

    # The parent handle works again after the parallel call.
    hold2 = result.pin()
    try:
        assert _consume(hold2, 5) == 5.0
    finally:
        hold2.close()


@with_numpy
def test_parallel_accepts_hold_nested_in_containers(tmp_path):
    memory, make, calls = _make_counted(tmp_path)
    result = make.call_and_shelve(0)
    hold = result.pin()
    try:
        outputs = Parallel(n_jobs=2, backend="loky", batch_size=1)(
            delayed(_consume_nested)(payload, 2)
            for payload in (
                [hold],
                (hold,),
                {"hold": hold},
            )
        )
        assert outputs == [2.0, 2.0, 2.0]
        assert len(calls) == 1
    finally:
        hold.close()


@with_numpy
def test_parallel_generator_parent_closes_before_consumption(tmp_path):
    memory, make, calls = _make_counted(tmp_path)
    result = make.call_and_shelve(0)
    hold = result.pin()

    generator = Parallel(
        n_jobs=2,
        backend="loky",
        return_as="generator",
        pre_dispatch="all",
        batch_size=1,
    )(delayed(_consume)(hold, i) for i in range(8))
    # Close the parent handle before consuming any result.
    hold.close()
    outputs = list(generator)
    assert outputs == [float(i) for i in range(8)]
    assert len(calls) == 1


@with_numpy
def test_parallel_many_tasks_share_one_hold_parent_closes(tmp_path):
    memory, make, calls = _make_counted(tmp_path)
    result = make.call_and_shelve(0)
    hold = result.pin()

    generator = Parallel(
        n_jobs=2,
        backend="loky",
        return_as="generator",
        pre_dispatch="all",
        batch_size=1,
    )(delayed(_consume)(hold, i % 4) for i in range(32))
    hold.close()
    outputs = list(generator)
    assert outputs == [float(i % 4) for i in range(32)]
    assert len(calls) == 1
    # No protection is left behind.
    assert lease_files(memory) == []


def _close_own_hold(hold, offset=0):
    try:
        return float(hold.get()[offset])
    finally:
        hold.close()


@with_numpy
def test_task_closing_its_own_handle_does_not_release_others(tmp_path):
    memory, make, calls = _make_counted(tmp_path)
    result = make.call_and_shelve(0)
    hold = result.pin()
    try:
        outputs = Parallel(n_jobs=2, backend="loky", batch_size=1)(
            delayed(_close_own_hold)(hold, i % 4) for i in range(6)
        )
        assert outputs == [float(i % 4) for i in range(6)]
        # The parent handle is unaffected and keeps protecting the value.
        with raises(CacheInUseError):
            result.clear()
        assert float(hold.get()[0]) == 0.0
    finally:
        hold.close()
    result.clear()


@with_numpy
def test_resubmitting_a_closed_hold_is_rejected(tmp_path):
    memory, make, _ = _make_counted(tmp_path)
    result = make.call_and_shelve(0)
    hold = result.pin()
    hold.close()
    for backend, n_jobs in [("sequential", 1), ("threading", 2)]:
        with raises(ResultReleasedError):
            Parallel(n_jobs=n_jobs, backend=backend, batch_size=1)(
                [delayed(_consume)(hold, 0)]
            )


def _return_bare_array(hold):
    return hold.get()


@with_numpy
def test_bare_arrays_returned_by_tasks_are_not_protected(tmp_path):
    memory, make, _ = _make_counted(tmp_path)
    result = make.call_and_shelve(0)
    hold = result.pin()
    try:
        arrays = Parallel(n_jobs=1, backend="sequential")(
            delayed(_return_bare_array)(hold) for _ in range(2)
        )
    finally:
        hold.close()
    # The tasks are over; the returned arrays carry no protection.  The
    # caller must take its own hold if it needs to keep using them.
    assert memory.reduce_size(items_limit=0) is True
    assert memory.store_backend.get_items() == []
    with raises(KeyError):
        result.get()
    # The arrays themselves are still usable in memory.
    np.testing.assert_array_equal(arrays[0], np.arange(16))


@with_numpy
def test_unsupported_backend_rejected_before_accepting_task(tmp_path):
    memory, make, _ = _make_counted(tmp_path)
    result = make.call_and_shelve(0)
    hold = result.pin()
    try:
        if sys.platform.startswith("linux"):
            with raises(Exception):
                Parallel(n_jobs=2, backend="multiprocessing", batch_size=1)(
                    [delayed(_consume)(hold, 0)]
                )
    finally:
        hold.close()


@with_numpy
def test_task_error_releases_reserved_protection(tmp_path):
    memory, make, _ = _make_counted(tmp_path)
    result = make.call_and_shelve(0)
    hold = result.pin()

    def failing(hold):
        hold.get()
        raise ValueError("boom")

    try:
        with raises(ValueError):
            Parallel(n_jobs=2, backend="loky", batch_size=1)(
                delayed(failing)(hold) for _ in range(4)
            )
    finally:
        hold.close()
    assert lease_files(memory) == []
    assert memory.reduce_size(items_limit=0) is True
    assert memory.store_backend.get_items() == []


class _UnpicklableArgument:
    def __reduce__(self):
        raise RuntimeError("cannot serialize")


@with_numpy
def test_argument_serialization_failure_releases_protection(tmp_path):
    memory, make, _ = _make_counted(tmp_path)
    result = make.call_and_shelve(0)
    hold = result.pin()

    def consume_two(hold, other):
        return float(hold.get()[0])

    try:
        with raises(Exception):
            Parallel(n_jobs=2, backend="loky", batch_size=1)(
                delayed(consume_two)(hold, _UnpicklableArgument()) for _ in range(2)
            )
    finally:
        hold.close()
    assert lease_files(memory) == []
    assert memory.reduce_size(items_limit=0) is True


@with_numpy
def test_generator_early_close_releases_protection(tmp_path):
    memory, make, _ = _make_counted(tmp_path)
    result = make.call_and_shelve(0)
    hold = result.pin()
    try:
        generator = Parallel(
            n_jobs=2,
            backend="loky",
            return_as="generator",
            pre_dispatch="all",
            batch_size=1,
        )(delayed(_consume)(hold, i % 4) for i in range(50))
        next(generator)
        generator.close()
    finally:
        hold.close()
    assert lease_files(memory) == []
    assert memory.reduce_size(items_limit=0) is True


@with_numpy
@pytest.mark.skipif(not HAVE_LOKY, reason="loky missing")
def test_killed_loky_worker_does_not_leave_permanent_hold(tmp_path):
    memory, make, _ = _make_counted(tmp_path)
    result = make.call_and_shelve(0)
    hold = result.pin()

    def slow(hold):
        time.sleep(30)
        return float(hold.get()[0])

    parallel = Parallel(
        n_jobs=2,
        backend="loky",
        return_as="generator",
        pre_dispatch="all",
        batch_size=1,
    )
    generator = parallel(delayed(slow)(hold) for _ in range(4))
    try:
        # Wait until the workers actually started.
        deadline = time.time() + 30
        while time.time() < deadline:
            workers = getattr(parallel._backend, "_workers", None)
            if workers is not None and len(workers._processes) >= 2:
                break
            time.sleep(0.05)
        procs = list(parallel._backend._workers._processes.values())
        assert len(procs) >= 1
        os.kill(procs[0].pid, signal.SIGKILL)
        with raises(Exception):
            for _ in range(4):
                next(generator)
    finally:
        try:
            generator.close()
        except Exception:  # pragma: no cover - defensive
            pass
        hold.close()

    # Parallel performed its error handling and the killed worker is gone;
    # maintenance can now reclaim everything.
    assert lease_files(memory) == []
    assert memory.reduce_size(items_limit=0) is True
    assert memory.store_backend.get_items() == []


@with_numpy
def test_other_live_task_result_remains_protected(tmp_path):
    memory = Memory(location=tmp_path, verbose=0)
    calls = []

    @memory.cache
    def make(x):
        calls.append(x)
        return np.arange(8, dtype=np.float64) + x

    held_result = make.call_and_shelve(0)
    # A second, unrelated result that must not be deleted while the held
    # result is active.
    make.call_and_shelve(1)
    hold = held_result.pin()

    def failing(hold):
        raise RuntimeError("task failed")

    try:
        with raises(RuntimeError):
            Parallel(n_jobs=2, backend="loky", batch_size=1)(
                delayed(failing)(hold) for _ in range(2)
            )
        # Failure cleaned up the tasks' own protection, but the unrelated
        # live hold still prevents deletion.
        with raises(CacheInUseError):
            memory.clear()
        np.testing.assert_array_equal(hold.get(), np.arange(8))
    finally:
        hold.close()
    memory.clear()


###############################################################################
# Cross-process timing: maintenance from another process

_MAINTENANCE_CHILD = textwrap.dedent(
    """
    import os, sys, time
    sys.path.insert(0, sys.argv[1])
    from joblib import Memory

    store, mode, ready, go = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    memory = Memory(location=store, verbose=0)
    with open(ready, "w") as f:
        f.write("ready")
    deadline = time.time() + 60
    while not os.path.exists(go):
        if time.time() > deadline:
            sys.exit(2)
        time.sleep(0.02)
    if mode == "clear":
        try:
            memory.clear()
            print("cleared")
        except Exception as exc:
            print(type(exc).__name__)
    else:
        print(memory.reduce_size(items_limit=0))
    """
)

_CONSUMER_CHILD = textwrap.dedent(
    """
    import os, sys, time
    sys.path.insert(0, sys.argv[1])
    import numpy as np
    from joblib import Memory

    store, ready, go, done = sys.argv[2:6]
    memory = Memory(location=store, mmap_mode="r", verbose=0)

    @memory.cache
    def make(x):
        return np.arange(16, dtype=np.float64) + x

    result = make.call_and_shelve(0)
    hold = result.pin()
    with open(ready, "w") as f:
        f.write("ready")
    deadline = time.time() + 60
    while not os.path.exists(go):
        if time.time() > deadline:
            sys.exit(2)
        time.sleep(0.02)
    # Read while the other process is maintaining.
    print(float(hold.get()[3]), flush=True)
    time.sleep(0.2)  # still reading while maintenance happens
    with open(done, "w") as f:
        f.write("done")
    hold.close()
    """
)


def _run_child(code, args, project_root):
    return subprocess.Popen(
        [sys.executable, "-c", code, os.fspath(project_root), *map(str, args)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait_for(path, timeout=60):
    deadline = time.time() + timeout
    while not os.path.exists(path):
        if time.time() > deadline:  # pragma: no cover - timeout guard
            raise AssertionError(f"timed out waiting for {path}")
        time.sleep(0.02)


@with_numpy
def test_other_process_clear_during_read(tmp_path):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    store = str(tmp_path / "cache")
    memory = Memory(location=store, verbose=0)

    @memory.cache
    def make(x):
        return np.arange(16, dtype=np.float64) + x

    make(0)  # populate in the parent too

    consumer_ready = str(tmp_path / "c_ready")
    consumer_go = str(tmp_path / "c_go")
    consumer_done = str(tmp_path / "c_done")

    consumer = _run_child(
        _CONSUMER_CHILD,
        (store, consumer_ready, consumer_go, consumer_done),
        project_root,
    )
    try:
        _wait_for(consumer_ready)

        # The consumer is now holding before it starts timing its read.
        # Trigger maintenance concurrently with the read.
        with open(consumer_go, "w") as f:
            f.write("go")
        # Give the consumer a head start so it is inside get() when clear runs.
        time.sleep(0.05)
        with raises(CacheInUseError):
            memory.clear()
        # The consumer read succeeded and finished.
        stdout, stderr = consumer.communicate(timeout=120)
        assert consumer.returncode == 0, stderr.decode()
        assert stdout.decode().strip().splitlines()[0] == "3.0"
        _wait_for(consumer_done)
        # After the hold is released, clearing works.
        memory.clear()
        assert memory.store_backend.get_items() == []
    finally:
        if consumer.poll() is None:  # pragma: no cover - timeout cleanup
            consumer.kill()
            consumer.wait()


@with_numpy
def test_other_process_reduce_size_during_read(tmp_path):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    store = str(tmp_path / "cache")
    memory = Memory(location=store, verbose=0)

    @memory.cache
    def make(x):
        return np.arange(16, dtype=np.float64) + x

    make(0)
    make(1)  # a second, unheld result that maintenance may reclaim

    consumer_ready = str(tmp_path / "c_ready")
    consumer_go = str(tmp_path / "c_go")
    consumer_done = str(tmp_path / "c_done")

    consumer = _run_child(
        _CONSUMER_CHILD,
        (store, consumer_ready, consumer_go, consumer_done),
        project_root,
    )
    try:
        _wait_for(consumer_ready)
        with open(consumer_go, "w") as f:
            f.write("go")
        time.sleep(0.05)
        # Cannot reach the limit while one result is held.
        assert memory.reduce_size(items_limit=0) is False
        stdout, stderr = consumer.communicate(timeout=120)
        assert consumer.returncode == 0, stderr.decode()
        assert stdout.decode().strip().splitlines()[0] == "3.0"
        _wait_for(consumer_done)
        time.sleep(0.1)
        assert memory.reduce_size(items_limit=0) is True
        assert memory.store_backend.get_items() == []
    finally:
        if consumer.poll() is None:  # pragma: no cover - timeout cleanup
            consumer.kill()
            consumer.wait()


@with_numpy
def test_other_process_maintenance_before_consumer_get(tmp_path):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    store = str(tmp_path / "cache")
    memory = Memory(location=store, verbose=0)

    @memory.cache
    def make(x):
        return np.arange(16, dtype=np.float64) + x

    make(0)
    make(1)

    consumer_ready = str(tmp_path / "c_ready")
    consumer_go = str(tmp_path / "c_go")
    consumer_done = str(tmp_path / "c_done")

    consumer = _run_child(
        _CONSUMER_CHILD,
        (store, consumer_ready, consumer_go, consumer_done),
        project_root,
    )
    try:
        _wait_for(consumer_ready)
        # Maintenance runs *before* the consumer calls get(): the held value
        # must still be there afterwards.
        assert memory.reduce_size(items_limit=0) is False
        with open(consumer_go, "w") as f:
            f.write("go")
        stdout, stderr = consumer.communicate(timeout=120)
        assert consumer.returncode == 0, stderr.decode()
        assert stdout.decode().strip().splitlines()[0] == "3.0"
        _wait_for(consumer_done)
    finally:
        if consumer.poll() is None:  # pragma: no cover - timeout cleanup
            consumer.kill()
            consumer.wait()


@with_numpy
def test_crashed_consumer_lease_is_reclaimed(tmp_path):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    store = str(tmp_path / "cache")
    memory = Memory(location=store, verbose=0)

    @memory.cache
    def make(x):
        return np.arange(16, dtype=np.float64) + x

    make(0)

    consumer_ready = str(tmp_path / "c_ready")
    consumer_go = str(tmp_path / "c_go")
    consumer_done = str(tmp_path / "c_done")

    consumer = _run_child(
        _CONSUMER_CHILD,
        (store, consumer_ready, consumer_go, consumer_done),
        project_root,
    )
    try:
        _wait_for(consumer_ready)
        # While the consumer is alive, the held result survives.
        assert memory.reduce_size(items_limit=0) is False
        consumer.kill()
        consumer.wait()
        time.sleep(0.1)
        # The kernel released the lease when the process died: no permanent
        # occupation, no timeout needed.
        assert memory.reduce_size(items_limit=0) is True
        assert memory.store_backend.get_items() == []
    finally:
        if consumer.poll() is None:  # pragma: no cover
            consumer.kill()
            consumer.wait()


###############################################################################
# Safety and behaviour of the hold handle itself


@with_numpy
def test_hold_get_does_not_return_a_recomputed_value_after_deletion(tmp_path):
    memory, make, calls = _make_counted(tmp_path)
    result = make.call_and_shelve(0)
    hold = result.pin()
    try:
        old_value = hold.get().copy()
        # Simulate a concurrent maintenance that deleted and recomputed the
        # value at the same path (forced recompute keeps the hold alive).
        make.call(0)
        np.testing.assert_array_equal(hold.get(), old_value)
    finally:
        hold.close()

    # After releasing, a fresh pin/read sees the recomputation; the old hold
    # is closed and cannot take it over.
    with raises(ResultReleasedError):
        hold.get()


@with_numpy
def test_hold_get_fails_when_value_unrecoverable(tmp_path):
    memory, make, _ = _make_counted(tmp_path)
    result = make.call_and_shelve(0)
    hold = result.pin()
    try:
        # Delete both the canonical and retired values behind the hold's back
        # (simulates external deletion of files it cannot prevent).
        import shutil

        shutil.rmtree(item_path(result), ignore_errors=True)
        shutil.rmtree(holds_dir(memory), ignore_errors=True)
        with raises((ResultUnavailableError, KeyError)):
            hold.get()
    finally:
        hold.close()


@with_numpy
def test_hold_is_picklable_and_reusable_after_roundtrip(tmp_path):
    memory, make, calls = _make_counted(tmp_path)
    result = make.call_and_shelve(0)
    hold = result.pin()
    try:
        restored = pickle.loads(pickle.dumps(hold))
        np.testing.assert_array_equal(restored.get(), np.arange(16))
        restored.close()
        assert len(calls) == 1
    finally:
        hold.close()


@with_numpy
def test_hold_keeps_cached_function_and_memory_in_sync(tmp_path):
    memory, make, calls = _make_counted(tmp_path)
    result = make.call_and_shelve(0)
    hold = result.pin()
    try:
        # A second, unheld call reuses the cache without recomputing.
        np.testing.assert_array_equal(make(0), np.arange(16))
        assert len(calls) == 1
        # Garbage collecting intermediate arrays does not drop the hold.
        gc.collect()
        np.testing.assert_array_equal(hold.get(), np.arange(16))
        assert len(calls) == 1
    finally:
        hold.close()


###############################################################################
# Generation identity of a shelf
#
# A shelf returned by ``call_and_shelve`` must keep referring to the
# generation of the value it observed or published.  These tests use an
# ``event``-driven interposition rather than sleeps: the replacement happens
# exactly inside the handoff window, at a point chosen by the test.


def _generation_file(result):
    return os.path.join(item_path(result), "generation")


def _drop_generation_markers(location):
    """Turn entries into ones as written by a joblib without markers."""
    for dirpath, _, filenames in os.walk(location):
        if "generation" in filenames:
            os.unlink(os.path.join(dirpath, "generation"))


def _stateful_cache(memory, state):
    calls = []

    @memory.cache
    def make(x):
        calls.append(x)
        return np.array([state["value"], x], dtype=np.int64)

    return make, calls


@with_numpy
def test_shelf_pin_rejects_deleted_then_recomputed_result(tmp_path):
    state = {"value": 11}
    memory = Memory(location=tmp_path, verbose=0)
    make, calls = _stateful_cache(memory, state)

    old = make.call_and_shelve(0)
    np.testing.assert_array_equal(old.get(), [11, 0])
    old.clear()
    state["value"] = 22
    new = make.call_and_shelve(0)
    np.testing.assert_array_equal(new.get(), [22, 0])
    assert len(calls) == 2

    with raises(KeyError):
        old.pin()
    # The new shelf, by contrast, protects the new generation.
    with new.pin() as hold:
        np.testing.assert_array_equal(hold.get(), [22, 0])


@with_numpy
def test_shelf_pin_rejects_same_value_recomputation(tmp_path):
    state = {"value": 11}
    memory = Memory(location=tmp_path, verbose=0)
    make, calls = _stateful_cache(memory, state)

    old = make.call_and_shelve(0)
    old.clear()
    # The recomputation produces byte-for-byte the same array.
    new = make.call_and_shelve(0)
    assert len(calls) == 2
    np.testing.assert_array_equal(old.get(), new.get())

    with raises(KeyError):
        old.pin()
    with new.pin() as hold:
        np.testing.assert_array_equal(hold.get(), [11, 0])


@with_numpy
def test_shelf_pin_rejects_recompute_after_restoring_old_pickle(tmp_path):
    state = {"value": 11}
    memory = Memory(location=tmp_path, verbose=0)
    make, _ = _stateful_cache(memory, state)

    old = make.call_and_shelve(0)
    blob = pickle.dumps(old)
    old.clear()
    state["value"] = 22
    make.call_and_shelve(0)

    restored = pickle.loads(blob)
    with raises(KeyError):
        restored.pin()


@with_numpy
def test_old_shelf_pins_retired_generation_after_replacement(tmp_path):
    state = {"value": 11}
    memory = Memory(location=tmp_path, mmap_mode="r", verbose=0)
    make, calls = _stateful_cache(memory, state)

    old = make.call_and_shelve(0)
    first = old.pin()
    old_file = output_filename(old)
    old_stat = os.stat(old_file)
    try:
        state["value"] = 22
        make.call(0)  # force a recomputation at the same path
        # Both handles read the full, correct old value.
        np.testing.assert_array_equal(first.get(), [11, 0])
        second = old.pin()  # re-pin the same old shelf: retired copy
        np.testing.assert_array_equal(second.get(), [11, 0])
        second.close()
        # The old generation is a renamed file: same inode and size, and the
        # canonical path now holds the new generation (different inode).
        retired = os.path.join(holds_dir(memory), "retired")
        retired_files = [
            os.path.join(dirpath, "output.pkl")
            for dirpath, _, names in os.walk(retired)
            if "output.pkl" in names
        ]
        assert len(retired_files) == 1
        retired_stat = os.stat(retired_files[0])
        assert (retired_stat.st_ino, retired_stat.st_size) == (
            old_stat.st_ino,
            old_stat.st_size,
        )
        new_stat = os.stat(old_file)
        assert new_stat.st_ino != old_stat.st_ino
        assert new_stat.st_size == old_stat.st_size
        # The new generation is readable at the same time, unaffected.
        fresh = make.call_and_shelve(0)
        np.testing.assert_array_equal(fresh.get(), [22, 0])
        assert len(calls) == 2
        # Clearing the old path is refused while the old generation is held.
        with raises(CacheInUseError):
            old.clear()
    finally:
        first.close()
    # Once the last old holder exits, maintenance reclaims the retired copy
    # and the canonical generation remains usable.
    assert memory.reduce_size(items_limit=0) is True
    assert memory.store_backend.get_items() == []
    # The retired copy and the bookkeeping directory are gone.
    retired = os.path.join(holds_dir(memory), "retired")
    assert not os.path.exists(retired)


@with_numpy
def test_shelf_from_cache_hit_refers_to_observed_generation(tmp_path):
    state = {"value": 11}
    memory = Memory(location=tmp_path, verbose=0)
    make, calls = _stateful_cache(memory, state)
    make(0)
    assert len(calls) == 1

    hit = make.call_and_shelve(0)
    assert len(calls) == 1  # a hit, not a recomputation
    state["value"] = 22
    make.call(0)
    assert len(calls) == 2
    # The shelf observed 11; 22 is a different generation at the same path.
    with raises(KeyError):
        hit.pin()


@with_numpy
def test_generation_marker_is_replaced_not_accumulated(tmp_path):
    state = {"value": 11}
    memory = Memory(location=tmp_path, verbose=0)
    make, _ = _stateful_cache(memory, state)
    make.call_and_shelve(0)
    result = make.call_and_shelve(0)
    markers_before = os.listdir(item_path(result))
    for value in range(12, 20):
        state["value"] = value
        make.call(0)
    # A constant-size marker per cache entry: no history of past values.
    item_dir = item_path(result)
    assert os.path.exists(_generation_file(result))
    assert sorted(os.listdir(item_dir)) == sorted(markers_before)
    with make.call_and_shelve(0).pin() as hold:
        np.testing.assert_array_equal(hold.get(), [19, 0])


@with_numpy
def test_handoff_after_publication_replacement_is_rejected(tmp_path):
    """Result written but shelf not returned yet: another writer replaces it."""
    from joblib import memory as memory_module

    state = {"value": 11}
    memory = Memory(location=tmp_path, verbose=0)
    make, _ = _stateful_cache(memory, state)
    original = memory_module.MemorizedFunc._get_memorized_result
    fired = []

    def racing(self, call_id, metadata=None, generation=None, generation_identity=None):
        if not fired:
            fired.append(True)
            state["value"] = 22
            self.call(0)  # replace before returning the shelf
        return original(
            self,
            call_id,
            metadata=metadata,
            generation=generation,
            generation_identity=generation_identity,
        )

    memory_module.MemorizedFunc._get_memorized_result = racing
    try:
        old = make.call_and_shelve(0)
    finally:
        memory_module.MemorizedFunc._get_memorized_result = original
    assert fired
    with raises(KeyError):
        old.pin()


@with_numpy
def test_handoff_after_hit_replacement_is_rejected(tmp_path):
    """Cache hit validated but shelf not returned: the hit is replaced."""
    from joblib import memory as memory_module

    state = {"value": 11}
    memory = Memory(location=tmp_path, verbose=0)
    make, calls = _stateful_cache(memory, state)
    make(0)
    original = memory_module.MemorizedFunc._get_memorized_result
    fired = []

    def racing(self, call_id, metadata=None, generation=None, generation_identity=None):
        if not fired:
            fired.append(True)
            state["value"] = 22
            self.call(0)
        return original(
            self,
            call_id,
            metadata=metadata,
            generation=generation,
            generation_identity=generation_identity,
        )

    memory_module.MemorizedFunc._get_memorized_result = racing
    try:
        hit = make.call_and_shelve(0)
    finally:
        memory_module.MemorizedFunc._get_memorized_result = original
    assert fired
    assert len(calls) == 2
    with raises(KeyError):
        hit.pin()


@with_numpy
def test_handoff_after_hit_replacement_keeps_held_generation(tmp_path):
    """Same window, but the observed hit is held and therefore retired."""
    from joblib import memory as memory_module

    state = {"value": 11}
    memory = Memory(location=tmp_path, verbose=0)
    make, _ = _stateful_cache(memory, state)
    first = make.call_and_shelve(0)
    keep = first.pin()
    original = memory_module.MemorizedFunc._get_memorized_result
    fired = []

    def racing(self, call_id, metadata=None, generation=None, generation_identity=None):
        if not fired:
            fired.append(True)
            state["value"] = 22
            self.call(0)  # retires the observed generation
        return original(
            self,
            call_id,
            metadata=metadata,
            generation=generation,
            generation_identity=generation_identity,
        )

    memory_module.MemorizedFunc._get_memorized_result = racing
    try:
        hit = make.call_and_shelve(0)
    finally:
        memory_module.MemorizedFunc._get_memorized_result = original
    assert fired
    hold = hit.pin()
    try:
        np.testing.assert_array_equal(hold.get(), [11, 0])
    finally:
        hold.close()
        keep.close()


@with_numpy
def test_unmarked_legacy_entry_is_pinned_only_if_unchanged(tmp_path):
    state = {"value": 11}
    memory = Memory(location=tmp_path, verbose=0)
    make, calls = _stateful_cache(memory, state)
    make(0)
    _drop_generation_markers(memory.store_backend.location)

    # Plain reads and hits still work without generation information.
    shelf = make.call_and_shelve(0)
    assert len(calls) == 1
    np.testing.assert_array_equal(shelf.get(), [11, 0])
    # Pinning the unchanged file seals exactly that file.
    with shelf.pin() as hold:
        np.testing.assert_array_equal(hold.get(), [11, 0])

    # Replacing it makes the shelf refuse rather than adopt the new file.
    state["value"] = 22
    make.call(0)
    with raises(KeyError):
        shelf.pin()
    with make.call_and_shelve(0).pin() as hold:
        np.testing.assert_array_equal(hold.get(), [22, 0])


@with_numpy
def test_legacy_shelf_pickle_without_generation_refuses_pin(tmp_path):
    state = {"value": 11}
    memory = Memory(location=tmp_path, verbose=0)
    make, _ = _stateful_cache(memory, state)

    shelf = make.call_and_shelve(0)
    state_dict = shelf.__getstate__()
    state_dict.pop("_generation", None)
    state_dict.pop("_generation_known", None)
    state_dict.pop("_generation_identity", None)
    legacy = MemorizedResult.__new__(MemorizedResult)
    legacy.__setstate__(state_dict)

    # Ordinary reads still work.
    np.testing.assert_array_equal(legacy.get(), [11, 0])
    # But the shelf cannot prove which generation it refers to.
    with raises(KeyError):
        legacy.pin()
    # And the current file is not adopted even though it is the same one.
    np.testing.assert_array_equal(shelf.pin().get(), [11, 0])


@with_numpy
def test_generation_is_stable_across_shelf_pickle(tmp_path):
    state = {"value": 11}
    memory = Memory(location=tmp_path, verbose=0)
    make, _ = _stateful_cache(memory, state)

    shelf = make.call_and_shelve(0)
    restored = pickle.loads(pickle.dumps(shelf))
    assert restored.generation == shelf.generation
    with restored.pin() as hold:
        np.testing.assert_array_equal(hold.get(), [11, 0])
    state["value"] = 22
    make.call(0)
    with raises(KeyError):
        restored.pin()


@with_numpy
@pytest.mark.parametrize(
    "backend,n_jobs",
    [
        ("sequential", 1),
        ("threading", 2),
        pytest.param(
            "loky", 2, marks=pytest.mark.skipif(not HAVE_LOKY, reason="loky missing")
        ),
    ],
)
def test_two_generations_through_parallel_and_parent_closes(tmp_path, backend, n_jobs):
    state = {"value": 11}
    memory = Memory(location=tmp_path, mmap_mode="r", verbose=0)
    make, calls = _stateful_cache(memory, state)

    old = make.call_and_shelve(0)
    hold_old = old.pin()
    state["value"] = 22
    make.call(0)
    hold_new = make.call_and_shelve(0).pin()

    generator = Parallel(
        n_jobs=n_jobs,
        backend=backend,
        return_as="generator",
        pre_dispatch="all",
        batch_size=1,
    )(delayed(_consume)(hold, 0) for _ in range(4) for hold in (hold_old, hold_new))
    if backend != "sequential":
        # Queued tasks own their protection; the parent may let go now.
        hold_old.close()
        hold_new.close()
    outputs = list(generator)
    if backend == "sequential":
        hold_old.close()
        hold_new.close()
    assert outputs == [11.0, 22.0] * 4
    assert len(calls) == 2
    # Nothing was left protected after the run.
    assert lease_files(memory) == []
    assert memory.reduce_size(items_limit=0) is True


@with_numpy
def test_failure_in_one_generation_does_not_corrupt_the_other(tmp_path):
    state = {"value": 11}
    memory = Memory(location=tmp_path, verbose=0)
    make, _ = _stateful_cache(memory, state)
    hold_old = make.call_and_shelve(0).pin()
    state["value"] = 22
    make.call(0)
    hold_new = make.call_and_shelve(0).pin()

    def maybe_fail(hold, fail):
        value = int(hold.get()[0])
        if fail:
            raise RuntimeError("boom")
        return value

    with pytest.raises(RuntimeError):
        Parallel(n_jobs=2, backend="loky", batch_size=1)(
            delayed(maybe_fail)(hold, fail)
            for fail in (False, True, False, True)
            for hold in (hold_old, hold_new)
        )
    try:
        # Both parent handles still read their own generation.
        np.testing.assert_array_equal(hold_old.get(), [11, 0])
        np.testing.assert_array_equal(hold_new.get(), [22, 0])
        # And the cache is usable for a new legal call after the failure.
        assert memory.reduce_size(items_limit=0) is False
    finally:
        hold_old.close()
        hold_new.close()
    assert memory.reduce_size(items_limit=0) is True


@with_numpy
def test_generation_identity_survives_new_process(tmp_path):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    store = str(tmp_path / "cache")
    state = {"value": 11}
    memory = Memory(location=store, verbose=0)
    make, _ = _stateful_cache(memory, state)
    old = make.call_and_shelve(0)
    blob = str(tmp_path / "old.pkl")
    with open(blob, "wb") as f:
        pickle.dump(old, f)
    old.clear()
    state["value"] = 22
    make.call_and_shelve(0)

    child = textwrap.dedent(
        """
        import pickle, sys
        sys.path.insert(0, sys.argv[1])
        with open(sys.argv[2], "rb") as f:
            shelf = pickle.load(f)
        try:
            shelf.pin()
        except KeyError:
            print("rejected")
        else:
            print("adopted")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", child, os.fspath(project_root), blob],
        capture_output=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    assert proc.stdout.decode().strip() == "rejected"


@with_numpy
def test_creating_or_pickling_a_shelf_does_not_create_a_hold(tmp_path):
    state = {"value": 11}
    memory = Memory(location=tmp_path, verbose=0)
    make, _ = _stateful_cache(memory, state)

    shelf = make.call_and_shelve(0)
    restored = pickle.loads(pickle.dumps(shelf))
    assert lease_files(memory) == []
    # An unheld shelf does not prevent maintenance.
    assert memory.reduce_size(items_limit=0) is True
    assert memory.store_backend.get_items() == []
    assert lease_files(memory) == []
    # And a shelf whose value is gone cannot be pinned, while get keeps the
    # usual KeyError behaviour.
    for candidate in (shelf, restored):
        with raises(KeyError):
            candidate.get()
        with raises(KeyError):
            candidate.pin()


@with_numpy
def test_directly_constructed_shelf_does_not_pin_current_file(tmp_path):
    state = {"value": 11}
    memory = Memory(location=tmp_path, verbose=0)
    make, _ = _stateful_cache(memory, state)
    make(0)
    original = make.call_and_shelve(0)
    # Rebuild the same reference the way the documentation allows for
    # copy-pasting repr(), but without generation information.
    rebuilt = MemorizedResult(original.store_backend, original._call_id, verbose=0)
    # It can read the current file...
    assert rebuilt.get()[0] == 11
    # ...but it must not claim that file for a hold.
    with raises(KeyError):
        rebuilt.pin()
    assert lease_files(memory) == []
