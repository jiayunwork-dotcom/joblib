"""Manual probe of generation identity scenarios (events, no sleeps)."""
import os, pickle, subprocess, sys, tempfile, threading, textwrap
import numpy as np
from joblib import Memory, Parallel, delayed
from joblib._result_holds import ResultUnavailableError, CacheInUseError, require_holds
from joblib._store_backends import FileSystemStoreBackend

STATE = {"v": 11}

def mk(tmp, **kw):
    mem = Memory(location=tmp, verbose=0, **kw)
    @mem.cache
    def make(x):
        return np.array([STATE["v"], x], dtype=np.int64)
    return mem, make

# 1 delete -> recompute: pin of old shelf must fail
with tempfile.TemporaryDirectory() as tmp:
    mem, make = mk(tmp, mmap_mode="r")
    old = make.call_and_shelve(0)
    old.clear()
    STATE["v"] = 22
    new = make.call_and_shelve(0)
    assert new.get()[0] == 22
    try:
        old.pin()
        raise SystemExit("FAIL 1: old pin succeeded")
    except KeyError:
        print("ok 1 delete-recompute: pin(old) -> KeyError")

# 2 same-value recompute: must still fail (content equality irrelevant)
STATE["v"] = 11
with tempfile.TemporaryDirectory() as tmp:
    mem, make = mk(tmp)
    old = make.call_and_shelve(0)
    old.clear()
    new = make.call_and_shelve(0)   # identical content
    np.testing.assert_array_equal(old.get(), new.get())
    try:
        old.pin().get()
        raise SystemExit("FAIL 2: old pin adopted equal-content new result")
    except KeyError:
        print("ok 2 same-value recompute: pin(old) -> KeyError")

# 3 old still held -> new publish; old shelf re-pin returns old
STATE["v"] = 11
with tempfile.TemporaryDirectory() as tmp:
    mem, make = mk(tmp, mmap_mode="r")
    old = make.call_and_shelve(0)
    h1 = old.pin()
    STATE["v"] = 22
    make.call(0)                 # forced recompute replaces canonical
    np.testing.assert_array_equal(h1.get(), [11, 0])
    h2 = old.pin()               # old shelf pinned again -> retired copy
    np.testing.assert_array_equal(h2.get(), [11, 0])
    h2.close()
    # two generations readable at once
    fresh = make.call_and_shelve(0)
    np.testing.assert_array_equal(fresh.get(), [22, 0])
    with np.testing.assert_raises(CacheInUseError):
        old.clear()
    h1.close()
    print("ok 3 held old + new generation coexist")

# 4 pickle of shelf survives; still identity bound
STATE["v"] = 11
with tempfile.TemporaryDirectory() as tmp:
    mem, make = mk(tmp)
    old = make.call_and_shelve(0)
    blob = pickle.dumps(old)
    old.clear()
    STATE["v"] = 22
    make.call_and_shelve(0)
    restored = pickle.loads(blob)
    try:
        restored.pin().get()
        raise SystemExit("FAIL 4")
    except KeyError:
        print("ok 4 pickled old shelf still fails after delete+recompute")

# 5 old pickle without generation info -> pin refuses, get works
STATE["v"] = 11
with tempfile.TemporaryDirectory() as tmp:
    mem, make = mk(tmp)
    old = make.call_and_shelve(0)
    st = old.__getstate__()
    st.pop("_generation", None); st.pop("_generation_known", None)
    legacy = type(old).__new__(type(old))
    legacy.__setstate__(st)
    np.testing.assert_array_equal(legacy.get(), [11, 0])
    try:
        legacy.pin()
        raise SystemExit("FAIL 5")
    except KeyError:
        print("ok 5 legacy shelf: get works, pin refused")

# 6 three Parallel execution modes with two generations, parent closes
STATE["v"] = 11
def consume(hold, idx):
    return int(hold.get()[0])
for backend, n_jobs in [("sequential", 1), ("threading", 2), ("loky", 2)]:
    with tempfile.TemporaryDirectory() as tmp:
        mem, make = mk(tmp, mmap_mode="r")
        old = make.call_and_shelve(0)
        hold_old = old.pin()
        STATE["v"] = 22
        make.call(0)
        new = make.call_and_shelve(0)
        hold_new = new.pin()
        gen = Parallel(n_jobs=n_jobs, backend=backend, return_as="generator",
                       pre_dispatch="all", batch_size=1)(
            delayed(consume)(h, i) for i in range(6) for h in (hold_old, hold_new))
        if backend != "sequential":
            # In-process execution accepts a task only when it runs it, so
            # the parent handle must stay open until consumption here.
            hold_old.close(); hold_new.close()
        out = list(gen)
        if backend == "sequential":
            hold_old.close(); hold_new.close()
        assert out == [11, 22] * 6, (backend, out)
        print(f"ok 6 {backend}: {out[:4]}...")
        STATE["v"] = 11

# 7 last old holder released -> maintenance reclaims old, new stays usable
STATE["v"] = 11
with tempfile.TemporaryDirectory() as tmp:
    mem, make = mk(tmp, mmap_mode="r")
    old = make.call_and_shelve(0)
    h = old.pin()
    STATE["v"] = 22
    make.call(0)
    old2 = make.call_and_shelve(0)
    np.testing.assert_array_equal(old2.get(), [22, 0])
    h.close()
    assert mem.reduce_size(items_limit=0) is True, "maintenance should reclaim"
    assert mem.store_backend.get_items() == []
    print("ok 7 maintenance reclaims after last old holder")
