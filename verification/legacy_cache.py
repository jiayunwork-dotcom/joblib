"""Pre-existing (unmarked) cache entries can be hit; pin only when unchanged."""
import glob, os, pickle, tempfile
import numpy as np
from joblib import Memory
from joblib._result_holds import CacheInUseError

STATE = {"v": 11}

with tempfile.TemporaryDirectory() as tmp:
    mem = Memory(location=tmp, verbose=0)
    @mem.cache
    def make(x):
        return np.array([STATE["v"], x], dtype=np.int64)
    make(0)
    # Simulate an entry written by an older joblib: drop the marker.
    markers = glob.glob(os.path.join(tmp, "**", "generation"), recursive=True)
    assert markers, "generation marker missing"
    for m in markers:
        os.unlink(m)
    STATE["v"] = 11
    # Plain call is a hit (function must not run) and must still work.
    calls = []
    @mem.cache
    def counted(x):
        calls.append(x)
        return np.array([99, x])
    # not used; just exercise hit on the unmarked function below
    STATE["v"] = 11
    # Hit path: get() returns cached 11 without recomputation
    out = make(0)
    np.testing.assert_array_equal(out, [11, 0])
    shelf = make.call_and_shelve(0)
    np.testing.assert_array_equal(shelf.get(), [11, 0])
    # pin on the unmarked but unchanged entry seals it and works
    h = shelf.pin()
    np.testing.assert_array_equal(h.get(), [11, 0])
    h.close()
    print("ok unmarked cache: hit/get/pin work while unchanged")

    # Replace it; a shelf that observed the unmarked entry must now refuse.
    STATE["v"] = 22
    make.call(0)
    try:
        shelf.pin().get()
        raise SystemExit("FAIL: unmarked shelf adopted the replacement")
    except KeyError:
        print("ok unmarked cache: shelf refuses after replacement")

    # A fresh shelf sees the new generation and pins it.
    fresh = make.call_and_shelve(0)
    np.testing.assert_array_equal(fresh.pin().get(), [22, 0])
    print("ok fresh shelf pins the new generation")
