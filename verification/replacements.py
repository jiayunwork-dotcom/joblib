"""Continuous replacement: hold old gen while publishing many new ones."""
import os, tempfile
import numpy as np
from joblib import Memory
from joblib._result_holds import CacheInUseError

STATE = {"v": 11}
with tempfile.TemporaryDirectory() as tmp:
    mem = Memory(location=tmp, mmap_mode="r", verbose=0)
    @mem.cache
    def make(x):
        return np.full(4, STATE["v"], dtype=np.int64)

    old = make.call_and_shelve(0)
    hold = old.pin()
    retired_root = os.path.join(
        os.path.abspath(mem.store_backend.location) + ".joblib_holds", "retired"
    )
    max_retired = 0
    for v in range(12, 60):
        STATE["v"] = v
        make.call(0)
        np.testing.assert_array_equal(hold.get(), [11]*4)
        if os.path.isdir(retired_root):
            max_retired = max(max_retired, len(os.listdir(retired_root)))
            assert len(os.listdir(retired_root)) <= 1, os.listdir(retired_root)
        np.testing.assert_array_equal(make.call_and_shelve(0).get(), [v]*4)
    print("ok 48 replacements with old held; max retired dirs:", max_retired)
    hold.close()
    assert mem.reduce_size(items_limit=0) is True
    assert mem.store_backend.get_items() == []
    print("ok reclamation after last old holder")
