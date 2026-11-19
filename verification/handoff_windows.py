"""Deterministic probes for the two handoff windows (events, no sleeps)."""
import os, tempfile
import numpy as np
from joblib import Memory
from joblib import _result_holds, memory as memory_mod
from joblib._result_holds import ResultUnavailableError

STATE = {"v": 11}

def mk(tmp):
    mem = Memory(location=tmp, verbose=0)
    @mem.cache
    def make(x):
        return np.array([STATE["v"], x], dtype=np.int64)
    return mem, make

# Window A: value written, shelf not yet returned; another writer replaces it.
# We interpose on _get_memorized_result so the replacement happens after
# publication but before the shelf is handed back to the caller.
with tempfile.TemporaryDirectory() as tmp:
    mem, make = mk(tmp)
    original = memory_mod.MemorizedFunc._get_memorized_result
    fired = {"once": False}
    def racing(self, call_id, metadata=None, generation=None, generation_identity=None):
        if not fired["once"]:
            fired["once"] = True
            STATE["v"] = 22
            self.call(*[], **{}) if False else None
            # another writer publishes at the same key
            self.call(0)
        return original(self, call_id, metadata=metadata,
                        generation=generation,
                        generation_identity=generation_identity)
    memory_mod.MemorizedFunc._get_memorized_result = racing
    try:
        STATE["v"] = 11
        old = make.call_and_shelve(0)
    finally:
        memory_mod.MemorizedFunc._get_memorized_result = original
    try:
        old.pin().get()
        raise SystemExit("FAIL window A: old shelf adopted the replacement")
    except KeyError:
        print("ok window A: shelf from first publication rejects the replacement")
    STATE["v"] = 11

# Window B: cache hit validated, shelf not yet returned; hit replaced.
with tempfile.TemporaryDirectory() as tmp:
    mem, make = mk(tmp)
    make(0)                     # populate, value 11
    original = memory_mod.MemorizedFunc._get_memorized_result
    def racing_hit(self, call_id, metadata=None, generation=None, generation_identity=None):
        STATE["v"] = 22
        self.call(0)            # replace the hit before returning the shelf
        return original(self, call_id, metadata=metadata,
                        generation=generation,
                        generation_identity=generation_identity)
    memory_mod.MemorizedFunc._get_memorized_result = racing_hit
    try:
        hit = make.call_and_shelve(0)   # must be a hit (no recompute for it)
    finally:
        memory_mod.MemorizedFunc._get_memorized_result = original
    try:
        hit.pin().get()
        raise SystemExit("FAIL window B: hit shelf adopted the replacement")
    except KeyError:
        print("ok window B: shelf from validated hit rejects the replacement")
    STATE["v"] = 11

# Window B variant: hit validated, replaced, but hit value still held so it
# was retired -> pin must still resolve the observed generation.
with tempfile.TemporaryDirectory() as tmp:
    mem, make = mk(tmp)
    first = make.call_and_shelve(0)
    keep = first.pin()          # protects generation 11
    original = memory_mod.MemorizedFunc._get_memorized_result
    def racing_hit(self, call_id, metadata=None, generation=None, generation_identity=None):
        STATE["v"] = 22
        self.call(0)            # 11 is retired, not deleted
        return original(self, call_id, metadata=metadata,
                        generation=generation,
                        generation_identity=generation_identity)
    memory_mod.MemorizedFunc._get_memorized_result = racing_hit
    try:
        hit = make.call_and_shelve(0)
    finally:
        memory_mod.MemorizedFunc._get_memorized_result = original
    np.testing.assert_array_equal(hit.pin().get(), [11, 0])
    np.testing.assert_array_equal(make.call_and_shelve(0).get(), [22, 0])
    keep.close()
    print("ok window B-retired: observed generation still resolvable while held")
    STATE["v"] = 11
