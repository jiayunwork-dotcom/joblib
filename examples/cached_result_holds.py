"""
=================================================
Reusing large cached arrays across parallel tasks
=================================================

This example shows the difference between a plain ``MemorizedResult``
reference and an explicit *hold* obtained with ``MemorizedResult.pin()``.

A pipeline step computes a large array once and caches it.  Several
downstream tasks reuse that array.  While the tasks run, a maintenance
thread shrinks the cache: without a hold the cached array may be deleted
under the tasks' feet; with a hold it stays readable without being copied
or recomputed.
"""

import threading
import time

import numpy as np

from joblib import Memory, Parallel, delayed

# %%
# A cached pipeline step
# ----------------------
memory = Memory(location="./cached_pipeline", verbose=0)
computation_count = 0


@memory.cache
def load_features(n_samples):
    """Expensive step: pretend this reads and transforms a big dataset."""
    global computation_count
    computation_count += 1
    return np.linspace(0.0, 1.0, n_samples)


def summarise(hold, start, stop):
    # The task only receives a light handle; the array is read (memory
    # mapped when the result was cached with ``mmap_mode``) from the cache.
    features = hold.get()
    return float(features[start:stop].mean())


# %%
# Without a hold, maintenance can delete the value at any time
# ------------------------------------------------------------
plain_result = load_features.call_and_shelve(1_000_000)
assert computation_count == 1

# Nothing prevents this from deleting the value that ``plain_result`` points
# to, in which case a later ``plain_result.get()`` would raise ``KeyError``.
memory.reduce_size(items_limit=0)

# %%
# With a hold, the value is protected until the last user is done
# ----------------------------------------------------------------
result = load_features.call_and_shelve(1_000_000)
assert computation_count == 2  # the previous value was really deleted

hold = result.pin()

# Hand the same handle to many tasks.  The parent handle can even be closed
# before consuming the results: every accepted task keeps a lease.
generator = Parallel(
    n_jobs=2,
    backend="loky",
    return_as="generator",
    pre_dispatch="all",
    batch_size=1,
)(delayed(summarise)(hold, i * 1000, (i + 1) * 1000) for i in range(16))

hold.close()  # queued and running tasks remain protected


# A concurrent maintenance cannot reclaim the held value.
def shrink_cache():
    for _ in range(3):
        time.sleep(0.01)
        # Returns False while the hold prevents reaching the target.
        memory.reduce_size(items_limit=0)


maintenance = threading.Thread(target=shrink_cache)
maintenance.start()
outputs = list(generator)
maintenance.join()

# The tasks all read the same cached array; the function ran twice overall
# (once per distinct value), never once per task.
assert computation_count == 2
assert len(outputs) == 16

# Once every task is done, maintenance can reclaim the value.
assert memory.reduce_size(items_limit=0) is True

memory.clear()
