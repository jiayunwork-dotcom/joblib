# Result-generation identity for `MemorizedResult.pin()`

Scope: local filesystem store (`backend="local"`), Linux, CPython 3.11,
NumPy `ndarray` results.  No source defect injection: every observed failure
came from the archived code paths and ordinary user calls.

## The bug, before the change

The archived workspace identified a cached value only by the stat identity of
`output.pkl` at its current canonical path.  A shelf (`MemorizedResult`) kept
only `(func_id, args_id)`, so it had no record of *which* value it had
observed.  Two reproductions (`verification/repro_bug.py`) on the unmodified
archive:

```
old.get() [11  0]
REPRO1 UNEXPECTED: old.pin().get() = [22  0]     # deleted, then recomputed
h.get() [11  0]
h.get() after replace [11  0]
REPRO2 old.pin().get() = [22  0]                  # held old, forced new value
```

Both are wrong: the first shelf should report the original value as
unavailable; the second should still yield `11`.

## The fix

Every publication writes a random **generation token** into a tiny
`generation` file inside the item directory.  The file records the token plus
the `(st_dev, st_ino)` of the value it was published with, so it is invalidated
by any out-of-band overwrite.  Publication order (under the store exclusive
lock) is fail-closed: remove the old marker, atomically rename the new
`output.pkl`, then atomically install the marker.  A crash in the middle leaves
an *unmarked* value that `pin` rejects.

`MemorizedResult` records the token captured when its computation published the
value, or when its cache hit was validated (`capture_item` does the existence
check and the marker read under one shared lock).  `pin` resolves that token to
the canonical directory or a retired copy; if the token is gone it raises
`ResultUnavailableError` (a `KeyError` subclass).  Leases now record the token
instead of a stat identity, so maintenance, retirement and cleanup all speak in
"which generation".

A shelf with no token (old pickle, or an entry written by an older joblib)
cannot prove which value it refers to:

* `get`, `clear` and ordinary cache hits keep working;
* `pin` refuses with `KeyError` — but if the shelf *observed* an unmarked entry
  and that exact file is still unchanged (same stat identity), `pin` seals it
  with a new token atomically with the lease, which is safe and keeps upgrades
  working.

### Designs compared

| | stat identity of `output.pkl` (before) | per-item refcount file | random token + inode binding (chosen) |
|---|---|---|---|
| survives shelf pickle / new process | no (not recorded) | only via the counter path | yes |
| detects same-content recomputation | only while the file is still canonical | yes | yes |
| out-of-band overwrite detected | no after the fact | no | yes (inode mismatch) |
| crash safety | publication rename only | leak a permanent ref on crash | fail-closed: unmarked value rejected |
| extra files | none | one per item, rewritten on read | one constant-size file per cache entry |
| extra copies | none | none | none (retire is a rename) |
| extra lock time | none | requires a counter lock | marker read under the existing shared lock; two extra renames per publish |
| cleanup of old generations | n/a | needs liveness scan | reclaimed with retired dirs, bounded by live holders |
| failure mode | silently adopts a newer value | permanently stuck | explicit `KeyError` / `ResultUnavailableError` |

The inode binding is what makes the token robust to a value being replaced by
something other than a joblib publication (for example a manual `cp`) while
still surviving the rename used to retire a held generation.

## Concurrency properties exercised

* Two handoff windows, driven by in-process interposition, never by sleeps:
  * `test_handoff_after_publication_replacement_is_rejected` — value written,
    shelf not yet returned, another writer replaces the same key;
  * `test_handoff_after_hit_replacement_is_rejected` and
    `..._keeps_held_generation` — cache hit validated, hit replaced before the
    shelf is returned (and the retired variant, where the old value is still
    held and therefore survives).
* Old and new generations read, published and maintained concurrently:
  `test_old_shelf_pins_retired_generation_after_replacement`,
  `verification/replacements.py` (48 replacements while the old generation is
  held; at most one retired directory exists at a time).
* `n_jobs=1`, `threading`, `loky` with two generations and the parent closing
  its handles after submission:
  `test_two_generations_through_parallel_and_parent_closes`.
* Failure in one generation's tasks does not corrupt the other, and the cache
  is reusable afterwards: `test_failure_in_one_generation_does_not_corrupt_the_other`.
* New-process pickle round trip: `test_generation_identity_survives_new_process`.
* `CacheInUseError` on explicit clear while a generation is held, and
  invalidation (code change / validation callback) never returned as a hit:
  pre-existing tests plus `test_shelf_pins_retired_generation_after_replacement`.
* Old pickle without generation info refuses to pin while `get` still works:
  `test_legacy_shelf_pickle_without_generation_refuses_pin`.
* Unmarked legacy cache entries: `test_unmarked_legacy_entry_is_pinned_only_if_unchanged`.

## Scale and cost (`verification/bench_generation.py`)

16 MiB and 256 MiB arrays, two live generations, 8 tasks per Parallel mode
(2 workers).  Numbers below are from this run; page-cache state makes the RSS
figures environment-specific.

| metric | 16 MiB | 256 MiB |
|---|---|---|
| cache bytes, one generation | 16 777 714 | 268 435 953 |
| cache overhead over array bytes | 498 | 497 |
| bytes added by a second generation (old held) | -1 (≈0) | 1 (≈0) |
| retired bytes after replacement (rename, not copy) | 16 777 595 | 268 435 834 |
| bookkeeping files, two generations held | 1 | 1 |
| bookkeeping bytes, two generations held | 161 | 161 |
| generation markers, before/after replacement | 1 / 1 | 1 / 1 |
| mapping RSS after `get()` (memmap, no touch) | 0 KiB | 0 KiB |
| mapping RSS after touching 16 pages | 2 500 KiB | 964 KiB |
| benchmark process RSS delta (own memory) | 844 KiB | 16 276 KiB |
| store-lock median / max | 4 µs / 21 µs | 17 µs / 79 µs |
| `pin` median | 36 µs | 217 µs |
| maintenance while held / after release | 0.2 ms / 2.6 ms | 0.4 ms / 13.9 ms |

The benchmark process RSS delta at 256 MiB is the test harness itself (the
`np.full` source array used to produce the value), not the hold: the mapping
figures show only touched pages, and `bytes added by a second generation` is
~0.  Run-to-run variation of a few µs/last-digit bytes is expected; the
qualitative results were identical across runs.

## Commands

```bash
# environment used for these results
python3 -m venv /tmp/venv
/tmp/venv/bin/pip install numpy pytest pytest-asyncio loky threadpoolctl ruff

# targeted suites
PYTHONPATH=/workspace /tmp/venv/bin/python -m pytest \
  joblib/test/test_memory.py joblib/test/test_store_backends.py \
  joblib/test/test_result_holds.py joblib/test/test_parallel.py -q

# whole suite
PYTHONPATH=/workspace /tmp/venv/bin/python -m pytest joblib/test/ -q

# manual probes and scale probe
for f in repro_bug scenarios handoff_windows legacy_cache legacy_repeat_pin replacements; do
  PYTHONPATH=/workspace /tmp/venv/bin/python verification/$f.py
done
PYTHONPATH=/workspace /tmp/venv/bin/python verification/bench_generation.py
```

## Results of this run

* `test_memory.py`, `test_store_backends.py`, `test_result_holds.py`,
  `test_parallel.py`: **758 passed, 15 skipped, 3 xpassed** (skips/xpasses
  pre-existing).
* Whole `joblib/test/` suite: **1550 passed, 23 skipped, 4 xpassed**.
* `doc/memory.rst` doctests: 76 examples, one pre-existing failure at line 254
  (`g.call_and_shelve(4)` prints its call banner under `verbose`); identical on
  the pristine document, so it is unrelated to this change.
* `ruff check` and `ruff format --check` clean on all changed files.

## Not verified here

* Windows / macOS: `flock` and the retirement rename are POSIX-only by design;
  `HoldNotSupportedError` is raised there.  Not exercised.
* Network filesystems (NFS/SMB): retirement requires the store and
  `.joblib_holds` to be on one filesystem; not exercised on a real share.
* Non-local store backends: generation hooks are opt-in, so third-party
  backends cannot be pinned; only the explicit-refusal path is covered.
* Python 3.13 / free-threaded builds, and NumPy < 2.
* Arrays with object dtype, compressed caches (`compress` disables memmap), and
  `mmap_mode` values other than `None`/`"r"` under generation identity.
* Adversarial interleavings beyond the two windows (e.g. process killed
  mid-publication): the fail-closed ordering is reasoned about in the design
  note but only the "unmarked value is rejected" direction was observed.
