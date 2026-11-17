"""Cross-process protection of cached results.

Explicit *holds* keep an individual entry of the joblib on-disk cache alive
while a consumer is using it.  A hold never copies the cached value: it only
adds a small, constant-size amount of bookkeeping plus a file descriptor.

Two independent mechanisms are used:

* Every holder owns a tiny, empty lease file under
  ``<store>.joblib_holds/leases`` and keeps a shared ``flock`` on it for the
  whole duration of the hold.  Maintenance can tell a live lease (``flock``
  held) from the leftover lease of a dead process (``flock`` free) and
  reclaim the latter.
* Mutations of the bookkeeping and cache deletions are serialized with a
  store-wide ``flock`` on the cache directory itself.  Creating a hold
  takes it in shared mode, deleting or retiring entries takes it in
  exclusive mode.  This makes the "check the result still exists, then
  register a hold" sequence atomic with respect to maintenance, and a store
  that is never pinned pays for no extra lock file.

Locks are released by the kernel when a file descriptor is closed, including
when the owning process is killed, so a crash can never leave a permanent
hold.  No timestamp, heartbeat or background daemon is involved: an active
hold is only released when its last holder closes it.

A cached value is identified by the ``(st_dev, st_ino, st_mtime_ns,
st_size)`` tuple of its ``output.pkl`` file.  Cache writes are atomic
renames, therefore recomputing a result at the same path changes its
identity.  When a held entry has to be replaced (function code change,
validation callback, forced ``call``) it is moved -- renamed, not copied --
to ``<store>.joblib_holds/retired`` so old holders keep reading the old
value while the canonical path receives the new result.

The first version supports POSIX local filesystems only.  ``require_holds``
raises :class:`HoldNotSupportedError` when ``flock`` is unavailable so that an
unsupported platform fails loudly instead of silently dropping protection.

Alternative organization considered
-----------------------------------
Two designs were evaluated:

1. **Reference-counted marker files (implemented here).**  Each hold is a
   lease file locked with ``flock``.  *Crash reclamation:* free (no
   ``flock``) files are reclaimed by the next maintenance call, at no cost
   when nothing is held.  *Concurrency:* a store-wide shared/exclusive lock
   makes "check existence then lease" atomic against deletion.
   *File overhead:* one empty file per hold, no file at all for an unpinned
   cache.
2. **Per-item counters inside the cache directory.**  A small JSON/text
   counter next to ``output.pkl`` would avoid a global lock, but increments
   are not atomic across processes without an extra lock; a crash between
   "increment" and "decrement" would leak a permanent reference, requiring
   either timestamps/heartbeats (violates "no timeout revocation") or a
   process-liveness scan.  It also writes inside every held item, and a
   reader must rewrite it to decrement, which is fragile on read-only
   mounts.

The lease design was chosen: crashes are handled by the kernel releasing
``flock`` (no timeout, no daemon), and the bookkeeping grows with the number
of live holds, never with array size.
"""

import hashlib
import json
import os
import shutil
import threading
import uuid

try:  # pragma: no cover - unavailable on non-POSIX platforms
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


HOLDS_SUFFIX = ".joblib_holds"
LEASES_DIRNAME = "leases"
RETIRED_DIRNAME = "retired"
OUTPUT_FILENAME = "output.pkl"


class HoldNotSupportedError(NotImplementedError):
    """Raised when a hold is requested on an unsupported backend/platform."""


class ResultHoldError(RuntimeError):
    """Base class for errors related to a result hold."""


class ResultReleasedError(ResultHoldError):
    """Raised when using a hold after it has been closed."""


class ResultUnavailableError(ResultHoldError, KeyError):
    """Raised when the pinned result no longer exists."""


class CacheInUseError(RuntimeError):
    """Raised when a clear operation would delete an actively held result.

    ``CacheInUseError`` is raised *before* anything in the deletion scope is
    removed, so the cache is left untouched.
    """


def holds_supported():
    """Return whether the running platform can provide cross-process holds."""
    return fcntl is not None and os.name == "posix"


def require_holds(what="result holds"):
    """Raise if holds cannot be provided on this platform."""
    if not holds_supported():
        raise HoldNotSupportedError(
            f"{what} require POSIX file locking (fcntl.flock), which is not "
            "available on this platform."
        )


def holds_root(store_location):
    """Return the bookkeeping directory associated with a store location."""
    return os.path.abspath(store_location).rstrip(os.sep) + HOLDS_SUFFIX


def _leases_dir(store_location):
    return os.path.join(holds_root(store_location), LEASES_DIRNAME)


def _retired_dir(store_location):
    return os.path.join(holds_root(store_location), RETIRED_DIRNAME)


def _mkdirp(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != getattr(os, "EEXIST", 17):
            if not os.path.isdir(path):
                raise


def item_key(rel_path):
    """Stable key for a cache item, based on its store-relative path."""
    return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()


def relative_item_path(call_id):
    """Convert a ``call_id`` tuple (func_id, args_id) to a relative path."""
    if isinstance(call_id, (list, tuple)):
        return os.path.join(*call_id)
    return call_id


def is_in_scope(rel_path, scope_rel):
    """Return whether ``rel_path`` is inside the ``scope_rel`` subtree."""
    if scope_rel in (None, "", "."):
        return True
    scope_rel = scope_rel.rstrip(os.sep)
    return rel_path == scope_rel or rel_path.startswith(scope_rel + os.sep)


def result_identity(output_path):
    """Return the identity of a cached value, or ``None`` if it is missing."""
    try:
        stat = os.stat(output_path)
    except OSError:
        return None
    return (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)


def identity_of_item(item_path):
    return result_identity(os.path.join(item_path, OUTPUT_FILENAME))


class _StoreLock:
    """A ``flock`` on the store directory, held for the block duration.

    Locking the store directory itself (instead of a dedicated lock file)
    means a store that is never pinned does not pay for any extra file.
    """

    def __init__(self, store_location, exclusive):
        self._path = os.path.abspath(store_location)
        self._exclusive = exclusive
        self._fd = None

    def __enter__(self):
        require_holds()
        _mkdirp(self._path)
        directory_flags = getattr(os, "O_DIRECTORY", 0)
        self._fd = os.open(self._path, os.O_RDONLY | directory_flags)
        fcntl.flock(self._fd, fcntl.LOCK_EX if self._exclusive else fcntl.LOCK_SH)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        fd, self._fd = self._fd, None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        return False


def store_lock(store_location, exclusive):
    """Context manager taking the store-wide shared or exclusive lock."""
    return _StoreLock(store_location, exclusive=exclusive)


def _read_lease_metadata(path):
    try:
        with open(path, "rb") as f:
            raw = f.read()
        if not raw:
            return None
        data = json.loads(raw.decode("utf-8"))
        return (data["rel"], tuple(data["identity"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


class Lease:
    """A live lease on one cache entry, materialized as a locked file.

    The lease stays valid as long as this object holds a file descriptor on
    its lease file.  :meth:`close` is idempotent.
    """

    def __init__(self, fd, path, rel_path, identity, unlink_on_close=True):
        self._fd = fd
        self._path = path
        self.rel_path = rel_path
        self.identity = identity
        self._unlink_on_close = unlink_on_close
        self._close_lock = threading.Lock()

    @property
    def path(self):
        return self._path

    @property
    def closed(self):
        return self._fd is None

    def close(self):
        with self._close_lock:
            fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:  # pragma: no cover - defensive
            pass
        finally:
            try:
                os.close(fd)
            except OSError:  # pragma: no cover - defensive
                pass
        if self._unlink_on_close:
            try:
                os.unlink(self._path)
            except OSError:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter shutdown
            pass


def create_lease(store_location, rel_path, identity):
    """Create a new, already locked lease for ``rel_path``/``identity``.

    The caller must hold the store shared lock so that a concurrent
    maintenance cannot miss the newly created lease.
    """
    require_holds()
    _mkdirp(_leases_dir(store_location))
    path = os.path.join(
        _leases_dir(store_location),
        f"{item_key(rel_path)}.{uuid.uuid4().hex}.lease",
    )
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        payload = json.dumps({"rel": rel_path, "identity": list(identity)})
        os.write(fd, payload.encode("utf-8"))
    except BaseException:
        try:
            os.close(fd)
        finally:
            try:
                os.unlink(path)
            except OSError:  # pragma: no cover - defensive
                pass
        raise
    return Lease(fd, path, rel_path, identity)


def _try_open_lease_for_scan(path):
    """Open a lease file for scanning without requiring write access."""
    try:
        return os.open(path, os.O_RDONLY)
    except OSError:
        return None


def scan_leases(store_location, clean_stale=True):
    """Return ``(rel_path, identity)`` pairs for active leases.

    A lease whose file is not locked anymore belongs to a process that
    exited or crashed; it is removed when ``clean_stale`` is true.  Callers
    must hold the store exclusive lock.
    """
    leases_dir = _leases_dir(store_location)
    try:
        names = os.listdir(leases_dir)
    except OSError:
        return []
    active = []
    for name in names:
        path = os.path.join(leases_dir, name)
        fd = _try_open_lease_for_scan(path)
        if fd is None:
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError):
                metadata = _read_lease_metadata(path)
                if metadata is not None:
                    active.append(metadata)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
                if clean_stale:
                    try:
                        os.unlink(path)
                    except OSError:  # pragma: no cover - defensive
                        pass
        finally:
            try:
                os.close(fd)
            except OSError:  # pragma: no cover - defensive
                pass
    return active


def protected_items(store_location):
    """Return the ``(rel_path, identity)`` pairs currently held.

    Must be called while holding the store exclusive lock.
    """
    return set(scan_leases(store_location))


def resolve_item_dir(store_location, rel_path, identity):
    """Locate the on-disk directory holding the value ``identity``.

    The canonical cache path wins when it still contains that exact value.
    Otherwise a retired copy with the same identity is looked up, which is
    what makes a hold survive cache invalidation at the same path.
    """
    canonical = os.path.join(store_location, rel_path)
    if identity_of_item(canonical) == identity:
        return canonical

    retired = _retired_dir(store_location)
    prefix = item_key(rel_path) + "."
    try:
        names = os.listdir(retired)
    except OSError:
        return None
    for name in names:
        if not name.startswith(prefix):
            continue
        candidate = os.path.join(retired, name)
        if identity_of_item(candidate) == identity:
            return candidate
    return None


def retire_item(store_location, rel_path, identity):
    """Move a held cache item out of the canonical path, keeping the value.

    Must be called while holding the store exclusive lock.  Returns the new
    directory, or ``None`` if the item was already gone or had changed.
    """
    item_path = os.path.join(store_location, rel_path)
    if identity_of_item(item_path) != identity:
        return None
    destination = os.path.join(
        _retired_dir(store_location),
        f"{item_key(rel_path)}.{uuid.uuid4().hex}",
    )
    _mkdirp(_retired_dir(store_location))
    try:
        os.rename(item_path, destination)
    except OSError as exc:
        raise HoldNotSupportedError(
            "Cannot retire a held cache entry: the cache store and its hold "
            f"bookkeeping directory must be on the same filesystem ({exc})."
        ) from exc
    return destination


def cleanup_retired(store_location, active_leases=None):
    """Remove retired copies that no live lease refers to.

    Must be called while holding the store exclusive lock.
    """
    if active_leases is None:
        active_leases = protected_items(store_location)
    live_identities = {identity for _, identity in active_leases}
    retired = _retired_dir(store_location)
    removed = 0
    try:
        names = os.listdir(retired)
    except OSError:
        names = []
    for name in names:
        path = os.path.join(retired, name)
        if identity_of_item(path) in live_identities:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    remove_holds_root_if_unused(store_location)
    return removed


def remove_holds_root(store_location):
    """Delete the bookkeeping directory, if it exists and is unused."""
    root = holds_root(store_location)
    if not os.path.exists(root):
        return
    shutil.rmtree(root, ignore_errors=True)


def remove_holds_root_if_unused(store_location):
    """Remove the bookkeeping directory when no lease is left.

    Must be called while holding the store exclusive lock, so that no lease
    can be created concurrently between the check and the deletion.
    """
    leases_dir = _leases_dir(store_location)
    try:
        if os.listdir(leases_dir):
            return
    except OSError:
        pass
    remove_holds_root(store_location)


class ResultHold(object):
    """Handle keeping a cached result alive.

    Instances are returned by :meth:`MemorizedResult.pin`.  They expose
    ``get()``, ``close()`` and the context-manager protocol.  A hold can be
    passed as a task argument to :class:`joblib.Parallel`; doing so transfers
    protection to the task without requiring the parent handle to stay open.
    """

    def __init__(
        self,
        store_backend,
        rel_path,
        identity,
        mmap_mode=None,
        verbose=0,
        timestamp=None,
        metadata=None,
        lease=None,
        task_view=False,
    ):
        self._backend = store_backend
        self._store_location = os.path.abspath(store_backend.location)
        self._rel_path = rel_path
        self._identity = tuple(identity)
        self._mmap_mode = mmap_mode
        self._verbose = verbose
        self._timestamp = timestamp
        self._metadata = metadata
        self._lease = lease
        self._closed = False
        self._task_view = task_view
        self._lock = threading.RLock()

    # -- public API -----------------------------------------------------
    @property
    def closed(self):
        with self._lock:
            return self._closed

    @property
    def item_path(self):
        """Canonical path of the held cache entry."""
        return os.path.join(self._store_location, self._rel_path)

    @property
    def identity(self):
        return self._identity

    def get(self):
        """Load the held value, honouring the result's ``mmap_mode``."""
        with self._lock:
            if self._closed:
                raise ResultReleasedError(
                    "This result hold has been closed; its value is not "
                    "protected anymore."
                )
            with store_lock(self._store_location, exclusive=False):
                item_dir = resolve_item_dir(
                    self._store_location, self._rel_path, self._identity
                )
                if item_dir is None:
                    raise ResultUnavailableError(
                        "The pinned result is no longer available at "
                        f"{self.item_path!r}. It has been deleted or replaced "
                        "by a different computation."
                    )
                loader = getattr(self._backend, "load_item_from_path", None)
                if loader is None:
                    raise HoldNotSupportedError(
                        "This store backend does not support protected result loading."
                    )
                return loader(
                    item_dir,
                    mmap_mode=self._mmap_mode,
                    verbose=self._verbose,
                    timestamp=self._timestamp,
                    metadata=self._metadata,
                )

    def close(self):
        """Release the hold.  Calling ``close`` more than once is a no-op."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            lease, self._lease = self._lease, None
        if lease is not None:
            lease.close()

    # -- context manager ------------------------------------------------
    def __enter__(self):
        with self._lock:
            if self._closed:
                raise ResultReleasedError("Cannot enter a closed result hold.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        # Last-resort cleanup for a handle that was never closed: releasing
        # it at garbage collection is safe, unlike an implicit timeout.
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter shutdown
            pass

    # -- Parallel support ----------------------------------------------
    def _acquire_lease_locked(self):
        return create_lease(self._store_location, self._rel_path, self._identity)

    def _reserve(self):
        """Create an independent lease for a task, in the submitting process."""
        require_holds()
        with store_lock(self._store_location, exclusive=False):
            item_dir = resolve_item_dir(
                self._store_location, self._rel_path, self._identity
            )
            if item_dir is None:
                raise ResultUnavailableError(
                    "Cannot submit a task using a hold whose result is no "
                    f"longer available at {self.item_path!r}."
                )
            return self._acquire_lease_locked()

    def _make_task_view(self):
        """Return the handle that is actually sent to a worker."""
        require_holds()
        if not getattr(self._backend, "supports_holds", False):
            raise HoldNotSupportedError(
                "The store backend of this result does not support holds."
            )
        with self._lock:
            if self._closed:
                raise ResultReleasedError(
                    "Cannot submit a task with a closed result hold. Create a "
                    "new hold with MemorizedResult.pin() before submitting."
                )
        return ResultHold(
            self._backend,
            self._rel_path,
            self._identity,
            mmap_mode=self._mmap_mode,
            verbose=self._verbose,
            timestamp=self._timestamp,
            metadata=self._metadata,
            lease=None,
            task_view=True,
        )

    def _release_lease(self):
        with self._lock:
            lease, self._lease = self._lease, None
        if lease is not None:
            lease.close()

    # -- pickling -------------------------------------------------------
    def __getstate__(self):
        return {
            "backend": self._backend,
            "rel_path": self._rel_path,
            "identity": self._identity,
            "mmap_mode": self._mmap_mode,
            "verbose": self._verbose,
            "timestamp": self._timestamp,
            "metadata": self._metadata,
            "closed": self._task_view and self._closed,
        }

    def __setstate__(self, state):
        self._backend = state["backend"]
        self._store_location = os.path.abspath(self._backend.location)
        self._rel_path = state["rel_path"]
        self._identity = tuple(state["identity"])
        self._mmap_mode = state["mmap_mode"]
        self._verbose = state["verbose"]
        self._timestamp = state["timestamp"]
        self._metadata = state["metadata"]
        self._lease = None
        self._closed = state["closed"]
        self._task_view = True
        self._lock = threading.RLock()

    def __repr__(self):
        state = "closed" if self.closed else "active"
        return (
            f"{self.__class__.__name__}(location={self._store_location!r}, "
            f"item={self._rel_path!r}, {state})"
        )


def replace_holds(obj, _seen=None):
    """Clone ``ResultHold`` objects nested in ``obj`` as task views.

    Returns ``(new_obj, originals)`` where ``originals`` lists the handles
    that were found, in the order the views were created.  Objects that do
    not contain any hold are returned *unchanged* (identical object), so
    task arguments with side effects keep working as before.

    Only lists, tuples, dict keys and dict values are traversed; cycles are
    left alone and left unchanged.  The caller is responsible for reserving
    a lease for each original handle.
    """
    if isinstance(obj, ResultHold):
        return obj._make_task_view(), [obj]
    if not isinstance(obj, (list, tuple, dict)):
        return obj, []
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return obj, []
    _seen.add(obj_id)

    if isinstance(obj, dict):
        new_obj = {}
        originals = []
        for key, value in obj.items():
            new_key, key_holds = replace_holds(key, _seen)
            new_value, value_holds = replace_holds(value, _seen)
            new_obj[new_key] = new_value
            originals.extend(key_holds)
            originals.extend(value_holds)
        if not originals:
            return obj, []
        return new_obj, originals
    new_items = []
    originals = []
    for value in obj:
        new_value, value_holds = replace_holds(value, _seen)
        new_items.append(new_value)
        originals.extend(value_holds)
    if not originals:
        return obj, []
    if isinstance(obj, tuple):
        if hasattr(obj, "_make"):  # namedtuple
            return type(obj)._make(new_items), originals
        return tuple(new_items), originals
    return new_items, originals
