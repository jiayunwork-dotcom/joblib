"""Storage providers backends for Memory caching."""

import collections
import datetime
import json
import operator
import os
import os.path
import re
import shutil
import threading
import time
import uuid
import warnings
from abc import ABCMeta, abstractmethod
from pickle import PicklingError

from . import _result_holds, numpy_pickle
from ._result_holds import (
    CacheInUseError,
    HoldNotSupportedError,
)
from .backports import concurrency_safe_rename
from .disk import memstr_to_bytes, mkdirp, rm_subdirs
from .logger import format_time

CacheItemInfo = collections.namedtuple("CacheItemInfo", "path size last_access")


class CacheWarning(Warning):
    """Warning to capture dump failures except for PicklingError."""

    pass


def concurrency_safe_write(object_to_write, filename, write_func):
    """Writes an object into a unique file in a concurrency-safe way."""
    # Temporary name is composed of UUID, process_id and thread_id to avoid
    # collisions due to concurrent write.
    # UUID is unique across nodes and time and help avoid collisions, even if
    # the cache folder is shared by several Python processes with the same pid and
    # thread id on different nodes of a cluster for instance.
    thread_id = id(threading.current_thread())
    temporary_filename = f"{filename}.{uuid.uuid4().hex}-{os.getpid()}-{thread_id}"

    write_func(object_to_write, temporary_filename)

    return temporary_filename


class StoreBackendBase(metaclass=ABCMeta):
    """Helper Abstract Base Class which defines all methods that
    a StorageBackend must implement."""

    location = None

    @abstractmethod
    def _open_item(self, f, mode):
        """Opens an item on the store and return a file-like object.

        This method is private and only used by the StoreBackendMixin object.

        Parameters
        ----------
        f: a file-like object
            The file-like object where an item is stored and retrieved
        mode: string, optional
            the mode in which the file-like object is opened allowed valued are
            'rb', 'wb'

        Returns
        -------
        a file-like object
        """

    @abstractmethod
    def _item_exists(self, location):
        """Checks if an item location exists in the store.

        This method is private and only used by the StoreBackendMixin object.

        Parameters
        ----------
        location: string
            The location of an item. On a filesystem, this corresponds to the
            absolute path, including the filename, of a file.

        Returns
        -------
        True if the item exists, False otherwise
        """

    @abstractmethod
    def _move_item(self, src, dst):
        """Moves an item from src to dst in the store.

        This method is private and only used by the StoreBackendMixin object.

        Parameters
        ----------
        src: string
            The source location of an item
        dst: string
            The destination location of an item
        """

    @abstractmethod
    def create_location(self, location):
        """Creates a location on the store.

        Parameters
        ----------
        location: string
            The location in the store. On a filesystem, this corresponds to a
            directory.
        """

    @abstractmethod
    def clear_location(self, location):
        """Clears a location on the store.

        Parameters
        ----------
        location: string
            The location in the store. On a filesystem, this corresponds to a
            directory or a filename absolute path
        """

    @abstractmethod
    def get_items(self):
        """Returns the whole list of items available in the store.

        Returns
        -------
        The list of items identified by their ids (e.g filename in a
        filesystem).
        """

    @abstractmethod
    def configure(self, location, verbose=0, backend_options=dict()):
        """Configures the store.

        Parameters
        ----------
        location: string
            The base location used by the store. On a filesystem, this
            corresponds to a directory.
        verbose: int
            The level of verbosity of the store
        backend_options: dict
            Contains a dictionary of named parameters used to configure the
            store backend.
        """


class StoreBackendMixin(object):
    """Class providing all logic for managing the store in a generic way.

    The StoreBackend subclass has to implement 3 methods: create_location,
    clear_location and configure. The StoreBackend also has to provide
    a private _open_item, _item_exists and _move_item methods. The _open_item
    method has to have the same signature as the builtin open and return a
    file-like object.
    """

    # Whether this backend can provide the cross-process protection used by
    # ``MemorizedResult.pin``.  Backends that do not set this to True reject
    # pinning explicitly instead of silently dropping the protection.
    supports_holds = False

    # -- holds ---------------------------------------------------------
    def _holds_scope(self, scope_rel=None):
        """Return ``(rel_path, identity)`` pairs held inside ``scope_rel``."""
        if not self.supports_holds or self.location is None:
            return set()
        active = set(_result_holds.protected_items(self.location))
        if scope_rel is None:
            return active
        return {
            entry for entry in active if _result_holds.is_in_scope(entry[0], scope_rel)
        }

    def _assert_not_held(self, scope_rel, description):
        held = self._holds_scope(scope_rel)
        if held:
            raise CacheInUseError(
                f"Cannot clear {description}: {len(held)} cached result(s) in "
                "the deletion scope are actively held (for instance by "
                "MemorizedResult.pin). Release the holds before clearing."
            )

    def holds_in_scope(self, scope_rel=None):
        """Public helper returning the held results inside a sub-path."""
        if not self.supports_holds or self.location is None:
            return set()
        with _result_holds.store_lock(self.location, exclusive=False):
            return self._holds_scope(scope_rel)

    def load_item(self, call_id, verbose=1, timestamp=None, metadata=None):
        """Load an item from the store given its id as a list of str."""
        full_path = os.path.join(self.location, *call_id)
        filename = os.path.join(full_path, "output.pkl")
        return self._load_item_file(
            filename,
            full_path,
            verbose=verbose,
            timestamp=timestamp,
            metadata=metadata,
        )

    def load_item_from_path(
        self, item_path, mmap_mode=None, verbose=1, timestamp=None, metadata=None
    ):
        """Load an item from an explicit directory.

        Used by protected result holds, which may resolve the value to a
        retired directory.  This is a filesystem-oriented default; backends
        that do not support holds never reach this code.
        """
        if not self.supports_holds:
            raise HoldNotSupportedError(
                f"{self.__class__.__name__} does not support protected result loading."
            )
        if mmap_mode is None:
            # Match the loading behaviour of the store: the result's own
            # mmap_mode is used, and the store configuration applies when the
            # result does not specify one.
            mmap_mode = getattr(self, "mmap_mode", None)
        filename = os.path.join(item_path, "output.pkl")
        if not self._item_exists(filename):
            raise KeyError(f"Non-existing item: {filename} does not exist")
        if mmap_mode is None:
            with self._open_item(filename, "rb") as f:
                return numpy_pickle.load(f)
        return numpy_pickle.load(filename, mmap_mode=mmap_mode)

    def _load_item_file(
        self, filename, full_path, verbose=1, timestamp=None, metadata=None
    ):
        if verbose > 1:
            ts_string = (
                "{: <16}".format(format_time(time.time() - timestamp))
                if timestamp is not None
                else ""
            )
            signature = os.path.basename(os.path.dirname(filename))
            if metadata is not None and "input_args" in metadata:
                kwargs = ", ".join(
                    "{}={}".format(*item) for item in metadata["input_args"].items()
                )
                signature += "({})".format(kwargs)
            msg = "[Memory]{}: Loading {}".format(ts_string, signature)
            if verbose < 10:
                print("{0}...".format(msg))
            else:
                print("{0} from {1}".format(msg, full_path))

        mmap_mode = None if not hasattr(self, "mmap_mode") else self.mmap_mode

        if not self._item_exists(filename):
            raise KeyError(
                "Non-existing item (may have been "
                "cleared).\nFile %s does not exist" % filename
            )

        # file-like object cannot be used when mmap_mode is set
        if mmap_mode is None:
            with self._open_item(filename, "rb") as f:
                item = numpy_pickle.load(f)
        else:
            item = numpy_pickle.load(filename, mmap_mode=mmap_mode)
        return item

    def dump_item(self, call_id, item, verbose=1):
        """Dump an item in the store at the id given as a list of str."""
        try:
            item_path = os.path.join(self.location, *call_id)
            if not self._item_exists(item_path):
                self.create_location(item_path)
            filename = os.path.join(item_path, "output.pkl")
            if verbose > 10:
                print("Persisting in %s" % item_path)

            def write_func(to_write, dest_filename):
                with self._open_item(dest_filename, "wb") as f:
                    try:
                        numpy_pickle.dump(to_write, f, compress=self.compress)
                    except PicklingError as e:
                        # TODO(1.5) turn into error
                        warnings.warn(
                            "Unable to cache to disk: failed to pickle "
                            "output. In version 1.5 this will raise an "
                            f"exception. Exception: {e}.",
                            FutureWarning,
                        )

            # The value is first fully written to a unique temporary file and
            # only then atomically renamed into place.  For a held item, the
            # previous value is retired (renamed, never copied) while the
            # exclusive lock is held, so holders never observe a missing
            # value and no holder sees the new value through the old identity.
            temporary_filename = concurrency_safe_write(item, filename, write_func)
            if not self.supports_holds:
                self._move_item(temporary_filename, filename)
                return
            with _result_holds.store_lock(self.location, exclusive=True):
                retired_dir = self._retire_held_locked(item_path, call_id)
                if retired_dir is not None:
                    # The temporary file was moved along with the item
                    # directory: pull it back into the canonical location.
                    temporary_filename = os.path.join(
                        retired_dir, os.path.basename(temporary_filename)
                    )
                    self.create_location(item_path)
                self._move_item(temporary_filename, filename)
        except Exception as e:  # noqa: E722
            warnings.warn(
                "Unable to cache to disk. Possibly a race condition in the "
                f"creation of the directory. Exception: {e}.",
                CacheWarning,
            )

    def _retire_held_locked(self, item_path, call_id):
        """Retire the canonical value if it is held; return the new directory."""
        rel_path = _result_holds.relative_item_path(call_id)
        active = _result_holds.protected_items(self.location)
        identities = {identity for rel, identity in active if rel == rel_path}
        if not identities:
            return None
        current = _result_holds.identity_of_item(item_path)
        if current in identities:
            return _result_holds.retire_item(self.location, rel_path, current)
        return None

    def _invalidate_item_locked(self, call_id):
        """Remove an item; a held item is retired instead of deleted."""
        item_path = os.path.join(self.location, *call_id)
        if not self._item_exists(item_path):
            return
        rel_path = _result_holds.relative_item_path(call_id)
        active = _result_holds.protected_items(self.location)
        current = _result_holds.identity_of_item(item_path)
        if any(rel == rel_path and identity == current for rel, identity in active):
            _result_holds.retire_item(self.location, rel_path, current)
        else:
            self.clear_location(item_path)

    def clear_item(self, call_id):
        """Clear the item at the id, given as a list of str."""
        if self.supports_holds:
            with _result_holds.store_lock(self.location, exclusive=True):
                rel_path = _result_holds.relative_item_path(call_id)
                self._assert_not_held(rel_path, f"item {rel_path!r}")
                self._clear_item_locked(call_id)
        else:
            self._clear_item_locked(call_id)

    def _clear_item_locked(self, call_id):
        item_path = os.path.join(self.location, *call_id)
        if self._item_exists(item_path):
            self.clear_location(item_path)

    def invalidate_item(self, call_id):
        """Invalidate an item without failing on active holds.

        Used by automatic invalidation (cache validation callbacks, function
        code changes): held values are moved aside so their holders can keep
        reading them, while unheld values are simply deleted.
        """
        if not self.supports_holds:
            self.clear_item(call_id)
            return
        with _result_holds.store_lock(self.location, exclusive=True):
            self._invalidate_item_locked(call_id)

    def invalidate_path(self, call_id):
        """Invalidate a whole function path, preserving actively held values."""
        if not self.supports_holds:
            self.clear_path(call_id)
            return
        with _result_holds.store_lock(self.location, exclusive=True):
            func_path = os.path.join(self.location, *call_id)
            rel_path = _result_holds.relative_item_path(call_id)
            if self._item_exists(func_path):
                for item in self.get_items():
                    item_rel = self._item_relpath(item.path)
                    if not _result_holds.is_in_scope(item_rel, rel_path):
                        continue
                    self._invalidate_item_locked(
                        os.path.relpath(item.path, self.location).split(os.sep)
                    )
                # Remove anything left that was not a recognised item
                # directory, e.g. stale metadata files.
                remaining = os.path.join(self.location, rel_path)
                if self._item_exists(remaining):
                    self.clear_location(remaining)
                    self.create_location(remaining)
            _result_holds.cleanup_retired(self.location)

    def contains_item(self, call_id):
        """Check if there is an item at the id, given as a list of str."""
        item_path = os.path.join(self.location, *call_id)
        filename = os.path.join(item_path, "output.pkl")

        return self._item_exists(filename)

    def get_item_info(self, call_id):
        """Return information about item."""
        return {"location": os.path.join(self.location, *call_id)}

    def get_metadata(self, call_id):
        """Return actual metadata of an item."""
        try:
            item_path = os.path.join(self.location, *call_id)
            filename = os.path.join(item_path, "metadata.json")
            with self._open_item(filename, "rb") as f:
                return json.loads(f.read().decode("utf-8"))
        except:  # noqa: E722
            return {}

    def store_metadata(self, call_id, metadata):
        """Store metadata of a computation."""
        try:
            item_path = os.path.join(self.location, *call_id)
            self.create_location(item_path)
            filename = os.path.join(item_path, "metadata.json")

            def write_func(to_write, dest_filename):
                with self._open_item(dest_filename, "wb") as f:
                    f.write(json.dumps(to_write).encode("utf-8"))

            self._concurrency_safe_write(metadata, filename, write_func)
        except:  # noqa: E722
            pass

    def contains_path(self, call_id):
        """Check cached function is available in store."""
        func_path = os.path.join(self.location, *call_id)
        return self.object_exists(func_path)

    def clear_path(self, call_id):
        """Clear all items with a common path in the store."""
        if self.supports_holds:
            with _result_holds.store_lock(self.location, exclusive=True):
                rel_path = _result_holds.relative_item_path(call_id)
                self._assert_not_held(rel_path, f"path {rel_path!r}")
                self._clear_path_locked(call_id)
                _result_holds.cleanup_retired(self.location)
        else:
            self._clear_path_locked(call_id)

    def _clear_path_locked(self, call_id):
        func_path = os.path.join(self.location, *call_id)
        if self._item_exists(func_path):
            self.clear_location(func_path)

    def store_cached_func_code(self, call_id, func_code=None):
        """Store the code of the cached function."""
        func_path = os.path.join(self.location, *call_id)
        if not self._item_exists(func_path):
            self.create_location(func_path)

        if func_code is not None:
            filename = os.path.join(func_path, "func_code.py")
            with self._open_item(filename, "wb") as f:
                f.write(func_code.encode("utf-8"))

    def get_cached_func_code(self, call_id):
        """Store the code of the cached function."""
        filename = os.path.join(self.location, *call_id, "func_code.py")
        try:
            with self._open_item(filename, "rb") as f:
                return f.read().decode("utf-8")
        except:  # noqa: E722
            raise

    def get_cached_func_info(self, call_id):
        """Return information related to the cached function if it exists."""
        return {"location": os.path.join(self.location, *call_id)}

    def clear(self):
        """Clear the whole store content."""
        if self.supports_holds:
            with _result_holds.store_lock(self.location, exclusive=True):
                self._assert_not_held(None, "the whole cache")
                self.clear_location(self.location)
                # cleanup_retired also drops the bookkeeping directory when
                # no lease is left.
                _result_holds.cleanup_retired(self.location)
        else:
            self.clear_location(self.location)

    def enforce_store_limits(self, bytes_limit, items_limit=None, age_limit=None):
        """
        Remove the store's oldest files to enforce item, byte, and age limits.

        Actively held results are never removed.  ``True`` is returned when the
        limits could be enforced and ``False`` when held results prevented it.
        """
        if not self.supports_holds:
            items_to_delete = self._get_items_to_delete(
                bytes_limit, items_limit, age_limit
            )
            for item in items_to_delete:
                if self.verbose > 10:
                    print("Deleting item {0}".format(item))
                try:
                    self.clear_location(item.path)
                except OSError:
                    # Even with ignore_errors=True shutil.rmtree can raise OSError
                    # with:
                    # [Errno 116] Stale file handle if another process has deleted
                    # the folder already.
                    pass
            return True

        with _result_holds.store_lock(self.location, exclusive=True):
            active = _result_holds.protected_items(self.location)
            _result_holds.cleanup_retired(self.location, active)

            held_paths = {rel for rel, _ in active}
            items_to_delete = self._get_items_to_delete(
                bytes_limit,
                items_limit,
                age_limit,
                protected_paths=held_paths,
            )

            for item in items_to_delete:
                if self.verbose > 10:
                    print("Deleting item {0}".format(item))
                try:
                    self.clear_location(item.path)
                except OSError:
                    # Even with ignore_errors=True shutil.rmtree can raise OSError
                    # with:
                    # [Errno 116] Stale file handle if another process has deleted
                    # the folder already.
                    pass

            return self._limits_satisfied(
                bytes_limit, items_limit, age_limit, items_to_delete
            )

    def _item_relpath(self, item_path):
        """Return the store-relative path of a cache item directory."""
        return os.path.relpath(item_path, self.location)

    def _limits_satisfied(self, bytes_limit, items_limit, age_limit, deleted_items):
        """Check whether the store now fits the requested limits."""
        if isinstance(bytes_limit, str):
            bytes_limit = memstr_to_bytes(bytes_limit)
        deleted_paths = {item.path for item in deleted_items}
        remaining = [
            item for item in self.get_items() if item.path not in deleted_paths
        ]
        if bytes_limit is not None and sum(i.size for i in remaining) > bytes_limit:
            return False
        if items_limit is not None and len(remaining) > items_limit:
            return False
        if age_limit is not None:
            deadline = datetime.datetime.now() - age_limit
            if any(item.last_access < deadline for item in remaining):
                return False
        return True

    def _get_items_to_delete(
        self,
        bytes_limit,
        items_limit=None,
        age_limit=None,
        items=None,
        protected_paths=None,
    ):
        """
        Get items to delete to keep the store under size, file, & age limits.

        Items whose store-relative path is listed in ``protected_paths`` are
        never selected, but they still count towards the current size and
        number of items: the limits can therefore only be met by deleting
        other items, and possibly not at all.
        """
        if isinstance(bytes_limit, str):
            bytes_limit = memstr_to_bytes(bytes_limit)

        if items is None:
            items = self.get_items()
        if not items:
            return []

        protected_paths = protected_paths or set()
        size = sum(item.size for item in items)
        nb_items = len(items)

        if bytes_limit is not None:
            to_delete_size = size - bytes_limit
        else:
            to_delete_size = 0

        if items_limit is not None:
            to_delete_items = nb_items - items_limit
        else:
            to_delete_items = 0

        if age_limit is not None:
            older_item = min(item.last_access for item in items)
            if age_limit.total_seconds() < 0:
                raise ValueError("age_limit has to be a positive timedelta")
            deadline = datetime.datetime.now() - age_limit
        else:
            deadline = None

        deletable_items = [
            item
            for item in items
            if self._item_relpath(item.path) not in protected_paths
        ]

        if not deletable_items and (
            to_delete_size > 0
            or to_delete_items > 0
            or (deadline is not None and older_item <= deadline)
        ):
            return []

        if (
            to_delete_size <= 0
            and to_delete_items <= 0
            and (deadline is None or older_item > deadline)
        ):
            return []

        # We want to delete first the cache items that were accessed a
        # long time ago
        deletable_items.sort(key=operator.attrgetter("last_access"))

        items_to_delete = []
        size_so_far = 0
        items_so_far = 0

        for item in deletable_items:
            if (
                (size_so_far >= to_delete_size)
                and items_so_far >= to_delete_items
                and (deadline is None or deadline < item.last_access)
            ):
                break

            items_to_delete.append(item)
            size_so_far += item.size
            items_so_far += 1

        return items_to_delete

    def _concurrency_safe_write(self, to_write, filename, write_func):
        """Writes an object into a file in a concurrency-safe way."""
        temporary_filename = concurrency_safe_write(to_write, filename, write_func)
        self._move_item(temporary_filename, filename)

    def __repr__(self):
        """Printable representation of the store location."""
        return '{class_name}(location="{location}")'.format(
            class_name=self.__class__.__name__, location=self.location
        )


class FileSystemStoreBackend(StoreBackendBase, StoreBackendMixin):
    """A StoreBackend used with local or network file systems."""

    # Local POSIX filesystems can provide cross-process result holds.
    supports_holds = _result_holds.holds_supported()

    _open_item = staticmethod(open)
    _item_exists = staticmethod(os.path.exists)
    _move_item = staticmethod(concurrency_safe_rename)

    def clear_location(self, location):
        """Delete location on store."""
        if location == self.location:
            rm_subdirs(location)
        else:
            shutil.rmtree(location, ignore_errors=True)

    def create_location(self, location):
        """Create object location on store"""
        mkdirp(location)

    def get_items(self):
        """Returns the whole list of items available in the store."""
        items = []

        for dirpath, _, filenames in os.walk(self.location):
            is_cache_hash_dir = re.match("[a-f0-9]{32}", os.path.basename(dirpath))

            if is_cache_hash_dir:
                output_filename = os.path.join(dirpath, "output.pkl")
                try:
                    last_access = os.path.getatime(output_filename)
                except OSError:
                    try:
                        last_access = os.path.getatime(dirpath)
                    except OSError:
                        # The directory has already been deleted
                        continue

                last_access = datetime.datetime.fromtimestamp(last_access)
                try:
                    full_filenames = [os.path.join(dirpath, fn) for fn in filenames]
                    dirsize = sum(os.path.getsize(fn) for fn in full_filenames)
                except OSError:
                    # Either output_filename or one of the files in
                    # dirpath does not exist any more. We assume this
                    # directory is being cleaned by another process already
                    continue

                items.append(CacheItemInfo(dirpath, dirsize, last_access))

        return items

    def configure(self, location, verbose=1, backend_options=None):
        """Configure the store backend.

        For this backend, valid store options are 'compress' and 'mmap_mode'
        """
        if backend_options is None:
            backend_options = {}

        # setup location directory
        self.location = location
        if not os.path.exists(self.location):
            mkdirp(self.location)

        # Automatically add `.gitignore` file to the cache folder.
        # XXX: the condition is necessary because in `Memory.__init__`, the user
        # passed `location` param is modified to be either `{location}` or
        # `{location}/joblib` depending on input type (`pathlib.Path` vs `str`).
        # The proper resolution of this inconsistency is tracked in:
        # https://github.com/joblib/joblib/issues/1684
        cache_directory = (
            os.path.dirname(location)
            if os.path.dirname(location) and os.path.basename(location) == "joblib"
            else location
        )
        with open(os.path.join(cache_directory, ".gitignore"), "w") as file:
            file.write("# Created by joblib automatically.\n")
            file.write("*\n")

        # item can be stored compressed for faster I/O
        self.compress = backend_options.get("compress", False)

        # FileSystemStoreBackend can be used with mmap_mode options under
        # certain conditions.
        mmap_mode = backend_options.get("mmap_mode")
        if self.compress and mmap_mode is not None:
            warnings.warn(
                "Compressed items cannot be memmapped in a "
                "filesystem store. Option will be ignored.",
                stacklevel=2,
            )

        self.mmap_mode = mmap_mode
        self.verbose = verbose
