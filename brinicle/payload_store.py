from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Final
from typing import Iterator
from typing import Sequence


def _shard_index(payload_id: str, shard_count: int) -> int:
    digest = hashlib.blake2b(
        payload_id.encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(digest, byteorder="big") % shard_count


_STORE_FORMAT: Final = "brinicle-payload-store"
_STORE_VERSION: Final = 1

_MANIFEST_FILENAME: Final = "manifest.json"
_SHARD_FILENAME_TEMPLATE: Final = "shard_{index:05d}.sqlite3"

_DEFAULT_SHARD_COUNT: Final = 2
_DEFAULT_CACHE_SIZE_KIB: Final = 512
_DEFAULT_BUSY_TIMEOUT_MS: Final = 5_000

_SQL_PARAMETER_CHUNK_SIZE: Final = 500


_ROUTING_NAME: Final = "blake2b-64"
_PAYLOAD_ENCODING: Final = "utf-8"


class _PayloadShard:
    """Manages one SQLite-backed payload shard."""

    _path: Path
    _cache_size_kib: int
    _busy_timeout_ms: int

    _connection: sqlite3.Connection
    _lock: threading.RLock
    _closed: bool

    def __init__(
        self,
        path: Path,
        *,
        create: bool,
        cache_size_kib: int = _DEFAULT_CACHE_SIZE_KIB,
        busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        if cache_size_kib <= 0:
            raise ValueError("cache_size_kib must be greater than zero")

        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms cannot be negative")

        self._path = path
        self._cache_size_kib = cache_size_kib
        self._busy_timeout_ms = busy_timeout_ms

        self._lock = threading.RLock()
        self._closed = False

        self._path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if create:
                database = str(self._path)
                use_uri = False
            else:
                database = f"{self._path.as_uri()}?mode=rw"
                use_uri = True

            self._connection = sqlite3.connect(
                database,
                uri=use_uri,
                timeout=self._busy_timeout_ms / 1_000,
                check_same_thread=False,
                isolation_level=None,
            )

            self._configure_connection()
            self._create_schema()

        except Exception:
            connection = getattr(self, "_connection", None)

            if connection is not None:
                connection.close()

            self._closed = True
            raise

    def retrieve_many(
        self,
        ids: Sequence[str],
    ) -> dict[str, bytes]:
        if not ids:
            return {}

        unique_ids = list(dict.fromkeys(ids))

        with self._lock:
            self._ensure_open()

            try:
                self._connection.execute("BEGIN")

                results: dict[str, bytes] = {}

                for start in range(
                    0,
                    len(unique_ids),
                    _SQL_PARAMETER_CHUNK_SIZE,
                ):
                    chunk = unique_ids[start : start + _SQL_PARAMETER_CHUNK_SIZE]

                    placeholders = ", ".join("?" for _ in chunk)

                    cursor = self._connection.execute(
                        f"""
                        SELECT id, value
                        FROM payloads
                        WHERE id IN ({placeholders})
                        """,
                        chunk,
                    )

                    for payload_id, value in cursor:
                        results[payload_id] = value

                self._connection.commit()
                return results

            except Exception:
                if self._connection.in_transaction:
                    self._connection.rollback()

                raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return

            try:
                if self._connection.in_transaction:
                    self._connection.rollback()

            finally:
                try:
                    self._connection.close()

                finally:
                    self._closed = True

    def _configure_connection(self) -> None:
        journal_mode = self._connection.execute("PRAGMA journal_mode = WAL").fetchone()

        if journal_mode is None or journal_mode[0].lower() != "wal":
            raise RuntimeError(
                f"Failed to enable WAL mode for payload shard: {self._path}"
            )

        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._connection.execute(f"PRAGMA cache_size = -{self._cache_size_kib}")
        self._connection.execute("PRAGMA temp_store = FILE")
        self._connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        self._connection.execute("PRAGMA mmap_size = 0")

    def _create_schema(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS payloads (
                id TEXT PRIMARY KEY NOT NULL,
                value BLOB NOT NULL
            ) WITHOUT ROWID
            """
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Payload shard is closed: {self._path}")

    def _acquire(self) -> None:
        self._lock.acquire()

        try:
            self._ensure_open()
        except BaseException:
            self._lock.release()
            raise

    def _release(self) -> None:
        self._lock.release()

    def _begin_write(self) -> None:
        self._ensure_open()

        if self._connection.in_transaction:
            raise RuntimeError(
                f"Payload shard already has an active transaction: " f"{self._path}"
            )

        self._connection.execute("BEGIN IMMEDIATE")

    def _commit(self) -> None:
        self._ensure_open()

        if not self._connection.in_transaction:
            raise RuntimeError(
                f"Payload shard has no active transaction to commit: " f"{self._path}"
            )

        self._connection.commit()

    def _rollback(self) -> None:
        if self._connection.in_transaction:
            self._connection.rollback()

    def _insert_many_uncommitted(
        self,
        records: Sequence[tuple[str, bytes]],
    ) -> None:
        self._ensure_write_transaction()

        self._connection.executemany(
            """
            INSERT INTO payloads (id, value)
            VALUES (?, ?)
            """,
            records,
        )

    def _upsert_many_uncommitted(
        self,
        records: Sequence[tuple[str, bytes]],
    ) -> None:
        self._ensure_write_transaction()

        self._connection.executemany(
            """
            INSERT INTO payloads (id, value)
            VALUES (?, ?)
            ON CONFLICT(id) DO UPDATE SET
                value = excluded.value
            """,
            records,
        )

    def _delete_many_uncommitted(
        self,
        ids: Sequence[str],
    ) -> int:
        self._ensure_write_transaction()

        unique_ids = list(dict.fromkeys(ids))
        deleted_count = 0

        for start in range(
            0,
            len(unique_ids),
            _SQL_PARAMETER_CHUNK_SIZE,
        ):
            chunk = unique_ids[start : start + _SQL_PARAMETER_CHUNK_SIZE]

            placeholders = ", ".join("?" for _ in chunk)

            cursor = self._connection.execute(
                f"""
                DELETE FROM payloads
                WHERE id IN ({placeholders})
                """,
                chunk,
            )

            deleted_count += cursor.rowcount

        return deleted_count

    def _ensure_write_transaction(self) -> None:
        self._ensure_open()

        if not self._connection.in_transaction:
            raise RuntimeError(
                f"Payload shard has no active write transaction: " f"{self._path}"
            )


class PayloadStore:

    _path: Path | None
    _manifest_path: Path | None

    _shard_count: int
    _cache_size_kib: int
    _busy_timeout_ms: int

    _shards: tuple[_PayloadShard, ...]

    _state_condition: threading.Condition
    _active_operations: int

    _initialized: bool
    _closing: bool
    _closed: bool
    _destroyed: bool

    def __init__(self) -> None:
        self._path = None
        self._manifest_path = None

        self._shard_count = 0
        self._cache_size_kib = _DEFAULT_CACHE_SIZE_KIB
        self._busy_timeout_ms = _DEFAULT_BUSY_TIMEOUT_MS

        self._shards = ()

        self._state_condition = threading.Condition(threading.RLock())
        self._active_operations = 0

        self._initialized = False
        self._closing = False
        self._closed = False
        self._destroyed = False

    def init(
        self,
        path: str | Path,
        *,
        shard_count: int = _DEFAULT_SHARD_COUNT,
        cache_size_kib: int = _DEFAULT_CACHE_SIZE_KIB,
        busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
    ) -> PayloadStore:
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or pathlib.Path")

        if isinstance(path, str) and not path.strip():
            raise ValueError("path cannot be empty")

        if (
            not isinstance(shard_count, int)
            or isinstance(shard_count, bool)
            or shard_count <= 0
        ):
            raise ValueError("shard_count must be a positive integer")

        if (
            not isinstance(cache_size_kib, int)
            or isinstance(cache_size_kib, bool)
            or cache_size_kib <= 0
        ):
            raise ValueError("cache_size_kib must be a positive integer")

        if (
            not isinstance(busy_timeout_ms, int)
            or isinstance(busy_timeout_ms, bool)
            or busy_timeout_ms < 0
        ):
            raise ValueError("busy_timeout_ms must be a non-negative integer")

        store_path = Path(path).expanduser().absolute()

        if store_path.exists() and store_path.is_symlink():
            raise RuntimeError(
                f"Payload store path cannot be a symbolic link: " f"{store_path}"
            )

        with self._state_condition:
            if self._destroyed:
                raise RuntimeError("Destroyed payload store cannot be initialized")

            if self._initialized:
                raise RuntimeError("Payload store is already initialized")

            if self._closed:
                raise RuntimeError("Closed payload store cannot be initialized")

            if self._closing:
                raise RuntimeError("Payload store is undergoing a lifecycle operation")

            self._path = store_path
            self._manifest_path = store_path / _MANIFEST_FILENAME

            self._cache_size_kib = cache_size_kib
            self._busy_timeout_ms = busy_timeout_ms

            try:
                is_new_store = self._load_or_create_manifest(
                    requested_shard_count=shard_count,
                )
                self._open_shards(
                    create=is_new_store,
                )

            except BaseException:
                for shard in reversed(self._shards):
                    try:
                        shard.close()
                    except BaseException:
                        pass

                self._shards = ()

                self._path = None
                self._manifest_path = None

                self._shard_count = 0
                self._cache_size_kib = _DEFAULT_CACHE_SIZE_KIB
                self._busy_timeout_ms = _DEFAULT_BUSY_TIMEOUT_MS

                self._initialized = False
                raise

            self._initialized = True

        return self

    def insert(
        self,
        ids: Sequence[str],
        values: Sequence[str],
    ) -> None:
        self._validate_write_inputs(ids, values)

        with self._operation():
            if not ids:
                return

            records_by_shard = self._group_records_by_shard(
                ids,
                values,
            )

            shard_indices = sorted(records_by_shard)
            acquired_shards: list[_PayloadShard] = []

            try:
                for shard_index in shard_indices:
                    shard = self._shards[shard_index]
                    shard._acquire()
                    acquired_shards.append(shard)

                for shard in acquired_shards:
                    shard._begin_write()

                for shard_index in shard_indices:
                    self._shards[shard_index]._insert_many_uncommitted(
                        records_by_shard[shard_index]
                    )

                for shard in acquired_shards:
                    shard._commit()

            except BaseException:
                for shard in reversed(acquired_shards):
                    try:
                        shard._rollback()
                    except BaseException:
                        pass
                raise
            finally:
                for shard in reversed(acquired_shards):
                    shard._release()

    def upsert(
        self,
        ids: Sequence[str],
        values: Sequence[str],
    ) -> None:
        self._validate_write_inputs(ids, values)

        with self._operation():
            if not ids:
                return

            records_by_shard = self._group_records_by_shard(
                ids,
                values,
            )

            shard_indices = sorted(records_by_shard)
            acquired_shards: list[_PayloadShard] = []

            try:
                for shard_index in shard_indices:
                    shard = self._shards[shard_index]
                    shard._acquire()
                    acquired_shards.append(shard)

                for shard in acquired_shards:
                    shard._begin_write()

                for shard_index in shard_indices:
                    self._shards[shard_index]._upsert_many_uncommitted(
                        records_by_shard[shard_index]
                    )

                for shard in acquired_shards:
                    shard._commit()

            except BaseException:
                for shard in reversed(acquired_shards):
                    try:
                        shard._rollback()
                    except BaseException:
                        pass

                raise

            finally:
                for shard in reversed(acquired_shards):
                    shard._release()

    def delete(
        self,
        ids: Sequence[str],
    ) -> int:
        self._validate_ids(ids)

        with self._operation():
            if not ids:
                return 0

            ids_by_shard = self._group_ids_by_shard(ids)
            shard_indices = sorted(ids_by_shard)

            acquired_shards: list[_PayloadShard] = []
            deleted_count = 0

            try:
                for shard_index in shard_indices:
                    shard = self._shards[shard_index]
                    shard._acquire()
                    acquired_shards.append(shard)

                for shard in acquired_shards:
                    shard._begin_write()

                for shard_index in shard_indices:
                    deleted_count += self._shards[shard_index]._delete_many_uncommitted(
                        ids_by_shard[shard_index]
                    )

                for shard in acquired_shards:
                    shard._commit()

                return deleted_count

            except BaseException:
                for shard in reversed(acquired_shards):
                    try:
                        shard._rollback()
                    except BaseException:
                        pass

                raise

            finally:
                for shard in reversed(acquired_shards):
                    shard._release()

    def retrieve(
        self,
        ids: Sequence[str],
    ) -> list[str | None]:
        self._validate_ids(ids)

        with self._operation():
            if not ids:
                return []

            ids_by_shard = self._group_ids_by_shard(ids)
            shard_indices = sorted(ids_by_shard)

            acquired_shards: list[_PayloadShard] = []
            payloads_by_id: dict[str, str] = {}

            try:
                for shard_index in shard_indices:
                    shard = self._shards[shard_index]
                    shard._acquire()
                    acquired_shards.append(shard)

                for shard_index in shard_indices:
                    shard_payloads = self._shards[shard_index].retrieve_many(
                        ids_by_shard[shard_index]
                    )

                    for payload_id, payload in shard_payloads.items():
                        payloads_by_id[payload_id] = payload.decode("utf-8")

            finally:
                for shard in reversed(acquired_shards):
                    shard._release()

            return [payloads_by_id.get(payload_id) for payload_id in ids]

    def close(self) -> None:
        with self._state_condition:
            while self._closing:
                self._state_condition.wait()

            if self._closed or self._destroyed:
                return

            if not self._initialized:
                return

            self._closing = True

            while self._active_operations > 0:
                self._state_condition.wait()

        try:
            self._close_shards()

        finally:
            with self._state_condition:
                self._closed = True
                self._closing = False
                self._state_condition.notify_all()

    def destroy(self) -> None:
        with self._state_condition:
            while self._closing:
                self._state_condition.wait()

            if self._destroyed:
                return

            if not self._initialized:
                raise RuntimeError(
                    "Payload store must be initialized before it can be destroyed"
                )

            if self._path is None:
                raise RuntimeError("Payload store path is unavailable")

            self._closing = True

            while self._active_operations > 0:
                self._state_condition.wait()

            store_path = self._path

        shard_closing_started = False

        try:
            self._validate_store_before_destroy()

            shard_closing_started = True
            self._close_shards()

            shutil.rmtree(store_path)

        except BaseException:
            with self._state_condition:
                if shard_closing_started:
                    self._closed = True

                self._closing = False
                self._state_condition.notify_all()

            raise

        else:
            with self._state_condition:
                self._shards = ()

                self._initialized = False
                self._closed = True
                self._destroyed = True
                self._closing = False

                self._state_condition.notify_all()

    def _close_shards(self) -> None:
        first_error: BaseException | None = None

        for shard in self._shards:
            try:
                shard.close()

            except BaseException as error:
                if first_error is None:
                    first_error = error

        if first_error is not None:
            raise first_error

    def _validate_store_before_destroy(self) -> None:
        if self._path is None or self._manifest_path is None:
            raise RuntimeError("Payload store paths are unavailable")

        if not self._path.exists():
            raise FileNotFoundError(
                f"Payload store directory does not exist: {self._path}"
            )

        if not self._path.is_dir():
            raise RuntimeError(f"Payload store path is not a directory: {self._path}")

        if self._path.is_symlink():
            raise RuntimeError(f"Refusing to destroy a symbolic link: {self._path}")

        expected_manifest_path = self._path / _MANIFEST_FILENAME

        if self._manifest_path != expected_manifest_path:
            raise RuntimeError(
                "Payload store manifest path does not match " "the store directory"
            )

        manifest = self._read_manifest()

        if manifest.get("format") != _STORE_FORMAT:
            raise RuntimeError(
                f"Directory is not a Brinicle payload store: " f"{self._path}"
            )

        if manifest.get("version") != _STORE_VERSION:
            raise RuntimeError(
                "Cannot destroy payload store with an unsupported "
                f"format version: {manifest.get('version')!r}"
            )

        manifest_shard_count = manifest.get("shard_count")

        if (
            not isinstance(manifest_shard_count, int)
            or isinstance(manifest_shard_count, bool)
            or manifest_shard_count <= 0
        ):
            raise RuntimeError("Payload store manifest contains an invalid shard count")

        if manifest_shard_count != self._shard_count:
            raise RuntimeError(
                "Payload store shard count does not match " "the initialized store"
            )

        for shard_index in range(manifest_shard_count):
            shard_path = self._path / _SHARD_FILENAME_TEMPLATE.format(index=shard_index)

            if not shard_path.is_file():
                raise RuntimeError(f"Payload store shard is missing: {shard_path}")

    def _open_shards(
        self,
        *,
        create: bool,
    ) -> None:
        if self._path is None:
            raise RuntimeError("Payload store path is unavailable")

        if self._shard_count <= 0:
            raise RuntimeError("Payload store shard count is invalid")

        if self._shards:
            raise RuntimeError("Payload store shards are already open")

        if not create:
            for shard_index in range(self._shard_count):
                shard_path = self._path / (
                    _SHARD_FILENAME_TEMPLATE.format(index=shard_index)
                )

                if not shard_path.is_file():
                    raise RuntimeError(
                        f"Payload store shard is missing: " f"{shard_path}"
                    )

        opened_shards: list[_PayloadShard] = []

        try:
            for shard_index in range(self._shard_count):
                shard_path = self._path / (
                    _SHARD_FILENAME_TEMPLATE.format(index=shard_index)
                )

                shard = _PayloadShard(
                    shard_path,
                    create=create,
                    cache_size_kib=self._cache_size_kib,
                    busy_timeout_ms=self._busy_timeout_ms,
                )

                opened_shards.append(shard)

        except BaseException:
            for shard in reversed(opened_shards):
                try:
                    shard.close()
                except BaseException:
                    pass

            raise

        self._shards = tuple(opened_shards)

    def _load_or_create_manifest(
        self,
        requested_shard_count: int,
    ) -> bool:
        if self._path is None or self._manifest_path is None:
            raise RuntimeError("Payload store paths are unavailable")

        if (
            not isinstance(requested_shard_count, int)
            or isinstance(requested_shard_count, bool)
            or requested_shard_count <= 0
        ):
            raise ValueError("shard_count must be a positive integer")

        if self._path.exists() and not self._path.is_dir():
            raise RuntimeError(f"Payload store path is not a directory: {self._path}")

        self._path.mkdir(parents=True, exist_ok=True)

        if self._manifest_path.exists():
            if not self._manifest_path.is_file():
                raise RuntimeError(
                    f"Payload store manifest is not a file: " f"{self._manifest_path}"
                )

            manifest = self._read_manifest()

            if manifest.get("format") != _STORE_FORMAT:
                raise RuntimeError(
                    f"Directory is not a Brinicle payload store: " f"{self._path}"
                )

            if manifest.get("version") != _STORE_VERSION:
                raise RuntimeError(
                    "Unsupported payload store format version: "
                    f"{manifest.get('version')!r}"
                )

            stored_shard_count = manifest.get("shard_count")

            if (
                not isinstance(stored_shard_count, int)
                or isinstance(stored_shard_count, bool)
                or stored_shard_count <= 0
            ):
                raise RuntimeError(
                    "Payload store manifest contains an invalid " "shard count"
                )

            if stored_shard_count != requested_shard_count:
                raise ValueError(
                    "Requested shard_count does not match the "
                    "existing payload store: "
                    f"requested={requested_shard_count}, "
                    f"stored={stored_shard_count}"
                )

            if manifest.get("routing") != _ROUTING_NAME:
                raise RuntimeError(
                    "Unsupported payload store routing strategy: "
                    f"{manifest.get('routing')!r}"
                )

            if manifest.get("encoding") != _PAYLOAD_ENCODING:
                raise RuntimeError(
                    "Unsupported payload store encoding: "
                    f"{manifest.get('encoding')!r}"
                )

            self._shard_count = stored_shard_count
            return False

        if any(self._path.iterdir()):
            raise RuntimeError(
                "Cannot create a payload store inside a non-empty "
                f"directory without a manifest: {self._path}"
            )

        self._shard_count = requested_shard_count

        try:
            self._write_manifest()
        except BaseException:
            self._shard_count = 0
            raise
        return True

    def _read_manifest(self) -> dict[str, object]:
        if self._manifest_path is None:
            raise RuntimeError("Payload store manifest path is unavailable")

        try:
            with self._manifest_path.open(
                "r",
                encoding="utf-8",
            ) as manifest_file:
                manifest = json.load(manifest_file)

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Payload store manifest does not exist: " f"{self._manifest_path}"
            ) from None

        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Payload store manifest contains invalid JSON: "
                f"{self._manifest_path}"
            ) from error

        except UnicodeDecodeError as error:
            raise RuntimeError(
                f"Payload store manifest is not valid UTF-8: " f"{self._manifest_path}"
            ) from error

        except OSError as error:
            raise RuntimeError(
                f"Failed to read payload store manifest: " f"{self._manifest_path}"
            ) from error

        if not isinstance(manifest, dict):
            raise RuntimeError("Payload store manifest must contain a JSON object")

        return manifest

    def _write_manifest(self) -> None:
        if self._path is None or self._manifest_path is None:
            raise RuntimeError("Payload store paths are unavailable")

        if self._shard_count <= 0:
            raise RuntimeError("Cannot write manifest with an invalid shard count")

        manifest: dict[str, object] = {
            "format": _STORE_FORMAT,
            "version": _STORE_VERSION,
            "shard_count": self._shard_count,
            "routing": _ROUTING_NAME,
            "encoding": _PAYLOAD_ENCODING,
        }

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path,
                prefix=f".{_MANIFEST_FILENAME}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

                json.dump(
                    manifest,
                    temporary_file,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )

                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(
                temporary_path,
                self._manifest_path,
            )

        except OSError as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

            raise RuntimeError(
                f"Failed to write payload store manifest: " f"{self._manifest_path}"
            ) from error

        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

            raise

    def _shard_index(
        self,
        payload_id: str,
    ) -> int:
        digest = hashlib.blake2b(
            payload_id.encode(_PAYLOAD_ENCODING),
            digest_size=8,
        ).digest()

        hash_value = int.from_bytes(
            digest,
            byteorder="big",
            signed=False,
        )

        return hash_value % self._shard_count

    def _group_ids_by_shard(
        self,
        ids: Sequence[str],
    ) -> dict[int, list[str]]:
        grouped: dict[int, list[str]] = {}

        for payload_id in ids:
            shard_index = self._shard_index(payload_id)

            grouped.setdefault(
                shard_index,
                [],
            ).append(payload_id)

        return grouped

    def _group_records_by_shard(
        self,
        ids: Sequence[str],
        values: Sequence[str],
    ) -> dict[int, list[tuple[str, bytes]]]:
        grouped: dict[int, list[tuple[str, bytes]]] = {}

        for payload_id, value in zip(ids, values):
            shard_index = self._shard_index(payload_id)
            encoded_value = value.encode(_PAYLOAD_ENCODING)

            grouped.setdefault(
                shard_index,
                [],
            ).append((payload_id, encoded_value))

        return grouped

    def _validate_ids(
        self,
        ids: Sequence[str],
    ) -> None:
        if isinstance(ids, (str, bytes)):
            raise TypeError("ids must be a sequence of strings, not a single string")

        for index, payload_id in enumerate(ids):
            if not isinstance(payload_id, str):
                raise TypeError(
                    f"ids[{index}] must be a string, "
                    f"got {type(payload_id).__name__}"
                )

            if not payload_id:
                raise ValueError(f"ids[{index}] cannot be empty")

    def _validate_write_inputs(
        self,
        ids: Sequence[str],
        values: Sequence[str],
    ) -> None:
        self._validate_ids(ids)

        if isinstance(values, (str, bytes)):
            raise TypeError(
                "values must be a sequence of strings, " "not a single string"
            )

        if len(ids) != len(values):
            raise ValueError(
                "ids and values must have the same length: "
                f"ids={len(ids)}, values={len(values)}"
            )

        seen_ids: dict[str, int] = {}

        for index, payload_id in enumerate(ids):
            previous_index = seen_ids.get(payload_id)

            if previous_index is not None:
                raise ValueError(
                    f"Duplicate payload ID {payload_id!r} at "
                    f"indices {previous_index} and {index}"
                )

            seen_ids[payload_id] = index

        for index, value in enumerate(values):
            if not isinstance(value, str):
                raise TypeError(
                    f"values[{index}] must be a string, " f"got {type(value).__name__}"
                )

    @contextmanager
    def _operation(self) -> Iterator[None]:
        with self._state_condition:
            self._ensure_ready()
            self._active_operations += 1

        try:
            yield

        finally:
            with self._state_condition:
                self._active_operations -= 1

                if self._active_operations == 0:
                    self._state_condition.notify_all()

    def _ensure_ready(self) -> None:
        if self._destroyed:
            raise RuntimeError("Payload store has been destroyed")

        if not self._initialized:
            raise RuntimeError("Payload store is not initialized")

        if self._closing:
            raise RuntimeError("Payload store is closing")

        if self._closed:
            raise RuntimeError("Payload store is closed")

        if self._shard_count <= 0 or len(self._shards) != self._shard_count:
            raise RuntimeError("Payload store is not fully initialized")
