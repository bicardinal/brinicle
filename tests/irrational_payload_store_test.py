import json
import os
import shutil
import sqlite3
import tempfile
import traceback


class PayloadStoreIrrationalTests:

    def __init__(self):
        self.test_dir = None
        self.test_count = 0
        self.passed_count = 0

    def setup(self):
        self.test_dir = tempfile.mkdtemp(prefix="payloadstore_irrational_test_")
        print(f"Test directory: {self.test_dir}")

    def teardown(self):
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"Cleaned up: {self.test_dir}")

    def _get_test_path(self, name):
        return os.path.join(
            self.test_dir,
            f"test_payload_store_{name}",
        )

    def _close_store(self, store):
        try:
            store.close()
        except Exception:
            pass

    def _assert_raises(
        self,
        expected_exception,
        operation,
        message=None,
    ):
        try:
            operation()

        except expected_exception as error:
            return error

        except Exception as error:
            expected_name = getattr(
                expected_exception,
                "__name__",
                str(expected_exception),
            )

            raise AssertionError(
                message
                or (
                    f"expected {expected_name}, " f"got {type(error).__name__}: {error}"
                )
            ) from error

        expected_name = getattr(
            expected_exception,
            "__name__",
            str(expected_exception),
        )

        raise AssertionError(
            message or f"expected {expected_name}, but no error was raised"
        )

    def _write_manifest(self, path, manifest):
        os.makedirs(path, exist_ok=True)

        manifest_path = os.path.join(
            path,
            "manifest.json",
        )

        with open(
            manifest_path,
            "w",
            encoding="utf-8",
        ) as manifest_file:
            json.dump(
                manifest,
                manifest_file,
                ensure_ascii=False,
                indent=2,
            )

    def _valid_manifest(self, shard_count=4):
        return {
            "format": "brinicle-payload-store",
            "version": 1,
            "shard_count": shard_count,
            "routing": "blake2b-64",
            "encoding": "utf-8",
        }

    def _find_id_for_shard(
        self,
        store,
        shard_index,
        prefix,
    ):
        candidate_index = 0

        while True:
            candidate = f"{prefix}-{candidate_index}"

            if store._shard_index(candidate) == shard_index:
                return candidate

            candidate_index += 1

    def _run_test(self, test_method):
        self.test_count += 1
        test_name = test_method.__name__

        try:
            print(f"\nRunning: {test_name}")
            test_method()
            self.passed_count += 1
            print(f"[OK] {test_name} PASSED")

        except Exception as error:
            print(f"[NOT OK] {test_name} FAILED: {error}")
            traceback.print_exc()

    def run_all(self):
        self.setup()

        try:
            test_methods = [
                getattr(self, method_name)
                for method_name in dir(self)
                if method_name.startswith("test_")
                and callable(getattr(self, method_name))
            ]

            for test_method in test_methods:
                self._run_test(test_method)

            print(f"\n{'=' * 60}")
            print(
                f"Results: " f"{self.passed_count}/{self.test_count} " f"tests passed"
            )
            print(f"{'=' * 60}")

        finally:
            self.teardown()

    def test_operations_before_init(self):
        store = brinicle.PayloadStore()

        self._assert_raises(
            RuntimeError,
            lambda: store.insert(
                ["item-1"],
                ["payload-1"],
            ),
        )

        self._assert_raises(
            RuntimeError,
            lambda: store.upsert(
                ["item-1"],
                ["payload-1"],
            ),
        )

        self._assert_raises(
            RuntimeError,
            lambda: store.retrieve(["item-1"]),
        )

        self._assert_raises(
            RuntimeError,
            lambda: store.delete(["item-1"]),
        )

        self._assert_raises(
            RuntimeError,
            store.destroy,
        )

        store.close()

        path = self._get_test_path("init_after_early_close")

        store.init(
            path,
            shard_count=2,
        )

        try:
            store.insert(
                ["item-1"],
                ["payload-1"],
            )

            assert store.retrieve(["item-1"]) == [
                "payload-1"
            ], "store could not initialize after an early close"

        finally:
            store.close()

    def test_invalid_init_arguments(self):
        path = self._get_test_path("invalid_arguments")

        store = brinicle.PayloadStore()

        self._assert_raises(
            TypeError,
            lambda: store.init(123),
        )

        self._assert_raises(
            ValueError,
            lambda: store.init(""),
        )

        self._assert_raises(
            ValueError,
            lambda: store.init(
                path,
                shard_count=0,
            ),
        )

        self._assert_raises(
            ValueError,
            lambda: store.init(
                path,
                shard_count=-1,
            ),
        )

        self._assert_raises(
            ValueError,
            lambda: store.init(
                path,
                shard_count=True,
            ),
        )

        self._assert_raises(
            ValueError,
            lambda: store.init(
                path,
                shard_count=1.5,
            ),
        )

        self._assert_raises(
            ValueError,
            lambda: store.init(
                path,
                cache_size_kib=0,
            ),
        )

        self._assert_raises(
            ValueError,
            lambda: store.init(
                path,
                cache_size_kib=True,
            ),
        )

        self._assert_raises(
            ValueError,
            lambda: store.init(
                path,
                cache_size_kib=2.5,
            ),
        )

        self._assert_raises(
            ValueError,
            lambda: store.init(
                path,
                busy_timeout_ms=-1,
            ),
        )

        self._assert_raises(
            ValueError,
            lambda: store.init(
                path,
                busy_timeout_ms=True,
            ),
        )

        self._assert_raises(
            ValueError,
            lambda: store.init(
                path,
                busy_timeout_ms=2.5,
            ),
        )

        assert not os.path.exists(path), "invalid init unexpectedly created a store"

    def test_init_path_is_a_file(self):
        path = self._get_test_path("path_is_file")

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as output_file:
            output_file.write("This is not a payload store.")

        store = brinicle.PayloadStore()

        self._assert_raises(
            RuntimeError,
            lambda: store.init(
                path,
                shard_count=4,
            ),
        )

        assert os.path.isfile(path), "invalid initialization changed the existing file"

    def test_non_empty_directory_without_manifest(self):
        path = self._get_test_path("non_empty_without_manifest")

        os.makedirs(path)

        unrelated_file = os.path.join(
            path,
            "important_user_file.txt",
        )

        with open(
            unrelated_file,
            "w",
            encoding="utf-8",
        ) as output_file:
            output_file.write("Please do not convert me into a database.")

        store = brinicle.PayloadStore()

        self._assert_raises(
            RuntimeError,
            lambda: store.init(
                path,
                shard_count=4,
            ),
        )

        assert os.path.isfile(unrelated_file), "unrelated file was removed or changed"

        with open(
            unrelated_file,
            "r",
            encoding="utf-8",
        ) as input_file:
            assert input_file.read() == (
                "Please do not convert me into a database."
            ), "unrelated file contents were changed"

    def test_symbolic_link_path_is_rejected(self):
        target_path = self._get_test_path("symlink_target")

        link_path = self._get_test_path("symlink_path")

        os.makedirs(target_path)

        try:
            os.symlink(
                target_path,
                link_path,
            )

        except (OSError, NotImplementedError) as error:
            print(f"Symbolic links unavailable; " f"test skipped: {error}")
            return

        store = brinicle.PayloadStore()

        self._assert_raises(
            RuntimeError,
            lambda: store.init(
                link_path,
                shard_count=4,
            ),
        )

        assert os.path.isdir(
            target_path
        ), "symbolic-link target was unexpectedly changed"

    def test_double_init_and_reinit_after_close(self):
        first_path = self._get_test_path("double_init_first")

        second_path = self._get_test_path("double_init_second")

        store = brinicle.PayloadStore().init(
            first_path,
            shard_count=4,
        )

        try:
            self._assert_raises(
                RuntimeError,
                lambda: store.init(
                    second_path,
                    shard_count=4,
                ),
            )

            store.insert(
                ["item-1"],
                ["payload-1"],
            )

        finally:
            store.close()

        self._assert_raises(
            RuntimeError,
            lambda: store.init(
                first_path,
                shard_count=4,
            ),
        )

        assert (
            store.retrieve
            if hasattr(
                store,
                "retrieve",
            )
            else True
        )

        assert not os.path.exists(
            second_path
        ), "failed second initialization created a directory"

    def test_wrong_shard_count_on_reopen(self):
        path = self._get_test_path("wrong_shard_count")

        first_store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            first_store.insert(
                ["item-1"],
                ["payload-1"],
            )

        finally:
            first_store.close()

        wrong_store = brinicle.PayloadStore()

        self._assert_raises(
            ValueError,
            lambda: wrong_store.init(
                path,
                shard_count=8,
            ),
        )

        correct_store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            assert correct_store.retrieve(["item-1"]) == [
                "payload-1"
            ], "wrong reopen attempt damaged the existing store"

        finally:
            correct_store.close()

    def test_ids_must_be_valid_sequences(self):
        path = self._get_test_path("invalid_ids")

        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            self._assert_raises(
                TypeError,
                lambda: store.retrieve("item-1"),
            )

            self._assert_raises(
                TypeError,
                lambda: store.delete(b"item-1"),
            )

            self._assert_raises(
                TypeError,
                lambda: store.insert(
                    "item-1",
                    ["payload-1"],
                ),
            )

            self._assert_raises(
                ValueError,
                lambda: store.retrieve([""]),
            )

            self._assert_raises(
                TypeError,
                lambda: store.retrieve(["item-1", 42]),
            )

            self._assert_raises(
                TypeError,
                lambda: store.delete(["item-1", None]),
            )

            assert (
                store.retrieve([]) == []
            ), "store became unusable after invalid ID inputs"

        finally:
            store.close()

    def test_values_must_be_valid_sequences(self):
        path = self._get_test_path("invalid_values")

        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            self._assert_raises(
                TypeError,
                lambda: store.insert(
                    ["item-1"],
                    "payload-1",
                ),
            )

            self._assert_raises(
                TypeError,
                lambda: store.upsert(
                    ["item-1"],
                    b"payload-1",
                ),
            )

            self._assert_raises(
                ValueError,
                lambda: store.insert(
                    ["item-1", "item-2"],
                    ["payload-1"],
                ),
            )

            self._assert_raises(
                TypeError,
                lambda: store.insert(
                    ["item-1"],
                    [123],
                ),
            )

            self._assert_raises(
                TypeError,
                lambda: store.upsert(
                    ["item-1"],
                    [None],
                ),
            )

            self._assert_raises(
                TypeError,
                lambda: store.insert(
                    ["item-1"],
                    [b"raw-bytes"],
                ),
            )

            store.insert(
                ["valid-item"],
                ["valid-payload"],
            )

            assert store.retrieve(["valid-item"]) == [
                "valid-payload"
            ], "store became unusable after invalid value inputs"

        finally:
            store.close()

    def test_duplicate_ids_in_write_batches(self):
        path = self._get_test_path("duplicate_write_ids")

        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            self._assert_raises(
                ValueError,
                lambda: store.insert(
                    [
                        "item-1",
                        "item-1",
                    ],
                    [
                        "payload-a",
                        "payload-b",
                    ],
                ),
            )

            self._assert_raises(
                ValueError,
                lambda: store.upsert(
                    [
                        "item-2",
                        "item-2",
                    ],
                    [
                        "payload-a",
                        "payload-b",
                    ],
                ),
            )

            assert store.retrieve(
                [
                    "item-1",
                    "item-2",
                ]
            ) == [
                None,
                None,
            ], "duplicate write batch partially modified the store"

        finally:
            store.close()

    def test_insert_existing_id_does_not_replace_value(self):
        path = self._get_test_path("existing_insert")

        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            store.insert(
                ["item-1"],
                ["original-payload"],
            )

            self._assert_raises(
                sqlite3.IntegrityError,
                lambda: store.insert(
                    ["item-1"],
                    ["replacement-payload"],
                ),
            )

            assert store.retrieve(["item-1"]) == [
                "original-payload"
            ], "failed insert replaced the existing payload"

        finally:
            store.close()

    def test_operations_after_close(self):
        path = self._get_test_path("after_close")

        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        store.insert(
            ["item-1"],
            ["payload-1"],
        )

        store.close()
        store.close()

        self._assert_raises(
            RuntimeError,
            lambda: store.insert(
                ["item-2"],
                ["payload-2"],
            ),
        )

        self._assert_raises(
            RuntimeError,
            lambda: store.upsert(
                ["item-1"],
                ["updated-payload"],
            ),
        )

        self._assert_raises(
            RuntimeError,
            lambda: store.retrieve(["item-1"]),
        )

        self._assert_raises(
            RuntimeError,
            lambda: store.delete(["item-1"]),
        )

        store.destroy()

        assert not os.path.exists(path), "destroy after close did not remove the store"

    def test_operations_after_destroy(self):
        path = self._get_test_path("after_destroy")

        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        store.insert(
            ["item-1"],
            ["payload-1"],
        )

        store.destroy()
        store.destroy()

        assert not os.path.exists(path), "destroy did not remove the store"

        self._assert_raises(
            RuntimeError,
            lambda: store.insert(
                ["item-2"],
                ["payload-2"],
            ),
        )

        self._assert_raises(
            RuntimeError,
            lambda: store.upsert(
                ["item-2"],
                ["payload-2"],
            ),
        )

        self._assert_raises(
            RuntimeError,
            lambda: store.retrieve(["item-1"]),
        )

        self._assert_raises(
            RuntimeError,
            lambda: store.delete(["item-1"]),
        )

        self._assert_raises(
            RuntimeError,
            lambda: store.init(
                path,
                shard_count=4,
            ),
        )

    def test_malformed_manifest(self):
        malformed_path = self._get_test_path("malformed_manifest")

        os.makedirs(malformed_path)

        with open(
            os.path.join(
                malformed_path,
                "manifest.json",
            ),
            "w",
            encoding="utf-8",
        ) as manifest_file:
            manifest_file.write('{"format": "brinicle-payload-store",')

        malformed_store = brinicle.PayloadStore()

        self._assert_raises(
            RuntimeError,
            lambda: malformed_store.init(
                malformed_path,
                shard_count=4,
            ),
        )

        list_path = self._get_test_path("manifest_is_list")

        self._write_manifest(
            list_path,
            [
                "This",
                "is",
                "not",
                "an",
                "object",
            ],
        )

        list_store = brinicle.PayloadStore()

        self._assert_raises(
            RuntimeError,
            lambda: list_store.init(
                list_path,
                shard_count=4,
            ),
        )

        assert os.path.isdir(malformed_path), "malformed manifest directory was removed"

        assert os.path.isdir(list_path), "invalid manifest directory was removed"

    def test_invalid_manifest_fields(self):
        cases = [
            (
                "wrong_format",
                {
                    **self._valid_manifest(),
                    "format": "definitely-not-brinicle",
                },
            ),
            (
                "wrong_version",
                {
                    **self._valid_manifest(),
                    "version": 999,
                },
            ),
            (
                "invalid_shard_count",
                {
                    **self._valid_manifest(),
                    "shard_count": 0,
                },
            ),
            (
                "boolean_shard_count",
                {
                    **self._valid_manifest(),
                    "shard_count": True,
                },
            ),
            (
                "wrong_routing",
                {
                    **self._valid_manifest(),
                    "routing": "python-random-hash",
                },
            ),
            (
                "wrong_encoding",
                {
                    **self._valid_manifest(),
                    "encoding": "mysterious-encoding",
                },
            ),
        ]

        for case_name, manifest in cases:
            path = self._get_test_path(f"manifest_{case_name}")

            self._write_manifest(
                path,
                manifest,
            )

            store = brinicle.PayloadStore()

            self._assert_raises(
                RuntimeError,
                lambda store=store, path=path: store.init(
                    path,
                    shard_count=4,
                ),
                message=(f"invalid manifest case " f"{case_name!r} was accepted"),
            )

            assert os.path.isdir(path), (
                f"invalid manifest case {case_name!r} " f"was unexpectedly deleted"
            )

    def test_destroy_refuses_tampered_manifest(self):
        path = self._get_test_path("tampered_destroy")

        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        manifest_path = os.path.join(
            path,
            "manifest.json",
        )

        try:
            store.insert(
                ["item-1"],
                ["payload-1"],
            )

            with open(
                manifest_path,
                "r",
                encoding="utf-8",
            ) as manifest_file:
                original_manifest = manifest_file.read()

            tampered_manifest = self._valid_manifest()
            tampered_manifest["format"] = "unrelated-directory"

            self._write_manifest(
                path,
                tampered_manifest,
            )

            self._assert_raises(
                RuntimeError,
                store.destroy,
            )

            assert os.path.isdir(path), (
                "destroy removed a directory with " "an invalid manifest"
            )

            assert store.retrieve(["item-1"]) == [
                "payload-1"
            ], "failed destroy left the store unusable"

            with open(
                manifest_path,
                "w",
                encoding="utf-8",
            ) as manifest_file:
                manifest_file.write(original_manifest)

            store.destroy()

            assert not os.path.exists(
                path
            ), "destroy failed after restoring the manifest"

        finally:
            self._close_store(store)

    def test_destroy_refuses_missing_shard(self):
        path = self._get_test_path("destroy_missing_shard")

        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        store.insert(
            ["item-1"],
            ["payload-1"],
        )

        store.close()

        missing_shard_path = os.path.join(
            path,
            "shard_00002.sqlite3",
        )

        os.remove(missing_shard_path)

        self._assert_raises(
            RuntimeError,
            store.destroy,
        )

        assert os.path.isdir(path), "destroy removed a store with a missing shard"

    def test_corrupted_payload_bytes(self):
        path = self._get_test_path("corrupted_payload")

        payload_id = "corrupted-item"

        first_store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        shard_index = first_store._shard_index(payload_id)

        try:
            first_store.insert(
                [payload_id],
                ["valid-payload"],
            )

        finally:
            first_store.close()

        shard_path = os.path.join(
            path,
            f"shard_{shard_index:05d}.sqlite3",
        )

        connection = sqlite3.connect(shard_path)

        try:
            connection.execute(
                """
                UPDATE payloads
                SET value = ?
                WHERE id = ?
                """,
                (
                    sqlite3.Binary(b"\xff\xfe\xfa"),
                    payload_id,
                ),
            )

            connection.commit()

        finally:
            connection.close()

        second_store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            self._assert_raises(
                UnicodeDecodeError,
                lambda: second_store.retrieve([payload_id]),
            )

        finally:
            second_store.close()

    def test_corrupted_sqlite_shard(self):
        path = self._get_test_path("corrupted_sqlite")

        first_store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        payload_id = self._find_id_for_shard(
            first_store,
            1,
            "corrupt-database",
        )

        try:
            first_store.insert(
                [payload_id],
                ["payload"],
            )

        finally:
            first_store.close()

        shard_path = os.path.join(
            path,
            "shard_00001.sqlite3",
        )

        with open(
            shard_path,
            "wb",
        ) as shard_file:
            shard_file.write(b"This is no longer an SQLite database.")

        second_store = brinicle.PayloadStore()

        self._assert_raises(
            sqlite3.DatabaseError,
            lambda: second_store.init(
                path,
                shard_count=4,
            ),
        )

        assert os.path.isdir(path), "failed corrupted-shard load removed the store"

    def test_missing_shard_must_not_be_recreated(self):
        path = self._get_test_path("missing_shard_reopen")

        first_store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        payload_id = self._find_id_for_shard(
            first_store,
            2,
            "missing-shard-item",
        )

        try:
            first_store.insert(
                [payload_id],
                ["important-payload"],
            )

        finally:
            first_store.close()

        missing_shard_path = os.path.join(
            path,
            "shard_00002.sqlite3",
        )

        os.remove(missing_shard_path)

        second_store = brinicle.PayloadStore()

        self._assert_raises(
            RuntimeError,
            lambda: second_store.init(
                path,
                shard_count=4,
            ),
            message=("existing store silently recreated " "a missing shard"),
        )

        assert not os.path.exists(
            missing_shard_path
        ), "missing shard was silently recreated"


if __name__ == "__main__":
    import brinicle

    tests = PayloadStoreIrrationalTests()
    tests.run_all()
