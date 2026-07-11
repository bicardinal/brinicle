import json
import os
import shutil
import tempfile
import traceback


class PayloadStoreSimpleTests:

    def __init__(self):
        self.test_dir = None
        self.test_count = 0
        self.passed_count = 0

    def setup(self):
        self.test_dir = tempfile.mkdtemp(prefix="payloadstore_simple_test_")
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

    def test_empty_store(self):
        path = self._get_test_path("empty")
        store = brinicle.PayloadStore()

        try:
            returned_store = store.init(
                path,
                shard_count=4,
            )

            assert returned_store is store, "init must return the PayloadStore instance"

            assert os.path.isdir(path), "payload store directory was not created"

            assert store.retrieve([]) == [], "empty retrieval must return an empty list"

            assert store.delete([]) == 0, "empty deletion must return zero"

            store.insert([], [])
            store.upsert([], [])

        finally:
            self._close_store(store)

    def test_simple_insert_and_retrieve(self):
        path = self._get_test_path("insert")
        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            ids = [
                "item-1",
                "item-2",
                "item-3",
                "item-4",
            ]

            values = [
                "payload one",
                "payload two",
                "payload three",
                "payload four",
            ]

            store.insert(ids, values)

            result = store.retrieve(ids)

            assert result == values, "retrieved payloads do not match inserted values"

        finally:
            self._close_store(store)

    def test_retrieve_preserves_order_and_duplicates(self):
        path = self._get_test_path("ordering")
        store = brinicle.PayloadStore().init(
            path,
            shard_count=8,
        )

        try:
            store.insert(
                [
                    "item-a",
                    "item-b",
                    "item-c",
                ],
                [
                    "payload-a",
                    "payload-b",
                    "payload-c",
                ],
            )

            result = store.retrieve(
                [
                    "item-c",
                    "missing-item",
                    "item-a",
                    "item-c",
                    "item-b",
                    "missing-item",
                ]
            )

            assert result == [
                "payload-c",
                None,
                "payload-a",
                "payload-c",
                "payload-b",
                None,
            ], (
                "retrieval did not preserve ordering, " "duplicates, or missing values"
            )

        finally:
            self._close_store(store)

    def test_simple_upsert(self):
        path = self._get_test_path("upsert")
        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            store.insert(
                [
                    "item-1",
                    "item-2",
                    "item-3",
                ],
                [
                    "old-value-1",
                    "old-value-2",
                    "old-value-3",
                ],
            )

            store.upsert(
                [
                    "item-2",
                    "item-4",
                ],
                [
                    "new-value-2",
                    "new-value-4",
                ],
            )

            result = store.retrieve(
                [
                    "item-1",
                    "item-2",
                    "item-3",
                    "item-4",
                ]
            )

            assert result == [
                "old-value-1",
                "new-value-2",
                "old-value-3",
                "new-value-4",
            ], "upsert did not correctly update and insert values"

        finally:
            self._close_store(store)

    def test_simple_delete(self):
        path = self._get_test_path("delete")
        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            store.insert(
                [
                    "item-1",
                    "item-2",
                    "item-3",
                    "item-4",
                ],
                [
                    "payload-1",
                    "payload-2",
                    "payload-3",
                    "payload-4",
                ],
            )

            deleted_count = store.delete(
                [
                    "item-2",
                    "item-4",
                    "item-2",
                    "missing-item",
                ]
            )

            assert deleted_count == 2, "delete returned an incorrect deleted count"

            result = store.retrieve(
                [
                    "item-1",
                    "item-2",
                    "item-3",
                    "item-4",
                ]
            )

            assert result == [
                "payload-1",
                None,
                "payload-3",
                None,
            ], "deleted payloads still appear in retrieval"

            second_deleted_count = store.delete(
                [
                    "item-2",
                    "item-4",
                ]
            )

            assert second_deleted_count == 0, "repeated deletion must return zero"

        finally:
            self._close_store(store)

    def test_persistence_after_reopen(self):
        path = self._get_test_path("persistence")

        ids = [
            "persistent-1",
            "persistent-2",
            "persistent-3",
        ]

        values = [
            "stored value one",
            "stored value two",
            "stored value three",
        ]

        first_store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            first_store.insert(ids, values)

        finally:
            first_store.close()

        second_store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            result = second_store.retrieve(ids)

            assert result == values, "payloads were not preserved after reopening"

            second_store.upsert(
                ["persistent-2"],
                ["updated after reopen"],
            )

        finally:
            second_store.close()

        third_store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            result = third_store.retrieve(ids)

            assert result == [
                "stored value one",
                "updated after reopen",
                "stored value three",
            ], "upsert was not preserved after another reopen"

        finally:
            third_store.close()

    def test_unicode_and_empty_payloads(self):
        path = self._get_test_path("unicode")
        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            ids = [
                "english-id",
                "شناسه-فارسی",
                "日本語-id",
                "emoji-🌨️",
                "empty-value",
            ]

            values = [
                "ordinary payload",
                "یک مقدار فارسی برای برینیکل",
                "日本語のペイロード",
                "snow engine 🌨️",
                "",
            ]

            store.insert(ids, values)

            result = store.retrieve(ids)

            assert result == values, "Unicode or empty payloads were not preserved"

        finally:
            self._close_store(store)

    def test_serialized_json_payloads(self):
        path = self._get_test_path("json")
        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            documents = [
                {
                    "name": "Brinicle",
                    "type": "vector engine",
                    "version": 1,
                },
                {
                    "name": "محصول",
                    "price": 125000,
                    "available": True,
                },
                {
                    "items": [1, 2, 3],
                    "metadata": {
                        "source": "test",
                        "nullable": None,
                    },
                },
            ]

            ids = [
                "document-1",
                "document-2",
                "document-3",
            ]

            serialized_documents = [
                json.dumps(
                    document,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                for document in documents
            ]

            store.insert(
                ids,
                serialized_documents,
            )

            retrieved = store.retrieve(ids)

            assert (
                retrieved == serialized_documents
            ), "serialized JSON strings were changed"

            deserialized = [json.loads(payload) for payload in retrieved]

            assert (
                deserialized == documents
            ), "retrieved JSON payloads could not be restored"

        finally:
            self._close_store(store)

    def test_large_single_shard_batch(self):
        path = self._get_test_path("large_batch")
        store = brinicle.PayloadStore().init(
            path,
            shard_count=1,
        )

        try:
            count = 1500

            ids = [f"large-item-{index}" for index in range(count)]

            values = [f"large-payload-{index}" for index in range(count)]

            store.insert(ids, values)

            reversed_ids = list(reversed(ids))
            reversed_values = list(reversed(values))

            retrieved = store.retrieve(reversed_ids)

            assert (
                retrieved == reversed_values
            ), "large batch retrieval returned invalid values"

            ids_to_delete = ids[:1100]

            deleted_count = store.delete(ids_to_delete)

            assert deleted_count == len(
                ids_to_delete
            ), "large chunked deletion returned invalid count"

            remaining_result = store.retrieve(
                [
                    ids[0],
                    ids[1099],
                    ids[1100],
                    ids[-1],
                ]
            )

            assert remaining_result == [
                None,
                None,
                values[1100],
                values[-1],
            ], "large batch deletion produced invalid state"

        finally:
            self._close_store(store)

    def test_multiple_shards_are_transparent(self):
        path = self._get_test_path("multiple_shards")
        store = brinicle.PayloadStore().init(
            path,
            shard_count=16,
        )

        try:
            count = 500

            ids = [f"sharded-item-{index}" for index in range(count)]

            values = [f"sharded-payload-{index}" for index in range(count)]

            store.insert(ids, values)

            requested_ids = ids[::3] + ids[1::3] + ids[2::3]

            expected_values_by_id = dict(zip(ids, values))

            expected = [
                expected_values_by_id[payload_id] for payload_id in requested_ids
            ]

            result = store.retrieve(requested_ids)

            assert result == expected, "sharding changed the public retrieval behavior"

            store.upsert(
                ids[::10],
                [f"updated-{index}" for index in range(len(ids[::10]))],
            )

            updated_result = store.retrieve(ids[::10])

            assert updated_result == [
                f"updated-{index}" for index in range(len(ids[::10]))
            ], "multi-shard upsert returned invalid values"

        finally:
            self._close_store(store)


    def test_empty_new_capabilities(self):
        path = self._get_test_path("empty_new_capabilities")
        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            store.insert_bytes([], [])
            store.upsert_bytes([], [])

            assert (
                store.retrieve_bytes([]) == []
            ), "empty binary retrieval must return an empty list"

            assert (
                store.exists([]) == []
            ), "empty existence check must return an empty list"

            scanned_ids, cursor = store.scan(limit=10)

            assert scanned_ids == [], "empty scan must return an empty list"
            assert cursor is None, "empty scan must not return a cursor"

        finally:
            self._close_store(store)

    def test_insert_and_retrieve_bytes(self):
        path = self._get_test_path("insert_bytes")
        store = brinicle.PayloadStore().init(
            path,
            shard_count=8,
        )

        try:
            ids = [
                "binary-empty",
                "binary-null",
                "binary-invalid-utf8",
                "binary-document",
            ]

            values = [
                b"",
                b"\x00\x01\x02\x00",
                b"\xff\xfe\x80\x81",
                bytes(range(256)),
            ]

            store.insert_bytes(ids, values)

            result = store.retrieve_bytes(
                [
                    "binary-document",
                    "missing-binary",
                    "binary-empty",
                    "binary-invalid-utf8",
                    "binary-document",
                    "binary-null",
                ]
            )

            assert result == [
                bytes(range(256)),
                None,
                b"",
                b"\xff\xfe\x80\x81",
                bytes(range(256)),
                b"\x00\x01\x02\x00",
            ], (
                "binary retrieval did not preserve bytes, "
                "ordering, duplicates, or missing values"
            )

        finally:
            self._close_store(store)

    def test_upsert_bytes(self):
        path = self._get_test_path("upsert_bytes")
        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            store.insert_bytes(
                [
                    "binary-1",
                    "binary-2",
                ],
                [
                    b"old-value-1",
                    b"old-value-2",
                ],
            )

            store.upsert_bytes(
                [
                    "binary-2",
                    "binary-3",
                ],
                [
                    b"new-value-2",
                    b"\x00new-value-3\xff",
                ],
            )

            result = store.retrieve_bytes(
                [
                    "binary-1",
                    "binary-2",
                    "binary-3",
                ]
            )

            assert result == [
                b"old-value-1",
                b"new-value-2",
                b"\x00new-value-3\xff",
            ], "binary upsert did not correctly update and insert values"

        finally:
            self._close_store(store)

    def test_bytes_persistence_after_reopen(self):
        path = self._get_test_path("bytes_persistence")

        ids = [
            "persistent-binary-1",
            "persistent-binary-2",
        ]

        values = [
            b"\x00\x10\x20",
            b"\xffstored\x00payload",
        ]

        first_store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            first_store.insert_bytes(ids, values)

        finally:
            self._close_store(first_store)

        second_store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            assert (
                second_store.retrieve_bytes(ids) == values
            ), "binary payloads were not preserved after reopening"

            second_store.upsert_bytes(
                ["persistent-binary-2"],
                [b"updated-after-reopen"],
            )

        finally:
            self._close_store(second_store)

        third_store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            assert third_store.retrieve_bytes(ids) == [
                b"\x00\x10\x20",
                b"updated-after-reopen",
            ], "binary upsert was not preserved after another reopen"

        finally:
            self._close_store(third_store)

    def test_exists_preserves_order_duplicates_and_missing(self):
        path = self._get_test_path("exists")
        store = brinicle.PayloadStore().init(
            path,
            shard_count=8,
        )

        try:
            store.insert(
                [
                    "text-item-1",
                    "text-item-2",
                ],
                [
                    "value-1",
                    "value-2",
                ],
            )

            store.insert_bytes(
                ["binary-item-1"],
                [b"\x00\xff"],
            )

            result = store.exists(
                [
                    "binary-item-1",
                    "missing-item",
                    "text-item-2",
                    "binary-item-1",
                    "text-item-1",
                    "missing-item",
                ]
            )

            assert result == [
                True,
                False,
                True,
                True,
                True,
                False,
            ], (
                "exists did not preserve ordering, duplicates, "
                "or missing-value behavior"
            )

            store.delete(["text-item-1"])

            assert store.exists(
                ["text-item-1", "text-item-2"]
            ) == [
                False,
                True,
            ], "exists returned stale results after deletion"

        finally:
            self._close_store(store)

    def test_scan_sorted_prefix_and_pagination(self):
        path = self._get_test_path("scan")
        store = brinicle.PayloadStore().init(
            path,
            shard_count=16,
        )

        try:
            user_ids = [
                f"user:{index:03d}"
                for index in range(17)
            ]

            # Insert in reverse order so scan ordering cannot accidentally
            # match insertion order.
            reversed_user_ids = list(reversed(user_ids))

            store.insert(
                reversed_user_ids,
                [
                    f"value-{payload_id}"
                    for payload_id in reversed_user_ids
                ],
            )

            store.insert(
                [
                    "book:001",
                    "book:002",
                    "question:001",
                ],
                [
                    "book-one",
                    "book-two",
                    "question-one",
                ],
            )

            collected: list[str] = []
            cursor = None

            while True:
                page, cursor = store.scan(
                    prefix="user:",
                    cursor=cursor,
                    limit=4,
                )

                collected.extend(page)

                if cursor is None:
                    break

            assert collected == user_ids, (
                "scan did not return all matching IDs in global "
                "lexicographic order"
            )

            all_ids, all_cursor = store.scan(limit=100)

            assert all_ids == sorted(
                user_ids
                + [
                    "book:001",
                    "book:002",
                    "question:001",
                ]
            ), "unfiltered scan did not return globally sorted IDs"

            assert all_cursor is None, (
                "scan returned a cursor when all results fit in one page"
            )

        finally:
            self._close_store(store)

    def test_scan_treats_prefix_characters_literally(self):
        path = self._get_test_path("scan_literal_prefix")
        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            store.insert(
                [
                    "user%:001",
                    "user_:001",
                    "userA:001",
                    "user%:002",
                ],
                [
                    "percent-one",
                    "underscore",
                    "letter",
                    "percent-two",
                ],
            )

            percent_ids, percent_cursor = store.scan(
                prefix="user%:",
                limit=10,
            )

            underscore_ids, underscore_cursor = store.scan(
                prefix="user_:",
                limit=10,
            )

            assert percent_ids == [
                "user%:001",
                "user%:002",
            ], "scan treated '%' as a wildcard"

            assert underscore_ids == [
                "user_:001",
            ], "scan treated '_' as a wildcard"

            assert percent_cursor is None
            assert underscore_cursor is None

        finally:
            self._close_store(store)

    def test_backup_copies_text_and_binary_and_is_independent(self):
        source_path = self._get_test_path("backup_source")
        backup_path = self._get_test_path("backup_snapshot")

        source_store = brinicle.PayloadStore().init(
            source_path,
            shard_count=4,
        )

        try:
            source_store.insert(
                [
                    "text-1",
                    "text-2",
                ],
                [
                    "version-1",
                    "persistent-text",
                ],
            )

            source_store.insert_bytes(
                ["binary-1"],
                [b"\x00\x01\xff"],
            )

            source_store.backup(backup_path)

            # Mutations after backup must not change the snapshot.
            source_store.upsert(
                ["text-1"],
                ["version-2"],
            )
            source_store.upsert_bytes(
                ["binary-1"],
                [b"changed"],
            )
            source_store.delete(["text-2"])

        finally:
            self._close_store(source_store)

        backup_store = brinicle.PayloadStore().init(
            backup_path,
            shard_count=4,
        )

        try:
            assert backup_store.retrieve(
                [
                    "text-1",
                    "text-2",
                ]
            ) == [
                "version-1",
                "persistent-text",
            ], "backup did not preserve text payloads at snapshot time"

            assert backup_store.retrieve_bytes(
                ["binary-1"]
            ) == [
                b"\x00\x01\xff"
            ], "backup did not preserve binary payloads at snapshot time"

            scanned_ids, cursor = backup_store.scan(limit=10)

            assert scanned_ids == [
                "binary-1",
                "text-1",
                "text-2",
            ], "backup did not preserve the complete ID set"

            assert cursor is None

        finally:
            self._close_store(backup_store)

    def test_backup_rejects_existing_destination(self):
        source_path = self._get_test_path(
            "backup_existing_source"
        )
        backup_path = self._get_test_path(
            "backup_existing_destination"
        )

        os.makedirs(backup_path)

        store = brinicle.PayloadStore().init(
            source_path,
            shard_count=2,
        )

        try:
            try:
                store.backup(backup_path)
            except FileExistsError:
                pass
            else:
                raise AssertionError(
                    "backup must reject an existing destination"
                )

        finally:
            self._close_store(store)

    def test_destroy(self):
        path = self._get_test_path("destroy")
        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        store.insert(
            ["item-1"],
            ["payload-1"],
        )

        assert os.path.isdir(path), "store directory does not exist before destroy"

        store.destroy()

        assert not os.path.exists(
            path
        ), "destroy did not remove the payload store directory"

        store.destroy()


if __name__ == "__main__":
    import brinicle

    tests = PayloadStoreSimpleTests()
    tests.run_all()
