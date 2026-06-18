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
