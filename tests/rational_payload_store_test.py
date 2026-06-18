import json
import os
import shutil
import sqlite3
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor


class PayloadStoreRationalTests:

    def __init__(self):
        self.test_dir = None
        self.test_count = 0
        self.passed_count = 0

    def setup(self):
        self.test_dir = tempfile.mkdtemp(prefix="payloadstore_rational_test_")
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

    def test_realistic_document_lifecycle(self):
        path = self._get_test_path("document_lifecycle")

        store = brinicle.PayloadStore().init(
            path,
            shard_count=8,
        )

        expected = {}

        try:
            ids = []
            values = []

            for index in range(300):
                payload_id = f"document-{index}"

                document = {
                    "id": payload_id,
                    "title": f"Document number {index}",
                    "category": f"category-{index % 12}",
                    "score": index / 10,
                    "active": index % 3 != 0,
                    "tags": [
                        f"tag-{index % 5}",
                        f"tag-{index % 7}",
                    ],
                    "metadata": {
                        "source": "rational-test",
                        "position": index,
                    },
                }

                serialized = json.dumps(
                    document,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

                ids.append(payload_id)
                values.append(serialized)
                expected[payload_id] = serialized

            store.insert(ids, values)

            requested_ids = [
                "document-250",
                "document-10",
                "missing-document",
                "document-10",
                "document-299",
            ]

            retrieved = store.retrieve(requested_ids)

            assert retrieved == [
                expected["document-250"],
                expected["document-10"],
                None,
                expected["document-10"],
                expected["document-299"],
            ], "initial realistic retrieval returned invalid values"

            update_ids = []
            update_values = []

            for index in range(0, 300, 4):
                payload_id = f"document-{index}"

                updated_document = {
                    "id": payload_id,
                    "title": f"Updated document {index}",
                    "revision": 2,
                    "active": True,
                }

                serialized = json.dumps(
                    updated_document,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

                update_ids.append(payload_id)
                update_values.append(serialized)
                expected[payload_id] = serialized

            for index in range(300, 325):
                payload_id = f"document-{index}"

                new_document = {
                    "id": payload_id,
                    "title": f"New document {index}",
                    "revision": 1,
                }

                serialized = json.dumps(
                    new_document,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

                update_ids.append(payload_id)
                update_values.append(serialized)
                expected[payload_id] = serialized

            store.upsert(
                update_ids,
                update_values,
            )

            delete_ids = [f"document-{index}" for index in range(25, 125, 5)]

            deleted_count = store.delete(delete_ids + delete_ids[:5])

            assert deleted_count == len(
                delete_ids
            ), "realistic deletion returned invalid count"

            for payload_id in delete_ids:
                expected.pop(payload_id)

            verification_ids = [f"document-{index}" for index in range(325)]

            result = store.retrieve(verification_ids)

            expected_result = [
                expected.get(payload_id) for payload_id in verification_ids
            ]

            assert (
                result == expected_result
            ), "document lifecycle produced an invalid final state"

        finally:
            self._close_store(store)

    def test_failed_insert_rolls_back_all_shards(self):
        path = self._get_test_path("rollback")

        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        try:
            new_id_shard_0 = self._find_id_for_shard(
                store,
                0,
                "new-zero",
            )

            new_id_shard_1 = self._find_id_for_shard(
                store,
                1,
                "new-one",
            )

            existing_id_shard_3 = self._find_id_for_shard(
                store,
                3,
                "existing-three",
            )

            store.insert(
                [existing_id_shard_3],
                ["original-value"],
            )

            error_raised = False

            try:
                store.insert(
                    [
                        new_id_shard_0,
                        new_id_shard_1,
                        existing_id_shard_3,
                    ],
                    [
                        "new-value-zero",
                        "new-value-one",
                        "must-fail",
                    ],
                )

            except sqlite3.IntegrityError:
                error_raised = True

            assert error_raised, "inserting an existing ID did not raise IntegrityError"

            result = store.retrieve(
                [
                    new_id_shard_0,
                    new_id_shard_1,
                    existing_id_shard_3,
                ]
            )

            assert result == [
                None,
                None,
                "original-value",
            ], "failed multi-shard insertion was not rolled back"

            store.insert(
                [new_id_shard_0],
                ["successful-after-rollback"],
            )

            assert store.retrieve([new_id_shard_0]) == [
                "successful-after-rollback"
            ], "store was unusable after transaction rollback"

        finally:
            self._close_store(store)

    def test_concurrent_batch_inserts(self):
        path = self._get_test_path("concurrent_inserts")

        store = brinicle.PayloadStore().init(
            path,
            shard_count=16,
        )

        worker_count = 6
        records_per_worker = 120
        batch_size = 20
        start_barrier = threading.Barrier(worker_count)

        def insert_worker(worker_id):
            start_barrier.wait()

            for batch_start in range(
                0,
                records_per_worker,
                batch_size,
            ):
                batch_end = min(
                    batch_start + batch_size,
                    records_per_worker,
                )

                ids = [
                    f"worker-{worker_id}-item-{index}"
                    for index in range(batch_start, batch_end)
                ]

                values = [
                    f"worker-{worker_id}-payload-{index}"
                    for index in range(batch_start, batch_end)
                ]

                store.insert(ids, values)

        try:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        insert_worker,
                        worker_id,
                    )
                    for worker_id in range(worker_count)
                ]

                for future in futures:
                    future.result()

            all_ids = []
            all_values = []

            for worker_id in range(worker_count):
                for index in range(records_per_worker):
                    all_ids.append(f"worker-{worker_id}-item-{index}")
                    all_values.append(f"worker-{worker_id}-payload-{index}")

            result = store.retrieve(all_ids)

            assert (
                result == all_values
            ), "concurrent insert operations lost or changed payloads"

        finally:
            self._close_store(store)

    def test_concurrent_readers(self):
        path = self._get_test_path("concurrent_readers")

        store = brinicle.PayloadStore().init(
            path,
            shard_count=16,
        )

        count = 600
        ids = [f"read-item-{index}" for index in range(count)]
        values = [f"read-payload-{index}" for index in range(count)]

        expected_by_id = dict(zip(ids, values))

        worker_count = 8
        start_barrier = threading.Barrier(worker_count)

        def read_worker(worker_id):
            start_barrier.wait()

            for round_index in range(30):
                requested_ids = [
                    ids[(worker_id * 17 + round_index * 11 + position * 13) % count]
                    for position in range(50)
                ]

                expected = [expected_by_id[payload_id] for payload_id in requested_ids]

                result = store.retrieve(requested_ids)

                assert result == expected, (
                    f"reader {worker_id} received invalid values "
                    f"during round {round_index}"
                )

        try:
            store.insert(ids, values)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        read_worker,
                        worker_id,
                    )
                    for worker_id in range(worker_count)
                ]

                for future in futures:
                    future.result()

        finally:
            self._close_store(store)

    def test_concurrent_independent_updates_and_deletes(self):
        path = self._get_test_path("mixed_concurrency")

        store = brinicle.PayloadStore().init(
            path,
            shard_count=16,
        )

        worker_count = 4
        records_per_worker = 100
        start_barrier = threading.Barrier(worker_count)

        initial_ids = []
        initial_values = []

        for worker_id in range(worker_count):
            for index in range(records_per_worker):
                initial_ids.append(f"owner-{worker_id}-item-{index}")
                initial_values.append(f"initial-{worker_id}-{index}")

        def mixed_worker(worker_id):
            owned_ids = [
                f"owner-{worker_id}-item-{index}" for index in range(records_per_worker)
            ]

            start_barrier.wait()

            for batch_start in range(
                0,
                records_per_worker,
                25,
            ):
                batch_ids = owned_ids[batch_start : batch_start + 25]

                batch_values = [
                    f"updated-{worker_id}-{index}"
                    for index in range(
                        batch_start,
                        batch_start + len(batch_ids),
                    )
                ]

                store.upsert(
                    batch_ids,
                    batch_values,
                )

            deleted_ids = owned_ids[::7]
            store.delete(deleted_ids)

            remaining_ids = [
                payload_id
                for payload_id in owned_ids
                if payload_id not in set(deleted_ids)
            ]

            expected_remaining = [
                f"updated-{worker_id}-{index}"
                for index in range(records_per_worker)
                if index % 7 != 0
            ]

            result = store.retrieve(remaining_ids)

            assert (
                result == expected_remaining
            ), f"worker {worker_id} observed an invalid final state"

        try:
            store.insert(
                initial_ids,
                initial_values,
            )

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        mixed_worker,
                        worker_id,
                    )
                    for worker_id in range(worker_count)
                ]

                for future in futures:
                    future.result()

            final_result = store.retrieve(initial_ids)

            expected_final_result = []

            for worker_id in range(worker_count):
                for index in range(records_per_worker):
                    if index % 7 == 0:
                        expected_final_result.append(None)
                    else:
                        expected_final_result.append(f"updated-{worker_id}-{index}")

            assert (
                final_result == expected_final_result
            ), "concurrent mixed workload produced invalid state"

        finally:
            self._close_store(store)

    def test_repeated_reopen_cycles(self):
        path = self._get_test_path("reopen_cycles")

        ids = [f"persistent-item-{index}" for index in range(200)]

        expected = {
            payload_id: f"initial-{index}" for index, payload_id in enumerate(ids)
        }

        store = brinicle.PayloadStore().init(
            path,
            shard_count=8,
        )

        try:
            store.insert(
                ids,
                [expected[payload_id] for payload_id in ids],
            )

        finally:
            store.close()

        for cycle in range(1, 5):
            store = brinicle.PayloadStore().init(
                path,
                shard_count=8,
            )

            try:
                update_ids = ids[cycle - 1 :: 4]

                update_values = [
                    f"cycle-{cycle}-{payload_id}" for payload_id in update_ids
                ]

                store.upsert(
                    update_ids,
                    update_values,
                )

                for payload_id, value in zip(
                    update_ids,
                    update_values,
                ):
                    expected[payload_id] = value

                result = store.retrieve(ids)

                assert result == [
                    expected[payload_id] for payload_id in ids
                ], f"data was invalid after reopen cycle {cycle}"

            finally:
                store.close()

        final_store = brinicle.PayloadStore().init(
            path,
            shard_count=8,
        )

        try:
            assert final_store.retrieve(ids) == [
                expected[payload_id] for payload_id in ids
            ], "final reopened store did not preserve all changes"

        finally:
            final_store.close()

    def test_large_payload_storage(self):
        path = self._get_test_path("large_payloads")

        store = brinicle.PayloadStore().init(
            path,
            shard_count=8,
        )

        try:
            ids = [f"large-document-{index}" for index in range(16)]

            values = [
                (f"document-{index}:" + chr(65 + index % 26) * (256 * 1024))
                for index in range(16)
            ]

            store.insert(ids, values)

            requested_ids = list(reversed(ids))
            result = store.retrieve(requested_ids)

            assert result == list(
                reversed(values)
            ), "large payloads were truncated or corrupted"

            updated_value = "updated-large-document:" + "Z" * (512 * 1024)

            store.upsert(
                [ids[0]],
                [updated_value],
            )

            assert store.retrieve([ids[0]]) == [
                updated_value
            ], "large payload upsert failed"

        finally:
            self._close_store(store)

    def test_close_then_destroy(self):
        path = self._get_test_path("close_destroy")

        store = brinicle.PayloadStore().init(
            path,
            shard_count=4,
        )

        store.insert(
            ["document-1", "document-2"],
            ["payload-1", "payload-2"],
        )

        store.close()

        assert os.path.isdir(path), "close unexpectedly removed the store directory"

        store.destroy()

        assert not os.path.exists(path), "destroy after close did not remove the store"


if __name__ == "__main__":
    import brinicle

    tests = PayloadStoreRationalTests()
    tests.run_all()
