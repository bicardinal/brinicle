import os
import shutil
import tempfile
import traceback

import numpy as np


class ItemSearchEngineSimpleTests:

    def __init__(self):
        self.test_dir = None
        self.test_count = 0
        self.passed_count = 0

    def setup(self):
        self.test_dir = tempfile.mkdtemp(prefix="itemsearch_test_")
        print(f"Test directory: {self.test_dir}")

    def teardown(self):
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"Cleaned up: {self.test_dir}")

    def _get_test_path(self, name):
        return os.path.join(self.test_dir, f"test_idx_{name}")

    def _run_test(self, test_method):
        self.test_count += 1
        test_name = test_method.__name__
        try:
            print(f"\nRunning: {test_name}")
            test_method()
            self.passed_count += 1
            print(f"[OK] {test_name} PASSED")
        except Exception as e:
            print(f"[NOT OK] {test_name} FAILED: {e}")
            traceback.print_exc()

    def run_all(self):
        self.setup()
        try:
            test_methods = [
                getattr(self, m)
                for m in dir(self)
                if m.startswith("test_") and callable(getattr(self, m))
            ]

            for test_method in test_methods:
                self._run_test(test_method)

            print(f"\n{'=' * 60}")
            print(f"Results: {self.passed_count}/{self.test_count} tests passed")
            print(f"{'=' * 60}")
        finally:
            self.teardown()

    def test_simple_build(self):
        n = 1000
        k = n

        engine = brinicle.ItemSearchEngine(
            self._get_test_path("simple_build"),
            dim=96,
            alpha=0.0,
            delta_ratio=0.1,
        )

        engine.init(mode="build")

        for eid in range(n):
            engine.ingest(
                external_id=str(eid),
                title=f"test product number {eid}",
                category="Test",
                subcategory="Products",
                attributes={
                    "index": eid,
                    "kind": "basic",
                },
            )

        engine.finalize()

        search = engine.search("test product", k=k)

        assert len(search) == k, "invalid search length"
        assert len(set(search)) == k, "duplicate results"
        assert sorted(search, key=int) == [str(i) for i in range(n)], "invalid ids"

        engine.close()

    def test_simple_insert(self):
        n = 1000
        k = n

        engine = brinicle.ItemSearchEngine(
            self._get_test_path("simple_insert"),
            dim=96,
            alpha=0.0,
            delta_ratio=0.1,
        )

        engine.init(mode="build")

        for eid in range(n):
            engine.ingest(
                external_id=str(eid),
                title=f"base product number {eid}",
                category="Base",
                subcategory="Products",
                attributes={
                    "group": "base",
                    "index": eid,
                },
            )

        engine.finalize()

        search = engine.search("base product", k=k)

        assert len(search) == k, "invalid initial search length"
        assert len(set(search)) == k, "duplicate initial results"
        assert sorted(search, key=int) == [
            str(i) for i in range(n)
        ], "invalid initial ids"

        engine.init(mode="insert")

        for eid in range(5):
            engine.ingest(
                external_id=f"{eid}x",
                title=f"inserted product number {eid}",
                category="Inserted",
                subcategory="Products",
                attributes={
                    "group": "inserted",
                    "index": eid,
                },
            )

        engine.finalize()

        search = engine.search("product", k=n + 5)

        expected_ids = [str(i) for i in range(n)] + [f"{i}x" for i in range(5)]

        assert len(search) == n + 5, "invalid search length after insert"
        assert len(set(search)) == n + 5, "duplicate results after insert"
        assert sorted(search) == sorted(expected_ids), "inserted ids not found"

        engine.close()

    def test_simple_upsert(self):
        engine = brinicle.ItemSearchEngine(
            self._get_test_path("simple_upsert"),
            dim=96,
            alpha=0.0,
            delta_ratio=0.1,
        )

        engine.init(mode="build")

        engine.ingest(
            external_id="p1",
            title="apple iphone 15 pro max",
            category="Electronics",
            subcategory="Smartphones",
            attributes={"brand": "Apple"},
        )

        engine.ingest(
            external_id="p2",
            title="samsung galaxy s24 ultra",
            category="Electronics",
            subcategory="Smartphones",
            attributes={"brand": "Samsung"},
        )

        engine.finalize()

        before = engine.search("google pixel", k=2)

        assert "p1" in before, "p1 should exist before upsert"
        assert "p2" in before, "p2 should exist before upsert"

        engine.init(mode="upsert")

        engine.ingest(
            external_id="p1",
            title="google pixel 8 pro",
            category="Electronics",
            subcategory="Smartphones",
            attributes={"brand": "Google"},
        )

        engine.finalize()

        after = engine.search("google pixel", k=1)

        assert len(after) == 1, "invalid search length after upsert"
        assert after[0] == "p1", "upserted item should be the best match"

        all_results = engine.search("smartphone", k=2)

        assert len(all_results) == 2, "invalid total result count after upsert"
        assert len(set(all_results)) == 2, "duplicate results after upsert"
        assert sorted(all_results) == ["p1", "p2"], "upsert changed ids incorrectly"

        engine.close()

    def test_simple_delete(self):
        engine = brinicle.ItemSearchEngine(
            self._get_test_path("simple_delete"),
            dim=96,
            alpha=0.0,
            delta_ratio=0.1,
        )

        engine.init(mode="build")

        for eid in range(10):
            engine.ingest(
                external_id=str(eid),
                title=f"delete test product {eid}",
                category="DeleteTest",
                subcategory="Products",
                attributes={"index": eid},
            )

        engine.finalize()

        deleted_count, not_found = engine.delete_items(
            ["1", "4", "missing"],
            return_not_found=True,
        )

        assert deleted_count == 2, "invalid delete count"
        assert not_found == ["missing"], "invalid not_found list"

        search = engine.search("delete test product", k=10)

        assert "1" not in search, "deleted id 1 still appears"
        assert "4" not in search, "deleted id 4 still appears"
        assert len(search) == 8, "invalid search length after delete"
        assert len(set(search)) == 8, "duplicate results after delete"

        expected_ids = [str(i) for i in range(10) if i not in (1, 4)]

        assert sorted(search, key=int) == expected_ids, "invalid ids after delete"

        engine.close()

    def test_rebuild_and_optimize(self):
        engine = brinicle.ItemSearchEngine(
            self._get_test_path("rebuild_optimize"),
            dim=96,
            alpha=0.0,
            delta_ratio=0.1,
        )

        engine.init(mode="build")

        for eid in range(100):
            engine.ingest(
                external_id=str(eid),
                title=f"rebuild product {eid}",
                category="Rebuild",
                subcategory="Products",
                attributes={"index": eid},
            )

        engine.finalize()

        engine.delete_items(["1", "2", "3"])

        before = engine.search("rebuild product", k=100)

        assert len(before) == 97, "invalid result count before rebuild"
        assert "1" not in before, "deleted id 1 appears before rebuild"
        assert "2" not in before, "deleted id 2 appears before rebuild"
        assert "3" not in before, "deleted id 3 appears before rebuild"

        engine.rebuild_compact(
            M=16,
            ef_construction=200,
            ef_search=64,
            build_n_threads=1,
            seed=0,
        )

        engine.optimize_graph()

        after = engine.search("rebuild product", k=100)

        assert len(after) == 97, "invalid result count after rebuild"
        assert len(set(after)) == 97, "duplicate results after rebuild"
        assert "1" not in after, "deleted id 1 appears after rebuild"
        assert "2" not in after, "deleted id 2 appears after rebuild"
        assert "3" not in after, "deleted id 3 appears after rebuild"

        expected_ids = [str(i) for i in range(100) if i not in (1, 2, 3)]

        assert sorted(after, key=int) == expected_ids, "invalid ids after rebuild"

        engine.close()

    def test_search_batch_queries_only(self):
        engine = brinicle.ItemSearchEngine(
            self._get_test_path("batch_queries_only"),
            dim=96,
            alpha=0.0,
            delta_ratio=0.1,
        )

        engine.init(mode="build")

        engine.ingest(
            external_id="p1",
            title="apple iphone 15 pro max",
            category="Electronics",
            subcategory="Smartphones",
            attributes={"brand": "Apple"},
        )

        engine.ingest(
            external_id="p2",
            title="nike running shoes size 42",
            category="Fashion",
            subcategory="Shoes",
            attributes={"brand": "Nike"},
        )

        engine.ingest(
            external_id="p3",
            title="logitech wireless mouse",
            category="Electronics",
            subcategory="Computer Accessories",
            attributes={"brand": "Logitech"},
        )

        engine.finalize()

        results = engine.search_batch(
            [
                "iphone 15",
                "running shoes",
                "wireless mouse",
            ],
            k=1,
        )

        assert len(results) == 3, "invalid batch result length"
        assert results[0][0] == "p1", "invalid result for iphone query"
        assert results[1][0] == "p2", "invalid result for shoes query"
        assert results[2][0] == "p3", "invalid result for mouse query"

        engine.close()

    def test_search_batch_independent_metadata(self):
        engine = brinicle.ItemSearchEngine(
            self._get_test_path("batch_independent_metadata"),
            dim=96,
            alpha=0.0,
            delta_ratio=0.1,
        )

        engine.init(mode="build")

        engine.ingest(
            external_id="p1",
            title="apple iphone 15 pro max",
            category="Electronics",
            subcategory="Smartphones",
            attributes={"brand": "Apple", "storage": "256GB"},
        )

        engine.ingest(
            external_id="p2",
            title="nike running shoes size 42",
            category="Fashion",
            subcategory="Shoes",
            attributes={"brand": "Nike", "size": "42"},
        )

        engine.ingest(
            external_id="p3",
            title="logitech wireless mouse",
            category="Electronics",
            subcategory="Computer Accessories",
            attributes={"brand": "Logitech", "connection": "wireless"},
        )

        engine.finalize()

        results = engine.search_batch(
            queries=[
                "iphone",
                "running shoes",
                "wireless mouse",
            ],
            categories=[
                "Electronics",
                "Fashion",
                "Electronics",
            ],
            subcategories=[
                "Smartphones",
                "Shoes",
                "Computer Accessories",
            ],
            attributes_list=[
                {"brand": "Apple"},
                {"brand": "Nike"},
                {"brand": "Logitech"},
            ],
            k=1,
        )

        assert len(results) == 3, "invalid batch result length"
        assert results[0][0] == "p1", "invalid result for independent metadata query 0"
        assert results[1][0] == "p2", "invalid result for independent metadata query 1"
        assert results[2][0] == "p3", "invalid result for independent metadata query 2"

        engine.close()

    def test_search_batch_queries_and_vectors(self):
        VECTOR_DIM = 4

        engine = brinicle.ItemSearchEngine(
            self._get_test_path("batch_queries_vectors"),
            dim=32,
            vector_dim=VECTOR_DIM,
            alpha=1.0,
            vector_normalized=False,
            delta_ratio=0.1,
        )

        p1_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        p2_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        p3_vec = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        engine.init(mode="build")

        engine.ingest(
            external_id="p1",
            title="item one",
            vector=p1_vec,
        )

        engine.ingest(
            external_id="p2",
            title="item two",
            vector=p2_vec,
        )

        engine.ingest(
            external_id="p3",
            title="item three",
            vector=p3_vec,
        )

        engine.finalize()

        query_vectors = np.array(
            [
                p1_vec,
                p2_vec,
                p3_vec,
            ],
            dtype=np.float32,
        )

        results = engine.search_batch(
            queries=[
                "query one",
                "query two",
                "query three",
            ],
            vectors=query_vectors,
            k=1,
        )

        assert len(results) == 3, "invalid batch result length"
        assert results[0][0] == "p1", "invalid vector result 0"
        assert results[1][0] == "p2", "invalid vector result 1"
        assert results[2][0] == "p3", "invalid vector result 2"

        engine.close()

    def test_search_batch_empty_queries(self):
        engine = brinicle.ItemSearchEngine(
            self._get_test_path("batch_empty"),
            dim=96,
            alpha=0.0,
            delta_ratio=0.1,
        )

        results = engine.search_batch([], k=10)

        assert results == [], "empty batch should return empty list"

        engine.close()

    def test_search_batch_length_validation(self):
        engine = brinicle.ItemSearchEngine(
            self._get_test_path("batch_length_validation"),
            dim=96,
            alpha=0.0,
            delta_ratio=0.1,
        )

        queries = ["iphone", "galaxy"]

        try:
            engine.search_batch(
                queries,
                categories=["Electronics"],
            )
            assert False, "categories length mismatch should raise ValueError"
        except ValueError as e:
            assert "categories length mismatch" in str(
                e
            ), "invalid categories error message"

        try:
            engine.search_batch(
                queries,
                subcategories=["Smartphones"],
            )
            assert False, "subcategories length mismatch should raise ValueError"
        except ValueError as e:
            assert "subcategories length mismatch" in str(
                e
            ), "invalid subcategories error message"

        try:
            engine.search_batch(
                queries,
                attributes_list=[{"brand": "Apple"}],
            )
            assert False, "attributes_list length mismatch should raise ValueError"
        except ValueError as e:
            assert "attributes_list length mismatch" in str(
                e
            ), "invalid attributes_list error message"

        try:
            engine.search_batch(
                queries,
                attributes_list={"brand": "Apple"},
            )
            assert False, "single mapping attributes_list should raise TypeError"
        except TypeError as e:
            assert "attributes_list must be a sequence" in str(
                e
            ), "invalid attributes_list type error"

        engine.close()

    def test_search_batch_vector_length_validation(self):
        VECTOR_DIM = 4

        engine = brinicle.ItemSearchEngine(
            self._get_test_path("batch_vector_length_validation"),
            dim=32,
            vector_dim=VECTOR_DIM,
            alpha=1.0,
            delta_ratio=0.1,
        )

        queries = ["q1", "q2"]

        bad_vectors_count = np.random.randn(1, VECTOR_DIM).astype(np.float32)

        try:
            engine.search_batch(
                queries,
                vectors=bad_vectors_count,
            )
            assert False, "vectors count mismatch should raise ValueError"
        except ValueError as e:
            assert "vectors length mismatch" in str(e), "invalid vectors length error"

        bad_vectors_dim = np.random.randn(2, VECTOR_DIM + 1).astype(np.float32)

        try:
            engine.search_batch(
                queries,
                vectors=bad_vectors_dim,
            )
            assert False, "vectors dimension mismatch should raise ValueError"
        except ValueError as e:
            assert "vectors dimension mismatch" in str(
                e
            ), "invalid vectors dimension error"

        engine.close()


if __name__ == "__main__":
    import brinicle

    tests = ItemSearchEngineSimpleTests()
    tests.run_all()
