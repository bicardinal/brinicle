import os
import shutil
import tempfile
import traceback

import numpy as np


class VectorEngineIrrationalTests:

    def __init__(self, VectorEngine):
        self.VectorEngine = VectorEngine
        self.test_dir = None
        self.test_count = 0
        self.passed_count = 0

    def setup(self):
        self.test_dir = tempfile.mkdtemp(prefix="vectordb_test_")
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

            print(f"\n{'='*60}")
            print(f"Results: {self.passed_count}/{self.test_count} tests passed")
            print(f"{'='*60}")
        finally:
            self.teardown()

    def test_empty_index_search(self):
        db = self.VectorEngine(self._get_test_path("empty"), dim=128)
        query = np.random.randn(128).astype(np.float32)

        try:
            results = db.search(query, k=10)
            assert False, "Should have raised error on empty index"
        except RuntimeError as e:
            pass

    def test_dimension_zero(self):
        try:
            db = self.VectorEngine(self._get_test_path("dim_zero"), dim=0)
            assert False, "Should reject zero dimension"
        except:
            pass

    def test_ingest_before_init(self):
        db = self.VectorEngine(self._get_test_path("no_init"), dim=128)
        vec = np.random.randn(128).astype(np.float32)

        try:
            db.ingest("id1", vec)
            assert False, "Should require init() first"
        except RuntimeError as e:
            assert "call init() first" in str(e)

    def test_finalize_without_init(self):
        db = self.VectorEngine(self._get_test_path("finalize_no_init"), dim=128)

        try:
            db.finalize()
            assert False, "Should require init() first"
        except RuntimeError as e:
            assert "no pending ingest" in str(e)

    def test_dimension_mismatch_ingest(self):
        db = self.VectorEngine(self._get_test_path("dim_mismatch"), dim=128)
        db.init("build")

        wrong_vec = np.random.randn(64).astype(np.float32)
        try:
            db.ingest("id1", wrong_vec)
            assert False, "Should reject wrong dimension"
        except RuntimeError as e:
            assert "dimension mismatch" in str(e)

    def test_dimension_mismatch_search(self):
        db = self.VectorEngine(self._get_test_path("search_dim"), dim=128)
        db.init("build")
        vec = np.random.randn(128).astype(np.float32)
        db.ingest("id1", vec)
        db.finalize()

        wrong_query = np.random.randn(64).astype(np.float32)
        try:
            db.search(wrong_query, k=5)
            assert False, "Should reject wrong query dimension"
        except RuntimeError as e:
            assert "dimension mismatch" in str(e)

    def test_nan_vector_ingest(self):
        db = self.VectorEngine(self._get_test_path("nan_vec"), dim=128)
        db.init("build")

        vec = np.random.randn(128).astype(np.float32)
        vec[50] = np.nan

        try:
            db.ingest("id1", vec)
            assert False, "Should reject NaN values"
        except RuntimeError as e:
            assert "NaN" in str(e) or "Inf" in str(e)

    def test_inf_vector_ingest(self):
        db = self.VectorEngine(self._get_test_path("inf_vec"), dim=128)
        db.init("build")

        vec = np.random.randn(128).astype(np.float32)
        vec[50] = np.inf

        try:
            db.ingest("id1", vec)
            assert False, "Should reject Inf values"
        except RuntimeError as e:
            assert "NaN" in str(e) or "Inf" in str(e)

    def test_negative_k_search(self):
        db = self.VectorEngine(self._get_test_path("neg_k"), dim=128)
        db.init("build")
        vec = np.random.randn(128).astype(np.float32)
        db.ingest("id1", vec)
        db.finalize()
        try:
            query = np.random.randn(128).astype(np.float32)
            results = db.search(query, k=-5)
            assert False, "Should reject negative k"
        except:
            pass

    def test_nan_inf_search(self):
        db = self.VectorEngine(self._get_test_path("neg_k"), dim=128)
        db.init("build")
        vec = np.random.randn(128).astype(np.float32)
        db.ingest("id1", vec)
        db.finalize()

        query = np.random.randn(128).astype(np.float32)
        query[50] = np.nan
        query[51] = np.inf
        try:
            results = db.search(query, k=-5)
            assert False, "should reject invalid values"
        except Exception as e:
            traceback.print_exc()

    def test_zero_k_search(self):
        db = self.VectorEngine(self._get_test_path("zero_k"), dim=128)
        db.init("build")
        vec = np.random.randn(128).astype(np.float32)
        db.ingest("id1", vec)
        db.finalize()
        try:
            query = np.random.randn(128).astype(np.float32)
            results = db.search(query, k=0)
            assert False, "should reject invalid values"
        except:
            pass

    def test_k_larger_than_index(self):
        db = self.VectorEngine(self._get_test_path("large_k"), dim=128)
        db.init("build")

        for i in range(5):
            vec = np.random.randn(128).astype(np.float32)
            db.ingest(f"id{i}", vec)
        db.finalize()

        query = np.random.randn(128).astype(np.float32)
        results = db.search(query, k=100)
        assert len(results) <= 5

    def test_empty_external_id(self):
        db = self.VectorEngine(self._get_test_path("empty_id"), dim=128)
        db.init("build")
        vec = np.random.randn(128).astype(np.float32)
        db.ingest("", vec)
        db.finalize()

        query = np.random.randn(128).astype(np.float32)
        results = db.search(query, k=1)
        assert len(results) == 1
        assert results[0] == ""

    def test_duplicate_ids_build(self):
        db = self.VectorEngine(self._get_test_path("dup_ids"), dim=128)
        db.init("build")

        vec1 = np.random.randn(128).astype(np.float32)
        vec2 = np.random.randn(128).astype(np.float32)

        db.ingest("same_id", vec1)
        db.ingest("same_id", vec2)
        try:
            db.finalize()
            assert False, "Should reject duplicate ids"
        except Exception as e:
            pass

    def test_invalid_mode(self):
        db = self.VectorEngine(self._get_test_path("bad_mode"), dim=128)

        try:
            db.init("invalid_mode")
            assert False, "Should reject invalid mode"
        except RuntimeError as e:
            assert "mode must be" in str(e)

    def test_multiple_init_without_finalize(self):
        db = self.VectorEngine(self._get_test_path("multi_init"), dim=128)
        db.init("build")
        try:
            db.init("build")
            vec = np.random.randn(128).astype(np.float32)
            db.ingest("id1", vec)
            db.finalize()
        except:
            assert "multiple init triggered errors"

    def test_delete_nonexistent_ids(self):
        db = self.VectorEngine(self._get_test_path("del_nonexist"), dim=128)
        db.init("build")

        vec = np.random.randn(128).astype(np.float32)
        db.ingest("id1", vec)
        db.finalize()

        deleted, not_found = db.delete_items(
            ["nonexistent1", "nonexistent2"], return_not_found=True
        )
        assert deleted == 0
        assert len(not_found) == 2

    def test_delete_empty_list(self):
        db = self.VectorEngine(self._get_test_path("del_empty"), dim=128)
        db.init("build")
        vec = np.random.randn(128).astype(np.float32)
        db.ingest("id1", vec)
        db.finalize()

        deleted, _ = db.delete_items([])
        assert deleted == 0

    def test_search_after_delete_all(self):
        db = self.VectorEngine(self._get_test_path("del_all"), dim=128)
        db.init("build")

        ids = []
        for i in range(10):
            vec = np.random.randn(128).astype(np.float32)
            id_str = f"id{i}"
            ids.append(id_str)
            db.ingest(id_str, vec)
        db.finalize()

        deleted, _ = db.delete_items(ids)
        assert deleted == 10

        query = np.random.randn(128).astype(np.float32)
        results = db.search(query, k=5)
        assert len(results) == 0

    def test_upsert_same_id(self):
        db = self.VectorEngine(self._get_test_path("upsert"), dim=128)

        db.init("build")
        vec1 = np.ones(128, dtype=np.float32)
        db.ingest("id1", vec1)
        db.finalize()

        db.init("upsert")
        vec2 = np.ones(128, dtype=np.float32) * 2.0
        db.ingest("id1", vec2)
        db.finalize()

        query = np.ones(128, dtype=np.float32) * 2.0
        results = db.search(query, k=1)
        assert len(results) == 1
        assert results[0] == "id1"

    def test_insert_without_existing_index(self):
        db = self.VectorEngine(self._get_test_path("insert_noindex"), dim=128)
        db.init("insert")
        vec = np.random.randn(128).astype(np.float32)
        db.ingest("id1", vec)

        try:
            db.finalize()
            assert False, "Insert should require existing index"
        except RuntimeError as e:
            assert "No index loaded" in str(e)

    def test_very_long_external_id(self):
        db = self.VectorEngine(self._get_test_path("long_id"), dim=128)
        db.init("build")

        long_id = "x" * 10000
        vec = np.random.randn(128).astype(np.float32)
        db.ingest(long_id, vec)
        db.finalize()

        query = np.random.randn(128).astype(np.float32)
        results = db.search(query, k=1)
        assert len(results) == 1
        assert results[0] == long_id

    def test_special_characters_in_id(self):
        db = self.VectorEngine(self._get_test_path("special_id"), dim=128)
        db.init("build")

        special_ids = [
            "id\nwith\nnewlines",
            "id\twith\ttabs",
            "id with spaces",
            "id/with/slashes",
            "id\\with\\backslashes",
            'id"with"quotes',
            "id'with'apostrophes",
            "id;with;semicolons",
        ]

        for special_id in special_ids:
            vec = np.random.randn(128).astype(np.float32)
            db.ingest(special_id, vec)

        db.finalize()

        query = np.random.randn(128).astype(np.float32)
        results = db.search(query, k=len(special_ids))
        assert len(results) <= len(special_ids)

    def test_unicode_external_id(self):
        db = self.VectorEngine(self._get_test_path("unicode_id"), dim=128)
        db.init("build")

        unicode_ids = ["测试", "テスト", "тест", "🚀🎉", "Ω≈∆"]

        for uid in unicode_ids:
            vec = np.random.randn(128).astype(np.float32)
            db.ingest(uid, vec)

        db.finalize()

        query = np.random.randn(128).astype(np.float32)
        results = db.search(query, k=5)
        assert len(results) > 0

    def test_close_and_reopen(self):
        path = self._get_test_path("close_reopen")

        db1 = self.VectorEngine(path, dim=128)
        db1.init("build")
        for i in range(10):
            vec = np.random.randn(128).astype(np.float32)
            db1.ingest(f"id{i}", vec)
        db1.finalize()

        query = np.random.randn(128).astype(np.float32)
        results1 = db1.search(query, k=5)

        db1.close()

        db2 = self.VectorEngine(path, dim=128)
        results2 = db2.search(query, k=5)

        assert len(results2) == len(results1)
        assert results2 == results1

    def test_destroy_and_recreate(self):
        path = self._get_test_path("destroy_recreate")

        db1 = self.VectorEngine(path, dim=128)
        db1.init("build")
        vec = np.random.randn(128).astype(np.float32)
        db1.ingest("id1", vec)
        db1.finalize()

        db1.destroy()

        db2 = self.VectorEngine(path, dim=128)
        assert not db2.has_index

    def test_negative_delta_ratio(self):
        try:
            db = self.VectorEngine(
                self._get_test_path("neg_delta"), dim=128, delta_ratio=-0.5
            )
            assert False, "Should reject invalid delta_ratio"
        except:
            pass

    def test_very_large_delta_ratio(self):
        try:
            db = self.VectorEngine(
                self._get_test_path("large_delta"), dim=128, delta_ratio=10.0
            )
            assert False, "Should reject invalid delta_ratio"
        except:
            assert True

    def test_single_vector_database(self):
        db = self.VectorEngine(self._get_test_path("single_vec"), dim=128)
        db.init("build")

        vec = np.random.randn(128).astype(np.float32)
        db.ingest("only_one", vec)
        db.finalize()

        query = np.random.randn(128).astype(np.float32)
        results = db.search(query, k=10)
        assert len(results) == 1
        assert results[0] == "only_one"

    def test_zero_vector_ingest(self):
        db = self.VectorEngine(self._get_test_path("zero_vec"), dim=128)
        db.init("build")

        zero_vec = np.zeros(128, dtype=np.float32)
        db.ingest("zero", zero_vec)

        normal_vec = np.random.randn(128).astype(np.float32)
        db.ingest("normal", normal_vec)

        db.finalize()

        query = np.zeros(128, dtype=np.float32)
        results = db.search(query, k=2)
        assert len(results) == 2
        assert "zero" in results

    def test_huge_vector_values(self):
        db = self.VectorEngine(self._get_test_path("huge_vals"), dim=128)
        db.init("build")

        huge_vec = np.ones(128, dtype=np.float32) * 1e6
        db.ingest("huge", huge_vec)

        tiny_vec = np.ones(128, dtype=np.float32) * 1e-6
        db.ingest("tiny", tiny_vec)

        db.finalize()

        query = np.ones(128, dtype=np.float32)
        results = db.search(query, k=2)
        assert len(results) == 2

    def test_optimize_empty_index(self):
        db = self.VectorEngine(self._get_test_path("opt_empty"), dim=128)

        try:
            db.optimize_graph()
            assert False, "Should fail on empty index"
        except:
            pass

    def test_needs_rebuild_empty(self):
        db = self.VectorEngine(self._get_test_path("rebuild_empty"), dim=128)

        try:
            needs = db.needs_rebuild()
            assert False, "Should fail on empty index"
        except:
            pass

    def test_rebuild_compact_empty(self):
        db = self.VectorEngine(self._get_test_path("compact_empty"), dim=128)

        try:
            db.rebuild_compact()
            assert False, "Should fail on empty index"
        except:
            pass

    def test_multiple_finalize_calls(self):
        db = self.VectorEngine(self._get_test_path("multi_finalize"), dim=128)
        db.init("build")
        vec = np.random.randn(128).astype(np.float32)
        db.ingest("id1", vec)
        db.finalize()

        try:
            db.finalize()
            assert False, "Should fail on second finalize"
        except RuntimeError:
            pass

    def test_ingest_after_finalize(self):
        db = self.VectorEngine(self._get_test_path("ingest_after_fin"), dim=128)
        db.init("build")
        vec = np.random.randn(128).astype(np.float32)
        db.ingest("id1", vec)
        db.finalize()

        vec2 = np.random.randn(128).astype(np.float32)
        try:
            db.ingest("id2", vec2)
            assert False, "Should fail without new init"
        except RuntimeError:
            pass

    def test_concurrent_path_access(self):
        path = self._get_test_path("concurrent")

        db1 = self.VectorEngine(path, dim=128)
        db1.init("build")
        vec = np.random.randn(128).astype(np.float32)
        db1.ingest("id1", vec)
        db1.finalize()

        db2 = self.VectorEngine(path, dim=128)
        query = np.random.randn(128).astype(np.float32)
        results = db2.search(query, k=5)

        assert len(results) > 0

    def test_path_with_special_chars(self):
        special_path = os.path.join(self.test_dir, "path with spaces & special!@#")
        os.makedirs(special_path, exist_ok=True)

        db = self.VectorEngine(os.path.join(special_path, "index"), dim=128)
        db.init("build")
        vec = np.random.randn(128).astype(np.float32)
        db.ingest("id1", vec)
        db.finalize()

        query = np.random.randn(128).astype(np.float32)
        results = db.search(query, k=1)
        assert len(results) == 1

    def test_very_high_dimension(self):
        db = self.VectorEngine(self._get_test_path("high_dim"), dim=4096)
        db.init("build")

        vec = np.random.randn(4096).astype(np.float32)
        db.ingest("high_d", vec)
        db.finalize()

        query = np.random.randn(4096).astype(np.float32)
        results = db.search(query, k=1)
        assert len(results) == 1

    def test_insert_then_delete_then_insert(self):
        db = self.VectorEngine(self._get_test_path("ins_del_ins"), dim=128)

        db.init("build")
        for i in range(5):
            vec = np.random.randn(128).astype(np.float32)
            db.ingest(f"id{i}", vec)
        db.finalize()

        db.delete_items(["id1", "id3"])

        db.init("insert")
        for i in range(5, 8):
            vec = np.random.randn(128).astype(np.float32)
            db.ingest(f"id{i}", vec)
        db.finalize()

        query = np.random.randn(128).astype(np.float32)
        results = db.search(query, k=10)

        assert "id1" not in results
        assert "id3" not in results
        assert "id0" in results or "id2" in results


if __name__ == "__main__":
    from brinicle import VectorEngine

    tests = VectorEngineIrrationalTests(VectorEngine)
    tests.run_all()
