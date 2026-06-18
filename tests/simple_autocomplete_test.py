import os
import shutil
import tempfile
import traceback


class AutocompleteEngineSimpleTests:

    def __init__(self):
        self.test_dir = None
        self.test_count = 0
        self.passed_count = 0

    def setup(self):
        self.test_dir = tempfile.mkdtemp(prefix="autocomplete_test_")
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
        suggestions = [
            "iphone 15 pro max",
            "iphone 15 case",
            "samsung galaxy s24 ultra",
            "google pixel 8 pro",
            "logitech wireless mouse",
        ]

        ac = brinicle.AutocompleteEngine(
            self._get_test_path("simple_build"),
            dim=48,
            delta_ratio=0.1,
        )

        ac.init(mode="build")

        for text in suggestions:
            ac.ingest(text, text)

        ac.finalize()

        search = ac.search("iphone", k=2)

        assert len(search) == 2, "invalid search length"
        assert len(set(search)) == 2, "duplicate results"
        assert "iphone 15 pro max" in search, "missing iphone pro suggestion"
        assert "iphone 15 case" in search, "missing iphone case suggestion"

        all_results = ac.search("", k=len(suggestions))

        assert len(all_results) == len(suggestions), "invalid all-result length"
        assert len(set(all_results)) == len(suggestions), "duplicate all-results"
        assert sorted(all_results) == sorted(suggestions), "invalid suggestion ids"

        ac.close()

    def test_simple_insert(self):
        base = [
            "iphone 15 pro max",
            "samsung galaxy s24 ultra",
            "google pixel 8 pro",
        ]

        inserted = [
            "iphone 15 case",
            "iphone charger cable",
        ]

        ac = brinicle.AutocompleteEngine(
            self._get_test_path("simple_insert"),
            dim=48,
            delta_ratio=0.1,
        )

        ac.init(mode="build")

        for text in base:
            ac.ingest(text, text)

        ac.finalize()

        search = ac.search("", k=len(base))

        assert len(search) == len(base), "invalid initial search length"
        assert len(set(search)) == len(base), "duplicate initial results"
        assert sorted(search) == sorted(base), "invalid initial ids"

        ac.init(mode="insert")

        for text in inserted:
            ac.ingest(text, text)

        ac.finalize()

        all_expected = base + inserted
        search = ac.search("", k=len(all_expected))

        assert len(search) == len(all_expected), "invalid search length after insert"
        assert len(set(search)) == len(all_expected), "duplicate results after insert"
        assert sorted(search) == sorted(all_expected), "inserted ids not found"

        iphone_results = ac.search("iph", k=4)

        assert (
            "iphone 15 pro max" in iphone_results
        ), "missing original iphone suggestion"
        assert (
            "iphone 15 case" in iphone_results
        ), "missing inserted iphone case suggestion"
        assert (
            "iphone charger cable" in iphone_results
        ), "missing inserted iphone charger suggestion"

        ac.close()

    def test_simple_upsert(self):
        ac = brinicle.AutocompleteEngine(
            self._get_test_path("simple_upsert"),
            dim=48,
            delta_ratio=0.1,
        )

        ac.init(mode="build")

        ac.ingest("s1", "iphone 15 pro max")
        ac.ingest("s2", "samsung galaxy s24 ultra")

        ac.finalize()

        before = ac.search("iph", k=2)

        assert "s1" in before, "s1 should exist before upsert"
        assert "s2" in ac.search("sam", k=2), "s2 should exist before upsert"

        ac.init(mode="upsert")

        ac.ingest("s1", "google pixel 8 pro")

        ac.finalize()

        pixel_results = ac.search("goo", k=1)

        assert len(pixel_results) == 1, "invalid search length after upsert"
        assert pixel_results[0] == "s1", "upserted suggestion should match new text"

        all_results = ac.search("", k=2)

        assert len(all_results) == 2, "invalid total result count after upsert"
        assert len(set(all_results)) == 2, "duplicate results after upsert"
        assert sorted(all_results) == ["s1", "s2"], "upsert changed ids incorrectly"

        ac.close()

    def test_simple_delete(self):
        suggestions = [
            "iphone 15 pro max",
            "iphone 15 case",
            "samsung galaxy s24 ultra",
            "google pixel 8 pro",
            "logitech wireless mouse",
        ]

        ac = brinicle.AutocompleteEngine(
            self._get_test_path("simple_delete"),
            dim=48,
            delta_ratio=0.1,
        )

        ac.init(mode="build")

        for text in suggestions:
            ac.ingest(text, text)

        ac.finalize()

        deleted_count, not_found = ac.delete_items(
            ["iphone 15 case", "google pixel 8 pro", "missing"],
            return_not_found=True,
        )

        assert deleted_count == 2, "invalid delete count"
        assert not_found == ["missing"], "invalid not_found list"

        search = ac.search("", k=len(suggestions))

        expected = [
            "iphone 15 pro max",
            "samsung galaxy s24 ultra",
            "logitech wireless mouse",
        ]

        assert len(search) == len(expected), "invalid search length after delete"
        assert len(set(search)) == len(expected), "duplicate results after delete"
        assert sorted(search) == sorted(expected), "invalid ids after delete"

        iphone_results = ac.search("iph", k=5)

        assert (
            "iphone 15 pro max" in iphone_results
        ), "remaining iphone suggestion missing"
        assert (
            "iphone 15 case" not in iphone_results
        ), "deleted iphone suggestion still appears"

        ac.close()

    def test_rebuild_and_optimize(self):
        suggestions = [f"rebuild suggestion {i}" for i in range(100)]

        ac = brinicle.AutocompleteEngine(
            self._get_test_path("rebuild_optimize"),
            dim=48,
            delta_ratio=0.1,
        )

        ac.init(mode="build")

        for text in suggestions:
            ac.ingest(text, text)

        ac.finalize()

        deleted_count, _ = ac.delete_items(
            [
                "rebuild suggestion 1",
                "rebuild suggestion 2",
                "rebuild suggestion 3",
            ],
            return_not_found=False,
        )

        assert deleted_count == 3, "invalid delete count before rebuild"

        before = ac.search("", k=100)

        assert len(before) == 97, "invalid result count before rebuild"
        assert (
            "rebuild suggestion 1" not in before
        ), "deleted id 1 appears before rebuild"
        assert (
            "rebuild suggestion 2" not in before
        ), "deleted id 2 appears before rebuild"
        assert (
            "rebuild suggestion 3" not in before
        ), "deleted id 3 appears before rebuild"

        ac.rebuild_compact(
            M=16,
            ef_construction=200,
            ef_search=64,
            build_n_threads=1,
            seed=0,
        )

        ac.optimize_graph()

        after = ac.search("", k=100)

        expected = [f"rebuild suggestion {i}" for i in range(100) if i not in (1, 2, 3)]

        assert len(after) == 97, "invalid result count after rebuild"
        assert len(set(after)) == 97, "duplicate results after rebuild"
        assert sorted(after) == sorted(expected), "invalid ids after rebuild"

        ac.close()

    def test_search_batch(self):
        suggestions = [
            "iphone 15 pro max",
            "iphone 15 case",
            "samsung galaxy s24 ultra",
            "google pixel 8 pro",
            "logitech wireless mouse",
        ]

        ac = brinicle.AutocompleteEngine(
            self._get_test_path("search_batch"),
            dim=48,
            delta_ratio=0.1,
        )

        ac.init(mode="build")

        for text in suggestions:
            ac.ingest(text, text)

        ac.finalize()

        results = ac.search_batch(
            [
                "iphone",
                "samsung",
                "google",
                "logi",
            ],
            k=1,
        )
        assert len(results) == 4, "invalid batch result length"
        assert results[0][0] in {
            "iphone 15 pro max",
            "iphone 15 case",
        }, "invalid iphone batch result"
        assert (
            results[1][0] == "samsung galaxy s24 ultra"
        ), "invalid samsung batch result"
        assert results[2][0] == "google pixel 8 pro", "invalid google batch result"
        assert (
            results[3][0] == "logitech wireless mouse"
        ), "invalid logitech batch result"

        ac.close()

    def test_search_batch_empty_queries(self):
        ac = brinicle.AutocompleteEngine(
            self._get_test_path("batch_empty"),
            dim=48,
            delta_ratio=0.1,
        )

        results = ac.search_batch([], k=10)

        assert results == [], "empty batch should return empty list"

        ac.close()

    def test_search_with_distance(self):
        suggestions = [
            "iphone 15 pro max",
            "iphone 15 case",
            "samsung galaxy s24 ultra",
        ]

        ac = brinicle.AutocompleteEngine(
            self._get_test_path("search_with_distance"),
            dim=48,
            delta_ratio=0.1,
        )

        ac.init(mode="build")

        for text in suggestions:
            ac.ingest(text, text)

        ac.finalize()

        results = ac.search_with_distance("iph", k=2)

        assert len(results) == 2, "invalid search_with_distance length"

        ids = [x[0] for x in results]
        distances = [x[1] for x in results]

        assert "iphone 15 pro max" in ids, "missing iphone pro suggestion"
        assert "iphone 15 case" in ids, "missing iphone case suggestion"

        for dist in distances:
            assert isinstance(dist, float), "distance should be float"

        ac.close()

    def test_external_id_can_differ_from_text(self):
        ac = brinicle.AutocompleteEngine(
            self._get_test_path("external_id_differs"),
            dim=48,
            delta_ratio=0.1,
        )

        ac.init(mode="build")

        ac.ingest("q1", "iphone 15 pro max")
        ac.ingest("q2", "samsung galaxy s24 ultra")
        ac.ingest("q3", "google pixel 8 pro")

        ac.finalize()

        results = ac.search("iphone", k=1)

        assert len(results) == 1, "invalid result length"
        assert results[0] in (
            "q1",
            "q2",
            "q3",
        ), "search should return external id, not text"

        all_results = ac.search("", k=3)

        assert sorted(all_results) == ["q1", "q2", "q3"], "invalid external ids"

        ac.close()

    def test_context_manager(self):
        with brinicle.AutocompleteEngine(
            self._get_test_path("context_manager"),
            dim=48,
            delta_ratio=0.1,
        ) as ac:
            ac.init(mode="build")
            ac.ingest("iphone", "iphone 15 pro max")
            ac.ingest("samsung", "samsung galaxy s24 ultra")
            ac.finalize()

            results = ac.search("iph", k=1)

            assert results == ["iphone"], "invalid context manager search result"


if __name__ == "__main__":
    import brinicle

    tests = AutocompleteEngineSimpleTests()
    tests.run_all()
