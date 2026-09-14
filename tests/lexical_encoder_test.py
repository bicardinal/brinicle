import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback

import numpy as np


class LexicalEncoderTests:
    """
    Covers the process-wide tokenizer cache and the encoder behaviour that
    depends on it: identity of the shared Tokenizer across encoders and
    engines, custom tokenizer paths, special-id handling, cache clearing,
    thread safety, and encoding equivalence with the uncached loader.
    """

    def __init__(self):
        self.test_dir = None
        self.test_count = 0
        self.passed_count = 0

    def setup(self):
        self.test_dir = tempfile.mkdtemp(prefix="lexenc_test_")
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
        return self.passed_count == self.test_count

    # ------------------------------------------------------------------ #

    def _bundled_tokenizer_path(self):
        import brinicle

        return os.path.join(os.path.dirname(brinicle.__file__), "tokenizer.json")

    def _write_custom_tokenizer(self, name):
        # A copy of the bundled file under a different path is a distinct
        # cache entry even though its content is identical.
        dst = os.path.join(self.test_dir, name)
        shutil.copyfile(self._bundled_tokenizer_path(), dst)
        return dst

    def test_bundled_tokenizer_is_shared_across_encoders(self):
        from brinicle import LexicalEncoder, clear_tokenizer_cache

        clear_tokenizer_cache()
        a = LexicalEncoder(max_dim=96)
        b = LexicalEncoder(max_dim=64, title_ratio=0.5)

        assert a.tokenizer is b.tokenizer, "bundled tokenizer not shared"
        assert a.special_ids == b.special_ids, "special ids differ"
        assert a.vocab_size == b.vocab_size

    def test_bundled_tokenizer_is_shared_across_engines(self):
        from brinicle import AutocompleteEngine, ItemSearchEngine

        item = ItemSearchEngine(self._get_test_path("share_item"), dim=64)
        auto = AutocompleteEngine(self._get_test_path("share_auto"), dim=32)
        item2 = ItemSearchEngine(self._get_test_path("share_item2"), dim=64)

        try:
            assert item.encoder.tokenizer is auto.encoder.tokenizer
            assert item.encoder.tokenizer is item2.encoder.tokenizer
        finally:
            item.close()
            auto.close()
            item2.close()

    def test_cached_load_is_faster_than_first_load(self):
        from brinicle import LexicalEncoder, clear_tokenizer_cache

        clear_tokenizer_cache()
        t0 = time.perf_counter()
        LexicalEncoder(max_dim=96)
        first = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(20):
            LexicalEncoder(max_dim=96)
        cached = (time.perf_counter() - t0) / 20

        print(f"  first load {first * 1000:.1f} ms, cached {cached * 1000:.3f} ms")
        assert cached * 5 < first, "cached construction should be far cheaper"

    def test_custom_path_is_distinct_cache_entry(self):
        from brinicle import LexicalEncoder, clear_tokenizer_cache

        clear_tokenizer_cache()
        custom = self._write_custom_tokenizer("custom_a.json")

        bundled = LexicalEncoder(max_dim=96)
        c1 = LexicalEncoder(max_dim=96, tokenizer_path=custom)
        c2 = LexicalEncoder(max_dim=96, tokenizer_path=custom)

        assert c1.tokenizer is c2.tokenizer, "same custom path not shared"
        assert c1.tokenizer is not bundled.tokenizer, "custom path collided with bundled"
        assert c1.vocab_size == bundled.vocab_size

    def test_custom_path_variants_resolve_to_same_entry(self):
        from pathlib import Path

        from brinicle import LexicalEncoder, clear_tokenizer_cache

        clear_tokenizer_cache()
        custom = self._write_custom_tokenizer("custom_b.json")

        as_str = LexicalEncoder(max_dim=96, tokenizer_path=custom)
        as_path = LexicalEncoder(max_dim=96, tokenizer_path=Path(custom))
        relative = os.path.relpath(custom, os.getcwd())
        as_rel = LexicalEncoder(max_dim=96, tokenizer_path=relative)

        assert as_str.tokenizer is as_path.tokenizer
        assert as_str.tokenizer is as_rel.tokenizer

    def test_missing_custom_path_raises(self):
        from brinicle import LexicalEncoder

        missing = os.path.join(self.test_dir, "does_not_exist.json")
        try:
            LexicalEncoder(max_dim=96, tokenizer_path=missing)
        except Exception:
            return
        raise AssertionError("missing tokenizer path should raise")

    def test_special_ids_match_uncached_loader(self):
        from tokenizers import Tokenizer

        from brinicle import LexicalEncoder, clear_tokenizer_cache

        clear_tokenizer_cache()
        enc = LexicalEncoder(max_dim=96)

        raw = Tokenizer.from_file(self._bundled_tokenizer_path())
        expected = {
            int(tid)
            for name in LexicalEncoder.special_token_names
            if (tid := raw.token_to_id(name)) is not None
        }
        assert enc.special_ids == expected, (enc.special_ids, expected)
        assert isinstance(enc.special_ids, set)

    def test_special_ids_follow_requested_names(self):
        from brinicle import get_cached_tokenizer, clear_tokenizer_cache

        clear_tokenizer_cache()
        tok_all, ids_all = get_cached_tokenizer(None, ("[CLS]", "[SEP]", "[PAD]"))
        tok_none, ids_none = get_cached_tokenizer(None, ())

        assert tok_all is tok_none, "different special sets must share Tokenizer"
        assert ids_none == frozenset()
        assert ids_all == frozenset(
            tok_all.token_to_id(t) for t in ("[CLS]", "[SEP]", "[PAD]")
            if tok_all.token_to_id(t) is not None
        )

    def test_clear_cache_reloads(self):
        from brinicle import LexicalEncoder, clear_tokenizer_cache

        clear_tokenizer_cache()
        before = LexicalEncoder(max_dim=96).tokenizer
        clear_tokenizer_cache()
        after = LexicalEncoder(max_dim=96).tokenizer

        assert before is not after, "clear_tokenizer_cache did not drop entry"
        assert before.get_vocab_size() == after.get_vocab_size()

    def test_encoding_equals_uncached_loader(self):
        from tokenizers import Tokenizer

        from brinicle import LexicalEncoder, clear_tokenizer_cache

        clear_tokenizer_cache()
        cached = LexicalEncoder(max_dim=96, vector_dim=0)

        uncached = LexicalEncoder(max_dim=96, vector_dim=0)
        uncached.tokenizer = Tokenizer.from_file(self._bundled_tokenizer_path())

        items = [
            ("Apple iPhone 15 Pro Max 256GB", {"color": "titanium"}, "phones", "apple"),
            ("running shoes men size 42", {"size": "42", "gender": "men"}, "shoes", None),
            ("", None, None, None),
            ("   spaced    out   title   ", None, None, None),
            ("unicode título ñ 日本語 🚀", {"k": "v"}, "cat", "sub"),
        ]
        for title, attrs, cat, sub in items:
            a = cached.encode_item_vector(title=title, attributes=attrs, category=cat, subcategory=sub)
            b = uncached.encode_item_vector(title=title, attributes=attrs, category=cat, subcategory=sub)
            assert np.array_equal(a, b), f"item encoding differs for {title!r}"

            qa = cached.encode_query_vector(query=title, attributes=attrs, category=cat, subcategory=sub)
            qb = uncached.encode_query_vector(query=title, attributes=attrs, category=cat, subcategory=sub)
            assert np.array_equal(qa, qb), f"query encoding differs for {title!r}"

    def test_concurrent_construction_yields_single_tokenizer(self):
        from brinicle import LexicalEncoder, clear_tokenizer_cache

        clear_tokenizer_cache()
        n_threads = 16
        barrier = threading.Barrier(n_threads)
        results = [None] * n_threads
        errors = []

        def worker(i):
            try:
                barrier.wait()
                results[i] = LexicalEncoder(max_dim=96).tokenizer
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        assert all(r is results[0] for r in results), "threads got different tokenizers"

    def test_concurrent_encoding_on_shared_tokenizer(self):
        from brinicle import LexicalEncoder, clear_tokenizer_cache

        clear_tokenizer_cache()
        reference = LexicalEncoder(max_dim=96)
        titles = [f"product number {i} with some words and {i * 7} extras" for i in range(200)]
        expected = [reference.encode_item_vector(title=t) for t in titles]

        errors = []

        def worker():
            try:
                enc = LexicalEncoder(max_dim=96)
                assert enc.tokenizer is reference.tokenizer
                for t, e in zip(titles, expected):
                    got = enc.encode_item_vector(title=t)
                    if not np.array_equal(got, e):
                        raise AssertionError(f"mismatch for {t!r}")
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors

    def test_item_search_end_to_end_with_shared_tokenizer(self):
        from brinicle import ItemSearchEngine, clear_tokenizer_cache

        clear_tokenizer_cache()
        engine = ItemSearchEngine(self._get_test_path("e2e"), dim=64, alpha=0.0)
        engine.init(mode="build")
        titles = {
            "a": "red running shoes",
            "b": "blue cotton shirt",
            "c": "wireless bluetooth headphones",
            "d": "red leather boots",
        }
        for eid, title in titles.items():
            engine.ingest(eid, title=title)
        engine.finalize()

        # A second engine on the same index shares the tokenizer and must
        # produce identical results.
        engine2 = ItemSearchEngine(self._get_test_path("e2e"), dim=64, alpha=0.0)
        try:
            assert engine.encoder.tokenizer is engine2.encoder.tokenizer
            r1 = engine.search("red shoes", k=2)
            r2 = engine2.search("red shoes", k=2)
            assert r1 == r2, (r1, r2)
            assert r1[0] == "a", r1
        finally:
            engine.close()
            engine2.close()


if __name__ == "__main__":
    import brinicle  # noqa: F401  (ensures the extension is importable)

    ok = LexicalEncoderTests().run_all()
    sys.exit(0 if ok else 1)
