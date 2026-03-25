# Tasks: Segmenter Robustness & Cache Correctness

**Spec ID:** 002

---

## US-1: ReferenceAudio cache regression test

- [x] T1: Fix `ReferenceAudio.__new__` to invalidate on language change (done in this branch)
- [x] T2: Create `tests/Audio/test_reference_audio_cache.py` with cache invalidation tests

## US-2: Punctuation attached to preceding CJK segment

- [x] T3: Update `segment_by_language()` to attach punctuation-only chunks to preceding segment
- [x] T4: Add unit tests in `tests/Utils/test_lang_detector.py` for punctuation handling

## US-3: Configurable minimum segment length

- [x] T5: Add `min_len` parameter to `segment_by_language()` (default=2)
- [x] T6: Add unit tests for `min_len=1` and `min_len=2` behaviour
