# Spec: Segmenter Robustness & Cache Correctness

**Spec ID:** 002
**Status:** Done
**Branch:** 001-spec-kit-multilang
**Created:** 2026-03-25

---

## Problem Statement

After shipping auto-detection (spec 001), three classes of bugs remain:

1. **ReferenceAudio cache language mismatch** — The `ReferenceAudio` LRU cache keyed on `prompt_wav`
   did not invalidate when the same wav was re-used with a different language. Phonemes were silently
   stale. (Fixed in this branch; this spec adds regression tests.)

2. **Segmenter punctuation leak** — `segment_by_language()` pre-splits on CJK/non-CJK boundaries.
   Punctuation-only non-CJK chunks (e.g. `"、"` full-width comma, `"。"` period) fall through to
   `detect_language()`, which labels them `"English"` because fastText cannot infer a language from
   a single punctuation character. This causes spurious language-boundary splits and routes
   punctuation through the English G2P.

3. **Min-segment length is hardcoded** — `_MIN_SEGMENT_LEN = 2` is a module-level constant with no
   public API. Callers cannot tune it for different use-cases (e.g. single-char CJK words).

---

## User Stories

### US-1: ReferenceAudio cache regression test
As a maintainer,
I want a unit test that exercises the `language`-change invalidation path in `ReferenceAudio`,
so that the fix is protected from future regressions.

**Acceptance criteria:**
- `tests/Audio/test_reference_audio_cache.py` exists
- Test: same wav + same text + different language → `set_text()` is called again
- Test: same wav + same text + same language → `set_text()` is NOT called again
- Tests run without loading actual model files (all model calls are mocked)

### US-2: Punctuation attached to preceding CJK segment
As a developer using `language="auto"`,
I want punctuation characters (CJK and ASCII) to be attached to the preceding language segment
rather than forming their own segment labelled 'English',
so that G2P processes punctuation together with the text it belongs to.

**Acceptance criteria:**
- `segment_by_language("你好，world")` does not produce a standalone `{language: 'English', content: '，'}` segment
- Punctuation between two same-language segments merges into that segment
- Punctuation between two different-language segments attaches to the preceding segment
- Unit tests in `tests/Utils/test_lang_detector.py` cover these cases

### US-3: Configurable minimum segment length
As a developer,
I want to pass `min_len` to `segment_by_language()` to override the default minimum segment size,
so that single-character CJK words are not silently merged into the previous segment.

**Acceptance criteria:**
- `segment_by_language(text, min_len=1)` respects segments of length 1
- Default behaviour (`min_len=2`) is unchanged
- Parameter is documented in the function docstring
- Unit tests verify both `min_len=1` and `min_len=2` behaviour

---

## Out of Scope
- Cantonese G2P implementation
- Multi-speaker caching
- GPU inference
