# Plan: Segmenter Robustness & Cache Correctness

**Spec ID:** 002

---

## T2 — ReferenceAudio cache test

`ReferenceAudio.__new__` uses a class-level LRU dict keyed on `prompt_wav`.
The test must:
- Stub `load_audio`, `soxr.resample`, `model_manager`, and `get_phones_and_bert`
- Instantiate two `ReferenceAudio` objects with same wav, same text, different language
- Assert `get_phones_and_bert` called twice (second call triggered by language change)
- Instantiate a third with same wav + text + language → assert call count unchanged

## T3 — Punctuation fix in `segment_by_language`

Current approach: split on CJK/non-CJK boundary, detect each chunk.
Issue: non-CJK punctuation (`，`, `。`, `、`, `,`, `.`, `!`, `？`, etc.) get labelled 'English'.

Fix strategy:
- After pre-split, classify each chunk as: CJK, Latin, or Punctuation-only
- Punctuation-only chunks are NOT sent to `detect_language()`
- Instead attach them to the last non-empty segment (if any), else to the next
- Define a `_RE_PUNCT` regex covering common CJK + ASCII punctuation

## T5 — `min_len` parameter

Change `segment_by_language(text)` signature to `segment_by_language(text, min_len=2)`.
Pass `min_len` through to the short-chunk guard (`len(part.strip()) < min_len`).
Update call sites (only `GetPhonesAndBert.py`) — they use the default so no change needed there.
