# Spec: Multilingual Support Improvement

**Spec ID:** 001
**Status:** Done
**Branch:** 001-spec-kit-multilang
**Created:** 2026-03-25

---

## Problem Statement

Genie-TTS currently supports Japanese, English, Chinese, Korean, and a basic Hybrid-Chinese-English
mode. The hybrid mode only handles Chinese/English mixing via a simple Latin-character regex split.
Users who supply text in an unsupported or unspecified language get silently routed to the Japanese
fallback, producing incorrect phonemes. There is no automatic language detection capability.

## User Stories

### US-1: Auto Language Detection
As a developer integrating Genie-TTS,
I want to pass `language="auto"` and have the engine detect the language automatically,
so that I do not need to know the language of the input text in advance.

**Acceptance criteria:**
- `language="auto"` is accepted by `tts()`, `tts_async()`, `set_reference_audio()`
- Language is detected per-segment using `fast_langdetect`
- Detected language is mapped to a supported canonical name; unsupported languages fall back to
  English with a logged warning
- Detection is cached per-segment to avoid redundant inference

### US-2: General Mixed-Language Segmentation
As a developer,
I want mixed-language text (e.g. Chinese + Japanese, English + Korean) to be processed correctly,
so that each segment uses the right G2P pipeline.

**Acceptance criteria:**
- `LangDetector.segment(text)` returns a list of `{language, content}` dicts
- Segments respect sentence boundaries (punctuation is attached to the preceding segment)
- Minimum segment length is configurable (default 2 chars) to avoid over-segmentation
- Works for Chinese/English, Japanese/English, Chinese/Japanese mixes

### US-3: Language Normalisation Completeness
As a developer,
I want `normalize_language()` to accept all common ISO 639-1/3 codes and BCP-47 tags,
so that I don't get silent fallback for standard codes like `"cmn"` or `"yue"`.

**Acceptance criteria:**
- `normalize_language("cmn")` -> `"Chinese"`
- `normalize_language("yue")` -> `"Cantonese"` (reserved; logs warning that Cantonese is not yet fully supported)
- `normalize_language("auto")` -> `"auto"`
- Unknown codes raise a `ValueError` with a helpful message listing supported languages

### US-4: TextSplitter Language Awareness
As a developer,
I want the text splitter to correctly count effective length for all supported scripts,
so that Japanese/Korean sentences are not over-split due to CJK char width assumptions.

**Acceptance criteria:**
- Korean Hangul characters are counted as width 2
- Japanese hiragana/katakana/kanji are counted as width 2
- ASCII-range Korean jamo (U+3130-U+318F) counted as width 1
- Existing Chinese/English behaviour is unchanged

### US-5: Structured Unit Tests
As a maintainer,
I want a pytest test suite covering G2P and language utilities,
so that regressions are caught automatically.

**Acceptance criteria:**
- `tests/` directory exists with `pytest` configuration
- Tests for `normalize_language()`, `LangDetector`, `TextSplitter`
- Tests for each G2P module (Chinese, English, Japanese, Korean) with at least 3 representative sentences
- All tests pass with `pytest tests/`

---

## Out of Scope
- Cantonese G2P implementation (symbols already in SymbolsV2, but full pipeline deferred)
- V3/V4 model support
- GPU inference
- New voice/character additions
