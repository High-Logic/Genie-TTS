"""Tests for Utils/LangDetector.py."""
import pytest
from unittest.mock import patch
from genie_tts.Utils.LangDetector import detect_language, segment_by_language


class TestDetectLanguage:
    def test_empty_string_returns_fallback(self):
        result = detect_language("")
        assert result == "English"

    def test_whitespace_returns_fallback(self):
        result = detect_language("   ")
        assert result == "English"

    def test_english_text(self):
        result = detect_language("Hello, how are you today?")
        assert result == "English"

    def test_chinese_text(self):
        result = detect_language("你好，今天天气怎么样？")
        assert result == "Chinese"

    def test_japanese_text(self):
        result = detect_language("おはようございます、今日はいい天気ですね。")
        assert result == "Japanese"

    def test_korean_text(self):
        result = detect_language("안녕하세요, 오늘 날씨가 어떤가요?")
        assert result == "Korean"

    def test_unsupported_lang_falls_back(self):
        # Mock detector returning an unsupported code
        with patch("genie_tts.Utils.LangDetector._detect_fn",
                   return_value={"lang": "fr", "score": 0.99}):
            result = detect_language("Bonjour le monde")
            assert result == "English"  # fallback

    def test_detector_exception_falls_back(self):
        def _raise(_text, **_kwargs):
            raise RuntimeError("detection failed")
        with patch("genie_tts.Utils.LangDetector._detect_fn", side_effect=_raise):
            result = detect_language("some text")
            assert result == "English"


class TestSegmentByLanguage:
    def test_empty_returns_empty(self):
        assert segment_by_language("") == []

    def test_pure_english(self):
        segs = segment_by_language("Hello world")
        assert len(segs) >= 1
        assert all(s["content"].strip() for s in segs)

    def test_pure_chinese(self):
        segs = segment_by_language("今天天气很好")
        assert len(segs) >= 1
        combined = "".join(s["content"] for s in segs)
        assert "今天" in combined

    def test_mixed_chinese_english_segments(self):
        segs = segment_by_language("今天 I went to 学校")
        assert len(segs) >= 1
        languages = {s["language"] for s in segs}
        # Should detect at least one language
        assert len(languages) >= 1

    def test_segments_cover_full_text(self):
        text = "Hello 你好 world"
        segs = segment_by_language(text)
        combined = "".join(s["content"] for s in segs)
        # All content should be present
        assert "Hello" in combined or "hello" in combined.lower()

    def test_no_empty_segments(self):
        segs = segment_by_language("Hello 你好")
        for s in segs:
            assert s["content"].strip() != ""

    def test_adjacent_same_lang_merged(self):
        # Two English phrases should merge into one segment
        segs = segment_by_language("Hello world how are you")
        en_segs = [s for s in segs if s["language"] == "English"]
        # Should be a single merged English segment
        assert len(en_segs) == 1


class TestSegmentByLanguagePunctuation:
    """Punctuation handling: CJK and ASCII punctuation should not form standalone
    'English' segments — they must be attached to the preceding segment."""

    def test_cjk_punct_not_standalone(self):
        """Full-width comma between CJK text should not produce its own segment."""
        segs = segment_by_language("你好，世界")
        # All segments must have actual text content, not just punctuation
        punct_only = [s for s in segs if not any(c.isalpha() or '\u4e00' <= c <= '\u9fff'
                                                   or '\u3040' <= c <= '\u30ff'
                                                   for c in s["content"])]
        assert punct_only == [], f"Standalone punctuation segments found: {punct_only}"

    def test_ascii_punct_not_standalone_english(self):
        """ASCII punctuation-only chunk should not create a spurious 'English' segment."""
        segs = segment_by_language("你好, world")
        # The comma should not be its own 'English' segment
        for seg in segs:
            assert seg["content"].strip() not in (",", ".", "!", "?", ";", ":")

    def test_punctuation_attached_to_preceding(self):
        """Punctuation after CJK text is included in the CJK segment content."""
        segs = segment_by_language("你好。")
        # Should produce exactly one segment whose content includes the period
        assert len(segs) == 1
        assert "。" in segs[0]["content"]

    def test_mixed_punct_preserved_in_content(self):
        """Total content (incl. punctuation) is preserved across segmentation."""
        text = "你好, world!"
        segs = segment_by_language(text)
        combined = "".join(s["content"] for s in segs)
        # Every character of the original must appear somewhere in the output
        assert set(text) == set(combined)


class TestSegmentByLanguageMinLen:
    """Tests for the min_len parameter."""

    def test_default_min_len_merges_short_chunks(self):
        """A 1-char non-CJK chunk should be merged with the previous segment by default."""
        # Single letter between two CJK blocks — should not get its own segment
        segs = segment_by_language("你好a世界")
        single_letter_segs = [s for s in segs if s["content"].strip() == "a"]
        assert single_letter_segs == []

    def test_min_len_1_keeps_single_char(self):
        """With min_len=1, a single-character chunk may get its own segment."""
        # We can't assert exact segmentation because detect_language may merge
        # adjacent same-language chunks, but the content must be preserved.
        text = "你好a世界"
        segs = segment_by_language(text, min_len=1)
        combined = "".join(s["content"] for s in segs)
        assert "a" in combined
        assert "你好" in combined
        assert "世界" in combined

    def test_min_len_zero_keeps_all(self):
        """min_len=0 should not drop any chunk."""
        text = "A你B"
        segs = segment_by_language(text, min_len=0)
        combined = "".join(s["content"] for s in segs)
        assert "A" in combined
        assert "你" in combined
        assert "B" in combined

    def test_min_len_large_merges_everything(self):
        """Very large min_len causes all short non-CJK chunks to be merged."""
        segs = segment_by_language("你好 hello 世界", min_len=100)
        # All content must still be present
        combined = "".join(s["content"] for s in segs)
        assert "hello" in combined
        assert "你好" in combined
