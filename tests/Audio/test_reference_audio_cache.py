"""Tests for ReferenceAudio LRU cache invalidation.

Verifies that:
  - same wav + same text + same language  -> G2P is NOT re-run
  - same wav + same text + new language   -> G2P IS re-run
  - same wav + new text  + same language  -> G2P IS re-run

All model I/O (load_audio, soxr, cn_hubert, model_manager) is mocked so
no actual model files are needed.
"""
from __future__ import annotations

import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call


# ---------------------------------------------------------------------------
# Module-level stubs so ReferenceAudio can be imported
# ---------------------------------------------------------------------------

for _mod in [
    "soxr",
    "genie_tts.Audio.Audio",
    "genie_tts.ModelManager",
]:
    sys.modules.setdefault(_mod, MagicMock())


@pytest.fixture(autouse=True)
def clear_cache():
    """Wipe ReferenceAudio._prompt_cache between tests."""
    from genie_tts.Audio.ReferenceAudio import ReferenceAudio
    ReferenceAudio._prompt_cache.clear()
    yield
    ReferenceAudio._prompt_cache.clear()


@pytest.fixture()
def fake_audio():
    """Return a tiny float32 array that stands in for loaded audio."""
    return np.zeros(100, dtype=np.float32)


def _make_ref(wav, text, language, get_phones_mock):
    """Construct a ReferenceAudio with all heavy deps stubbed out."""
    import genie_tts.Audio.ReferenceAudio as mod
    audio = np.zeros(100, dtype=np.float32)
    audio_2d = np.zeros((1, 100), dtype=np.float32)

    mm_stub = MagicMock()
    mm_stub.cn_hubert = MagicMock()
    mm_stub.cn_hubert.run.return_value = [np.zeros((1, 256), dtype=np.float32)]

    with (
        patch("genie_tts.Audio.ReferenceAudio.load_audio", return_value=audio),
        patch("genie_tts.Audio.ReferenceAudio.soxr") as mock_soxr,
        patch("genie_tts.Audio.ReferenceAudio.model_manager", mm_stub),
        patch("genie_tts.Audio.ReferenceAudio.get_phones_and_bert", get_phones_mock),
    ):
        mock_soxr.resample.return_value = audio
        instance = mod.ReferenceAudio(
            prompt_wav=wav,
            prompt_text=text,
            language=language,
        )
    return instance


class TestReferenceAudioCache:

    def test_same_wav_text_language_no_extra_g2p(self):
        """Second construction with identical args must not re-run G2P."""
        g2p = MagicMock(return_value=(np.zeros((1, 3), dtype=np.int64),
                                      np.zeros((3, 1024), dtype=np.float32)))
        _make_ref("a.wav", "hello", "English", g2p)
        _make_ref("a.wav", "hello", "English", g2p)
        assert g2p.call_count == 1

    def test_same_wav_different_language_reruns_g2p(self):
        """Same wav + same text but different language must re-run G2P."""
        g2p = MagicMock(return_value=(np.zeros((1, 3), dtype=np.int64),
                                      np.zeros((3, 1024), dtype=np.float32)))
        _make_ref("a.wav", "hello", "English", g2p)
        _make_ref("a.wav", "hello", "Japanese", g2p)
        assert g2p.call_count == 2

    def test_same_wav_different_text_reruns_g2p(self):
        """Same wav but different text must re-run G2P."""
        g2p = MagicMock(return_value=(np.zeros((1, 3), dtype=np.int64),
                                      np.zeros((3, 1024), dtype=np.float32)))
        _make_ref("a.wav", "hello", "English", g2p)
        _make_ref("a.wav", "world", "English", g2p)
        assert g2p.call_count == 2

    def test_different_wav_creates_new_entry(self):
        """Different wav path must create a fresh cache entry (G2P called twice)."""
        g2p = MagicMock(return_value=(np.zeros((1, 3), dtype=np.int64),
                                      np.zeros((3, 1024), dtype=np.float32)))
        _make_ref("a.wav", "hello", "English", g2p)
        _make_ref("b.wav", "hello", "English", g2p)
        assert g2p.call_count == 2

    def test_language_stored_on_instance(self):
        """After construction, instance.language matches the argument."""
        g2p = MagicMock(return_value=(np.zeros((1, 2), dtype=np.int64),
                                      np.zeros((2, 1024), dtype=np.float32)))
        inst = _make_ref("c.wav", "おはよう", "Japanese", g2p)
        assert inst.language == "Japanese"

    def test_language_updated_after_change(self):
        """After a language change, instance.language reflects the new value."""
        g2p = MagicMock(return_value=(np.zeros((1, 2), dtype=np.int64),
                                      np.zeros((2, 1024), dtype=np.float32)))
        _make_ref("d.wav", "hello", "English", g2p)
        inst = _make_ref("d.wav", "hello", "Chinese", g2p)
        assert inst.language == "Chinese"
