import io
import math
import os
from typing import List, Iterator, Tuple, Dict

from ...Core.Resources import English_G2P_DIR


class WordSegmenter:
    """
    Contains the core logic for word segmentation, adapted from the original library.
    """
    ALPHABET = set('abcdefghijklmnopqrstuvwxyz0123456789')
    TOTAL = 1024908267229.0
    LIMIT = 24

    def __init__(self):
        self.unigrams: Dict[str, float] = {}
        self.bigrams: Dict[str, float] = {}
        self.words: List[str] = []
        self.total: float = 0.0

    def load(self, data_directory: str):
        """
        Load unigram, bigram, and word counts from the specified data directory.
        This is the main modification from the original library.
        """
        unigrams_path = os.path.join(data_directory, 'unigrams.txt')
        bigrams_path = os.path.join(data_directory, 'bigrams.txt')
        words_path = os.path.join(data_directory, 'words.txt')

        for file_path in [unigrams_path, bigrams_path, words_path]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Word segmentation data file not found: {file_path}. "
                    "Please ensure the data directory is correct."
                )

        self.unigrams.update(self._parse(unigrams_path))
        self.bigrams.update(self._parse(bigrams_path))
        with io.open(words_path, encoding='utf-8') as reader:
            self.words.extend(reader.read().splitlines())

        self.total = self.TOTAL

    @staticmethod
    def _parse(filename: str) -> Dict[str, float]:
        """Read `filename` and parse tab-separated file of word and count pairs."""
        with io.open(filename, encoding='utf-8') as reader:
            # CORRECTED LINE: The generator now defines `line` before the comprehension uses it.
            lines = (line.split('\t') for line in reader)
            return {word: float(number) for word, number in lines if len(word) > 0 and len(number) > 0}

    def score(self, word: str, previous: str = None) -> float:
        """Score `word` in the context of `previous` word."""
        if previous is None:
            if word in self.unigrams:
                return self.unigrams[word] / self.total
            return 10.0 / (self.total * 10 ** len(word))

        bigram = f'{previous} {word}'
        if bigram in self.bigrams and previous in self.unigrams:
            return self.bigrams[bigram] / self.total / self.score(previous)

        return self.score(word)

    def isegment(self, text: str) -> Iterator[str]:
        """Return iterator of words that is the best segmenation of `text`."""
        memo = {}

        def search(text: str, previous: str = '<s>') -> Tuple[float, List[str]]:
            if text == '':
                return 0.0, []

            def candidates() -> Iterator[Tuple[float, List[str]]]:
                for prefix, suffix in self._divide(text):
                    prefix_score = math.log10(self.score(prefix, previous))

                    pair = (suffix, prefix)
                    if pair not in memo:
                        memo[pair] = search(suffix, prefix)
                    suffix_score, suffix_words = memo[pair]

                    yield prefix_score + suffix_score, [prefix] + suffix_words

            return max(candidates())

        clean_text = self._clean(text)

        # Original logic to avoid recursion limits by chunking
        size = 250
        prefix = ''
        if len(clean_text) > size:
            for offset in range(0, len(clean_text), size):
                chunk = clean_text[offset:(offset + size)]
                _, chunk_words = search(prefix + chunk)

                if len(chunk_words) > 5:
                    prefix = ''.join(chunk_words[-5:])
                    del chunk_words[-5:]
                else:  # handle case where chunk is small
                    prefix = ''.join(chunk_words)
                    chunk_words = []

                for word in chunk_words:
                    yield word

            _, prefix_words = search(prefix)
            for word in prefix_words:
                yield word
        else:
            _, words = search(clean_text)
            for word in words:
                yield word

    def segment(self, text: str) -> List[str]:
        """Return list of words that is the best segmenation of `text`."""
        return list(self.isegment(text))

    def _divide(self, text: str) -> Iterator[Tuple[str, str]]:
        """Yield `(prefix, suffix)` pairs from `text`."""
        for pos in range(1, min(len(text), self.LIMIT) + 1):
            yield text[:pos], text[pos:]

    @classmethod
    def _clean(cls, text: str) -> str:
        """Return `text` lower-cased with non-alphanumeric characters removed."""
        text_lower = text.lower()
        return ''.join(letter for letter in text_lower if letter in cls.ALPHABET)


# --- Public Interface ---
# Create a single instance to be used by the importing module.

_segmenter = WordSegmenter()
_segmenter.load(os.path.join(English_G2P_DIR, 'wordsegment'))


def segment_text(text: str) -> List[str]:
    """
    Public function to segment a text string into a list of words.
    """
    return _segmenter.segment(text)
