"""Thai text pipeline pieces between the LLM and TTS (ARCHITECTURE.md §4.6).

``LLM TextDelta → EmotionTagExtractor → ThaiSpeechChunker → is_speakable → SafetyGate →
normalize_cloud | normalize_local → Segment``. Everything here is pure Python and imports
without pythainlp; the tokenizer and number reading load lazily (``warm_up_pythainlp``).
"""

from __future__ import annotations

from aivtube.text.chunker import (
    LOOKAHEAD,
    ChunkerConfig,
    ChunkerSettings,
    ThaiSpeechChunker,
    WordTokenizer,
    no_word_split,
)
from aivtube.text.normalize import (
    FILTERED,
    apply_lexicon,
    is_speakable,
    normalize_cloud,
    normalize_local,
)
from aivtube.text.tags import EmotionTagExtractor, TagEvent
from aivtube.text.thai import (
    NameMatcher,
    ThaiWordTokenizer,
    collapse_repeats,
    estimate_tokens,
    is_question,
    newmm,
    nfkc_casefold,
    strip_zero_width,
    thai_digits_to_arabic,
    warm_up_pythainlp,
)

__all__ = [
    "FILTERED",
    "LOOKAHEAD",
    "ChunkerConfig",
    "ChunkerSettings",
    "EmotionTagExtractor",
    "NameMatcher",
    "TagEvent",
    "ThaiSpeechChunker",
    "ThaiWordTokenizer",
    "WordTokenizer",
    "apply_lexicon",
    "collapse_repeats",
    "estimate_tokens",
    "is_question",
    "is_speakable",
    "newmm",
    "nfkc_casefold",
    "no_word_split",
    "normalize_cloud",
    "normalize_local",
    "strip_zero_width",
    "thai_digits_to_arabic",
    "warm_up_pythainlp",
]
