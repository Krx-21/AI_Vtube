"""NamePostProcessor: alias map, short-audio drop and echo drop (voice.stt acceptance)."""

from __future__ import annotations

from aivtube.contracts.types import Transcript
from aivtube.contracts.voice import TranscriptPostProcessor
from aivtube.voice.stt import NamePostProcessor, echo_similarity

ALIASES = {"ไพลิน": ["ไทลิน", "ไทยลิน", "ไทลิล", "ไภลิน"]}


def tr(text: str, audio_s: float = 2.0) -> Transcript:
    return Transcript(text=text, is_final=True, audio_s=audio_s, latency_ms=12.0, engine="rt")


def test_is_a_transcript_post_processor() -> None:
    assert isinstance(NamePostProcessor(ALIASES), TranscriptPostProcessor)


def test_maps_misheard_names_to_the_canonical_name() -> None:
    post = NamePostProcessor(ALIASES)
    out = post(tr("สวัสดีครับ ไทลิน วันนี้เราจะเล่นเกม"), "")
    assert out is not None and out.text == "สวัสดีครับ ไพลิน วันนี้เราจะเล่นเกม"
    for wrong in ("ไทยลิน", "ไทลิล", "ไภลิน"):
        got = post(tr(f"{wrong}ช่วยอ่านแชทหน่อย"), "")
        assert got is not None and got.text == "ไพลินช่วยอ่านแชทหน่อย"
    # the other fields survive
    assert out.engine == "rt" and out.audio_s == 2.0 and out.latency_ms == 12.0


def test_canonical_name_is_left_intact_and_unchanged_transcript_is_returned_as_is() -> None:
    post = NamePostProcessor({"ไพลิน": ["ไพลิ", "ไทลิน"]})  # an alias that is a prefix
    t = tr("ไพลินน่ารักมาก")
    assert post(t, "") is t


def test_latin_aliases_match_case_insensitively() -> None:
    post = NamePostProcessor({"Pailin": ["pailyn", "PIE LIN"]})
    out = post(tr("hello PAILYN and pie lin"), "")
    assert out is not None and out.text == "hello Pailin and Pailin"


def test_drops_audio_shorter_than_min_audio_and_empty_text() -> None:
    post = NamePostProcessor(ALIASES, min_audio_s=0.3)
    assert post(tr("ไทลิน", audio_s=0.25), "") is None
    assert post(tr("ไทลิน", audio_s=0.3), "") is not None
    assert post(tr("   "), "") is None
    out = post(tr("  สวัสดี \n  ครับ  "), "")
    assert out is not None and out.text == "สวัสดี ครับ"


def test_drops_echoes_of_recent_tts_text() -> None:
    post = NamePostProcessor(ALIASES)
    recent = "สวัสดีค่ะทุกคน วันนี้ไพลินจะมาเล่นเกมกันนะคะ ขอบคุณที่มาดูนะ"
    # her own words heard back through speakers, with an STT mishearing of her name
    assert post(tr("วันนี้ไทลินจะมาเล่นเกมกันนะคะ"), recent) is None
    # the streamer saying something different is kept
    kept = post(tr("ไพลินช่วยอ่านคอมเมนต์ในแชทหน่อยได้ไหม"), recent)
    assert kept is not None
    # no recent TTS text: never an echo
    assert post(tr("วันนี้ไพลินจะมาเล่นเกมกันนะคะ"), "") is not None


def test_echo_ratio_threshold_is_configurable() -> None:
    recent = "วันนี้อากาศดีมากเลยนะคะ"
    text = "วันนี้อากาศไม่ดีเลย"  # similar, not identical
    score = echo_similarity(text, recent)
    assert 0.6 < score < 0.95
    assert NamePostProcessor({}, echo_ratio=0.95)(tr(text), recent) is not None
    assert NamePostProcessor({}, echo_ratio=0.6)(tr(text), recent) is None
    assert echo_similarity("วันนี้อากาศดีมาก", recent) == 1.0  # a window matches exactly


def test_echo_similarity_ignores_spaces_and_punctuation() -> None:
    assert echo_similarity("สวัสดี ค่ะ!", "สวัสดีค่ะ") == 1.0
    assert echo_similarity("", "สวัสดี") == 0.0
    assert echo_similarity("abc", "") == 0.0
    assert echo_similarity("กินข้าวหรือยัง", "Minecraft เป็นเกมที่สนุก") < 0.4
