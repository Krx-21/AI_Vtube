"""EmotionTagExtractor (modules.json text.normalize; §4.6, §4.8 persona rules)."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from aivtube.config.schema import EMOTIONS
from aivtube.text.tags import EmotionTagExtractor, TagEvent

KNOWN = frozenset(EMOTIONS)


def run(deltas: list[str], **kwargs: str | None) -> tuple[list[TagEvent], str | None]:
    x = EmotionTagExtractor(KNOWN, **kwargs)
    out: list[TagEvent] = []
    for d in deltas:
        out += x.feed(d)
    out += x.flush()
    return out, x.last_emotion


def flatten(events: list[TagEvent]) -> tuple[str, list[tuple[int, str]]]:
    """Text plus (offset, emotion) changes: independent of how events were grouped."""
    text = ""
    changes: list[tuple[int, str]] = []
    for chunk, emotion in events:
        if emotion is not None:
            changes.append((len(text), emotion))
        text += chunk
    return text, changes


def test_tag_split_across_deltas_is_extracted() -> None:
    events, last = run(["[hap", "py] สวัสดีค่ะ"])
    assert events == [("สวัสดีค่ะ", "happy")]
    assert last == "happy"
    events, _ = run(["สวัสดี [", "s", "a", "d", "]", " ทุกคน"])
    assert flatten(events) == ("สวัสดี ทุกคน", [(7, "sad")])


def test_emits_only_on_change() -> None:
    events, last = run(["[happy] สวัสดี [happy] ค่ะ [sad]เสียใจ [sad] จัง [happy]เย้"])
    assert flatten(events) == ("สวัสดี ค่ะ เสียใจ จัง เย้", [(0, "happy"), (11, "sad"), (22, "happy")])
    assert last == "happy"


def test_initial_emotion_suppresses_a_repeat() -> None:
    events, last = run(["[neutral] สวัสดี"], current="neutral")
    assert events == [("สวัสดี", None)]
    assert last == "neutral"


def test_unknown_tags_are_stripped() -> None:
    events, _ = run(["สวัสดี [หัวเราะ] ทุกคน [แชท] [wink]โอเค []"])
    assert flatten(events) == ("สวัสดี ทุกคน โอเค ", [])


def test_known_tags_are_case_and_space_insensitive_and_canonical() -> None:
    events, last = run(["[ HAPPY ] ok [Surprised]ว้าว"])
    assert flatten(events) == ("ok ว้าว", [(0, "happy"), (3, "surprised")])
    assert last == "surprised"
    x = EmotionTagExtractor(["Happy", " sad ", ""])
    assert x.known == frozenset({"Happy", "sad"})
    assert x.feed("[happy]a") == [("a", "Happy")]


def test_brackets_that_are_not_tags_stay_text() -> None:
    long_body = "y" * (EmotionTagExtractor.MAX_TAG_CHARS + 5)
    events, _ = run([f"a [{long_body}] b"])
    assert flatten(events) == (f"a [{long_body}] b", [])
    events, _ = run(["a [b\nc] d"])
    assert flatten(events) == ("a [b\nc] d", [])
    events, _ = run(["[[happy]] ok"])
    assert flatten(events) == ("[] ok", [(1, "happy")])


def test_flush_drops_a_dangling_known_prefix_and_keeps_other_text() -> None:
    assert run(["สวัสดี [hap"])[0] == [("สวัสดี ", None)]
    assert run(["ราคา ["])[0] == [("ราคา ", None)]
    assert run(["ราคา [xyz"])[0] == [("ราคา ", None), ("[xyz", None)]


def test_whitespace_around_a_removed_tag_collapses() -> None:
    assert flatten(run(["สวัสดี [happy] ทุกคน"])[0])[0] == "สวัสดี ทุกคน"
    assert flatten(run(["[happy]  สวัสดี"])[0])[0] == "สวัสดี"
    assert flatten(run(["สวัสดี[happy] ทุกคน"])[0])[0] == "สวัสดี ทุกคน"
    assert flatten(run(["สวัสดี [happy]\nทุกคน"])[0])[0] == "สวัสดี \nทุกคน"


def test_emotion_change_at_delta_end_is_an_empty_event() -> None:
    x = EmotionTagExtractor(KNOWN)
    assert x.feed("ดีใจ [happy]") == [("ดีใจ ", None), ("", "happy")]
    assert x.feed(" มาก") == [("มาก", None)]


def test_reset_and_flush_keep_last_emotion() -> None:
    x = EmotionTagExtractor(KNOWN)
    x.feed("[shy] อาย [hap")
    x.reset()
    assert x.last_emotion == "shy"
    assert x.feed("py]") == [("py]", None)]  # the partial tag was forgotten
    assert x.feed("[shy] ต่อ") == [(" ต่อ", None)]  # unchanged emotion: no event
    assert x.flush() == []


TAGGED = st.lists(
    st.one_of(
        st.sampled_from(["[happy]", "[sad]", "[Shy]", "[ smug ]", "[หัวเราะ]", "[x", "]", "["]),
        st.sampled_from(["สวัสดี", "ค่ะ", " ", "\n", "ok", "เที่ยว", "น้ำ", "55", "a]b"]),
    ),
    max_size=30,
).map("".join)


@settings(max_examples=300, deadline=None)
@given(text=TAGGED, cuts=st.lists(st.integers(min_value=1, max_value=6), min_size=1, max_size=20))
def test_split_invariant(text: str, cuts: list[int]) -> None:
    whole, last_whole = run([text])
    deltas: list[str] = []
    pos = i = 0
    while pos < len(text):
        n = cuts[i % len(cuts)]
        deltas.append(text[pos : pos + n])
        pos += n
        i += 1
    split, last_split = run(deltas)
    assert flatten(split) == flatten(whole)
    assert last_split == last_whole
    assert "[happy]" not in flatten(whole)[0]
