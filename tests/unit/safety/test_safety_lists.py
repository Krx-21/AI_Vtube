"""Filter list files: the compact and table formats, validation errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivtube.safety.lists import (
    CATEGORIES,
    FilterListError,
    RuleSet,
    load_rule_dir,
    load_rule_file,
)


def write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_base_lists_parse_and_cover_every_category(base_dir: Path) -> None:
    rules = load_rule_dir(base_dir, required=True)
    counts = rules.counts()
    assert set(counts) == set(CATEGORIES)
    assert all(n > 0 for n in counts.values()), counts
    assert rules.replace, "replace.toml must define speech patches"
    for path in rules.files:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# config/filters/base/"), f"{path.name} needs a header"


def test_compact_form(tmp_path: Path) -> None:
    rules = load_rule_file(
        write(
            tmp_path / "a.toml",
            """
category = "slur"
token = ["คำหนึ่ง"]
substring = ["badword"]
regex = ['b[a4]d']
allow = ["คำดี"]
""",
        )
    )
    assert [(r.text, r.match, r.category) for r in rules.deny] == [
        ("คำหนึ่ง", "token", "slur"),
        ("badword", "substring", "slur"),
        ("b[a4]d", "regex", "slur"),
    ]
    assert rules.allow == ("คำดี",)
    assert rules.deny[0].source == "a.toml"


def test_table_form_matches_the_character_overlay_format(tmp_path: Path) -> None:
    rules = load_rule_file(
        write(
            tmp_path / "filters.toml",
            """
[[allow]]
text = "หีบ"

[[deny]]
category = "sexual"
match = "substring"
text = "xyz"

[[deny]]
category = "slur"
text = "คำใหม่"

[[replace]]
match = "regex"
text = "(?i)as an ai language model"
with = ""

[[replace]]
match = "substring"
text = "ไพลินเป็นบอท"
with = "ไพลิน"
directions = ["out", "memory"]
""",
        )
    )
    assert rules.allow == ("หีบ",)
    assert [(r.category, r.match) for r in rules.deny] == [
        ("sexual", "substring"),
        ("slur", "token"),
    ]
    assert rules.replace[0].directions == frozenset({"out"})
    assert rules.replace[1].directions == frozenset({"out", "memory"})
    assert rules.replace[1].match == "substring"


def test_the_committed_character_overlay_parses() -> None:
    repo = Path(__file__).resolve().parents[3]
    rules = load_rule_file(repo / "characters" / "pailin" / "filters.toml")
    assert rules == RuleSet(files=(repo / "characters" / "pailin" / "filters.toml",))


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ('category = "slurs"\ntoken = ["x"]', "unknown category"),
        ('token = ["x"]', "file-level category"),
        ('category = "slur"\nwords = ["x"]', "unknown keys"),
        ("category = \"slur\"\nregex = ['(unclosed']", "bad regex"),
        ('category = "pii"\ntoken = ["x"]', "substring or regex"),
        ('[[deny]]\ncategory = "slur"\nmatch = "fuzzy"\ntext = "x"', "match must be"),
        ('[[deny]]\ncategory = "slur"', "text is required"),
        ('[[replace]]\ntext = "x"\ndirections = ["sideways"]', "directions"),
        ('[[replace]]\nmatch = "glob"\ntext = "x"', "regex or substring"),
        ('category = "slur"\ntoken = [1, 2]', "list of strings"),
        ("category = [", "invalid TOML"),
    ],
)
def test_invalid_files_are_rejected(tmp_path: Path, body: str, fragment: str) -> None:
    path = write(tmp_path / "bad.toml", body)
    with pytest.raises(FilterListError) as info:
        load_rule_file(path)
    assert fragment in str(info.value)
    assert info.value.path == str(path)
    assert info.value.message_th


def test_directories(tmp_path: Path) -> None:
    assert load_rule_dir(tmp_path / "missing", required=False) == RuleSet()
    with pytest.raises(FilterListError):
        load_rule_dir(tmp_path / "missing", required=True)
    write(tmp_path / "b.toml", 'category = "slur"\ntoken = ["b"]')
    write(tmp_path / "a.toml", 'category = "slur"\ntoken = ["a"]')
    write(tmp_path / "notes.txt", "ignored")
    rules = load_rule_dir(tmp_path, required=True)
    assert [r.text for r in rules.deny] == ["a", "b"]  # file-name order
    with pytest.raises(FilterListError):
        load_rule_file(tmp_path / "nope.toml")


def test_unreadable_file_is_a_filter_list_error(tmp_path: Path) -> None:
    folder = tmp_path / "folder.toml"
    folder.mkdir()
    with pytest.raises(FilterListError):
        load_rule_file(folder)
