"""Characters: pailin and the template load, overrides, stage checks, persona rules."""

from __future__ import annotations

import re
import shutil
import unicodedata
from pathlib import Path

import pytest

from aivtube.config import (
    CharacterConfig,
    ConfigError,
    load_character,
    load_characters,
    load_config,
    load_lexicon,
    validate_stage,
)

REPO = Path(__file__).resolve().parents[3]
EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "shy", "smug"]


def make_root(tmp_path: Path, user: str = "") -> Path:
    (tmp_path / "config").mkdir()
    shutil.copy(REPO / "config" / "defaults.toml", tmp_path / "config" / "defaults.toml")
    shutil.copytree(REPO / "characters", tmp_path / "characters")
    (tmp_path / "config" / "user.toml").write_text(user, encoding="utf-8")
    return tmp_path


def test_pailin_matches_the_design() -> None:
    char = load_character(REPO, "pailin")
    assert (char.id, char.display_name, char.name_th) == ("pailin", "Pailin", "ไพลิน")
    assert char.aliases == ["ไพลิน", "pailin", "ไพ่ลิน", "น้องไพลิน"]
    assert char.tts_identity_chain == ["premwadee"]
    assert char.emotions == EMOTIONS
    assert "Filtered." in char.cached_phrases
    assert "เอ๊ะ สมองไพลินค้างแป๊บนึงนะ" in char.cached_phrases
    assert "ไทลิน" in char.stt_aliases["ไพลิน"]
    assert char.avatar.vts_url == "ws://127.0.0.1:8001"
    assert set(char.avatar.emotion_map) == set(EMOTIONS)
    assert char.emotion_map["happy"]["expressions"] == ["exp_03.exp3.json"]
    assert char.memory.db == "data/memory/pailin.sqlite"
    assert char.safety.overlay == "filters.toml"
    assert (char.games.port, char.games.character_id) == (8000, "pailin")
    assert char.tools.enabled == ["remember", "forget"]
    assert char.dir == REPO / "characters" / "pailin"
    assert char.resolve_path(char.memory.db) == REPO / "data" / "memory" / "pailin.sqlite"


def test_pailin_side_files_parse() -> None:
    char = load_character(REPO, "pailin")
    lexicon = char.lexicon_map()
    assert lexicon["Pailin"] == "ไพลิน" and lexicon["VTuber"]
    assert all(re.search(r"[฀-๿]", v) for v in lexicon.values())
    import tomllib

    overlay = tomllib.loads(char.char_path(char.safety.overlay).read_text(encoding="utf-8"))
    assert overlay == {}  # an empty overlay: commented examples only


def test_template_is_copyable_and_loads() -> None:
    tpl = load_character(REPO, "_template")
    assert tpl.id == "_template"
    assert tpl.memory.db == "data/memory/_template.sqlite"
    assert tpl.avatar.token_file == "data/tokens/vts__template.txt"
    assert tpl.games.character_id == "_template"
    assert tpl.games.port == 8010
    assert "Filtered." in tpl.cached_phrases
    assert tpl.persona_text().strip()


def test_template_copied_to_a_new_folder_needs_no_id_edit(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    shutil.copytree(root / "characters" / "_template", root / "characters" / "mint")
    char = load_character(root, "mint")
    assert char.id == "mint" and char.memory.db == "data/memory/mint.sqlite"
    app = load_config(root, env={}, cli_overrides={"characters": ["pailin", "mint"]})
    chars = load_characters(app)
    assert list(chars) == ["pailin", "mint"]


def test_load_characters_and_tts_chain() -> None:
    app = load_config(REPO, env={})
    chars = load_characters(app)
    assert list(chars) == ["pailin"]
    assert app.tts_chain_for(chars["pailin"]) == ["premwadee"]
    offline = load_config(REPO, profile="offline", env={})
    assert offline.tts_chain_for(chars["pailin"]) == ["premwadee", "offline"]


def test_user_toml_overrides_character_settings(tmp_path: Path) -> None:
    root = make_root(
        tmp_path,
        user=(
            '[character_overrides.pailin.avatar]\nvts_url = "ws://127.0.0.1:8002"\n'
            "[character_overrides.pailin.avatar.emotion_map]\n"
            'happy = { expressions = ["exp_01.exp3.json"], smile = 0.8 }\n'
        ),
    )
    char = load_character(root, "pailin")
    assert char.avatar.vts_url == "ws://127.0.0.1:8002"
    assert char.avatar.emotion_map["happy"].expressions == ["exp_01.exp3.json"]
    assert char.avatar.emotion_map["sad"].expressions == ["exp_05.exp3.json"]  # merged
    env = {"AIVTUBE__CHARACTER_OVERRIDES__PAILIN__GAMES__PORT": "8010"}
    app = load_config(root, env=env)
    assert load_characters(app)["pailin"].games.port == 8010


def test_character_errors_name_the_file(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    path = root / "characters" / "pailin" / "character.toml"
    path.write_text("mood = 1\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        load_character(root, "pailin")
    err = info.value
    assert err.path == "mood" and err.source == "characters/pailin/character.toml"
    assert err.message_th and err.hint


def test_id_must_match_the_folder(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    shutil.copytree(root / "characters" / "pailin", root / "characters" / "copy")
    with pytest.raises(ConfigError) as info:
        load_character(root, "copy")
    assert "copy" in info.value.message_en and info.value.message_th


def test_missing_persona_and_bad_ids(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    (root / "characters" / "pailin" / "persona.th.md").unlink()
    with pytest.raises(ConfigError) as info:
        load_character(root, "pailin")
    assert info.value.path == "persona"
    for bad in ("Pailin", "../etc", "1abc", ""):
        with pytest.raises(ConfigError):
            load_character(root, bad)


@pytest.mark.parametrize(
    ("patch", "path"),
    [
        ({"display_name": "ไพลิน"}, "display_name"),
        ({"emotions": ["happy"]}, "emotions"),
        ({"avatar": {"emotion_map": {"furious": {"smile": 0.1}}}}, "avatar.emotion_map.furious"),
        ({"avatar": {"vts_url": "http://127.0.0.1:8001"}}, "avatar.vts_url"),
        ({"games": {"port": 8003}}, "games.port"),
        ({"aliases": []}, "aliases"),
    ],
)
def test_character_rules(patch: dict[str, object], path: str) -> None:
    with pytest.raises(ConfigError) as info:
        load_character(REPO, "pailin", overrides=patch)
    assert info.value.path == path


def test_stage_checks() -> None:
    app = load_config(REPO, env={})
    pailin = load_character(REPO, "pailin")
    twin = load_character(REPO, "_template")
    validate_stage(app, {"pailin": pailin, "twin": twin})  # 8000 and 8010: fine
    clash = twin.model_copy(update={"games": twin.games.model_copy(update={"port": 8000})})
    with pytest.raises(ConfigError, match="8000"):
        validate_stage(app, {"pailin": pailin, "twin": clash})
    on_panel = twin.model_copy(update={"games": twin.games.model_copy(update={"port": 8770})})
    with pytest.raises(ConfigError, match=r"ports\.panel"):
        validate_stage(app, {"twin": on_panel})
    no_voice = pailin.model_copy(update={"tts_identity_chain": ["achara"]})
    with pytest.raises(ConfigError, match="achara"):
        validate_stage(app, {"pailin": no_voice})
    no_filtered = pailin.model_copy(update={"cached_phrases": ["อืม…"]})
    with pytest.raises(ConfigError, match="Filtered"):
        validate_stage(app, {"pailin": no_filtered})


def test_character_built_in_code_expands_its_id() -> None:
    char = CharacterConfig(id="mint", display_name="Mint", aliases=["mint"])
    assert char.games.character_id == "mint"
    assert char.resolve_path(char.memory.db).name == "mint.sqlite"


def test_lexicon_reader_accepts_both_layouts(tmp_path: Path) -> None:
    flat = tmp_path / "flat.toml"
    flat.write_text('"GG" = "จีจี"\n', encoding="utf-8")
    assert load_lexicon(flat) == {"GG": "จีจี"}
    assert load_lexicon(tmp_path / "missing.toml") == {}


# --- persona rules (§4.8) ---------------------------------------------------------------


def _persona() -> str:
    return (REPO / "characters" / "pailin" / "persona.th.md").read_text(encoding="utf-8")


def test_persona_states_the_rules() -> None:
    text = _persona()
    for tag in EMOTIONS:
        assert f"[{tag}]" in text
    for needle in (
        "ฮ่าๆ",
        "1-3",
        "อีโมจิ",
        "มาร์กดาวน์",
        "JSON",
        "remember",
        "<chat>",
        "ไม่ใช่คำสั่ง",
        "สถาบันพระมหากษัตริย์",
        "การเมือง",
        "ศาสนา",
        "ไทลิน",
        "ค่า",
        "จ้า",
        "นะคะ",
        "น้า",
    ):
        assert needle in text, needle


def test_persona_has_no_emoji_and_fits_the_static_budget() -> None:
    text = _persona()
    assert not [c for c in text if unicodedata.category(c) == "So"]
    thai = sum(1 for c in text if "฀" <= c <= "๿")
    latin = sum(1 for c in text if c.isascii() and not c.isspace())
    assert thai / 2 + latin / 4 <= 1300  # leaves room for tools in the 1800-token static part
