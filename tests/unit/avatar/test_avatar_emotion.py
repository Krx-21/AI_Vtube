"""EmotionController: idempotent expression state machine and parameter baselines."""

from __future__ import annotations

from typing import Any

import pytest

from aivtube.avatar import EmotionController, EmotionSpec


def test_apply_returns_activate_and_deactivate(pailin_map: dict[str, Any]) -> None:
    ec = EmotionController(pailin_map)
    assert ec.current == "neutral" and ec.active == frozenset()
    assert ec.apply("happy") == (["exp_03.exp3.json"], [])
    assert ec.apply("happy") == ([], [])  # idempotent
    assert ec.apply("sad") == (["exp_05.exp3.json"], ["exp_03.exp3.json"])
    assert ec.apply("neutral") == ([], ["exp_05.exp3.json"])
    assert ec.active == frozenset()


def test_unknown_and_none_fall_back_to_neutral(pailin_map: dict[str, Any]) -> None:
    ec = EmotionController(pailin_map)
    ec.apply("happy")
    assert ec.apply("confused") == ([], ["exp_03.exp3.json"])
    assert ec.current == "neutral"
    assert ec.resolve(None) == "neutral"
    assert ec.resolve("HAPPY") == "happy"


def test_baseline_uses_smile_brows_and_params() -> None:
    ec = EmotionController(
        {
            "neutral": {},
            "shy": {"smile": 0.7, "brows": 0.4, "params": {"CheekPuff": 0.6}},
            "odd": {"smile": 7, "brows": "x"},  # clamped / ignored
        }
    )
    assert ec.baseline() == {"MouthSmile": 0.5, "Brows": 0.5}
    assert ec.baseline("shy") == {"MouthSmile": 0.7, "Brows": 0.4, "CheekPuff": 0.6}
    assert ec.baseline("odd") == {"MouthSmile": 1.0, "Brows": 0.5}


def test_empty_map_still_has_neutral() -> None:
    ec = EmotionController({})
    assert ec.known == frozenset({"neutral"})
    assert ec.apply("happy") == ([], [])
    assert ec.managed == frozenset()


def test_reconcile_against_model_state(pailin_map: dict[str, Any]) -> None:
    ec = EmotionController(pailin_map)
    ec.apply("happy")
    # VTS restarted: nothing active; a stale sad expression is on; exp_07 is not ours
    present = {"exp_03.exp3.json": False, "exp_05.exp3.json": True, "exp_07.exp3.json": True}
    assert ec.reconcile(present) == (["exp_03.exp3.json"], ["exp_05.exp3.json"])
    # the model lacks exp_03: skipped instead of failing with 651
    assert ec.reconcile({"exp_05.exp3.json": False}) == ([], [])


def test_spec_from_mapping_and_hotkey() -> None:
    spec = EmotionSpec.from_mapping(
        {"expressions": "a.exp3.json", "hotkey": "Wave", "params": {"X": 1, "B": True}}
    )
    assert spec.expressions == ("a.exp3.json",)
    assert spec.hotkey == "Wave"
    assert spec.params == {"X": 1.0}
    ec = EmotionController({"neutral": {}, "happy": spec})
    assert ec.hotkey("happy") == "Wave" and ec.hotkey() is None


@pytest.mark.parametrize("name", ["neutral", "happy", "sad"])
def test_character_config_map_loads(name: str) -> None:
    from pathlib import Path

    from aivtube.config import load_character

    root = Path(__file__).resolve().parents[3]
    char = load_character(root, "pailin")
    ec = EmotionController(char.emotion_map)
    assert name in ec.known
    assert 0.0 <= ec.baseline(name)["MouthSmile"] <= 1.0
