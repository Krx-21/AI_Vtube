"""Tests for the config module."""

from ai_vtube.config import (
    DEFAULT_CACHE_SIZE,
    DEFAULT_MODEL,
    ERROR_PATTERNS,
    PAILIN_SYSTEM_PROMPT_MODEL,
    PAILIN_SYSTEM_PROMPT_USER,
    setup_logging,
)


class TestConfig:
    """Tests for configuration constants."""

    def test_default_model_is_set(self):
        assert DEFAULT_MODEL == "gemini-2.0-flash"

    def test_default_cache_size(self):
        assert isinstance(DEFAULT_CACHE_SIZE, int)
        assert DEFAULT_CACHE_SIZE > 0

    def test_system_prompt_not_empty(self):
        assert len(PAILIN_SYSTEM_PROMPT_USER) > 0
        assert len(PAILIN_SYSTEM_PROMPT_MODEL) > 0

    def test_system_prompt_contains_pailin(self):
        assert "ไพลิน" in PAILIN_SYSTEM_PROMPT_USER
        assert "ไพลิน" in PAILIN_SYSTEM_PROMPT_MODEL

    def test_error_patterns_is_list(self):
        assert isinstance(ERROR_PATTERNS, list)
        assert len(ERROR_PATTERNS) > 0

    def test_setup_logging_does_not_raise(self):
        # Should not raise any exceptions
        setup_logging()
