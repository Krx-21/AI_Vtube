"""Tests for configuration module."""

import logging

import pytest

from pailin_core.config import (
    DEFAULT_CACHE_SIZE,
    DEFAULT_LANGUAGE_STT,
    DEFAULT_MODEL,
    setup_logging,
)


class TestConfig:
    """Tests for configuration constants and functions."""

    def test_default_model_is_set(self) -> None:
        assert DEFAULT_MODEL == "gemini-2.0-flash"

    def test_default_language_stt_is_thai(self) -> None:
        assert DEFAULT_LANGUAGE_STT == "th-TH"

    def test_default_cache_size_is_positive(self) -> None:
        assert DEFAULT_CACHE_SIZE > 0

    def test_setup_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        setup_logging(level=logging.INFO)
        logger = logging.getLogger("test")
        logger.info("Test message")
        assert "Test message" in caplog.text
