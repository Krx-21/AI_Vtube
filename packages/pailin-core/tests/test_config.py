"""Tests for configuration module."""

import logging

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

    def test_setup_logging(self) -> None:
        # Just verify it doesn't raise an exception
        setup_logging(level=logging.INFO)
        logger = logging.getLogger("test_logger")
        logger.info("Test message")
        # Test passes if no exception is raised
