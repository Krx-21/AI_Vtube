"""
Configuration constants and shared settings for Pailin applications.
"""

import logging

# Logging configuration
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default Gemini model
DEFAULT_MODEL = "gemini-2.0-flash"

# Speech recognition settings
DEFAULT_LANGUAGE_STT = "th-TH"
DEFAULT_ENERGY_THRESHOLD = 300
DEFAULT_PAUSE_THRESHOLD = 0.7

# TTS settings
DEFAULT_TTS_LANGUAGE = "th"
DEFAULT_TTS_TLD = "co.th"
DEFAULT_CACHE_SIZE = 20


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application."""
    logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=level)
