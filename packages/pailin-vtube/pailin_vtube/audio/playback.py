"""
Audio playback utilities.
"""

import contextlib
import glob
import logging
import os
import tempfile

import pygame

logger = logging.getLogger(__name__)


def init_pygame_mixer() -> None:
    """Initialize pygame mixer for audio playback."""
    if pygame.mixer.get_init():
        pygame.mixer.quit()
    pygame.mixer.init()


def cleanup_temp_files() -> None:
    """Clean up temporary audio files created by TTS modules."""
    with contextlib.suppress(pygame.error):
        pygame.mixer.music.unload()

    try:
        temp_dir = tempfile.gettempdir()
        temp_files = glob.glob(os.path.join(temp_dir, "tts_temp_*.mp3"))
        if os.path.exists("audio_responses"):
            temp_files.extend(glob.glob(os.path.join("audio_responses", "output_*.mp3")))

        files_deleted = 0
        for filepath in temp_files:
            try:
                os.remove(filepath)
                files_deleted += 1
            except OSError:
                pass

        if files_deleted > 0:
            logger.info("Cleaned up %d temporary audio files", files_deleted)
    except Exception as e:
        logger.error("Error during cleanup: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cleanup_temp_files()
