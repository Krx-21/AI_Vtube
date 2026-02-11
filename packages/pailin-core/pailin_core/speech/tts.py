"""
Text-to-speech module using edge-tts for async support.
"""

import hashlib
import logging
import os
import tempfile
from typing import Optional

import edge_tts
import pygame

from pailin_core.config import DEFAULT_CACHE_SIZE, DEFAULT_TTS_LANGUAGE
from pailin_core.text.sanitizer import remove_special_characters

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Text-to-speech engine using Microsoft Edge TTS for async support."""

    def __init__(
        self,
        voice: str = "th-TH-PremwadeeNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        cache_size: int = DEFAULT_CACHE_SIZE,
    ):
        """
        Initialize the text-to-speech engine using Edge TTS.

        Args:
            voice: Voice identifier. Default is "th-TH-PremwadeeNeural" (Thai female).
            rate: Speech rate adjustment (e.g., "+20%", "-10%").
            volume: Volume adjustment (e.g., "+50%", "-25%").
            cache_size: Maximum number of TTS responses to cache.
        """
        self.voice = voice
        self.rate = rate
        self.volume_adjust = volume
        self.language = DEFAULT_TTS_LANGUAGE

        # LRU cache for TTS audio files
        self.cache: dict[str, str] = {}
        self.cache_size = cache_size
        self.cache_order: list[str] = []

        # Initialize pygame mixer
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        pygame.mixer.init()

        logger.info("Using Microsoft Edge TTS for Thai language")

    def _update_cache(self, text: str, file_path: str) -> None:
        """Add or refresh *text* → *file_path* in the LRU cache."""
        if text in self.cache:
            self.cache_order.remove(text)
        elif len(self.cache) >= self.cache_size and self.cache_order:
            oldest_text = self.cache_order.pop(0)
            oldest_path = self.cache.pop(oldest_text)
            try:
                if os.path.exists(oldest_path) and oldest_path != file_path:
                    os.remove(oldest_path)
            except OSError:
                pass

        self.cache[text] = file_path
        self.cache_order.append(text)

    async def speak(self, text: str) -> None:
        """
        Convert text to speech and play it.

        Args:
            text: The text to convert to speech.
        """
        filtered_text = remove_special_characters(text)

        if filtered_text in self.cache and os.path.exists(self.cache[filtered_text]):
            temp_path = self.cache[filtered_text]
            self.cache_order.remove(filtered_text)
            self.cache_order.append(filtered_text)
        else:
            text_hash = hashlib.md5(filtered_text.encode()).hexdigest()[:10]
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"tts_temp_{text_hash}.mp3")

            # Generate speech using edge-tts
            communicate = edge_tts.Communicate(
                filtered_text, self.voice, rate=self.rate, volume=self.volume_adjust
            )
            await communicate.save(temp_path)
            self._update_cache(filtered_text, temp_path)

        try:
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()

            clock = pygame.time.Clock()
            while pygame.mixer.music.get_busy():
                clock.tick(20)

            pygame.mixer.music.unload()

        except Exception as e:
            logger.error("Error playing audio: %s", e)
            if filtered_text in self.cache:
                self.cache.pop(filtered_text, None)
                if filtered_text in self.cache_order:
                    self.cache_order.remove(filtered_text)

    def set_voice(self, voice: str) -> None:
        """Set the TTS voice."""
        self.voice = voice

    def set_rate(self, rate: str) -> None:
        """Set the speech rate (e.g., '+20%', '-10%')."""
        self.rate = rate

    def set_volume(self, volume: str) -> None:
        """Set the volume adjustment (e.g., '+50%', '-25%')."""
        self.volume_adjust = volume

    @staticmethod
    async def get_available_voices() -> list[dict]:
        """Get list of available voices from Edge TTS."""
        voices = await edge_tts.list_voices()
        return voices

    @staticmethod
    async def print_available_voices() -> None:
        """Print available Thai voices."""
        voices = await edge_tts.list_voices()
        thai_voices = [v for v in voices if v["Locale"].startswith("th")]

        print("Available Thai voices:")
        for voice in thai_voices:
            print(
                f"  {voice['ShortName']}: {voice['FriendlyName']} "
                f"({voice['Gender']})"
            )


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def main() -> None:
        print("Text-to-Speech Module (Edge TTS)")
        print("================================")

        tts = TextToSpeech()

        print("\nAvailable Voices:")
        await tts.print_available_voices()

        test_message = (
            "ฮายยย~! ไพลินเองจ้า! พร้อมเมคเฟรนด์แล้วน้า! "
            "มีอะไรอยากเม้าท์มอยหรือปรึกษาไพลินไหมเอ่ย? "
            "บอกมาได้เล้ย! วันนี้ไพลินอารมณ์ดี๊ดี อยากหาเพื่อนคุยสุดๆ ไปเลยค่า!"
        )

        print("\nSpeaking test message...")
        await tts.speak(test_message)

    asyncio.run(main())
