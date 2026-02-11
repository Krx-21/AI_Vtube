"""
Main entry point for the ไพลิน (Pailin) AI VTuber application.
"""

import argparse
import asyncio
import glob
import logging
import os
import random
import tempfile
import time

from dotenv import load_dotenv

from pailin_core.ai.chatbot import Chatbot
from pailin_core.config import DEFAULT_CACHE_SIZE, setup_logging
from pailin_core.speech.stt import ERROR_PATTERNS, SpeechToText
from pailin_core.speech.tts import TextToSpeech
from pailin_core.text.sanitizer import remove_special_characters
from pailin_vtube.audio.playback import cleanup_temp_files

load_dotenv()

logger = logging.getLogger(__name__)


class AIVtuber:
    """AI VTuber application orchestrating chatbot, STT, and TTS."""

    def __init__(self, cache_size: int = DEFAULT_CACHE_SIZE):
        """
        Initialize the AI VTuber with all components.

        Args:
            cache_size: Maximum number of TTS responses to cache.
        """
        logger.info("Initializing AI VTuber...")

        self._cleanup_temp_files()

        logger.info("Initializing chatbot...")
        self.chatbot = Chatbot()

        logger.info("Initializing speech recognition...")
        self.speech_to_text = SpeechToText()

        logger.info("Initializing text-to-speech engine...")
        self.text_to_speech = TextToSpeech(cache_size=cache_size)

        logger.info("AI VTuber initialized successfully!")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def adjust_for_ambient_noise(self) -> None:
        """Adjust microphone for ambient noise."""
        self.speech_to_text.adjust_for_ambient_noise()

    @staticmethod
    def _cleanup_temp_files() -> None:
        """Clean up temporary audio files before initialization."""
        try:
            temp_dir = tempfile.gettempdir()
            temp_files = glob.glob(os.path.join(temp_dir, "tts_temp_*.mp3"))

            if os.path.exists("audio_responses"):
                temp_files.extend(
                    glob.glob(os.path.join("audio_responses", "output_*.mp3"))
                )

            files_deleted = 0
            for filepath in temp_files:
                try:
                    os.remove(filepath)
                    files_deleted += 1
                except OSError:
                    pass

            if files_deleted > 0:
                logger.info(
                    "Cleaned up %d temporary audio files during startup", files_deleted
                )
        except Exception as e:
            logger.warning("Error during initial cleanup: %s", e)

    def cleanup(self) -> None:
        """Clean up temporary audio files during runtime."""
        cleanup_temp_files()

    # ------------------------------------------------------------------
    # Conversation loop
    # ------------------------------------------------------------------

    async def listen_and_respond(self) -> bool:
        """
        Listen for user input, process it, and respond.

        Returns:
            True if the user wants to exit, False otherwise.
        """
        user_input = await self.speech_to_text.listen_for_speech()

        is_error = any(pattern in user_input for pattern in ERROR_PATTERNS)
        if is_error:
            logger.info("Speech recognition error: %s", user_input)
            await self.text_to_speech.speak(user_input)
            return False

        logger.info("You said: %s", user_input)

        if len(user_input) > 20:
            acknowledgments = ["อืมมมม~", "เข้าใจแล้วค่าา", "โอเคเบยยย", "รับทราบจ้าาา"]
            ack = random.choice(acknowledgments)
            logger.info("Acknowledgment: %s", ack)
            await self.text_to_speech.speak(ack)

        response = await self.chatbot.chat_with_gemini(user_input)
        logger.info("AI response: %s", response)
        await self.text_to_speech.speak(response)

        if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
            return True

        return False

    async def run(self) -> None:
        """Run the AI VTuber in a continuous async loop."""
        self.cleanup()

        welcome_messages = [
            "ฮายยย~ ไพลินพร้อมเมคเฟรนด์แล้วค่า มีอะไรอยากคุยป่ะเนี่ย",
            "ฮายยย~ ไพลินเองน้า วันนี้อารมณ์ดี๊ดี อยากคุยกับตัวเองจังเลยค่า",
            "ฮายยย~ หนูชื่อไพลิน ยินดีที่ได้รู้จักจ้า มีอะไรอยากเม้าท์มอยกันไหมเอ่ย",
        ]
        welcome = remove_special_characters(random.choice(welcome_messages))
        logger.info(welcome)
        await self.text_to_speech.speak(welcome)

        self.adjust_for_ambient_noise()

        should_exit = False
        while not should_exit:
            try:
                should_exit = await self.listen_and_respond()
                await asyncio.sleep(0.2)
            except KeyboardInterrupt:
                logger.info("Exiting...")
                break
            except Exception as e:
                logger.error("Error in main loop: %s", e)
                error_msg = remove_special_characters(
                    "อุ๊ย เหมือนจะมีอะไรติดขัดนิดหน่อยอ่า ไพลินขอโทษด้วยน้า~ ลองใหม่อีกทีได้ป่าวคะ"
                )
                await self.text_to_speech.speak(error_msg)

        goodbye_messages = [
            "บายบายจ้า ไว้มาคุยกับไพลินใหม่น้าา~ เดี๋ยวไพลินรอเลย",
            "ขอบคุณที่มาคุยกับไพลินนะคะ บ๊ายบายค่า ไว้มาเม้าท์กันใหม่น้า",
            "แล้วเจอกันใหม่นะจ๊ะ บายๆ ค่า อย่าลืมกลับมาคุยกับไพลินอีกน้า คิดถึงแย่เลย~",
        ]
        goodbye = remove_special_characters(random.choice(goodbye_messages))
        logger.info(goodbye)
        await self.text_to_speech.speak(goodbye)

        self.cleanup()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="AI VTuber Application")
    parser.add_argument(
        "--cache-size",
        type=int,
        default=DEFAULT_CACHE_SIZE,
        help="Maximum number of TTS responses to cache",
    )
    args = parser.parse_args()

    setup_logging()

    if not os.getenv("GEMINI_API_KEY"):
        logger.error(
            "GEMINI_API_KEY environment variable not set. "
            "Please create a .env file with your Gemini API key."
        )
        raise SystemExit(1)

    vtuber = AIVtuber(cache_size=args.cache_size)

    try:
        asyncio.run(vtuber.run())
    except KeyboardInterrupt:
        logger.info("Exiting gracefully...")
    finally:
        vtuber.cleanup()
        logger.info("Cleanup complete. Goodbye!")


if __name__ == "__main__":
    main()
