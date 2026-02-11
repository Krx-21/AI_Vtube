"""
Main entry point for the ไพลิน (Pailin) AI VTuber application.
"""

import argparse
import glob
import logging
import os
import random
import tempfile
import time

from dotenv import load_dotenv

from ai_vtube.chatbot import Chatbot
from ai_vtube.config import DEFAULT_CACHE_SIZE, ERROR_PATTERNS, setup_logging
from ai_vtube.speech_to_text import SpeechToText
from ai_vtube.text_to_speech import TextToSpeech
from ai_vtube.text_utils import remove_special_characters

load_dotenv()

logger = logging.getLogger(__name__)


class AIVtuber:
    def __init__(self, cache_size=DEFAULT_CACHE_SIZE):
        """
        Initialize the AI VTuber with all components.

        Args:
            cache_size (int): Maximum number of TTS responses to cache.
        """
        logger.info("Initializing AI VTuber...")

        self._cleanup_temp_files()

        logger.info("Initializing chatbot...")
        self.chatbot = Chatbot()

        logger.info("Initializing speech recognition...")
        self.speech_to_text = SpeechToText()

        logger.info("Initializing text-to-speech engine...")
        self.text_to_speech = TextToSpeech(cache_size=cache_size)

        if not self.text_to_speech.set_female_voice():
            logger.info("No female voice found, using default voice")
            voices = self.text_to_speech.get_available_voices()
            if voices:
                self.text_to_speech.set_voice_by_index(0)

        logger.info("AI VTuber initialized successfully!")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def adjust_for_ambient_noise(self):
        """Adjust microphone for ambient noise."""
        self.speech_to_text.adjust_for_ambient_noise()

    @staticmethod
    def _cleanup_temp_files():
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

    def cleanup_temp_files(self):
        """Clean up temporary audio files during runtime."""
        self.text_to_speech.cleanup_temp_files()

    # ------------------------------------------------------------------
    # Conversation loop
    # ------------------------------------------------------------------

    def listen_and_respond(self):
        """
        Listen for user input, process it, and respond.

        Returns:
            bool: True if the user wants to exit, False otherwise.
        """
        user_input = self.speech_to_text.listen_for_speech()

        is_error = any(pattern in user_input for pattern in ERROR_PATTERNS)
        if is_error:
            logger.info("Speech recognition error: %s", user_input)
            self.text_to_speech.speak(user_input)
            return False

        logger.info("You said: %s", user_input)

        if len(user_input) > 20:
            acknowledgments = ["อืมมมม~", "เข้าใจแล้วค่าา", "โอเคเบยยย", "รับทราบจ้าาา"]
            ack = random.choice(acknowledgments)
            logger.info("Acknowledgment: %s", ack)
            self.text_to_speech.speak(ack)

        response = self.chatbot.chat_with_gemini(user_input)
        logger.info("AI response: %s", response)
        self.text_to_speech.speak(response)

        if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
            return True

        return False

    def run(self):
        """Run the AI VTuber in a continuous loop."""
        self.cleanup_temp_files()

        welcome_messages = [
            "ฮายยย~ ไพลินพร้อมเมคเฟรนด์แล้วค่า มีอะไรอยากคุยป่ะเนี่ย",
            "ฮายยย~ ไพลินเองน้า วันนี้อารมณ์ดี๊ดี อยากคุยกับตัวเองจังเลยค่า",
            "ฮายยย~ หนูชื่อไพลิน ยินดีที่ได้รู้จักจ้า มีอะไรอยากเม้าท์มอยกันไหมเอ่ย",
        ]
        welcome = remove_special_characters(random.choice(welcome_messages))
        logger.info(welcome)
        self.text_to_speech.speak(welcome)

        self.adjust_for_ambient_noise()

        should_exit = False
        while not should_exit:
            try:
                should_exit = self.listen_and_respond()
                time.sleep(0.2)
            except KeyboardInterrupt:
                logger.info("Exiting...")
                break
            except Exception as e:
                logger.error("Error in main loop: %s", e)
                error_msg = remove_special_characters(
                    "อุ๊ย เหมือนจะมีอะไรติดขัดนิดหน่อยอ่า ไพลินขอโทษด้วยน้า~ ลองใหม่อีกทีได้ป่าวคะ"
                )
                self.text_to_speech.speak(error_msg)

        goodbye_messages = [
            "บายบายจ้า ไว้มาคุยกับไพลินใหม่น้าา~ เดี๋ยวไพลินรอเลย",
            "ขอบคุณที่มาคุยกับไพลินนะคะ บ๊ายบายค่า ไว้มาเม้าท์กันใหม่น้า",
            "แล้วเจอกันใหม่นะจ๊ะ บายๆ ค่า อย่าลืมกลับมาคุยกับไพลินอีกน้า คิดถึงแย่เลย~",
        ]
        goodbye = remove_special_characters(random.choice(goodbye_messages))
        logger.info(goodbye)
        self.text_to_speech.speak(goodbye)

        self.cleanup_temp_files()


def main():
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
        vtuber.run()
    except KeyboardInterrupt:
        logger.info("Exiting gracefully...")
    finally:
        vtuber.cleanup_temp_files()
        logger.info("Cleanup complete. Goodbye!")


if __name__ == "__main__":
    main()
