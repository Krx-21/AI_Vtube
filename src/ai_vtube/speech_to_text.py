"""
Speech-to-text module using Google Speech Recognition API.
"""

import logging
import random

import speech_recognition as sr

from ai_vtube.config import (
    DEFAULT_ENERGY_THRESHOLD,
    DEFAULT_LANGUAGE_STT,
    DEFAULT_PAUSE_THRESHOLD,
)
from ai_vtube.text_utils import remove_special_characters

logger = logging.getLogger(__name__)


class SpeechToText:
    def __init__(
        self,
        language=DEFAULT_LANGUAGE_STT,
        energy_threshold=DEFAULT_ENERGY_THRESHOLD,
        pause_threshold=DEFAULT_PAUSE_THRESHOLD,
    ):
        """
        Initialize the speech-to-text converter.

        Args:
            language (str): The language code for speech recognition.
                Default is "th-TH" for Thai language.
            energy_threshold (int): Minimum audio energy to consider for recording.
            pause_threshold (float): Seconds of non-speaking audio before a phrase
                is considered complete.
        """
        self.recognizer = sr.Recognizer()
        self.language = language
        self.microphone = None

        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = pause_threshold

        self._initialize_microphone()

    def _initialize_microphone(self):
        """Initialize the microphone instance once to speed up subsequent uses."""
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.1)
            logger.info("Microphone initialized successfully")
        except Exception as e:
            logger.warning("Could not initialize microphone: %s", e)
            self.microphone = None

    def _get_source(self):
        """Return the microphone source (reuse stored or create new)."""
        return self.microphone if self.microphone else sr.Microphone()

    def adjust_for_ambient_noise(self, duration=1):
        """
        Adjust the recognizer sensitivity to ambient noise.

        Args:
            duration (int): The duration in seconds to sample ambient noise.
        """
        try:
            with self._get_source() as source:
                logger.info(
                    "Adjusting for ambient noise... (please be quiet for %d seconds)",
                    duration,
                )
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)
                logger.info(
                    "Ambient noise adjustment complete. Energy threshold: %s",
                    self.recognizer.energy_threshold,
                )
        except Exception as e:
            logger.error("Error adjusting for ambient noise: %s", e)
            self.recognizer.dynamic_energy_threshold = False
            self.recognizer.energy_threshold = DEFAULT_ENERGY_THRESHOLD

    def _random_error(self, messages):
        """Pick a random Pailin-style error message and filter it."""
        return remove_special_characters(random.choice(messages))

    def listen_for_speech(self, timeout=None, phrase_time_limit=None):
        """
        Listen for speech and convert it to text.

        Args:
            timeout (int, optional): Maximum seconds to wait before giving up.
            phrase_time_limit (int, optional): Maximum seconds for a phrase.

        Returns:
            str: The recognized text or a Pailin-style error message.
        """
        try:
            with self._get_source() as source:
                logger.info("Listening...")
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )

            logger.info("Processing speech...")
            return self.recognizer.recognize_google(audio, language=self.language)

        except sr.WaitTimeoutError:
            return self._random_error([
                "อืมม~ ไพลินไม่ได้ยินเยยอ่ะ ลองพูดอีกทีได้ป่าวคะ",
                "ตะเองพูดว่าอะไรน้า ไพลินฟังไม่ค่อยชัดเลย พูดอีกทีได้มั้ยอ่า",
                "เงียบจังเยย~ พูดอะไรหน่อยสิค้า ไพลินรอฟังอยู่น้า~",
            ])
        except sr.UnknownValueError:
            return self._random_error([
                "ขอโทษน้า~ ไพลินฟังไม่ค่อยชัดเยยอ่า พูดอีกทีให้ไพลินฟังหน่อยได้มั้ยคะ",
                "เมื่อกี้ว่าไงน้า ไพลินฟังไม่ทันเบย ช่วยพูดอีกทีให้ชัดๆ หน่อยจ้า",
                "ไพลินไม่ค่อยเข้าใจที่ตัวเองพูดอ่ะจิ ลองพูดใหม่อีกรอบได้ป่ะคะ",
            ])
        except sr.RequestError as e:
            logger.error("Speech recognition request error: %s", e)
            return self._random_error([
                "อุ๊ย เหมือนจะมีปัญหานิดหน่อยอ่ะค่ะ ลองใหม่อีกทีน้า ไพลินรอฟังอยู่จ้า",
                "ขอโทษด้วยน้า~ ระบบมีปัญหาจิ๊ดนึง ลองพูดอีกทีให้ไพลินฟังได้ป่าวคะ",
                "ไพลินเจอปัญหานิดหน่อยค่า ลองใหม่อีกทีได้มั้ยอ่า",
            ])
        except Exception as e:
            logger.error("Error in speech recognition: %s", e)
            return self._random_error([
                "แง~ ขอโทษน้า ไพลินเจอข้อผิดพลาดนิดหน่อยค่า ลองใหม่ได้ป่าวคะ",
                "ไพลินมีปัญหาจิ๊ดนึงอ่ะค่ะ ลองพูดอีกทีได้มั้ยอ่า",
                "อุ๊ยตาย ไพลินมีปัญหานิดหน่อยค่า ลองใหม่อีกทีนะจ๊ะ นะคะ",
            ])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    stt = SpeechToText()
    stt.adjust_for_ambient_noise()

    print("Say something...")
    text = stt.listen_for_speech()
    print(f"You said: {text}")
