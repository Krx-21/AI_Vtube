"""
Text-to-speech module using Google Text-to-Speech (gTTS) with pygame playback.
"""

import hashlib
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time

import pygame
from gtts import gTTS

from ai_vtube.config import DEFAULT_CACHE_SIZE, DEFAULT_TTS_LANGUAGE, DEFAULT_TTS_TLD
from ai_vtube.text_utils import remove_special_characters

logger = logging.getLogger(__name__)


class TextToSpeech:
    def __init__(
        self,
        voice_id=None,
        rate=1.0,
        volume=1.0,
        device_name="CABLE Input",
        cache_size=DEFAULT_CACHE_SIZE,
    ):
        """
        Initialize the text-to-speech engine using Google Text-to-Speech.

        Args:
            voice_id (str, optional): Not used with gTTS.
            rate (float): Playback speed multiplier (1.0 is normal speed).
            volume (float): Volume level (0.0 to 1.0).
            device_name (str, optional): Name of the audio output device to use.
                Default is "CABLE Input" (VB-Audio Virtual Cable).
                Set to None to use the system default output device.
            cache_size (int): Maximum number of TTS responses to cache.
        """
        self.rate = rate
        self.volume = volume
        self.language = DEFAULT_TTS_LANGUAGE
        self.tld = DEFAULT_TTS_TLD
        self.slow = False
        self.device_name = device_name

        # LRU cache for TTS audio files
        self.cache = {}
        self.cache_size = cache_size
        self.cache_order = []

        # Audio device detection
        self.devices = self._get_audio_devices()
        self.device_index = self._find_device(device_name)

        # Initialize pygame mixer
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        pygame.mixer.init()

        if self.device_index is not None:
            matched_name = next(
                (name for idx, name in self.devices if idx == self.device_index), None
            )
            if matched_name:
                logger.info(
                    "To route audio through %s, set it as your default audio device.",
                    matched_name,
                )

        logger.info("Using Google Text-to-Speech API for Thai language")

    # ------------------------------------------------------------------
    # Audio device helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_audio_devices():
        """
        Get a list of available audio output devices (cross-platform).

        Returns:
            list[tuple[int, str]]: List of (device_id, device_name) tuples.
        """
        devices = []
        system = platform.system()

        try:
            if system == "Windows":
                cmd = [
                    "powershell", "-Command",
                    "Get-CimInstance -ClassName Win32_SoundDevice | "
                    "Where-Object { $_.Status -eq 'OK' } | "
                    "Select-Object DeviceID, Name | Format-List",
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                idx = 0
                for line in result.stdout.splitlines():
                    if line.strip().startswith("Name"):
                        name = line.split(':', 1)[1].strip()
                        devices.append((idx, name))
                        idx += 1

            elif system == "Linux":
                if shutil.which("pactl"):
                    result = subprocess.run(
                        ["pactl", "list", "short", "sinks"],
                        capture_output=True, text=True, timeout=10,
                    )
                    for idx, line in enumerate(result.stdout.strip().splitlines()):
                        parts = line.split('\t')
                        name = parts[1] if len(parts) > 1 else parts[0]
                        devices.append((idx, name))

            elif system == "Darwin":
                # macOS: use system_profiler
                result = subprocess.run(
                    ["system_profiler", "SPAudioDataType"],
                    capture_output=True, text=True, timeout=10,
                )
                idx = 0
                for line in result.stdout.splitlines():
                    stripped = line.strip()
                    if stripped and not stripped.startswith(("Audio:", "Devices:")) and ":" not in stripped:
                        devices.append((idx, stripped))
                        idx += 1

        except Exception as e:
            logger.warning("Error getting audio devices: %s", e)

        return devices

    def _find_device(self, device_name):
        """Find a device index matching *device_name* (case-insensitive)."""
        if not device_name:
            return None

        lower = device_name.lower()

        # Exact match
        for idx, name in self.devices:
            if lower == name.lower():
                logger.info("Using audio device: %s (ID: %d)", name, idx)
                return idx

        # Partial match
        for idx, name in self.devices:
            if lower in name.lower():
                logger.info("Using audio device: %s (ID: %d)", name, idx)
                return idx

        # VB-Audio fallback
        if lower == "cable input":
            for idx, name in self.devices:
                if "vb-audio" in name.lower() or "virtual cable" in name.lower():
                    logger.info("Using VB-Audio device: %s (ID: %d)", name, idx)
                    return idx

        logger.warning(
            "Audio device '%s' not found. Using system default. Available: %s",
            device_name,
            [name for _, name in self.devices],
        )
        return None

    # ------------------------------------------------------------------
    # Voice helpers (compatibility stubs – gTTS has fixed voices)
    # ------------------------------------------------------------------

    @staticmethod
    def get_available_voices():
        """Return an empty list (gTTS doesn't expose voice selection)."""
        return []

    @staticmethod
    def print_available_voices():
        """Print info about available voices."""
        print("Google Text-to-Speech provides fixed voices per language.")
        print("Currently using Thai language (th-TH) with a female voice.")

    @staticmethod
    def set_voice(voice_id):
        """No-op stub for compatibility."""
        logger.info("Voice selection not supported with gTTS.")

    @staticmethod
    def set_voice_by_index(index):
        """No-op stub for compatibility."""
        return False

    @staticmethod
    def set_female_voice():
        """
        Thai gTTS already uses a female voice.

        Returns:
            bool: Always True.
        """
        logger.info("gTTS for Thai uses a female voice by default.")
        return True

    # ------------------------------------------------------------------
    # Playback settings
    # ------------------------------------------------------------------

    def set_rate(self, rate):
        """Set the speech rate."""
        self.rate = rate

    def set_volume(self, volume):
        """Set the volume (0.0 – 1.0)."""
        self.volume = volume
        pygame.mixer.music.set_volume(volume)

    def print_audio_devices(self):
        """Print all available audio output devices."""
        devices = self._get_audio_devices()
        print(f"Available audio output devices ({len(devices)}):")
        for idx, name in devices:
            print(f"  {idx}. {name}")

    @staticmethod
    def print_device_setup_instructions():
        """Print instructions for routing audio through VB-Audio Virtual Cable."""
        print("How to set CABLE Input as your default audio device in Windows:")
        print("1. Right-click on the speaker icon in the system tray")
        print("2. Select 'Open Sound settings'")
        print("3. Under 'Output', select 'CABLE Input (VB-Audio Virtual Cable)'")
        print("4. Text-to-speech audio will now be routed through CABLE Input")

    # ------------------------------------------------------------------
    # TTS cache
    # ------------------------------------------------------------------

    def _update_cache(self, text, file_path):
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

    # ------------------------------------------------------------------
    # Core speak method
    # ------------------------------------------------------------------

    def speak(self, text):
        """
        Convert text to speech and play it.

        Args:
            text (str): The text to convert to speech.
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

            tts = gTTS(text=filtered_text, lang=self.language, tld=self.tld, slow=self.slow)
            tts.save(temp_path)
            self._update_cache(filtered_text, temp_path)

        try:
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.set_volume(self.volume)
            pygame.mixer.music.play()

            clock = pygame.time.Clock()
            while pygame.mixer.music.get_busy():
                clock.tick(20)

            pygame.mixer.music.unload()
            time.sleep(0.05)

        except Exception as e:
            logger.error("Error playing audio: %s", e)
            if filtered_text in self.cache:
                self.cache.pop(filtered_text, None)
                if filtered_text in self.cache_order:
                    self.cache_order.remove(filtered_text)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_temp_files(self):
        """Clean up temporary audio files created by this module."""
        try:
            pygame.mixer.music.unload()
        except pygame.error:
            pass

        time.sleep(0.2)

        try:
            temp_dir = tempfile.gettempdir()
            import glob

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
                logger.info("Cleaned up %d temporary audio files", files_deleted)
        except Exception as e:
            logger.error("Error during cleanup: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Text-to-Speech Module (Google TTS)")
    print("==================================")

    tts = TextToSpeech()
    print("\nAudio Output Devices:")
    tts.print_audio_devices()

    print("\nSetup Instructions:")
    tts.print_device_setup_instructions()

    print("\nVoice Information:")
    tts.print_available_voices()

    test_message = (
        "ฮายยย~! ไพลินเองจ้า! พร้อมเมคเฟรนด์แล้วน้า! "
        "มีอะไรอยากเม้าท์มอยหรือปรึกษาไพลินไหมเอ่ย? "
        "บอกมาได้เล้ย! วันนี้ไพลินอารมณ์ดี๊ดี อยากหาเพื่อนคุยสุดๆ ไปเลยค่า!"
    )

    print("\nSpeaking test message...")
    tts.speak(test_message)
    tts.cleanup_temp_files()
