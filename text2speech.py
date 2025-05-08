import os
import datetime
import tempfile
import glob
import pygame
from gtts import gTTS
import subprocess
from text_utils import remove_special_characters

class TextToSpeech:
    def __init__(self, voice_id=None, rate=1.0, volume=1.0, device_name="CABLE Input", cache_size=20):
        """
        Initialize the text-to-speech engine using Google Text-to-Speech

        Args:
            voice_id (str, optional): Not used with gTTS
            rate (float): Playback speed multiplier (1.0 is normal speed)
            volume (float): Volume level (0.0 to 1.0)
            device_name (str, optional): Name of the audio output device to use
                Default is "CABLE Input" (VB-Audio Virtual Cable)
                Set to None to use the system default output device
            cache_size (int): Maximum number of TTS responses to cache
        """
        # Store settings
        self.rate = rate
        self.volume = volume
        self.language = 'th'  # Default to Thai language
        self.tld = 'co.th'    # Use Thai TLD for more natural Thai voice
        self.slow = False     # Normal speed
        self.device_name = device_name

        # Initialize TTS cache
        self.cache = {}
        self.cache_size = cache_size
        self.cache_order = []  # To track LRU (least recently used)

        # Get available audio devices using our custom method
        self.devices = self.get_audio_devices()
        self.device_index = None

        # Find the device index for the specified device name
        if device_name:
            # First try to find an exact match
            for idx, name in self.devices:
                if device_name.lower() == name.lower():
                    self.device_index = idx
                    print(f"Using audio device: {name} (ID: {idx})")
                    break

            # If no exact match, try partial match
            if self.device_index is None:
                for idx, name in self.devices:
                    if device_name.lower() in name.lower():
                        self.device_index = idx
                        print(f"Using audio device: {name} (ID: {idx})")
                        break

            # Special case for VB-Audio Virtual Cable (CABLE Input)
            if self.device_index is None and device_name.lower() == "cable input":
                for idx, name in self.devices:
                    if "vb-audio" in name.lower() or "virtual cable" in name.lower():
                        self.device_index = idx
                        print(f"Using VB-Audio device: {name} (ID: {idx})")
                        break

            # If still not found, use default
            if self.device_index is None:
                print(f"Warning: Audio device '{device_name}' not found. Using system default.")
                print("Available devices:")
                for idx, name in self.devices:
                    print(f"  ID {idx}: {name}")

        # Initialize pygame mixer with the selected device
        # First make sure pygame mixer is not initialized
        if pygame.mixer.get_init():
            pygame.mixer.quit()

        # Initialize pygame mixer with default settings
        # We'll use the system's default audio device, but provide instructions for the user
        pygame.mixer.init()

        # If we found a CABLE Input device, print a message
        if self.device_index is not None:
            device_name = None
            for idx, name in self.devices:
                if idx == self.device_index:
                    device_name = name
                    break

            if device_name:
                print(f"\nTo route audio through {device_name}, please set it as your default audio device:")
                print("1. Right-click on the speaker icon in the system tray")
                print("2. Select 'Open Sound settings'")
                print("3. Under 'Choose your output device', select '{}'".format(device_name))
                print("4. The AI VTuber's voice will now be routed through this device\n")

        print("Using Google Text-to-Speech API for Thai language")

    def get_audio_devices(self):
        """
        Get a list of available audio output devices

        Returns:
            list: List of tuples (device_id, device_name)
        """
        devices = []

        try:
            cmd = ['powershell', '-Command',
                   "Get-CimInstance -ClassName Win32_SoundDevice | " +
                   "Where-Object { $_.Status -eq 'OK' } | " +
                   "Select-Object DeviceID, Name | Format-List"]

            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout

            current_id = 0
            for line in output.splitlines():
                if line.strip().startswith("Name"):
                    name = line.split(':', 1)[1].strip()
                    devices.append((current_id, name))
                    current_id += 1

        except Exception as e:
            print(f"Error getting audio devices: {e}")

        return devices

    def get_available_voices(self):
        """
        Get a list of available voices (dummy method for compatibility)

        Returns:
            list: Empty list since gTTS doesn't provide voice selection
        """
        # gTTS doesn't have voice selection like pyttsx3
        # It uses the Google TTS service which has fixed voices per language
        return []

    def print_available_voices(self):
        """Print information about available voices (limited with gTTS)"""
        print("Google Text-to-Speech provides fixed voices per language.")
        print("Currently using Thai language (th-TH) with a female voice.")
        print("To change language, modify the 'language' property.")

    def set_voice(self, voice_id):
        """
        Set the voice (dummy method for compatibility)

        Args:
            voice_id (str): Not used with gTTS
        """
        # gTTS doesn't support voice selection
        print("Voice selection not supported with Google Text-to-Speech.")
        print("The voice is determined by the language setting.")

    def set_voice_by_index(self, index):
        """
        Set the voice by index (dummy method for compatibility)

        Args:
            index (int): Not used with gTTS
        """
        # gTTS doesn't support voice selection
        return False

    def set_female_voice(self):
        """
        Set a female voice (always returns True with gTTS Thai)

        Returns:
            bool: Always True since Thai gTTS uses a female voice
        """
        # Thai gTTS already uses a female voice
        print("Google Text-to-Speech for Thai language uses a female voice by default.")
        return True

    def set_rate(self, rate):
        """
        Set the speech rate

        Args:
            rate (float): Playback speed multiplier
        """
        self.rate = rate

    def set_volume(self, volume):
        """
        Set the volume

        Args:
            volume (float): Volume level (0.0 to 1.0)
        """
        self.volume = volume
        pygame.mixer.music.set_volume(volume)

    def print_audio_devices(self):
        """Print information about all available audio output devices"""
        devices = self.get_audio_devices()

        print(f"Available audio output devices ({len(devices)}):")
        for idx, name in devices:
            print(f"{idx}. {name}")
        print("---")

    @staticmethod
    def print_device_setup_instructions():
        """Print instructions for setting up audio devices"""
        print("How to set CABLE Input as your default audio device in Windows:")
        print("1. Right-click on the speaker icon in the system tray")
        print("2. Select 'Open Sound settings'")
        print("3. Under 'Output', click on the dropdown menu")
        print("4. Select 'CABLE Input (VB-Audio Virtual Cable)'")
        print("5. Your text-to-speech audio will now be routed through CABLE Input")
        print()
        print("Note: After using the application, you may want to switch back to your regular speakers.")

    def _update_cache(self, text, file_path):
        """
        Update the TTS cache with a new entry

        Args:
            text (str): The text that was converted
            file_path (str): Path to the audio file
        """
        if text in self.cache:
            self.cache_order.remove(text)
        elif len(self.cache) >= self.cache_size and self.cache_order:
            oldest_text = self.cache_order[0]
            oldest_path = self.cache.pop(oldest_text)
            self.cache_order.pop(0)
            try:
                if os.path.exists(oldest_path) and oldest_path != file_path:
                    os.remove(oldest_path)
            except Exception:
                pass

        self.cache[text] = file_path
        self.cache_order.append(text)

    def speak(self, text):
        """
        Convert text to speech and play it

        Args:
            text (str): The text to convert to speech
        """
        import time
        import hashlib

        filtered_text = remove_special_characters(text)
        text = filtered_text

        if text in self.cache and os.path.exists(self.cache[text]):
            temp_path = self.cache[text]
            self.cache_order.remove(text)
            self.cache_order.append(text)
        else:
            text_hash = hashlib.md5(text.encode()).hexdigest()[:10]
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"tts_temp_{text_hash}.mp3")

            tts = gTTS(text=text, lang=self.language, tld=self.tld, slow=self.slow)
            tts.save(temp_path)

            self._update_cache(text, temp_path)

        try:
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.set_volume(self.volume)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            pygame.mixer.music.unload()
            time.sleep(0.1)

        except Exception as e:
            print(f"Error playing audio: {e}")
            if text in self.cache:
                self.cache.pop(text)
                if text in self.cache_order:
                    self.cache_order.remove(text)
                if temp_path != self.cache.get(text, None):
                    return self.speak(text)

    def cleanup_temp_files(self):
        """
        Clean up temporary audio files created by this module
        """
        import time

        try:
            pygame.mixer.music.unload()
        except:
            pass

        time.sleep(0.2)

        try:
            temp_dir = tempfile.gettempdir()
            temp_files = glob.glob(os.path.join(temp_dir, "tts_temp_*.mp3"))

            if os.path.exists("audio_responses"):
                temp_files.extend(glob.glob(os.path.join("audio_responses", "output_*.mp3")))

            files_deleted = 0
            for file in temp_files:
                try:
                    try:
                        with open(file, 'a'):
                            pass
                    except:
                        continue

                    os.remove(file)
                    files_deleted += 1
                except Exception:
                    pass

            if files_deleted > 0:
                print(f"Cleaned up {files_deleted} temporary audio files")
        except Exception as e:
            print(f"Error during cleanup: {e}")

# Example usage
if __name__ == "__main__":
    print("Text-to-Speech Module (Google TTS)")
    print("==================================")

    # Create TextToSpeech instance with default settings
    tts = TextToSpeech()

    # Print available audio devices
    print("\nAudio Output Devices:")
    tts.print_audio_devices()

    # Print setup instructions
    print("\nSetup Instructions:")
    tts.print_device_setup_instructions()

    # Print voice information
    print("\nVoice Information:")
    tts.print_available_voices()

    # Test message
    test_message = "สวัสดีค่า~ หนูเป็น AI VTuber ตัวน้อยที่พร้อมจะพูดคุยกับทุกคนแล้วนะคะ! วันนี้อารมณ์ดีมากเลยล่ะ มีอะไรให้หนูช่วยไหมคะ?"

    # Speak the text
    print("\nSpeaking test message...")
    tts.speak(test_message)

    # Clean up any temporary files
    tts.cleanup_temp_files()

    print("\nNote: Google Text-to-Speech requires an internet connection.")
    print("If you don't hear any audio, check your internet connection and audio device settings.")
