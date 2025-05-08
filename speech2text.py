import speech_recognition as sr
from text_utils import remove_special_characters

class SpeechToText:
    def __init__(self, language="th-TH", energy_threshold=300, pause_threshold=0.7):
        """
        Initialize the speech-to-text converter

        Args:
            language (str): The language code for speech recognition
                Default is "th-TH" for Thai language
                Other options include:
                - "en-US" for English (United States)
                - "ja-JP" for Japanese
                - "zh-CN" for Chinese (Simplified)
            energy_threshold (int): Minimum audio energy to consider for recording
                Higher values mean the microphone is less sensitive
                Default is 300, which is a good starting point
            pause_threshold (float): Seconds of non-speaking audio before a phrase is considered complete
                Default is 0.7 (reduced from 1.0 for more responsive conversation)
        """
        self.recognizer = sr.Recognizer()
        self.language = language

        # Store the microphone instance
        self.microphone = None

        # Configure recognizer settings
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = pause_threshold

        # Initialize the microphone once to speed up subsequent uses
        self._initialize_microphone()

    def _initialize_microphone(self):
        """
        Initialize the microphone instance once to avoid repeated initialization
        """
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.1)
            print("Microphone initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize microphone: {e}")
            self.microphone = None

    def adjust_for_ambient_noise(self, duration=1):
        """
        Adjust the recognizer sensitivity to ambient noise

        Args:
            duration (int): The duration in seconds to sample ambient noise
        """
        try:
            if self.microphone:
                with self.microphone as source:
                    print(f"Adjusting for ambient noise... (please be quiet for {duration} seconds)")
                    self.recognizer.adjust_for_ambient_noise(source, duration=duration)
                    print("Ambient noise adjustment complete.")
                    print(f"Energy threshold set to {self.recognizer.energy_threshold}")
            else:
                with sr.Microphone() as source:
                    print(f"Adjusting for ambient noise... (please be quiet for {duration} seconds)")
                    self.recognizer.adjust_for_ambient_noise(source, duration=duration)
                    print("Ambient noise adjustment complete.")
                    print(f"Energy threshold set to {self.recognizer.energy_threshold}")
        except Exception as e:
            print(f"Error adjusting for ambient noise: {e}")
            self.recognizer.dynamic_energy_threshold = False
            self.recognizer.energy_threshold = 300

    def listen_for_speech(self, timeout=None, phrase_time_limit=None):
        """
        Listen for speech and convert it to text

        Args:
            timeout (int, optional): Maximum seconds to wait before giving up
            phrase_time_limit (int, optional): Maximum seconds for a phrase

        Returns:
            str: The recognized text or error message
        """
        try:
            if self.microphone:
                with self.microphone as source:
                    print("Listening...")
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            else:
                with sr.Microphone() as source:
                    print("Listening...")
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

            print("Processing speech...")
            text = self.recognizer.recognize_google(audio, language=self.language)
            return text

        except sr.WaitTimeoutError:
            # Shorter, more natural error messages for better flow
            error_messages = [
                "อืมม ไพลินไม่ได้ยินเลยอ่ะ ลองพูดอีกทีป่ะ?",
                "ไพลินฟังไม่ได้ยินเลย พูดอีกทีได้ป่ะ?",
                "เงียบจังเลย ลองพูดใหม่ให้ไพลินฟังนะจ๊ะ"
            ]
            import random
            error_msg = random.choice(error_messages)
            return remove_special_characters(error_msg)
        except sr.UnknownValueError:
            error_messages = [
                "ขอโทษนะ ไพลินฟังไม่ชัดอ่ะ พูดอีกทีได้ป่ะ?",
                "ไพลินฟังไม่ทันอ่ะ ช่วยพูดอีกทีนะจ๊ะ",
                "ไพลินไม่เข้าใจที่พูดอ่ะ ลองพูดใหม่ได้ป่ะ?"
            ]
            import random
            error_msg = random.choice(error_messages)
            return remove_special_characters(error_msg)
        except sr.RequestError as e:
            error_messages = [
                "มีปัญหานิดหน่อยอ่ะ ลองใหม่อีกทีนะ ไพลินรอฟังอยู่",
                "ขอโทษนะ มีปัญหาเล็กน้อย ลองพูดอีกทีให้ไพลินฟังป่ะ?",
                "ไพลินมีปัญหาเล็กน้อยอ่ะ ลองใหม่ได้ป่ะ?"
            ]
            import random
            error_msg = random.choice(error_messages)
            return remove_special_characters(error_msg)
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            error_messages = [
                "ขอโทษนะ ไพลินมีข้อผิดพลาด ลองใหม่ป่ะ?",
                "ไพลินมีปัญหาเล็กน้อยอ่ะ ลองพูดอีกทีได้ป่ะ?",
                "อุย ไพลินมีปัญหานิดหน่อย ลองใหม่อีกทีนะจ๊ะ"
            ]
            import random
            error_msg = random.choice(error_messages)
            return remove_special_characters(error_msg)

# Example usage
if __name__ == "__main__":
    speech_to_text = SpeechToText()

    # Adjust for ambient noise
    speech_to_text.adjust_for_ambient_noise()

    print("Say something...")
    text = speech_to_text.listen_for_speech()

    print(f"You said: {text}")