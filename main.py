import os
import time
import argparse
from dotenv import load_dotenv
from chatbot import Chatbot
from speech2text import SpeechToText
from text2speech import TextToSpeech
from text_utils import remove_special_characters

# Load environment variables
load_dotenv()

class AIVtuber:
    def __init__(self, cache_size=20):
        """
        Initialize the AI VTuber with all components

        Args:
            cache_size (int): Maximum number of TTS responses to cache
        """
        print("Initializing AI VTuber...")

        # Clean up any existing temporary files before starting
        self._cleanup_temp_files()

        # Initialize components
        print("Initializing chatbot...")
        self.chatbot = Chatbot()

        print("Initializing speech recognition...")
        self.speech_to_text = SpeechToText()

        print("Initializing text-to-speech engine...")
        self.text_to_speech = TextToSpeech(cache_size=cache_size)

        # Set up voice (use female voice if available)
        if not self.text_to_speech.set_female_voice():
            # If no female voice found, use the first available voice
            print("No female voice found, using default voice")
            voices = self.text_to_speech.get_available_voices()
            if voices:
                self.text_to_speech.set_voice_by_index(0)

        print("AI VTuber initialized successfully!")

    def adjust_for_ambient_noise(self):
        """Adjust microphone for ambient noise"""
        self.speech_to_text.adjust_for_ambient_noise()

    def _cleanup_temp_files(self):
        """Clean up temporary audio files before initialization"""
        import tempfile
        import glob
        import os

        try:
            temp_dir = tempfile.gettempdir()
            temp_files = glob.glob(os.path.join(temp_dir, "tts_temp_*.mp3"))

            if os.path.exists("audio_responses"):
                temp_files.extend(glob.glob(os.path.join("audio_responses", "output_*.mp3")))

            files_deleted = 0
            for file in temp_files:
                try:
                    os.remove(file)
                    files_deleted += 1
                except:
                    pass

            if files_deleted > 0:
                print(f"Cleaned up {files_deleted} temporary audio files during startup")
        except Exception as e:
            print(f"Warning: Error during initial cleanup: {e}")

    def cleanup_temp_files(self):
        """Clean up temporary audio files during runtime"""
        # Use the TextToSpeech class's cleanup method
        self.text_to_speech.cleanup_temp_files()

    def listen_and_respond(self):
        """Listen for user input, process it, and respond"""
        user_input = self.speech_to_text.listen_for_speech()

        # Check for common error message patterns with more flexible matching
        error_patterns = ["ไม่ได้ยิน", "ฟังไม่", "ขอโทษ", "มีปัญหา", "ข้อผิดพลาด", "ลองใหม่", "ลองพูด", "เงียบจัง"]
        is_error = any(pattern in user_input for pattern in error_patterns)

        if is_error:
            print(f"Speech recognition error: {user_input}")
            self.text_to_speech.speak(user_input)
            return False

        print(f"You said: {user_input}")

        # Add a quick acknowledgment for longer inputs to improve conversation flow
        if len(user_input) > 20:
            acknowledgments = ["อืมม", "เข้าใจละ", "โอเคๆ", "รับทราบจ้า"]
            import random
            ack = random.choice(acknowledgments)
            print(f"Acknowledgment: {ack}")
            self.text_to_speech.speak(ack)

        response = self.chatbot.chat_with_gemini(user_input)
        print(f"AI response: {response}")

        self.text_to_speech.speak(response)

        if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
            return True

        return False

    def run(self):
        """Run the AI VTuber in a continuous loop"""
        self.cleanup_temp_files()

        welcome_messages = [
            "ฮายยย~! ไพลินพร้อมเมคเฟรนด์แล้วค่า! มีอะไรอยากคุยป่ะ?",
            "ฮายยย~! ไพลินชื่อนะ! วันนี้อารมณ์ดี๊ดี อยากคุยกับเธอจังเลยค่า!",
            "ฮายยย~! หนูชื่อไพลิน ยินดีที่ได้รู้จักจ้า! เล่นเกมอะไรอยู่เหรอ?"
        ]
        import random
        welcome_message = random.choice(welcome_messages)
        filtered_welcome = remove_special_characters(welcome_message)
        print(filtered_welcome)
        self.text_to_speech.speak(filtered_welcome)

        self.adjust_for_ambient_noise()

        should_exit = False
        while not should_exit:
            try:
                should_exit = self.listen_and_respond()
                time.sleep(0.2)  # Reduced delay between conversation turns

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                error_message = "อุย เหมือนจะมีอะไรขัดข้องนิดหน่อยอ่ะ ไพลินขอโทษด้วยนะ ลองใหม่อีกทีได้ป่ะ?"
                filtered_error = remove_special_characters(error_message)
                self.text_to_speech.speak(filtered_error)

        goodbye_messages = [
            "บายบายจ้า! ไว้มาคุยกับไพลินใหม่นะ เดี๋ยวไพลินรอเลย!",
            "ขอบคุณที่มาคุยกับไพลินนะ บายบายค่า! ไว้มาเล่นเกมด้วยกันนะ!",
            "แล้วเจอกันใหม่นะจ๊ะ บายบาย~! อย่าลืมกลับมาเล่นกับไพลินอีกนะ!"
        ]
        import random
        goodbye_message = random.choice(goodbye_messages)
        filtered_goodbye = remove_special_characters(goodbye_message)
        print(filtered_goodbye)
        self.text_to_speech.speak(filtered_goodbye)

        self.cleanup_temp_files()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI VTuber Application")
    parser.add_argument("--cache-size", type=int, default=20, help="Maximum number of TTS responses to cache")
    args = parser.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please create a .env file with your Gemini API key.")
        exit(1)

    vtuber = AIVtuber(
        cache_size=args.cache_size
    )

    try:
        vtuber.run()
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        vtuber.cleanup_temp_files()
        print("Cleanup complete. Goodbye!")
