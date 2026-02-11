"""
Chatbot module using Google Gemini API with Pailin personality.
"""

import logging
import os

import google.generativeai as genai
from dotenv import load_dotenv

from ai_vtube.config import (
    DEFAULT_MODEL,
    PAILIN_SYSTEM_PROMPT_MODEL,
    PAILIN_SYSTEM_PROMPT_USER,
)
from ai_vtube.text_utils import remove_special_characters

load_dotenv()

logger = logging.getLogger(__name__)


class Chatbot:
    def __init__(self, model=DEFAULT_MODEL):
        """
        Initialize the chatbot with Google Gemini API.

        Args:
            model (str): The Gemini model to use.
                Default is "gemini-2.0-flash".
                Other options include:
                - "gemini-1.5-pro" (more powerful but slower)
                - "gemini-1.5-flash" (stable model)
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=self.api_key)
        self.model = model
        self._start_chat()

    def _build_history(self):
        """Build the initial conversation history with Pailin's personality."""
        return [
            {"role": "user", "parts": [PAILIN_SYSTEM_PROMPT_USER]},
            {"role": "model", "parts": [PAILIN_SYSTEM_PROMPT_MODEL]},
        ]

    def _start_chat(self):
        """Start a new chat session with the configured history."""
        self.chat = genai.GenerativeModel(self.model).start_chat(
            history=self._build_history()
        )

    def chat_with_gemini(self, user_input):
        """
        Send user input to the Gemini chatbot and get a response.

        Args:
            user_input (str): The user's message.

        Returns:
            str: The chatbot's response with special characters removed.
        """
        try:
            filtered_input = remove_special_characters(user_input)
            response = self.chat.send_message(filtered_input)
            return remove_special_characters(response.text)
        except Exception as e:
            logger.error("Error in chatbot: %s", e)
            return "ขอโทษนะ ไพลินมีปัญหาในการประมวลผลคำขอของเธออ่ะ ลองใหม่อีกทีได้ป่ะ"

    def reset_conversation(self):
        """Reset the conversation history."""
        self._start_chat()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chatbot = Chatbot()

    print("Chatbot initialized. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        response = chatbot.chat_with_gemini(user_input)
        print(f"Chatbot: {response}")
