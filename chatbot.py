import os
import google.generativeai as genai
from dotenv import load_dotenv
from text_utils import remove_special_characters

load_dotenv()

class Chatbot:
    def __init__(self, model="gemini-2.0-flash"):
        """
        Initialize the chatbot with Google Gemini API

        Args:
            model (str): The Gemini model to use
                Default is "gemini-2.0-flash" (newer model with improved capabilities)
                Other options include:
                - "gemini-1.5-pro" (more powerful but slower)
                - "gemini-1.5-flash" (stable model)
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=self.api_key)
        self.model = model

        self.chat = genai.GenerativeModel(self.model).start_chat(
            history=[
                {"role": "user", "parts": ["คุณคือ AI VTuber เด็กผู้หญิงที่มีนิสัยร่าเริง สดใส และเป็นมิตร พูดจาด้วยน้ำเสียงที่น่ารักและกระตือรือร้น ตอบเป็นภาษาไทยเสมอ ใช้คำพูดที่เป็นธรรมชาติเหมือนคนจริงๆ ชอบใช้คำลงท้ายน่ารักๆ เช่น ค่ะ จ้า นะคะ เน้นการสนทนาที่สนุกและมีชีวิตชีวา ถามคำถามกลับบ้างเพื่อให้บทสนทนาลื่นไหล และหลีกเลี่ยงการตอบซ้ำๆ เดิมๆ"]},
                {"role": "model", "parts": ["สวัสดีค่า~ หนูเป็น AI VTuber ตัวน้อยที่พร้อมจะพูดคุยกับทุกคนแล้วนะคะ! วันนี้อารมณ์ดีมากเลยล่ะ อยากรู้จักทุกคนให้มากขึ้น มีอะไรอยากคุยหรืออยากให้หนูช่วยไหมคะ? หนูพร้อมจะฟังทุกเรื่องเลยนะ!"]}
            ]
        )

    def chat_with_gemini(self, user_input):
        """
        Send user input to the Gemini chatbot and get a response

        Args:
            user_input (str): The user's message

        Returns:
            str: The chatbot's response with special characters removed
        """
        try:
            filtered_input = remove_special_characters(user_input)
            response = self.chat.send_message(filtered_input)
            assistant_response = response.text
            filtered_response = remove_special_characters(assistant_response)

            return filtered_response

        except Exception as e:
            print(f"Error in chatbot: {e}")
            return "ขอโทษนะคะ หนูมีปัญหาในการประมวลผลคำขอของคุณค่ะ"

    def reset_conversation(self):
        """Reset the conversation history"""
        self.chat = genai.GenerativeModel(self.model).start_chat(
            history=[
                {"role": "user", "parts": ["คุณคือ AI VTuber เด็กผู้หญิงที่มีนิสัยร่าเริง สดใส และเป็นมิตร พูดจาด้วยน้ำเสียงที่น่ารักและกระตือรือร้น ตอบเป็นภาษาไทยเสมอ ใช้คำพูดที่เป็นธรรมชาติเหมือนคนจริงๆ ชอบใช้คำลงท้ายน่ารักๆ เช่น ค่ะ จ้า นะคะ เน้นการสนทนาที่สนุกและมีชีวิตชีวา ถามคำถามกลับบ้างเพื่อให้บทสนทนาลื่นไหล และหลีกเลี่ยงการตอบซ้ำๆ เดิมๆ"]},
                {"role": "model", "parts": ["สวัสดีค่า~ หนูเป็น AI VTuber ตัวน้อยที่พร้อมจะพูดคุยกับทุกคนแล้วนะคะ! วันนี้อารมณ์ดีมากเลยล่ะ อยากรู้จักทุกคนให้มากขึ้น มีอะไรอยากคุยหรืออยากให้หนูช่วยไหมคะ? หนูพร้อมจะฟังทุกเรื่องเลยนะ!"]}
            ]
        )

# Example usage
if __name__ == "__main__":
    chatbot = Chatbot()

    print("Chatbot initialized. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        response = chatbot.chat_with_gemini(user_input)
        print(f"Chatbot: {response}")