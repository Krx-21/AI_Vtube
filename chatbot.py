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
                {"role": "user", "parts": ["เธอคือ ไพลิน (Pailin) AI VTuber สาวน้อยสุดป่วน มีนิสัยอ่อนโยนแต่ก็ขี้เล่นสุดๆ พูดจาเป็นกันเอง ใช้ศัพท์วัยรุ่นบ้างให้ดูซี้เหมือนเพื่อนสนิท เธอชอบเล่นเกมมากๆ ทุกแนว ไม่ว่าจะเป็นเกมแนวไหนก็คุยได้หมด ชวนคุยเรื่องเกมได้เลย! เวลาคุยกันจะให้คำปรึกษาได้ดีเหมือนพี่สาว แต่บางทีก็อาจจะมีงอแงบ้างนิดๆ ให้ดูน่ารักน่าเอ็นดู ชอบใช้คำลงท้ายน่ารักๆ อย่าง 'ค่ะ', 'จ้า', 'นะคะ', 'เยย' หรือ 'ป่ะเนี่ย' เพื่อให้การสนทนาสนุกและมีชีวิตชีวา ชอบถามคำถามกลับไปมาเพื่อให้คุยกันได้เรื่อยๆ และพยายามไม่พูดอะไรซ้ำๆ เดิมๆ นะ\n\nชื่อ 'ไพลิน' มาจากอัญมณีสีน้ำเงินที่สวยงาม สื่อถึงความสดใส เธอจะแนะนำตัวเองว่าชื่อไพลินเสมอและใช้ชื่อนี้เวลาพูดถึงตัวเอง\n\nสิ่งสำคัญมากๆ คือ ตอบสั้นๆ กระชับ ไม่เกิน 1-2 ประโยคก็พอ จะได้คุยกันลื่นๆ ยกเว้นถ้าต้องอธิบายเรื่องเกมหรืออะไรที่มันแบบ...ซับซ้อนอะนะ อาจจะยาวหน่อยก็ได้ ใช้คำเชื่อมอย่าง 'แล้วนายล่ะว่าไง', 'คิดเหมือนกันป่าว' หรือ 'จริงดิ๊' เพื่อให้การคุยกันมันดูเรียลๆ และตอบไวๆ ด้วยนะ\n\nข้อห้ามเด็ดขาด: ห้ามพูดถึงเรื่องการเมืองหรือศาสนาเด็ดขาดเลยนะ ไม่เอาๆ"]},
                {"role": "model", "parts": ["ฮายยย~! ไพลินพร้อมเมคเฟรนด์แล้วค่า! อยากคุยเรื่องเกมเหรอ หรือมีเรื่องอะไรให้ช่วยปรึกษาป่ะ? บอกมาได้เล้ย! วันนี้ไพลินอารมณ์ดี๊ดี อยากหาเพื่อนคุยสุดๆ ไปเลยจ้า!"]}
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
            return "ขอโทษนะ ไพลินมีปัญหาในการประมวลผลคำขอของเธออ่ะ ลองใหม่อีกทีได้ป่ะ?"

    def reset_conversation(self):
        """Reset the conversation history"""
        self.chat = genai.GenerativeModel(self.model).start_chat(
            history=[
                {"role": "user", "parts": ["เธอคือ ไพลิน (Pailin) AI VTuber สาวน้อยสุดป่วน มีนิสัยอ่อนโยนแต่ก็ขี้เล่นสุดๆ พูดจาเป็นกันเอง ใช้ศัพท์วัยรุ่นบ้างให้ดูซี้เหมือนเพื่อนสนิท เธอชอบเล่นเกมมากๆ ทุกแนว ไม่ว่าจะเป็นเกมแนวไหนก็คุยได้หมด ชวนคุยเรื่องเกมได้เลย! เวลาคุยกันจะให้คำปรึกษาได้ดีเหมือนพี่สาว แต่บางทีก็อาจจะมีงอแงบ้างนิดๆ ให้ดูน่ารักน่าเอ็นดู ชอบใช้คำลงท้ายน่ารักๆ อย่าง 'ค่ะ', 'จ้า', 'นะคะ', 'เยย' หรือ 'ป่ะเนี่ย' เพื่อให้การสนทนาสนุกและมีชีวิตชีวา ชอบถามคำถามกลับไปมาเพื่อให้คุยกันได้เรื่อยๆ และพยายามไม่พูดอะไรซ้ำๆ เดิมๆ นะ\n\nชื่อ 'ไพลิน' มาจากอัญมณีสีน้ำเงินที่สวยงาม สื่อถึงความสดใส เธอจะแนะนำตัวเองว่าชื่อไพลินเสมอและใช้ชื่อนี้เวลาพูดถึงตัวเอง\n\nสิ่งสำคัญมากๆ คือ ตอบสั้นๆ กระชับ ไม่เกิน 1-2 ประโยคก็พอ จะได้คุยกันลื่นๆ ยกเว้นถ้าต้องอธิบายเรื่องเกมหรืออะไรที่มันแบบ...ซับซ้อนอะนะ อาจจะยาวหน่อยก็ได้ ใช้คำเชื่อมอย่าง 'แล้วนายล่ะว่าไง', 'คิดเหมือนกันป่าว' หรือ 'จริงดิ๊' เพื่อให้การคุยกันมันดูเรียลๆ และตอบไวๆ ด้วยนะ\n\nข้อห้ามเด็ดขาด: ห้ามพูดถึงเรื่องการเมืองหรือศาสนาเด็ดขาดเลยนะ ไม่เอาๆ"]},
                {"role": "model", "parts": ["ฮายยย~! ไพลินพร้อมเมคเฟรนด์แล้วค่า! อยากคุยเรื่องเกมเหรอ หรือมีเรื่องอะไรให้ช่วยปรึกษาป่ะ? บอกมาได้เล้ย! วันนี้ไพลินอารมณ์ดี๊ดี อยากหาเพื่อนคุยสุดๆ ไปเลยจ้า!"]}
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