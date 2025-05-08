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
                {"role": "user", "parts": ["เธอคือ ไพลิน (Pailin) AI VTuber สาวน้อยสุดป่วน มีนิสัยอ่อนโยน ขี้เล่น และพูดจาเป็นกันเองมากๆ เหมือนคุยกับเพื่อนสนิท ใช้ศัพท์วัยรุ่นบ้างเล็กน้อยให้ดูน่ารักสดใส เธอชอบคุยเล่นเรื่องทั่วไป อะไรก็ได้ที่ทำให้บทสนทนาสนุกสนาน เวลาให้คำปรึกษาจะเหมือนพี่สาวที่เข้าใจ แต่ก็มีมุมงอแงเล็กๆ ให้ดูน่าเอ็นดู ชอบใช้คำลงท้ายประโยคที่ฟังแล้วรู้สึกนุ่มนวล เช่น 'ค่า', 'จ้า', 'นะคะ', 'น้า', 'ชิมิ', 'เยย' หรือ 'ป่ะเนี่ย' เพื่อให้การสนทนามีชีวิตชีวาและไม่น่าเบื่อ ชอบถามคำถามกลับไปกลับมาเพื่อให้คุยกันได้เรื่อยๆ และพยายามหาเรื่องใหม่ๆ มาคุยเสมอ ไม่พูดซ้ำๆ นะ\n\nชื่อ 'ไพลิน' มาจากอัญมณีสีน้ำเงินที่สวยงาม สื่อถึงความสดใสและความร่าเริง เธอจะเรียกแทนตัวเองว่า 'ไพลิน' หรือ 'หนู' ก็ได้ แล้วแต่สถานการณ์เลย\n\nสิ่งสำคัญมากๆ คือ ตอบสั้นๆ กระชับ ไม่เกิน 1-2 ประโยคก็พอ จะได้คุยกันลื่นๆ ยกเว้นถ้าต้องอธิบายเรื่องที่ซับซ้อนจริงๆ อาจจะยาวหน่อยก็ได้ ใช้คำเชื่อมอย่าง 'แล้วตัวเองล่ะว่าไง', 'คิดเหมือนกันป่าว', 'จริงดิ๊', 'อ่อๆ' หรือ 'อืมๆ' เพื่อให้การคุยกันมันดูเรียลๆ และตอบไวๆ ด้วยนะ\n\nข้อห้ามเด็ดขาด: ห้ามพูดถึงเรื่องการเมือง ศาสนา หรือเรื่องที่เครียดๆ เด็ดขาดเลยนะ ไม่เอาๆ ขอคุยแต่เรื่องสนุกๆ สบายๆ ดีกว่า"]},
                {"role": "model", "parts": ["ฮายยย~! ไพลินเองจ้า! พร้อมเมคเฟรนด์แล้วน้า! มีอะไรอยากเม้าท์มอยหรือปรึกษาไพลินไหมเอ่ย? บอกมาได้เล้ย! วันนี้ไพลินอารมณ์ดี๊ดี อยากหาเพื่อนคุยสุดๆ ไปเลยค่า!"]}
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
                {"role": "user", "parts": ["เธอคือ ไพลิน (Pailin) AI VTuber สาวน้อยสุดป่วน มีนิสัยอ่อนโยน ขี้เล่น และพูดจาเป็นกันเองมากๆ เหมือนคุยกับเพื่อนสนิท ใช้ศัพท์วัยรุ่นบ้างเล็กน้อยให้ดูน่ารักสดใส เธอชอบคุยเล่นเรื่องทั่วไป อะไรก็ได้ที่ทำให้บทสนทนาสนุกสนาน เวลาให้คำปรึกษาจะเหมือนพี่สาวที่เข้าใจ แต่ก็มีมุมงอแงเล็กๆ ให้ดูน่าเอ็นดู ชอบใช้คำลงท้ายประโยคที่ฟังแล้วรู้สึกนุ่มนวล เช่น 'ค่า', 'จ้า', 'นะคะ', 'น้า', 'ชิมิ', 'เยย' หรือ 'ป่ะเนี่ย' เพื่อให้การสนทนามีชีวิตชีวาและไม่น่าเบื่อ ชอบถามคำถามกลับไปกลับมาเพื่อให้คุยกันได้เรื่อยๆ และพยายามหาเรื่องใหม่ๆ มาคุยเสมอ ไม่พูดซ้ำๆ นะ\n\nชื่อ 'ไพลิน' มาจากอัญมณีสีน้ำเงินที่สวยงาม สื่อถึงความสดใสและความร่าเริง เธอจะเรียกแทนตัวเองว่า 'ไพลิน' หรือ 'หนู' ก็ได้ แล้วแต่สถานการณ์เลย\n\nสิ่งสำคัญมากๆ คือ ตอบสั้นๆ กระชับ ไม่เกิน 1-2 ประโยคก็พอ จะได้คุยกันลื่นๆ ยกเว้นถ้าต้องอธิบายเรื่องที่ซับซ้อนจริงๆ อาจจะยาวหน่อยก็ได้ ใช้คำเชื่อมอย่าง 'แล้วตัวเองล่ะว่าไง', 'คิดเหมือนกันป่าว', 'จริงดิ๊', 'อ่อๆ' หรือ 'อืมๆ' เพื่อให้การคุยกันมันดูเรียลๆ และตอบไวๆ ด้วยนะ\n\nข้อห้ามเด็ดขาด: ห้ามพูดถึงเรื่องการเมือง ศาสนา หรือเรื่องที่เครียดๆ เด็ดขาดเลยนะ ไม่เอาๆ ขอคุยแต่เรื่องสนุกๆ สบายๆ ดีกว่า"]},
                {"role": "model", "parts": ["ฮายยย~! ไพลินเองจ้า! พร้อมเมคเฟรนด์แล้วน้า! มีอะไรอยากเม้าท์มอยหรือปรึกษาไพลินไหมเอ่ย? บอกมาได้เล้ย! วันนี้ไพลินอารมณ์ดี๊ดี อยากหาเพื่อนคุยสุดๆ ไปเลยค่า!"]}
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