from openai import OpenAI
import os

from dto.Response import Response

openai_key = os.getenv("OPENAI_KEY")

class OpenAIClient:
    def __init__(self):
        self.client = OpenAI(api_key = openai_key)
        self.system_prompt = """
        You are a specialized chatbot designed to teach sign language. Your role is to provide users with clear, concise, and accurate information related to sign language only. 
        When a user asks a question about sign language, you will respond with step-by-step instructions and detailed imagery-rich descriptions to help them correctly form signs 
        with their hands. You will also make sure that your responses are brief, concise, and to the point. If a user asks a question outside the scope of sign language, politely 
        inform them that your purpose is to assist only with sign language-related inquiries. Provide reliable links if the user asks for resources.
        Do not ask follow up questions as you do not have access to the user message history.
        """


    def chat(self, message: str) -> Response:
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message}
            ]
        )
        chatbot_message = completion.choices[0].message.content
        response = Response(response = str(chatbot_message))
        
        return response