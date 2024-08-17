from fastapi import FastAPI
from dto.Prompt import Prompt
import uvicorn
from OpenAIClient import OpenAIClient

app = FastAPI()
openai_client = OpenAIClient()

@app.post("/chat")
def chat(prompt: Prompt):
    response = openai_client.chat(message=prompt.prompt)
    return response

if __name__ == "__main__":
    uvicorn.run(app)