from fastapi import FastAPI
from dto.Prompt import Prompt
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from OpenAIClient import OpenAIClient

app = FastAPI()
openai_client = OpenAIClient()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def hello():
    return "hello world"

@app.post("/chat")
def chat(prompt: Prompt):
    response = openai_client.chat(message=prompt.prompt)
    return response
