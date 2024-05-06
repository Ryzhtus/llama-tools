from fastapi import FastAPI
from pydantic import BaseModel
from src.model import GenerativeModel

app = FastAPI(title="LLM", description="LLM Interface")

llm = GenerativeModel()


class InputData(BaseModel):
    prompt: str


class OutputData(BaseModel):
    response: str


@app.post("/generate", response_model=OutputData)
def generate(input_data: InputData):
    prompt = input_data.prompt

    response = llm.generate_response(prompt)

    return OutputData(response=response)
