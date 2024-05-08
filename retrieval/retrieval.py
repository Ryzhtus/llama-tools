from src.retrieval_engine import RetrievalEngine
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os


engine = RetrievalEngine()

app = FastAPI(
    title="Vector Storage",
    description="FAISS Vector Storage. Adds and retrieves text data",
    version="0.1",
)


class DocumentParameters(BaseModel):
    name: str


class DocumentResponse(BaseModel):
    text: str


class RetrievalSearchParameters(BaseModel):
    prompt: str
    top_k: int


class RetrievalResponse(BaseModel):
    response: str


class SearchParameters(BaseModel):
    fragment: str


class SearchResponse(BaseModel):
    search_results: list[str]


@app.post("/search")
def search(request: RetrievalSearchParameters, response_model=RetrievalResponse):
    response = engine.search(request.prompt, request.top_k)

    return RetrievalResponse(response=response)


@app.post("/search_similar")
def search_similar_documents(request: SearchParameters, response_model=SearchResponse):
    response = engine.get_list_of_similar_documents(request.fragment)

    return SearchResponse(response=response)


@app.post("/get_document")
def get_document(request: DocumentParameters):
    response = engine.get_document(request.name)

    return DocumentResponse(response=response)


@app.post("/add_document")
async def add_document(file: UploadFile = File(...)):
    try:
        # Save the file or just read the contents
        save_directory = "./saved_files"
        os.makedirs(save_directory, exist_ok=True)

        file_path = os.path.join(save_directory, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        engine.add_document(file_path)

        return JSONResponse(
            status_code=200, content={"message": "File uploaded successfully"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": "An error occurred", "details": str(e)}
        )
