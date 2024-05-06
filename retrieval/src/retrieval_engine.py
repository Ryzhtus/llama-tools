from transformers import AutoTokenizer, AutoModel
from src.chunk_storage import ChunkStorage, Document

import torch
import faiss

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class RetrievalEngine:
    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()

        self.index = faiss.IndexFlatL2(
            self.model.embeddings.word_embeddings.embedding_dim
        )
        self.chunk_storage = ChunkStorage(model_id)
        self.documents = []

    def __get_embeddings(self, text: str) -> torch.Tensor:
        encoded_input = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]

        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1
        )

        return sentence_embeddings

    def check_document(self, name: str) -> bool:
        """Checks that document is already in the database"""
        if len(self.documents) == 0:
            return False

        for document in self.documents:
            if name == document.name:
                print(document.name, name)
                return True

        return False

    def add_document(self, path: str) -> None:
        document = Document(path)

        if not self.check_document(document.name):
            self.documents.append(document)
            self.chunk_storage.chunk_document(document)

            embeddings = []
            for chunk in self.chunk_storage:
                embeddings.append(self.__get_embeddings(chunk.text))

            embeddings = torch.concat(embeddings, dim=0).data.cpu().numpy()

            # due to faiss architecture, we have to reset and rebuild index each time
            self.index.reset()
            self.index.add(embeddings)

    def get_document(self, name: str) -> str:
        for document in self.documents:
            if name == document.name:
                pages = document.pages
                return " ".join(pages)

        return None

    def search(self, prompt: str, top_k: int = 1) -> str:
        prompt_embedding = self.__get_embeddings(prompt)

        distances, indices = self.index.search(
            prompt_embedding.data.cpu().numpy(), k=top_k
        )

        context = ""
        for idx in indices[0]:
            context += self.chunk_storage[idx].text

        return context

    def get_list_of_similar_documents(self, fragment: str) -> list[str]:
        fragment_embedding = self.__get_embeddings(fragment)

        distances, indices = self.index.search(
            fragment_embedding.data.cpu().numpy(), k=int(len(self.chunk_storage) / 2)
        )

        search_result = set()
        for distance_id, distance in enumerate(distances[0]):
            if distance <= 0.5:
                chunk_id = indices[0][distance_id]
                search_result.add(self.chunk_storage[chunk_id].document_name)

        return search_result
