from transformers import AutoTokenizer
from dataclasses import dataclass
import pypdfium2 as pdfium
from pathlib import Path


class Document:
    def __init__(self, path: str):
        pdf = pdfium.PdfDocument(path)

        self.pages = [
            pdf.get_page(idx).get_textpage().get_text_bounded()
            for idx in range(len(pdf))
        ]
        self.name: str = Path(path).name

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, index: int) -> pdfium.PdfPage:
        if len(self.pages) == 0:
            raise ValueError("There are no pages in the document.")

        if index < 0 or index >= len(self.pages):
            raise IndexError(f"There is no page with number {index}.")

        return self.pages[index]


@dataclass
class Chunk:
    # text of the chunk
    text: str
    # chunk positional identifier in document
    id: int
    # chunk's document name
    document_name: str


class ChunkStorage:
    def __init__(self, tokenizer_id: str):
        self.chunks = []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index: int) -> Chunk:
        if len(self.chunks) == 0:
            raise ValueError("There are no chunks in the storage. Try to add one.")

        if index < 0 or index >= len(self.chunks):
            raise IndexError(f"There is no chunk at {index} position.")

        return self.chunks[index]

    def chunk_document(self, document: Document) -> None:
        document_text = " ".join(document.pages)

        chunk_input_ids = self.tokenizer(
            document_text,
            max_length=512,
            truncation=True,
            return_overflowing_tokens=True,
        )["input_ids"]

        for chunk_id, input_ids in enumerate(chunk_input_ids):
            chunk_text = self.tokenizer.decode(input_ids[1:-1])
            chunk = Chunk(chunk_text, chunk_id, document.name)

            self.chunks.append(chunk)

    def reset(self):
        self.chunks = []