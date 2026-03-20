import io
import re
from typing import List


def process_document(uploaded_file) -> str:
    """Extract raw text from a PDF or TXT file."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    elif filename.endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks by word count."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    words = text.split()
    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += step

    return chunks