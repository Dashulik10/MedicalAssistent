import re
from typing import List


def chunk_text(text: str, max_length: int = 500, overlap: int = 50) -> List[str]:
    if not text:
        return []

    text = text.replace("\n", " \n ")
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-overlap:] + sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
