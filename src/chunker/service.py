import re
from typing import List


def chunk_text(text: str, max_length: int = 500, overlap: int = 50) -> List[str]:
    """
    Разбивает текст на чанки для векторной базы данных.

    Args:
        text (str): Исходный текст.
        max_length (int): Максимальная длина чанка в символах.
        overlap (int): Количество символов, которые будут повторяться между соседними чанками.

    Returns:
        List[str]: Список текстовых чанков.
    """
    if not text:
        return []

    # Разбиваем текст по абзацам или предложениям
    # Сначала заменяем все переводы строки на маркеры
    text = text.replace("\n", " \n ")
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += sentence + " "
        else:
            # Заканчиваем текущий чанк
            chunks.append(current_chunk.strip())
            # Новый чанк с overlap
            current_chunk = current_chunk[-overlap:] + sentence + " "

    # Добавляем последний чанк
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
