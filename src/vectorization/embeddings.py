from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SBERTAdapter:
    """
    Класс-адаптер для генерации эмбеддингов (векторов) текста
    с помощью модели Sentence-BERT.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize: bool = True,
    ):
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Модель {model_name} успешно загружена.")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_name}: {e}")
            raise
        self.batch_size = batch_size
        self.normalize = normalize

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Преобразует список текстов в список эмбеддингов.

        Процесс:
        1. Делим текст на батчи
        2. Прогоняем через модель
        3. При необходимости делаем нормализацию
        4. Склеиваем результат в общий массив
        """
        if not texts:
            logger.warning("Пустой список текстов для эмбеддинга.")
            return []

        vectors = []
        try:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                logger.info(
                    f"Обработка батча {i // self.batch_size + 1} из {len(texts) // self.batch_size + 1} (размер: {len(batch)})"
                )

                # кодируем батч → получаем векторы (numpy)
                batch_vecs = self.model.encode(batch, convert_to_numpy=True)

                # опциональная L2 нормализация
                if self.normalize:
                    norms = np.linalg.norm(batch_vecs, axis=1, keepdims=True)
                    batch_vecs = batch_vecs / norms

                # сохраняем в result
                vectors.extend(batch_vecs.tolist())

            return vectors
        except Exception as e:
            logger.error(f"Ошибка при генерации эмбеддингов: {e}")
            raise
