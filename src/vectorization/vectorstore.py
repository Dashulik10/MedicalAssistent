import logging
from typing import Any, Dict, List

import chromadb

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChromaAdapter:
    """
    Адаптер для ChromaDB — хранилища векторов
    """

    def __init__(self, persist_directory: str, collection_name: str):
        self.persist_directory = persist_directory

        try:
            self.client = chromadb.PersistentClient(path=persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}, 
            )
            logger.info(
                f"Коллекция '{collection_name}' в '{persist_directory}' готова"
            )
        except Exception as e:
            logger.error(f"Ошибка инициализации ChromaDB: {e}")
            raise

    def add(
        self,
        ids: List[str],
        vectors: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """
        Добавляет документы в Chroma.
        ids - уникальные ID документов
        vectors - эмбеддинги
        texts - оригинальные тексты (для RAG выдачи)
        metadatas - любая дополнительная информация о документе (айдишник пациента)
        """
        try:
            self.collection.add(
                ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts
            )
            logger.info(f"Добавлено {len(ids)} документов в коллекцию")
        except Exception as e:
            logger.error(f"Ошибка добавления в ChromaDB: {e}")
            raise

    def query(
        self, query_vector: List[float], k: int = 3, where: Dict[str, Any] = None
    ) -> Dict[str, List]:
        """
        Ищет k ближайших документов по вектору
        Пример фильтра: {"patient_id": "12345", "date": {"$gte": "2024-01-01"}}
        """
        try:
            result = self.collection.query(
                query_embeddings=[query_vector], n_results=k, where=where
            )
            logger.info(
                f"Запрос выполнен: найдено {len(result['ids'][0])} результатов"
            )
            return result
        except Exception as e:
            logger.error(f"Ошибка запроса в ChromaDB: {e}")
            raise

    def update(
        self,
        ids: List[str],
        vectors: List[List[float]] = None,
        texts: List[str] = None,
        metadatas: List[Dict[str, Any]] = None,
    ):
        """
        Обновляет документы по ID
        """
        try:
            self.collection.update(
                ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas
            )
            logger.info(f"Обновлено {len(ids)} документов")
        except Exception as e:
            logger.error(f"Ошибка обновления в ChromaDB: {e}")
            raise

    def delete(self, where: Dict[str, Any] = None):
        """
        Удаляет документы по фильтру
        Пример фильтра: фильтр {"patient_id": "12345"}
        """
        try:
            self.collection.delete(where=where)
            logger.info("Документы удалены по фильтру")
        except Exception as e:
            logger.error(f"Ошибка удаления в ChromaDB: {e}")
            raise

    def persist(self):
        """
        Chroma сама пишет изменения на диск
        """
        logger.info("Persist вызван, но в PersistentClient ничего не требуется")
        pass


    def get_count(self) -> int:
        """Возвращает количество документов в коллекции"""
        return self.collection.count()
