import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .embeddings import SBERTAdapter
from .vectorstore import ChromaAdapter

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    distance: float


class EmbeddingAndStoreService:
    def __init__(
        self, embedder: SBERTAdapter, store: ChromaAdapter, batch_size: int = 32
    ):
        self.embedder = embedder
        self.store = store
        self.batch_size = batch_size

    def index_chunks(
        self,
        chunks: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> List[str]:
        if len(chunks) != len(metadatas):
            logger.error("Длина chunks и metadatas не совпадает!")
            raise ValueError("Длина chunks и metadatas должна быть одинаковой")

        ids = [str(uuid.uuid4()) for _ in chunks]
        logger.info(f"Генерация {len(ids)} уникальных ID завершена")

        try:
            for i in tqdm(
                range(0, len(chunks), self.batch_size), desc="Индексация батчей"
            ):
                chunk_batch = chunks[i : i + self.batch_size]
                meta_batch = metadatas[i : i + self.batch_size]
                id_batch = ids[i : i + self.batch_size]

                vectors = self.embedder.embed(chunk_batch)
                logger.info(
                    f"Эмбеддинги для батча {i // self.batch_size + 1} сгенерированы (размер: {len(vectors)})."
                )

                # Очищаем метаданные от None значений
                clean_meta_batch = [self._sanitize_metadata(m) for m in meta_batch]

                self.store.add(
                    ids=id_batch,
                    vectors=vectors,
                    texts=chunk_batch,
                    metadatas=clean_meta_batch,
                )
                logger.info(f"Батч {i // self.batch_size + 1} добавлен в ChromaDB")

            self.store.persist()
            logger.info("Индексация завершена")
            return ids
        except Exception as e:
            logger.error(f"Ошибка при индексации: {e}")
            raise

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any],
    ) -> str:
        try:
            vector = self.embedder.embed([text])[0]
            clean_metadata = self._sanitize_metadata(metadata)
            self.store.add(
                ids=[doc_id],
                vectors=[vector],
                texts=[text],
                metadatas=[clean_metadata],
            )
            logger.info(f"Документ {doc_id} добавлен в ChromaDB.")
            return doc_id
        except Exception as e:
            logger.error(f"Ошибка при добавлении документа {doc_id}: {e}")
            raise

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        try:
            query_vector = self.embedder.embed([query])[0]

            raw_results = self.store.query(
                query_vector=query_vector,
                k=top_k,
                where=filters,
            )

            results: List[SearchResult] = []
            if raw_results["ids"] and raw_results["ids"][0]:
                for i, doc_id in enumerate(raw_results["ids"][0]):
                    results.append(
                        SearchResult(
                            doc_id=doc_id,
                            text=raw_results["documents"][0][i]
                            if raw_results.get("documents")
                            else "",
                            metadata=raw_results["metadatas"][0][i]
                            if raw_results.get("metadatas")
                            else {},
                            distance=raw_results["distances"][0][i]
                            if raw_results.get("distances")
                            else 0.0,
                        )
                    )

            logger.info(f"Семантический поиск: найдено {len(results)} результатов")
            return results
        except Exception as e:
            logger.error(f"Ошибка семантического поиска: {e}")
            raise
