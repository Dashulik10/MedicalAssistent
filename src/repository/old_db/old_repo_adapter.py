import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from repository.old_db.old_db import client, collection, database

logger = logging.getLogger(__name__)


class OldReportRepository:
    """
    Репозиторий для хранения извлечённых медицинских данных в MongoDB.

    Схема документа:
    {
        "_id": ObjectId,
        "patient_id": int,
        "patient_name": str | None,
        "report": Dict[str, Any],  # структурированные данные
        "raw_text": str,           # полный текст для векторизации
        "created_at": datetime,
        "test_date": str | None,   # дата анализа из report
    }
    """

    def __init__(self):
        self.client = client
        self.db = database
        self.collection = collection

        # Удаляем старый уникальный индекс, если он существует
        try:
            existing_indexes = self.collection.list_indexes()
            for index in existing_indexes:
                if index.get("name") == "patient_id_1" and index.get("unique"):
                    logger.info("Удаление старого уникального индекса patient_id_1...")
                    self.collection.drop_index("patient_id_1")
                    break
        except Exception as e:
            logger.warning(f"Не удалось проверить/удалить старый индекс: {e}")

        # Создаём индексы для быстрого поиска (без уникальности - много записей на пациента)
        # Используем явные имена, чтобы избежать конфликтов
        try:
            self.collection.create_index(
                [("patient_id", ASCENDING)], name="patient_id_idx", unique=False
            )
        except Exception as e:
            logger.warning(f"Индекс patient_id уже существует или ошибка: {e}")

        try:
            self.collection.create_index(
                [("patient_name", ASCENDING)], name="patient_name_idx"
            )
        except Exception as e:
            logger.warning(f"Индекс patient_name уже существует или ошибка: {e}")

        try:
            self.collection.create_index(
                [("created_at", DESCENDING)], name="created_at_idx"
            )
        except Exception as e:
            logger.warning(f"Индекс created_at уже существует или ошибка: {e}")

        try:
            self.collection.create_index(
                [("test_date", DESCENDING)], name="test_date_idx"
            )
        except Exception as e:
            logger.warning(f"Индекс test_date уже существует или ошибка: {e}")

        logger.info("Индексы MongoDB инициализированы")

    def save_report(
        self,
        report: Dict[str, Any],
        patient_id: int,
        raw_text: str,
        patient_name: Optional[str] = None,
    ) -> str:
        """
        Сохраняет новый отчёт в базу данных.

        :param report: структурированные данные отчёта
        :param patient_id: ID пациента
        :param raw_text: полный текст для векторизации
        :param patient_name: имя пациента
        :return: ID созданной записи (строка ObjectId)
        """
        try:
            document = {
                "patient_id": patient_id,
                "patient_name": patient_name,
                "report": report,
                "raw_text": raw_text,
                "created_at": datetime.utcnow(),
                "test_date": report.get("test_date"),
            }

            result = self.collection.insert_one(document)
            record_id = str(result.inserted_id)

            logger.info(
                f"MongoDB: сохранён отчёт {record_id} для patient_id={patient_id}"
            )
            return record_id

        except Exception as e:
            logger.exception("Ошибка сохранения отчёта в MongoDB")
            raise RuntimeError("Failed to save report") from e

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """
        Получает отчёт по ID записи.

        :param report_id: ID записи (строка ObjectId)
        :return: документ или None
        """
        try:
            obj_id = ObjectId(report_id)
            doc = self.collection.find_one({"_id": obj_id})
            if doc:
                doc["_id"] = str(doc["_id"])
            return doc
        except Exception as e:
            logger.exception(f"Ошибка получения отчёта {report_id}")
            raise RuntimeError("Failed to get report") from e

    def get_reports_by_patient(
        self,
        patient_id: int,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Получает все отчёты пациента с опциональной фильтрацией по датам.

        :param patient_id: ID пациента
        :param date_from: начало периода
        :param date_to: конец периода
        :return: список отчётов, отсортированных по дате создания
        """
        try:
            query: Dict[str, Any] = {"patient_id": patient_id}

            # Добавляем фильтр по дате создания
            if date_from or date_to:
                date_filter: Dict[str, Any] = {}
                if date_from:
                    date_filter["$gte"] = date_from
                if date_to:
                    date_filter["$lte"] = date_to
                query["created_at"] = date_filter

            cursor = self.collection.find(query).sort("created_at", DESCENDING)
            results = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                results.append(doc)

            logger.info(
                f"MongoDB: найдено {len(results)} отчётов для patient_id={patient_id}"
            )
            return results

        except Exception as e:
            logger.exception(f"Ошибка поиска отчётов для patient_id={patient_id}")
            raise RuntimeError("Failed to get reports by patient") from e

    def get_reports_by_patient_name(
        self,
        patient_name: str,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Получает все отчёты по имени пациента (частичное совпадение).

        :param patient_name: имя пациента для поиска
        :param date_from: начало периода
        :param date_to: конец периода
        :return: список отчётов
        """
        try:
            # Регистронезависимый поиск по частичному совпадению
            query: Dict[str, Any] = {
                "patient_name": {"$regex": patient_name, "$options": "i"}
            }

            if date_from or date_to:
                date_filter: Dict[str, Any] = {}
                if date_from:
                    date_filter["$gte"] = date_from
                if date_to:
                    date_filter["$lte"] = date_to
                query["created_at"] = date_filter

            cursor = self.collection.find(query).sort("created_at", DESCENDING)
            results = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                results.append(doc)

            logger.info(
                f"MongoDB: найдено {len(results)} отчётов для patient_name={patient_name}"
            )
            return results

        except Exception as e:
            logger.exception(f"Ошибка поиска отчётов для patient_name={patient_name}")
            raise RuntimeError("Failed to get reports by patient name") from e

    def get_all_patient_ids(self) -> List[int]:
        """Возвращает список уникальных patient_id."""
        try:
            return self.collection.distinct("patient_id")
        except Exception as e:
            logger.exception("Ошибка получения списка patient_id")
            raise RuntimeError("Failed to get patient ids") from e
