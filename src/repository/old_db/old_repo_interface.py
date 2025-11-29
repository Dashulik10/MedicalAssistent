from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from bson import ObjectId


class OldRepoInterface(Protocol):
    """Интерфейс репозитория для хранения извлечённых медицинских данных."""

    def save_report(
        self,
        report: Dict[str, Any],
        patient_id: int,
        raw_text: str,
        patient_name: Optional[str] = None,
    ) -> str:
        """
        Сохраняет отчёт в базу данных.

        :param report: структурированные данные отчёта
        :param patient_id: ID пациента
        :param raw_text: полный текст для векторизации
        :param patient_name: имя пациента (опционально)
        :return: ID созданной записи
        """
        ...

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Получает отчёт по ID записи."""
        ...

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
        :return: список отчётов
        """
        ...

    def get_reports_by_patient_name(
        self,
        patient_name: str,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Получает все отчёты по имени пациента.

        :param patient_name: имя пациента (частичное совпадение)
        :param date_from: начало периода
        :param date_to: конец периода
        :return: список отчётов
        """
        ...
