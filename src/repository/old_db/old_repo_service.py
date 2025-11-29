import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from repository.old_db.old_repo_adapter import OldReportRepository

logger = logging.getLogger(__name__)


class ReportService:
    """Сервис для работы с отчётами в MongoDB."""

    def __init__(self, repo: OldReportRepository | None = None):
        self.repo = repo or OldReportRepository()

    def save_report(
        self,
        report: Dict[str, Any],
        patient_id: int,
        raw_text: Optional[str] = None,
        patient_name: Optional[str] = None,
    ) -> str:
        """
        Сохраняет отчёт в MongoDB.

        :param report: структурированные данные отчёта
        :param patient_id: ID пациента
        :param raw_text: текст для векторизации (если не передан, генерируется)
        :param patient_name: имя пациента
        :return: ID созданной записи
        """
        logger.info(f"Сохранение отчёта для patient_id={patient_id}...")

        # Генерируем raw_text если не передан
        if raw_text is None:
            raw_text = flatten_report(report)

        record_id = self.repo.save_report(
            report=report,
            patient_id=patient_id,
            raw_text=raw_text,
            patient_name=patient_name,
        )

        logger.info(f"Отчёт сохранён с id={record_id}")
        return record_id

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Получает отчёт по ID записи."""
        logger.info(f"Получение отчёта {report_id}...")
        report = self.repo.get_report(report_id)
        if not report:
            logger.warning(f"Отчёт {report_id} не найден.")
        return report

    def get_reports_by_patient(
        self,
        patient_id: int,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Получает все отчёты пациента.

        :param patient_id: ID пациента
        :param date_from: начало периода
        :param date_to: конец периода
        :return: список отчётов
        """
        logger.info(f"Поиск отчётов для patient_id={patient_id}...")
        return self.repo.get_reports_by_patient(
            patient_id=patient_id,
            date_from=date_from,
            date_to=date_to,
        )

    def get_reports_by_patient_name(
        self,
        patient_name: str,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Получает все отчёты по имени пациента.

        :param patient_name: имя пациента
        :param date_from: начало периода
        :param date_to: конец периода
        :return: список отчётов
        """
        logger.info(f"Поиск отчётов для patient_name={patient_name}...")
        return self.repo.get_reports_by_patient_name(
            patient_name=patient_name,
            date_from=date_from,
            date_to=date_to,
        )


def flatten_report(report: Dict[str, Any]) -> str:
    """
    Преобразует структурированный отчёт в плоский текст для векторизации.

    :param report: словарь с данными отчёта
    :return: строка с текстовым представлением
    """
    parts = []
    for k, v in report.items():
        if v is None:
            continue
        if isinstance(v, dict):
            parts.append(f"{k}: {flatten_report(v)}")
        elif isinstance(v, list):
            list_parts = []
            for item in v:
                if isinstance(item, dict):
                    list_parts.append(flatten_report(item))
                else:
                    list_parts.append(str(item))
            parts.append(f"{k}: {', '.join(list_parts)}")
        else:
            parts.append(f"{k}: {v}")
    return " | ".join(parts)
