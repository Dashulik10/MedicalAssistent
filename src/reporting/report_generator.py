import logging
from datetime import datetime
from typing import Optional

from configuration.settings import settings

logger = logging.getLogger(__name__)


class ReportGenerator:
    def __init__(self, clinic_name: Optional[str] = None):
        self.clinic_name = clinic_name or settings.clinic_name
        logger.info(f"ReportGenerator инициализирован: клиника={self.clinic_name}")

    def generate_from_llm(
        self,
        llm_response: str,
        patient_name: Optional[str] = None,
        patient_id: Optional[int] = None,
    ) -> str:
        sections = []

        # Заголовок с названием поликлиники
        sections.append(self._generate_header(patient_name, patient_id))

        # Основной контент от LLM
        sections.append(llm_response.strip())

        # Футер с датой
        sections.append(self._generate_footer())

        report = "\n\n".join(sections)
        logger.info(f"Сгенерирован отчёт: {len(report)} символов")
        return report

    def _generate_header(
        self, patient_name: Optional[str], patient_id: Optional[int]
    ) -> str:
        lines = [
            f"# {self.clinic_name}",
            "",
        ]

        if patient_name:
            lines.append(f"**Пациент:** {patient_name}")
        if patient_id:
            lines.append(f"**ID пациента:** {patient_id}")

        lines.append("")  # Пустая строка перед основным контентом

        return "\n".join(lines)

    def _generate_footer(self) -> str:
        current_date = datetime.now().strftime("%d.%m.%Y %H:%M")
        lines = [
            "---",
            "",
            f"*Дата формирования отчёта: {current_date}*",
        ]
        return "\n".join(lines)
