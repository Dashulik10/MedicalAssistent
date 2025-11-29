"""Построитель контекста для RAG-генерации."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContextSources:
    """Источники данных, использованные для построения контекста."""

    sqlite_patient: bool = False
    mongodb_records_count: int = 0
    vector_search_results: int = 0
    record_ids: List[str] = field(default_factory=list)


@dataclass
class PatientContext:
    """Контекст пациента для LLM."""

    text: str
    sources: ContextSources
    patient_id: int
    patient_name: Optional[str] = None


class ContextBuilder:
    """
    Построитель контекста для RAG-запросов.
    Агрегирует данные из SQLite, MongoDB и векторной БД.
    """

    def __init__(
        self,
        sqlite_adapter,
        report_service,
        embed_store_service,
    ):
        """
        Инициализация построителя.

        :param sqlite_adapter: адаптер SQLite для данных пациентов
        :param report_service: сервис MongoDB для медицинских записей
        :param embed_store_service: сервис векторного поиска
        """
        self.sqlite_adapter = sqlite_adapter
        self.report_service = report_service
        self.embed_store_service = embed_store_service
        logger.info("ContextBuilder инициализирован")

    def build_context(
        self,
        patient_id: int,
        query: str,
        vector_top_k: int = 5,
        max_mongo_records: int = 10,
    ) -> PatientContext:
        """
        Строит полный контекст для RAG-запроса.

        :param patient_id: ID пациента
        :param query: запрос пользователя для семантического поиска
        :param vector_top_k: количество результатов из векторного поиска
        :param max_mongo_records: максимум записей из MongoDB
        :return: контекст с текстом и метаданными источников
        """
        logger.info(f"Построение контекста для patient_id={patient_id}")

        sources = ContextSources()
        context_parts = []

        # 1. Базовая информация из SQLite
        patient_info = self._get_sqlite_data(patient_id)
        if patient_info:
            sources.sqlite_patient = True
            context_parts.append(self._format_patient_info(patient_info))
            patient_name = self._extract_patient_name(patient_info)
        else:
            patient_name = None
            context_parts.append(f"### Пациент ID: {patient_id}\n*Базовая информация не найдена в системе*")

        # 2. Семантический поиск в векторной БД
        vector_results = self._semantic_search(patient_id, query, vector_top_k)
        if vector_results:
            sources.vector_search_results = len(vector_results)
            context_parts.append(self._format_vector_results(vector_results))
            for result in vector_results:
                if result.get("record_id"):
                    sources.record_ids.append(result["record_id"])

        # 3. Записи из MongoDB (дополнительно, если нужно)
        mongo_records = self._get_mongo_records(patient_id, max_mongo_records)
        if mongo_records:
            # Фильтруем уже включённые через векторный поиск
            new_records = [
                r for r in mongo_records
                if str(r.get("_id")) not in sources.record_ids
            ]
            if new_records:
                sources.mongodb_records_count = len(new_records)
                context_parts.append(self._format_mongo_records(new_records))
                for record in new_records:
                    sources.record_ids.append(str(record.get("_id")))

        # Объединяем контекст
        full_context = "\n\n---\n\n".join(context_parts)

        logger.info(
            f"Контекст построен: SQLite={sources.sqlite_patient}, "
            f"Vector={sources.vector_search_results}, MongoDB={sources.mongodb_records_count}"
        )

        return PatientContext(
            text=full_context,
            sources=sources,
            patient_id=patient_id,
            patient_name=patient_name,
        )

    def _get_sqlite_data(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """Получает данные пациента из SQLite."""
        try:
            return self.sqlite_adapter.get_full_patient_data(patient_id)
        except Exception as e:
            logger.warning(f"Не удалось получить данные SQLite: {e}")
            return None

    def _semantic_search(
        self, patient_id: int, query: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """Выполняет семантический поиск в векторной БД."""
        try:
            results = self.embed_store_service.semantic_search(
                query=query,
                top_k=top_k,
                filters={"patient_id": str(patient_id)},
            )
            return [
                {
                    "text": r.text,
                    "record_id": r.metadata.get("record_id"),
                    "test_type": r.metadata.get("test_type"),
                    "distance": r.distance,
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Ошибка семантического поиска: {e}")
            return []

    def _get_mongo_records(
        self, patient_id: int, max_records: int
    ) -> List[Dict[str, Any]]:
        """Получает записи из MongoDB."""
        try:
            records = self.report_service.get_reports_by_patient(patient_id)
            return records[:max_records]
        except Exception as e:
            logger.warning(f"Не удалось получить записи MongoDB: {e}")
            return []

    def _extract_patient_name(self, patient_info: Dict[str, Any]) -> Optional[str]:
        """Извлекает имя пациента из данных."""
        basic = patient_info.get("basic_info", {})
        surname = basic.get("surname", "")
        firstname = basic.get("firstname", "")
        if surname or firstname:
            return f"{surname} {firstname}".strip()
        return None

    def _format_patient_info(self, data: Dict[str, Any]) -> str:
        """Форматирует информацию о пациенте из SQLite."""
        lines = ["### Информация о пациенте"]

        basic = data.get("basic_info", {})
        if basic:
            if basic.get("surname") or basic.get("firstname"):
                lines.append(f"**ФИО:** {basic.get('surname', '')} {basic.get('firstname', '')}")
            if basic.get("dob"):
                lines.append(f"**Дата рождения:** {basic['dob']}")

        medical = data.get("medical_card", {})
        if medical:
            lines.append("\n**Медицинская карта:**")
            if medical.get("gender"):
                gender_map = {"M": "Мужской", "F": "Женский"}
                lines.append(f"- Пол: {gender_map.get(medical['gender'], medical['gender'])}")
            if medical.get("blood_group"):
                lines.append(f"- Группа крови: {medical['blood_group']}")
            if medical.get("rhesus"):
                lines.append(f"- Резус-фактор: {medical['rhesus']}")
            if medical.get("allergic_history"):
                lines.append(f"- Аллергический анамнез: {medical['allergic_history']}")

        diagnostics = data.get("diagnostics", [])
        if diagnostics:
            lines.append(f"\n**История диагностик:** {len(diagnostics)} записей")
            for diag in diagnostics[:5]:  # Показываем первые 5
                date = diag.get("date", "Дата неизвестна")
                test_type = diag.get("test_type", "Неизвестный тип")
                lines.append(f"- {date}: {test_type}")

        return "\n".join(lines)

    def _format_vector_results(self, results: List[Dict[str, Any]]) -> str:
        """Форматирует результаты векторного поиска."""
        lines = ["### Релевантные медицинские записи (семантический поиск)"]

        for i, result in enumerate(results, 1):
            test_type = result.get("test_type", "Неизвестно")
            text = result.get("text", "")
            # Ограничиваем длину текста
            if len(text) > 500:
                text = text[:500] + "..."
            lines.append(f"\n**Запись {i}** (тип: {test_type}):")
            lines.append(text)

        return "\n".join(lines)

    def _format_mongo_records(self, records: List[Dict[str, Any]]) -> str:
        """Форматирует записи из MongoDB."""
        lines = ["### Дополнительные медицинские записи"]

        for i, record in enumerate(records, 1):
            report = record.get("report", {})
            test_type = report.get("test_type", "Неизвестно")
            test_date = report.get("test_date", "Дата неизвестна")
            created_at = record.get("created_at", "")

            lines.append(f"\n**Запись {i}** (тип: {test_type}, дата: {test_date}):")

            # Форматируем данные анализов
            if test_type == "blood_count" and report.get("blood_count"):
                lines.append(self._format_blood_count(report["blood_count"]))
            elif test_type == "biochemistry" and report.get("biochemistry"):
                lines.append(self._format_biochemistry(report["biochemistry"]))
            else:
                # Общий формат
                lines.append(self._format_generic_report(report))

        return "\n".join(lines)

    def _format_blood_count(self, data: Dict[str, Any]) -> str:
        """Форматирует результаты общего анализа крови."""
        lines = ["Общий анализ крови:"]
        params = [
            ("hemoglobin", "Гемоглобин"),
            ("rbc_count", "Эритроциты"),
            ("wbc_count", "Лейкоциты"),
            ("platelet_count", "Тромбоциты"),
        ]
        for key, name in params:
            if data.get(key):
                param = data[key]
                value = param.get("value", "-")
                unit = param.get("unit", "")
                ref = param.get("reference_range", "")
                flag = param.get("flag", "")
                flag_str = f" ({flag})" if flag else ""
                lines.append(f"- {name}: {value} {unit} (норма: {ref}){flag_str}")
        return "\n".join(lines)

    def _format_biochemistry(self, data: Dict[str, Any]) -> str:
        """Форматирует результаты биохимии."""
        lines = ["Биохимический анализ:"]
        params = [
            ("glucose", "Глюкоза"),
            ("creatinine", "Креатинин"),
            ("urea", "Мочевина"),
            ("bilirubin_total", "Билирубин общий"),
            ("sgot_ast", "АСТ"),
            ("sgpt_alt", "АЛТ"),
        ]
        for key, name in params:
            if data.get(key):
                param = data[key]
                value = param.get("value", "-")
                unit = param.get("unit", "")
                ref = param.get("reference_range", "")
                flag = param.get("flag", "")
                flag_str = f" ({flag})" if flag else ""
                lines.append(f"- {name}: {value} {unit} (норма: {ref}){flag_str}")
        return "\n".join(lines)

    def _format_generic_report(self, report: Dict[str, Any]) -> str:
        """Форматирует отчёт общего типа."""
        # Исключаем большие вложенные объекты
        exclude_keys = {"blood_count", "biochemistry", "other_parameters"}
        lines = []
        for key, value in report.items():
            if key not in exclude_keys and value is not None:
                if isinstance(value, dict):
                    lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False)[:200]}")
                else:
                    lines.append(f"- {key}: {value}")
        return "\n".join(lines) if lines else "Данные отсутствуют"

