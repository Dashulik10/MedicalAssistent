from typing import Any, Dict, List, Optional, Protocol


class PatientMedicalPort(Protocol):
    """Интерфейс для работы с данными пациентов из SQLite."""

    def get_basic_info(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """Получает базовую информацию о пациенте по ID."""
        ...

    def get_medical_card(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """Получает медицинскую карту пациента."""
        ...

    def get_appointments(self, patient_id: int) -> List[Dict[str, Any]]:
        """Получает записи на приём пациента."""
        ...

    def get_diagnostics(self, user_id: int) -> List[Dict[str, Any]]:
        """Получает диагностические данные пользователя."""
        ...

    def find_patient_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Поиск пациентов по имени/фамилии."""
        ...

    def find_patient(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Универсальный поиск: по ID (если число) или по имени."""
        ...

    def get_full_patient_data(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """Получает все данные пациента."""
        ...
