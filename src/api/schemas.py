from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# === Запросы ===


class GenerateReportRequest(BaseModel):
    patient_identifier: str = Field(
        ...,
        description="ID пациента (число) или имя/фамилия для поиска",
        examples=["123", "Иванов"],
    )
    query: Optional[str] = Field(
        None,
        description="Текстовый запрос для семантического поиска",
        examples=["краткий отчёт о состоянии здоровья"],
    )
    date_from: Optional[datetime] = Field(
        None,
        description="Начало периода для фильтрации записей",
    )
    date_to: Optional[datetime] = Field(
        None,
        description="Конец периода для фильтрации записей",
    )


# === Ответы ===


class UploadImageResponse(BaseModel):
    status: str = Field(..., description="Статус операции: success или error")
    message: str = Field(..., description="Описание результата")
    record_id: Optional[str] = Field(None, description="ID созданной записи в MongoDB")
    patient_id: Optional[int] = Field(None, description="ID пациента")
    patient_name: Optional[str] = Field(None, description="Имя пациента из документа")
    test_type: Optional[str] = Field(None, description="Тип анализа")
    test_date: Optional[str] = Field(None, description="Дата анализа")


class ExtractedDataResponse(BaseModel):
    patient_name: Optional[str] = None
    patient_id: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    test_date: Optional[str] = None
    test_type: Optional[str] = None
    doctor_name: Optional[str] = None
    lab_name: Optional[str] = None


class UploadImageDetailedResponse(BaseModel):
    status: str
    message: str
    record_id: Optional[str] = None
    extracted_data: Optional[ExtractedDataResponse] = None


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    detail: Optional[str] = None


class PatientInfoResponse(BaseModel):
    id: int
    surname: Optional[str] = None
    firstname: Optional[str] = None
    dob: Optional[str] = None


class ReportRecordResponse(BaseModel):
    id: str
    patient_id: int
    test_type: Optional[str] = None
    test_date: Optional[str] = None
    created_at: datetime


# === RAG Query ===


class QueryPatientRequest(BaseModel):
    patient_identifier: str = Field(
        ...,
        description="ID пациента (число) или имя/фамилия для поиска",
        examples=["1", "Иванов"],
    )
    query: str = Field(
        ...,
        description="Вопрос пользователя о пациенте",
        examples=[
            "Какие анализы были сданы?",
            "Есть ли отклонения в показателях крови?",
            "Краткое резюме состояния здоровья",
        ],
    )


class ContextSourcesResponse(BaseModel):
    sqlite_patient: bool = Field(..., description="Использованы ли данные из SQLite")
    mongodb_records_count: int = Field(..., description="Количество записей из MongoDB")
    vector_search_results: int = Field(
        ..., description="Количество результатов семантического поиска"
    )


class QueryPatientResponse(BaseModel):
    status: str = Field(..., description="Статус: success или error")
    answer: str = Field(..., description="Сгенерированный ответ LLM")
    patient_id: int = Field(..., description="ID пациента")
    patient_name: Optional[str] = Field(None, description="Имя пациента")
    sources: ContextSourcesResponse = Field(
        ..., description="Использованные источники данных"
    )
    model: str = Field(..., description="Использованная модель LLM")
    tokens_used: int = Field(..., description="Количество использованных токенов")
    generation_time: float = Field(..., description="Время генерации в секундах")
