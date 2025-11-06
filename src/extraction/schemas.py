from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, validator


# Базовый класс для всех отчетов
class BaseReport(BaseModel):
    """Общие поля для всех типов отчетов"""

    document_type: str
    patient_name: Optional[str] = None
    patient_age: Optional[int] = None
    patient_gender: Optional[Literal["M", "F", "Unknown"]] = None
    test_date: Optional[date] = None
    lab_name: Optional[str] = None
    doctor_name: Optional[str] = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @validator("patient_age")
    def validate_age(cls, v):
        if v is not None and (v < 0 or v > 120):
            raise ValueError("Возраст должен быть от 0 до 120")
        return v


# Параметр анализа с референсными значениями
class TestParameter(BaseModel):
    name: str
    value: str  # Может быть число или текст типа "Negative"
    unit: Optional[str] = None
    reference_range: Optional[str] = None  # "3.5-5.5" или "< 10"
    is_normal: Optional[bool] = None
    notes: Optional[str] = None


# ===== АНАЛИЗ КРОВИ =====
class BloodTestReport(BaseReport):
    """Общий анализ крови"""

    document_type: Literal["blood_test"] = "blood_test"

    # Основные показатели
    hemoglobin: Optional[TestParameter] = None  # HGB
    rbc: Optional[TestParameter] = None  # Эритроциты
    wbc: Optional[TestParameter] = None  # Лейкоциты
    platelets: Optional[TestParameter] = None  # Тромбоциты
    hematocrit: Optional[TestParameter] = None  # HCT

    # Лейкоцитарная формула
    neutrophils: Optional[TestParameter] = None
    lymphocytes: Optional[TestParameter] = None
    monocytes: Optional[TestParameter] = None
    eosinophils: Optional[TestParameter] = None
    basophils: Optional[TestParameter] = None

    # Прочие параметры
    other_parameters: List[TestParameter] = Field(default_factory=list)

    interpretation: Optional[str] = None


# ===== БИОХИМИЧЕСКИЙ АНАЛИЗ =====
class BiochemistryReport(BaseReport):
    """Биохимический анализ крови"""

    document_type: Literal["biochemistry"] = "biochemistry"

    # Основные показатели
    glucose: Optional[TestParameter] = None
    total_cholesterol: Optional[TestParameter] = None
    hdl_cholesterol: Optional[TestParameter] = None
    ldl_cholesterol: Optional[TestParameter] = None
    triglycerides: Optional[TestParameter] = None

    # Функция печени
    alt: Optional[TestParameter] = None  # АЛТ
    ast: Optional[TestParameter] = None  # АСТ
    bilirubin_total: Optional[TestParameter] = None
    bilirubin_direct: Optional[TestParameter] = None

    # Функция почек
    creatinine: Optional[TestParameter] = None
    urea: Optional[TestParameter] = None
    uric_acid: Optional[TestParameter] = None

    # Прочие
    other_parameters: List[TestParameter] = Field(default_factory=list)
    interpretation: Optional[str] = None


# ===== АНАЛИЗ МОЧИ =====
class UrineTestReport(BaseReport):
    """Общий анализ мочи"""

    document_type: Literal["urine_test"] = "urine_test"

    # Физические свойства
    color: Optional[TestParameter] = None
    clarity: Optional[TestParameter] = None
    specific_gravity: Optional[TestParameter] = None
    ph: Optional[TestParameter] = None

    # Химические свойства
    protein: Optional[TestParameter] = None
    glucose: Optional[TestParameter] = None
    ketones: Optional[TestParameter] = None
    blood: Optional[TestParameter] = None
    bilirubin: Optional[TestParameter] = None

    # Микроскопия
    rbc_microscopy: Optional[TestParameter] = None
    wbc_microscopy: Optional[TestParameter] = None
    epithelial_cells: Optional[TestParameter] = None

    other_parameters: List[TestParameter] = Field(default_factory=list)
    interpretation: Optional[str] = None


class OtherReport(BaseReport):
    """Для документов, не попадающих в основные категории"""

    document_type: Literal["other"] = "other"
    raw_text: str
    extracted_parameters: List[TestParameter] = Field(default_factory=list)
