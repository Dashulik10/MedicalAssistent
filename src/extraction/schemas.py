from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class TestParameter(BaseModel):
    """Represents a single test parameter with its value and reference range."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str
    value: str | None = None
    unit: str | None = None
    reference_range: str | None = None
    flag: str | None = None


class BloodCountTest(BaseModel):
    """Common blood count (CBC) test parameters."""

    model_config = ConfigDict(str_strip_whitespace=True)

    hemoglobin: TestParameter | None = None
    rbc_count: TestParameter | None = None
    wbc_count: TestParameter | None = None
    platelet_count: TestParameter | None = None
    hematocrit: TestParameter | None = None
    mcv: TestParameter | None = None
    mch: TestParameter | None = None
    mchc: TestParameter | None = None
    neutrophils: TestParameter | None = None
    lymphocytes: TestParameter | None = None
    monocytes: TestParameter | None = None
    eosinophils: TestParameter | None = None
    basophils: TestParameter | None = None
    other_parameters: list[TestParameter] = []


class BiochemistryTest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    glucose: TestParameter | None = None
    creatinine: TestParameter | None = None
    urea: TestParameter | None = None
    bun: TestParameter | None = None
    uric_acid: TestParameter | None = None
    sodium: TestParameter | None = None
    potassium: TestParameter | None = None
    chloride: TestParameter | None = None
    calcium: TestParameter | None = None
    phosphorus: TestParameter | None = None
    total_protein: TestParameter | None = None
    albumin: TestParameter | None = None
    globulin: TestParameter | None = None
    bilirubin_total: TestParameter | None = None
    bilirubin_direct: TestParameter | None = None
    sgot_ast: TestParameter | None = None
    sgpt_alt: TestParameter | None = None
    alkaline_phosphatase: TestParameter | None = None
    other_parameters: list[TestParameter] = []


class MedicalRecord(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    patient_name: str | None = None
    patient_id: str | None = None
    age: str | None = None
    gender: str | None = None

    test_date: str | None = None
    report_date: str | None = None
    test_type: Literal["blood_count", "biochemistry", "other"] | None = None

    blood_count: BloodCountTest | None = None
    biochemistry: BiochemistryTest | None = None

    doctor_name: str | None = None
    lab_name: str | None = None
    notes: str | None = None

    @field_validator("test_date", "report_date")
    @classmethod
    def validate_date_format(cls, v: str | None) -> str | None:
        if v is None:
            return v

        # Популярные форматы дат для парсинга
        date_formats = [
            "%Y-%m-%d",
            "%d.%m.%Y",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%Y/%m/%d",
            "%d.%m.%y",
            "%d/%m/%y",
            "%Y.%m.%d",
        ]

        for date_format in date_formats:
            try:
                parsed_date = datetime.strptime(v, date_format)
                # Преобразуем к единому формату YYYY-MM-DD
                return parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # Если ни один формат не подошел, возвращаем None
        return None
