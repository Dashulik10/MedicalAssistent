"""Pydantic models for medical data extraction."""

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
    flag: str | None = None  # "H" for high, "L" for low, "N" for normal, etc.


class BloodCountTest(BaseModel):
    """Common blood count (CBC) test parameters."""

    model_config = ConfigDict(str_strip_whitespace=True)

    hemoglobin: TestParameter | None = None
    rbc_count: TestParameter | None = None  # Red Blood Cell count
    wbc_count: TestParameter | None = None  # White Blood Cell count
    platelet_count: TestParameter | None = None
    hematocrit: TestParameter | None = None
    mcv: TestParameter | None = None  # Mean Corpuscular Volume
    mch: TestParameter | None = None  # Mean Corpuscular Hemoglobin
    mchc: TestParameter | None = None  # Mean Corpuscular Hemoglobin Concentration
    neutrophils: TestParameter | None = None
    lymphocytes: TestParameter | None = None
    monocytes: TestParameter | None = None
    eosinophils: TestParameter | None = None
    basophils: TestParameter | None = None
    other_parameters: list[TestParameter] = []  # For any additional parameters


class BiochemistryTest(BaseModel):
    """Common biochemistry test parameters."""

    model_config = ConfigDict(str_strip_whitespace=True)

    glucose: TestParameter | None = None
    creatinine: TestParameter | None = None
    urea: TestParameter | None = None
    bun: TestParameter | None = None  # Blood Urea Nitrogen
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
    sgot_ast: TestParameter | None = None  # SGOT/AST
    sgpt_alt: TestParameter | None = None  # SGPT/ALT
    alkaline_phosphatase: TestParameter | None = None
    other_parameters: list[TestParameter] = []  # For any additional parameters


class MedicalRecord(BaseModel):
    """Represents a complete medical/lab test record extracted from a document."""

    model_config = ConfigDict(str_strip_whitespace=True)

    # Patient Information
    patient_name: str | None = None
    patient_id: str | None = None
    age: str | None = None
    gender: str | None = None

    # Test Information
    test_date: str | None = None  # Date when test was performed (YYYY-MM-DD)
    report_date: str | None = None  # Date when report was issued (YYYY-MM-DD)
    test_type: Literal["blood_count", "biochemistry", "other"] | None = None

    # Test Results
    blood_count: BloodCountTest | None = None
    biochemistry: BiochemistryTest | None = None

    # Additional Information
    doctor_name: str | None = None
    lab_name: str | None = None
    notes: str | None = None

    @field_validator("test_date", "report_date")
    @classmethod
    def validate_date_format(cls, v: str | None) -> str | None:
        """Validate that date is in YYYY-MM-DD format."""
        if v is None:
            return v

        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            # If date doesn't match format, return None instead of raising error
            return None
        else:
            return v


# Legacy aliases for backward compatibility
Medication = TestParameter
VitalSigns = BloodCountTest
