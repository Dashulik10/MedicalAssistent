def get_extraction_prompt(test_type: str = "other") -> str:
    """
    Get the prompt for extracting medical data from document images.

    Args:
        test_type: Type of test to extract. Options: "auto", "blood_count", "biochemistry".
                   "auto" will detect the test type automatically.

    Returns:
        str: The extraction prompt with instructions and schema.

    """
    base_instructions = """You are a medical laboratory report extraction assistant. Your task is to extract structured information from lab test reports and return it as valid JSON.

PATIENT INFORMATION:
Extract the following patient details:
- patient_name: Full name of the patient
- patient_id: Patient ID, medical record number, or lab number
- age: Patient age (with unit if provided, e.g., "45 years", "3 months")
- gender: Patient gender (Male/Female/Other)

TEST INFORMATION:
- test_date: Date when the test was performed (YYYY-MM-DD format)
- report_date: Date when the report was issued (YYYY-MM-DD format)
- test_type: Type of test - "blood_count", "biochemistry", or "other"
- doctor_name: Name of the referring/attending physician
- lab_name: Name of the laboratory

TEST RESULTS:
For each test parameter found, extract:
- name: Parameter name (e.g., "Hemoglobin", "Glucose")
- value: Measured value
- unit: Unit of measurement (e.g., "g/dL", "mg/dL", "cells/μL")
- reference_range: Normal/reference range (e.g., "12-16", "70-110 mg/dL")
- flag: Status flag if present ("H" for High, "L" for Low, "N" for Normal, "*" for abnormal)
"""

    blood_count_details = """
BLOOD COUNT (CBC) PARAMETERS:
Common parameters to look for:
- Hemoglobin (Hb/HGB)
- RBC Count (Red Blood Cell count)
- WBC Count (White Blood Cell count)
- Platelet Count
- Hematocrit (HCT)
- MCV (Mean Corpuscular Volume)
- MCH (Mean Corpuscular Hemoglobin)
- MCHC (Mean Corpuscular Hemoglobin Concentration)
- Neutrophils (absolute or percentage)
- Lymphocytes (absolute or percentage)
- Monocytes
- Eosinophils
- Basophils
- Any other blood count parameters found
"""

    biochemistry_details = """
BIOCHEMISTRY PARAMETERS:
Common parameters to look for:
- Glucose (Blood Sugar/FBS/RBS/PPBS)
- Creatinine
- Urea/Blood Urea
- BUN (Blood Urea Nitrogen)
- Uric Acid
- Sodium (Na)
- Potassium (K)
- Chloride (Cl)
- Calcium (Ca)
- Phosphorus
- Total Protein
- Albumin
- Globulin
- Bilirubin Total
- Bilirubin Direct
- SGOT/AST (Aspartate Aminotransferase)
- SGPT/ALT (Alanine Aminotransferase)
- Alkaline Phosphatase (ALP)
- Any other biochemistry parameters found
"""

    schema = """
JSON SCHEMA STRUCTURE:
{
  "patient_name": "string or null",
  "patient_id": "string or null",
  "age": "string or null",
  "gender": "string or null",
  "test_date": "string (YYYY-MM-DD) or null",
  "report_date": "string (YYYY-MM-DD) or null",
  "test_type": "blood_count" | "biochemistry" | "other" | null,
  "doctor_name": "string or null",
  "lab_name": "string or null",
  "notes": "string or null",
  "blood_count": {
    "hemoglobin": {"name": "Hemoglobin", "value": "14.5", "unit": "g/dL", "reference_range": "12-16", "flag": "N"},
    "rbc_count": {"name": "RBC Count", "value": "4.5", "unit": "million/μL", "reference_range": "4-6", "flag": null},
    "wbc_count": {...},
    "platelet_count": {...},
    ... (other blood count parameters),
    "other_parameters": [
      {"name": "Custom Parameter", "value": "...", "unit": "...", "reference_range": "...", "flag": "..."}
    ]
  } or null,
  "biochemistry": {
    "glucose": {"name": "Glucose", "value": "95", "unit": "mg/dL", "reference_range": "70-110", "flag": "N"},
    "creatinine": {...},
    "urea": {...},
    ... (other biochemistry parameters),
    "other_parameters": [...]
  } or null
}
"""

    rules = """
IMPORTANT EXTRACTION RULES:
1. Return ONLY valid JSON without markdown code blocks, comments, or formatting
2. Use null for missing, unclear, or not applicable information
3. Dates must be in YYYY-MM-DD format (convert if needed)
4. Preserve original parameter names in the "name" field
5. Extract values and units separately
6. If a parameter is not found, set it to null (don't create empty objects)
7. Use "other_parameters" array for any parameters not explicitly listed
8. Determine test_type based on the parameters found in the report
9. Ensure all JSON is properly formatted and parseable
10. If multiple values exist for same parameter, use the most recent or primary value
"""

    example = """
EXAMPLE - Blood Count Report:
Input: Lab report for patient Rajesh Kumar, ID: BLR-0425-PA-0041660, Age: 45 years, Male
Test Date: 27-04-2025
Hemoglobin: 13.5 g/dL (Normal Range: 13-17)
WBC Count: 8500 cells/μL (4000-11000) *Normal
Platelet Count: 250000 cells/μL (150000-450000)

Output:
{
  "patient_name": "Rajesh Kumar",
  "patient_id": "BLR-0425-PA-0041660",
  "age": "45 years",
  "gender": "Male",
  "test_date": "2025-04-27",
  "report_date": null,
  "test_type": "blood_count",
  "doctor_name": null,
  "lab_name": null,
  "notes": null,
  "blood_count": {
    "hemoglobin": {
      "name": "Hemoglobin",
      "value": "13.5",
      "unit": "g/dL",
      "reference_range": "13-17",
      "flag": "N"
    },
    "rbc_count": null,
    "wbc_count": {
      "name": "WBC Count",
      "value": "8500",
      "unit": "cells/μL",
      "reference_range": "4000-11000",
      "flag": "N"
    },
    "platelet_count": {
      "name": "Platelet Count",
      "value": "250000",
      "unit": "cells/μL",
      "reference_range": "150000-450000",
      "flag": null
    },
    "hematocrit": null,
    "mcv": null,
    "mch": null,
    "mchc": null,
    "neutrophils": null,
    "lymphocytes": null,
    "monocytes": null,
    "eosinophils": null,
    "basophils": null,
    "other_parameters": []
  },
  "biochemistry": null
}
"""

    closing = """
Now extract ALL the medical data from the provided lab report image and return ONLY the JSON output.
Be thorough and extract all visible parameters, patient information, and dates."""

    # Build the full prompt based on test type
    if test_type == "blood_count":
        return (
            base_instructions + blood_count_details + schema + rules + example + closing
        )

    if test_type == "biochemistry":
        return (
            base_instructions
            + biochemistry_details
            + schema
            + rules
            + example.replace("Blood Count", "Biochemistry")
            + closing
        )

    # other
    return (
        base_instructions
        + blood_count_details
        + biochemistry_details
        + schema
        + rules
        + example
        + closing
    )


def get_classification_prompt() -> str:
    """
    Get the prompt for classifying the type of medical test.

    Returns:
        str: The classification prompt.
    """
    return """You are a medical document classifier. Analyze this laboratory report image and determine the type of test.

Look for these indicators:

BLOOD COUNT (CBC):
- Parameters like: Hemoglobin, RBC, WBC, Platelets, Hematocrit, MCV, MCH, MCHC
- Differential counts: Neutrophils, Lymphocytes, Monocytes, Eosinophils, Basophils
- Keywords: "Complete Blood Count", "CBC", "Hemogram"

BIOCHEMISTRY:
- Parameters like: Glucose, Creatinine, Urea, BUN, Uric Acid, Electrolytes (Na, K, Cl)
- Liver function: SGOT, SGPT, Bilirubin, Alkaline Phosphatase, Total Protein, Albumin
- Keywords: "Biochemistry", "Chemistry Panel", "Metabolic Panel"

Respond with ONLY ONE WORD:
- "blood_count" if it's a CBC/blood count test
- "biochemistry" if it's a biochemistry/chemistry test
- "other" if it's neither or unclear

Your response:"""
