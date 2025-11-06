# prompts.py
# Updated prompts with user's improved versions.
# Simplified JSON structure for efficiency: Flat dicts for parameters, no deep nesting.
# Prompts remain concise for low-end hardware.
# Fixed: Escaped curly braces {{ and }} in JSON examples to prevent str.format() errors.

CLASSIFICATION_PROMPT = """
You are an expert in analyzing medical documents. Your task is to identify the type of medical report in the image.
Possible document types:
1. blood_test - Complete Blood Count (CBC)
Signs: hemoglobin, red blood cells, white blood cells, platelets, leukocyte formula, ESR
2. biochemistry - Blood biochemistry analysis
Signs: glucose, cholesterol, triglycerides, ALT, AST, creatinine, urea, bilirubin
3. urine_test - General urine analysis
Signs: color, clarity, pH, specific gravity, protein, glucose, microscopy
4. other - Other types of medical documents or documents that are difficult to classify
INSTRUCTIONS:
- Carefully examine the image
- Determine which parameters are present in the document
- Choose the most appropriate type from the list above
- If in doubt, choose "other" with low confidence
Respond only with the type in JSON format: {"type": "blood_test"}
Do not add any extra text.
"""

EXTRACTION_BASE_PROMPT = """
You are an expert in analyzing medical documents. Your task is to extract ALL data from the image.
CRITICALLY IMPORTANT:
1. Carefully examine the image and find ALL information
2. Extract patient data: name (patient_name), age (patient_age), gender (patient_gender: "M" or "F")
3. Find test date (test_date in YYYY-MM-DD format), lab name (lab_name), doctor name (doctor_name)
4. For EACH parameter, extract as flat key-value pairs (e.g., "hemoglobin_value": "14.5", "hemoglobin_unit": "g/dL", "hemoglobin_reference": "13-17", "hemoglobin_is_normal": true)
5. If a field or parameter is missing, set it to null (DON'T make up data!)
6. Add any additional parameters not in the main list to "other_parameters" as a flat dict similar to main parameters
7. is_normal: true if value within reference, false if not, null if unknown
8. Be precise with numbers and units
{specific_instructions}
Return the result STRICTLY in the following JSON format (without additional text):
{{
"patient_name": "string or null",
"patient_age": "number or null",
"patient_gender": "M/F or null",
"test_date": "YYYY-MM-DD or null",
"lab_name": "string or null",
"doctor_name": "string or null",
"main_parameters": {{ "param1_value": "value", "param1_unit": "unit", "param1_reference": "range", "param1_is_normal": true/false/null, ... }},
"other_parameters": {{ "extra_param_value": "value", ... }},
"diagnosis": "string or null",
"raw_text": "full extracted text as string"
}}
RETURN ONLY VALID JSON WITHOUT ADDITIONAL TEXT OR EXPLANATIONS!
"""

BLOOD_TEST_PROMPT = EXTRACTION_BASE_PROMPT.format(
    specific_instructions="""
MAIN PARAMETERS (extract as flat keys if present):
- hemoglobin_value, hemoglobin_unit, hemoglobin_reference, hemoglobin_is_normal: Hemoglobin (HGB, Hb)
- rbc_value, rbc_unit, rbc_reference, rbc_is_normal: Red Blood Cells (RBC, Erythrocytes)
- wbc_value, wbc_unit, wbc_reference, wbc_is_normal: White Blood Cells (WBC, Leukocytes)
- platelets_value, platelets_unit, platelets_reference, platelets_is_normal: Platelets (PLT, Thrombocytes)
- hematocrit_value, hematocrit_unit, hematocrit_reference, hematocrit_is_normal: Hematocrit (HCT)
- neutrophils_value, neutrophils_unit, neutrophils_reference, neutrophils_is_normal: Neutrophils (segmented + band)
- lymphocytes_value, lymphocytes_unit, lymphocytes_reference, lymphocytes_is_normal: Lymphocytes
- monocytes_value, monocytes_unit, monocytes_reference, monocytes_is_normal: Monocytes
- eosinophils_value, eosinophils_unit, eosinophils_reference, eosinophils_is_normal: Eosinophils
- basophils_value, basophils_unit, basophils_reference, basophils_is_normal: Basophils
Additional parameters may include:
- esr_value, esr_unit, etc.: ESR (erythrocyte sedimentation rate)
- mcv_value, mcv_unit, etc.: MCV, MCH, MCHC (erythrocyte indices)
- reticulocytes_value, etc.: Reticulocytes
"""
)

BIOCHEMISTRY_PROMPT = EXTRACTION_BASE_PROMPT.format(
    specific_instructions="""
MAIN BIOCHEMISTRY PARAMETERS (extract as flat keys if present):
CARBOHYDRATE METABOLISM:
- glucose_value, glucose_unit, glucose_reference, glucose_is_normal: Glucose (Glu)
LIPID METABOLISM:
- total_cholesterol_value, total_cholesterol_unit, total_cholesterol_reference, total_cholesterol_is_normal: Total Cholesterol (Chol)
- hdl_cholesterol_value, hdl_cholesterol_unit, etc.: HDL (high-density lipoproteins)
- ldl_cholesterol_value, ldl_cholesterol_unit, etc.: LDL (low-density lipoproteins)
- triglycerides_value, triglycerides_unit, etc.: Triglycerides (TG)
LIVER FUNCTION:
- alt_value, alt_unit, etc.: ALT/ALAT (alanine aminotransferase)
- ast_value, ast_unit, etc.: AST/ASAT (aspartate aminotransferase)
- bilirubin_total_value, bilirubin_total_unit, etc.: Total Bilirubin
- bilirubin_direct_value, bilirubin_direct_unit, etc.: Direct Bilirubin
KIDNEY FUNCTION:
- creatinine_value, creatinine_unit, etc.: Creatinine (Crea)
- urea_value, urea_unit, etc.: Urea
- uric_acid_value, uric_acid_unit, etc.: Uric Acid
Additional parameters may include:
- total_protein_value, etc.: Total protein, Albumin
- alp_value, etc.: Alkaline phosphatase (ALP)
- ggt_value, etc.: GGT (gamma-glutamyl transferase)
- Electrolytes: na_value, k_value, cl_value, etc.
"""
)

URINE_TEST_PROMPT = EXTRACTION_BASE_PROMPT.format(
    specific_instructions="""
MAIN PARAMETERS (extract as flat keys if present):
PHYSICAL PROPERTIES:
- color_value, color_unit, color_reference, color_is_normal: Urine color (e.g., yellow, amber, straw)
- clarity_value, clarity_unit, etc.: Clarity (clear, slightly cloudy, cloudy, turbid)
- specific_gravity_value, specific_gravity_unit, etc.: Specific gravity (e.g., 1.010-1.025)
- ph_value, ph_unit, etc.: pH - acidity level (e.g., 5.0-7.0)
CHEMICAL PROPERTIES:
- protein_value, protein_unit, etc.: Protein (negative, trace, +, ++, +++, or value)
- glucose_value, glucose_unit, etc.: Glucose (negative or value)
- ketones_value, ketones_unit, etc.: Ketones/Ketone bodies (negative or value)
- blood_value, blood_unit, etc.: Blood/Hemoglobin (negative or value)
- bilirubin_value, bilirubin_unit, etc.: Bilirubin (negative or value)
- urobilinogen_value, urobilinogen_unit, etc.: Urobilinogen (may be present)
- nitrites_value, nitrites_unit, etc.: Nitrites (negative/positive)
- leukocyte_esterase_value, leukocyte_esterase_unit, etc.: Leukocyte esterase (negative/positive)
MICROSCOPY (SEDIMENT):
- rbc_microscopy_value, rbc_microscopy_unit, etc.: Red blood cells in sediment (per field of view)
- wbc_microscopy_value, wbc_microscopy_unit, etc.: White blood cells in sediment (per field of view)
- epithelial_cells_value, epithelial_cells_unit, etc.: Epithelial cells (type and amount)
- casts_value, casts_unit, etc.: Casts if present (hyaline, granular, etc.)
- crystals_value, crystals_unit, etc.: Crystals if present (type)
- bacteria_value, bacteria_unit, etc.: Bacteria presence
"""
)

OTHER_PROMPT = EXTRACTION_BASE_PROMPT.format(
    specific_instructions="""
INSTRUCTIONS:
1. Extract basic information: patient name, age, gender, date
2. Extract ALL visible text into "raw_text" field
3. If there are any numerical parameters or indicators:
   - Add them to "main_parameters" as flat dict (e.g., "param_value": "value", "param_unit": "unit", etc.)
4. Try to identify what type of medical document this is and include in raw_text
"""
)
