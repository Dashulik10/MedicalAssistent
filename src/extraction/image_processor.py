import ollama  # For local LLM inference
import json
from typing import Dict, Any
from .prompts import (  # Import updated prompts
    CLASSIFICATION_PROMPT,
    BLOOD_TEST_PROMPT,
    BIOCHEMISTRY_PROMPT,
    URINE_TEST_PROMPT,
    OTHER_PROMPT,
)


class DataExtractorAgent:
    def __init__(self, model_name: str = "llama3.2-vision:latest"):
        self.model_name = model_name

    def _query_llm(self, prompt: str, image_path: str) -> str:
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_path],
                options={"temperature": 0.0},  # Deterministic for speed
            )
            return response["response"].strip()
        except Exception as e:
            print(f"LLM query failed: {str(e)}")
            return "{}"

    def _classify_type(self, image_path: str) -> str:
        raw_response = self._query_llm(CLASSIFICATION_PROMPT, image_path)
        try:
            print(raw_response)
            classified = json.loads(raw_response)
            return classified.get("type", "other")
        except json.JSONDecodeError:
            print("Classification failed; defaulting to 'other'")
            return "other"

    def _get_extraction_prompt(self, report_type: str) -> str:
        if report_type == "blood_test":
            return BLOOD_TEST_PROMPT
        elif report_type == "biochemistry":
            return BIOCHEMISTRY_PROMPT
        elif report_type == "urine_test":
            return URINE_TEST_PROMPT
        else:
            return OTHER_PROMPT

    def extract_data(self, image_path: str) -> Dict[str, Any]:
        report_type = self._classify_type(image_path)
        print(f"Report type: {report_type}")
        prompt = self._get_extraction_prompt(report_type)
        raw_extraction = self._query_llm(prompt, image_path)
        print(f"Raw extraction: {raw_extraction}")
        try:
            extracted_data = json.loads(raw_extraction)
            extracted_data["report_type"] = (
                report_type  # Add for vector storage metadata
            )
            return extracted_data
        except json.JSONDecodeError:
            print("Extraction parsing failed; returning empty dict")
            return {"report_type": report_type, "error": "Parsing failed"}
