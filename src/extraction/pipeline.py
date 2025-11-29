"""Pipeline for orchestrating medical data extraction."""

import logging
from io import BytesIO
from typing import Any, Union

import numpy as np
from PIL import Image

from .extractor import GroqExtractor
from .schemas import MedicalRecord
from .validator import ImageValidator, JsonValidator

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    Orchestrates the complete medical data extraction pipeline.
    This pipeline validates images, extracts data using Groq API,
    and validates the resulting JSON data.
    """

    def __init__(self):
        """Initialize the pipeline components."""
        self.image_validator = ImageValidator()
        self.extractor = GroqExtractor()
        self.json_validator = JsonValidator()
        logger.info("ExtractionPipeline initialized")

    def process(
        self, image: Union[bytes, BytesIO, Image.Image, np.ndarray]
    ) -> dict[str, Any]:
        """
        Process a medical document image and extract structured data.

        Args:
            image: Image as bytes, BytesIO, PIL Image, or numpy ndarray.

        Returns:
            dict: Result dictionary with keys:
                - status: "success" or "error"
                - data: MedicalRecord object if successful, None otherwise
                - error: Error message if failed, None otherwise
        """
        logger.info("Processing image...")

        try:
            # Step 1: Validate and prepare image
            logger.info("Step 1: Validating image...")
            validated_image_bytes = self.image_validator.validate_image(image)

            # Step 2: Extract data using Groq API
            logger.info("Step 2: Extracting data...")
            raw_response = self.extractor.extract(validated_image_bytes)

            # Step 3: Validate and parse JSON
            logger.info("Step 3: Validating and parsing JSON...")
            print(raw_response)
            medical_record = self.json_validator.validate_and_parse(raw_response)

            logger.info("Successfully processed image")
            return {
                "status": "success",
                "data": medical_record,
                "error": None,
            }

        except ValueError as e:
            error_msg = f"Validation error: {e}"
            logger.error(error_msg)
            return {
                "status": "error",
                "data": None,
                "error": error_msg,
            }

        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(error_msg)
            return {
                "status": "error",
                "data": None,
                "error": error_msg,
            }
