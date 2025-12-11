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
    def __init__(self):
        self.image_validator = ImageValidator()
        self.extractor = GroqExtractor()
        self.json_validator = JsonValidator()
        logger.info("ExtractionPipeline initialized")

    def process(
        self, image: Union[bytes, BytesIO, Image.Image, np.ndarray]
    ) -> dict[str, Any]:
        logger.info("Processing image...")

        try:
            # Шаг 1: Валидация и подготовка изображения
            logger.info("Step 1: Validating image...")
            validated_image_bytes = self.image_validator.validate_image(image)

            # Шаг 2: Извлечение данных с помощью Groq API
            logger.info("Step 2: Extracting data...")
            raw_response = self.extractor.extract(validated_image_bytes)

            # Шаг 3: Валидация и парсинг JSON
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
