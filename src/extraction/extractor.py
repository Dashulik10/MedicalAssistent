import base64
import logging
import time
from io import BytesIO
from typing import Union

import cv2
import numpy as np
from groq import Groq

from configuration.settings import settings

from .prompts import get_extraction_prompt, get_classification_prompt

logger = logging.getLogger(__name__)


class GroqExtractor:
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.model_name
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        logger.info("GroqExtractor initialized")

    def extract(self, image: Union[bytes, BytesIO, np.ndarray]) -> str:
        logger.info("Starting extraction process")

        # Преобразование изображения в формат base64
        base64_image = self._encode_image(image)

        # Шаг 1: Классификация типа теста
        test_type = self._classify_test_type(base64_image)
        logger.info(f"Detected test type: {test_type}")

        # Шаг 2: Извлечение данных с соответствующим промптом
        result = self._extract_with_type(base64_image, test_type)
        logger.info("Data extracted successfully")

        return result

    def _classify_test_type(self, base64_image: str) -> str:
        logger.info("Classifying test type")

        # Получение промпта для классификации
        prompt = get_classification_prompt()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ]

        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=50,  # Нам нужен короткий ответ
                )
                result = response.choices[0].message.content.strip().lower()

                if "blood" in result or "cbc" in result or "hemoglobin" in result:
                    return "blood_count"
                elif (
                    "biochem" in result or "glucose" in result or "creatinine" in result
                ):
                    return "biochemistry"
                else:
                    return "other"

            except Exception as e:
                logger.error(
                    f"Classification failed (attempt {attempt + 1}/{max_attempts}): {e}"
                )
                if attempt < max_attempts - 1:
                    logger.info("Retrying classification in 1 second...")
                    time.sleep(1)
                else:
                    logger.warning(
                        f"Failed to classify test type, defaulting to 'other': {e}"
                    )
                    return "other"

    def _extract_with_type(self, base64_image: str, test_type: str) -> str:
        logger.info(f"Extracting data for test type: {test_type}")

        prompt = get_extraction_prompt(test_type=test_type)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ]

        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content

            except Exception as e:
                logger.error(
                    f"Extraction failed (attempt {attempt + 1}/{max_attempts}): {e}"
                )
                if attempt < max_attempts - 1:
                    logger.info("Retrying extraction in 2 seconds...")
                    time.sleep(2)
                else:
                    raise ValueError(
                        f"Failed to extract data after {max_attempts} attempts: {e}"
                    )

    def _encode_image(self, image: Union[bytes, BytesIO, np.ndarray]) -> str:
        try:
            if isinstance(image, bytes):
                return base64.b64encode(image).decode("utf-8")

            if isinstance(image, np.ndarray):
                _, buffer = cv2.imencode(".jpg", image)
                image_bytes = buffer.tobytes()
                return base64.b64encode(image_bytes).decode("utf-8")

            if hasattr(image, "read"):
                image_bytes = image.read()
                if hasattr(image, "seek"):
                    image.seek(0)
                return base64.b64encode(image_bytes).decode("utf-8")

            error_msg = (
                f"Unsupported image type: {type(image)}. "
                "Expected bytes, BytesIO, or np.ndarray."
            )
            raise TypeError(error_msg)

        except (TypeError, ValueError):
            raise
        except Exception as e:
            error_msg = f"Failed to encode image: {e}"
            raise ValueError(error_msg) from e
