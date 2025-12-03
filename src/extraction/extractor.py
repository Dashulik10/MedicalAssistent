"""Medical data extractor using Groq API."""

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
    """Extracts medical data from images using Groq API with vision model."""

    def __init__(self):
        """Initialize the Groq client."""
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.model_name
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        logger.info("GroqExtractor initialized")

    def extract(self, image: Union[bytes, BytesIO, np.ndarray]) -> str:
        """
        Extract medical data from an image.

        Args:
            image: Image as bytes, BytesIO, or numpy ndarray.

        Returns:
            str: Raw text response from the API.

        Raises:
            ValueError: If extraction fails after retry.
        """
        logger.info("Starting extraction process")

        # Encode image to base64 once
        base64_image = self._encode_image(image)

        # Step 1: Classify the test type
        test_type = self._classify_test_type(base64_image)
        logger.info(f"Detected test type: {test_type}")

        # Step 2: Extract data with appropriate prompt
        result = self._extract_with_type(base64_image, test_type)
        logger.info("Data extracted successfully")

        return result

    def _classify_test_type(self, base64_image: str) -> str:
        """
        Classify the type of medical test from the image.

        Args:
            base64_image: Base64 encoded image string.

        Returns:
            str: Test type - "blood_count", "biochemistry", or "other".

        Raises:
            ValueError: If classification fails.
        """
        logger.info("Classifying test type")

        # Get classification prompt
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
                    temperature=0.0,  # Use 0 for classification for consistency
                    max_tokens=50,  # We only need a short response
                )
                result = response.choices[0].message.content.strip().lower()

                # Validate and normalize the result
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
        """
        Extract data using the appropriate prompt for the test type.

        Args:
            base64_image: Base64 encoded image string.
            test_type: Type of test detected.

        Returns:
            str: Raw text response from the API.

        Raises:
            ValueError: If extraction fails after retry.
        """
        logger.info(f"Extracting data for test type: {test_type}")

        # Get specialized extraction prompt
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
        """
        Encode image to base64 string.

        Args:
            image: Image as bytes, BytesIO, or numpy ndarray.

        Returns:
            str: Base64 encoded image.

        Raises:
            ValueError: If encoding fails.
        """
        try:
            # If image is already bytes
            if isinstance(image, bytes):
                return base64.b64encode(image).decode("utf-8")

            # If image is a numpy array
            if isinstance(image, np.ndarray):
                # Encode ndarray to JPEG bytes
                _, buffer = cv2.imencode(".jpg", image)
                image_bytes = buffer.tobytes()
                return base64.b64encode(image_bytes).decode("utf-8")

            # If image is a file-like object (BytesIO)
            if hasattr(image, "read"):
                image_bytes = image.read()
                # Reset file pointer if possible
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
