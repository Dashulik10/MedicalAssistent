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

from .prompts import get_extraction_prompt

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
        logger.info("Extracting data from image")

        # Encode image to base64
        base64_image = self._encode_image(image)

        # Get extraction prompt
        prompt = get_extraction_prompt()

        # Prepare messages for Groq API
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

        # Try to get completion with one retry
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                result = response.choices[0].message.content
                logger.info("Data extracted successfully")
                return result

            except Exception as e:
                logger.error(
                    f"API call failed (attempt {attempt + 1}/{max_attempts}): {e}"
                )
                if attempt < max_attempts - 1:
                    logger.info("Retrying in 2 seconds...")
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
