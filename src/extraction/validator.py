"""Validators for image and JSON data."""

import json
import logging
import re
from io import BytesIO
from typing import Any, Union

import numpy as np
from PIL import Image

from configuration.settings import settings

from .schemas import MedicalRecord

logger = logging.getLogger(__name__)


class ImageValidator:
    """Validates and processes medical document images."""

    def validate_image(
        self, image: Union[bytes, BytesIO, Image.Image, np.ndarray]
    ) -> bytes:
        """
        Validate image and resize if necessary.

        Args:
            image: Image as bytes, BytesIO, PIL Image, or numpy ndarray.

        Returns:
            bytes: Validated (and potentially resized) image as bytes.

        Raises:
            ValueError: If image format is invalid or processing fails.
        """
        # Convert input to PIL Image
        pil_image = self._load_image(image)

        # Check image dimensions
        width, height = pil_image.size
        needs_resize = max(width, height) > settings.max_image_size

        # Check file size if we have bytes
        image_bytes = self._image_to_bytes(pil_image)
        file_size_mb = len(image_bytes) / (1024 * 1024)

        if file_size_mb > settings.max_file_size_mb:
            needs_resize = True
            logger.info(
                f"Image file too large ({file_size_mb:.2f}MB) or dimensions ({width}x{height}) "
                f"exceed limits. Resizing and compressing..."
            )
        elif needs_resize:
            logger.info(
                f"Image dimensions ({width}x{height}) exceed max size. Resizing..."
            )

        if needs_resize:
            return self.resize_image(pil_image, settings.max_image_size)

        logger.info(f"Image validated successfully")
        return image_bytes

    def resize_image(
        self, image: Union[bytes, BytesIO, Image.Image, np.ndarray], max_size: int
    ) -> bytes:
        """
        Resize image while maintaining aspect ratio and optimize compression.

        Args:
            image: Image as bytes, BytesIO, PIL Image, or numpy ndarray.
            max_size: Maximum width or height in pixels.

        Returns:
            bytes: Resized and compressed image as bytes.

        Raises:
            ValueError: If image processing fails.
        """
        try:
            # Load image as PIL Image
            pil_image = self._load_image(image)

            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Calculate new dimensions
            width, height = pil_image.size
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))

            # Resize with high-quality LANCZOS filter
            resized_img = pil_image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )

            # Convert to bytes with optimized JPEG compression
            output = BytesIO()
            resized_img.save(output, "JPEG", quality=85, optimize=True)
            image_bytes = output.getvalue()

            # Check final file size
            final_size_mb = len(image_bytes) / (1024 * 1024)
            logger.info(
                f"Image resized to {new_width}x{new_height}, "
                f"compressed to {final_size_mb:.2f}MB"
            )

            return image_bytes
        except Exception as e:
            raise ValueError(f"Failed to resize image: {e}")

    def _load_image(
        self, image: Union[bytes, BytesIO, Image.Image, np.ndarray]
    ) -> Image.Image:
        """
        Load image from various input types into PIL Image.

        Args:
            image: Image as bytes, BytesIO, PIL Image, or numpy ndarray.

        Returns:
            Image.Image: PIL Image object.

        Raises:
            ValueError: If image cannot be loaded or format is invalid.
        """
        try:
            # If already a PIL Image
            if isinstance(image, Image.Image):
                return image

            # If image is bytes
            if isinstance(image, bytes):
                return Image.open(BytesIO(image))

            # If image is a numpy array
            if isinstance(image, np.ndarray):
                # Handle different channel formats
                if len(image.shape) == 2:
                    # Grayscale
                    return Image.fromarray(image)
                if len(image.shape) == 3:
                    if image.shape[2] == 3:
                        # RGB
                        return Image.fromarray(image, mode="RGB")
                    if image.shape[2] == 4:
                        # RGBA
                        return Image.fromarray(image, mode="RGBA")
                    error_msg = f"Unsupported number of channels: {image.shape[2]}"
                    raise ValueError(error_msg)
                error_msg = f"Unsupported image shape: {image.shape}"
                raise ValueError(error_msg)

            # If image is a file-like object (BytesIO)
            if hasattr(image, "read"):
                # Save current position
                current_pos = image.tell() if hasattr(image, "tell") else None
                pil_image = Image.open(image)
                # Reset file pointer if possible
                if hasattr(image, "seek") and current_pos is not None:
                    image.seek(current_pos)
                return pil_image

            error_msg = (
                f"Unsupported image type: {type(image)}. "
                "Expected bytes, BytesIO, PIL Image, or np.ndarray."
            )
            raise TypeError(error_msg)

        except (TypeError, ValueError):
            raise
        except Exception as e:
            error_msg = f"Failed to load image: {e}"
            raise ValueError(error_msg) from e

    def _image_to_bytes(self, pil_image: Image.Image) -> bytes:
        """
        Convert PIL Image to bytes.

        Args:
            pil_image: PIL Image object.

        Returns:
            bytes: Image as bytes in JPEG format.
        """
        output = BytesIO()
        # Convert to RGB if necessary
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pil_image.save(output, "JPEG", quality=95)
        return output.getvalue()


class JsonValidator:
    """Validates and parses JSON responses."""

    def validate_and_parse(self, text: str) -> MedicalRecord:
        """
        Extract JSON from text, parse it, and create a MedicalRecord object.

        Args:
            text: Raw text response potentially containing JSON.

        Returns:
            MedicalRecord: Parsed and validated medical record.

        Raises:
            ValueError: If JSON cannot be extracted or parsed.
        """
        # Extract JSON from text
        json_str = self._extract_json(text)

        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")

        # Clean "N/A" values to None
        cleaned_data = self._clean_data(data)

        # Create and validate Pydantic object
        try:
            medical_record = MedicalRecord(**cleaned_data)
            # TODO: Remove this after testing
            medical_record = self._change_record_patient_id(medical_record)
            logger.info("Medical record parsed and validated successfully")
            return medical_record
        except Exception as e:
            raise ValueError(f"Failed to create MedicalRecord: {e}")

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from text that may contain markdown or other formatting.

        Args:
            text: Text potentially containing JSON.

        Returns:
            str: Extracted JSON string.

        Raises:
            ValueError: If no valid JSON found.
        """
        # Try to find JSON in markdown code blocks first
        markdown_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(markdown_pattern, text, re.DOTALL)
        if match:
            return match.group(1)

        # Try to find JSON object directly
        json_pattern = r"\{.*\}"
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            return match.group(0)

        raise ValueError("No JSON found in response")

    def _clean_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively clean data by converting "N/A" strings to None.

        Args:
            data: Dictionary to clean.

        Returns:
            dict: Cleaned dictionary.
        """
        if isinstance(data, dict):
            return {k: self._clean_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_data(item) for item in data]
        elif isinstance(data, str) and data.upper() in ["N/A", "NA", "NONE", ""]:
            return None
        return data

    def _change_record_patient_id(self, medical_record: MedicalRecord) -> MedicalRecord:
        medical_record.patient_id = 1
        return medical_record
