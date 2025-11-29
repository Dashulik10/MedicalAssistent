import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Union

import cv2
import numpy as np


@dataclass
class PreprocessingConfig:
    """Конфигурация параметров предобработки"""

    max_dimension: int = 3000

    # Удаление фона
    background_kernel_divisor: int = 30
    background_weight: float = 0.7

    # Удаление шума
    median_blur_size: int = 3
    bilateral_filter: bool = False
    bilateral_d: int = 9
    bilateral_sigma_color: int = 75
    bilateral_sigma_space: int = 75

    # CLAHE параметры
    clahe_clip_limit: float = 3.0
    clahe_tile_size: int = 8

    # Резкость
    sharpen_amount: float = 1.5
    sharpen_blur_sigma: float = 1.0

    # Бинаризация (опционально - для LLM лучше grayscale!)
    use_binarization: bool = False
    binarization_block_size: int = 51
    binarization_c: int = 5

    # Контрастность (альтернатива бинаризации)
    contrast_enhancement: bool = True
    gamma_correction: float = 1.0

    # Вывод
    output_format: str = "PNG"
    output_quality: int = 100


ImageInput = Union[np.ndarray, bytes, BytesIO]


# Доступные пресеты:
# - "medical_llm" - ЛУЧШИЙ для LLM (сбалансированный)
# - "medical_llm_sharp" - МАКСИМАЛЬНАЯ четкость для LLM
# - "medical_ocr" - для традиционного OCR (С бинаризацией)
# - "aggressive" - для очень плохих изображений
# - "balanced" - универсальный
# - "gentle" - для хороших изображений
class MedicalImagePreprocessor:
    """
    Продвинутый препроцессор медицинских отчетов.

    Использует state-of-the-art методы для максимальной читаемости текста:
    - CLAHE для адаптивного контраста
    - Билатеральная фильтрация для удаления шума
    - Адаптивная бинаризация
    - Морфологические операции
    - Deskewing (выравнивание наклона)
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.logger = logging.getLogger(__name__)

    def preprocess(
        self,
        image_input: ImageInput,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Предобработка медицинского отчета - ОПТИМИЗИРОВАНО ДЛЯ LLM.

        Args:
            image_input: Изображение в виде numpy array, bytes или BytesIO
            save_path: Путь для сохранения (опционально)

        Returns:
            Обработанное изображение (RGB numpy array)
        """
        try:
            # 1. Загрузка изображения
            image = self._load_image(image_input)

            # 2. Изменение размера
            image = self._resize_image(image)

            # 3. Конвертация в grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # 4. Проверка и исправление инверсии
            gray = self._fix_inversion(gray)

            # 5. Удаление фона (выравнивание освещения)
            gray = self._remove_background(gray)

            # 6. Удаление шума
            if self.config.bilateral_filter:
                gray = self._bilateral_filter(gray)
            else:
                gray = cv2.medianBlur(gray, self.config.median_blur_size)

            # 7. CLAHE - адаптивное увеличение контраста
            gray = self._apply_clahe(gray)

            # 8. Легкое повышение резкости
            gray = self._sharpen_light(gray)

            # 9A. Бинаризация (опционально, для OCR систем)
            if self.config.use_binarization:
                result = self._smart_binarize(gray)
            # 9B. Улучшение контраста (для LLM - ЛУЧШЕ!)
            else:
                result = self._enhance_contrast(gray)

            # 10. Конвертация в RGB
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

            # 11. Сохранение
            if save_path:
                self._save_image(result, save_path)

            return result

        except Exception as e:
            self.logger.error(f"Ошибка предобработки: {e}", exc_info=True)
            raise

    def _load_image(self, image_input: ImageInput) -> np.ndarray:
        """Загрузка изображения из bytes, BytesIO или ndarray"""
        if isinstance(image_input, np.ndarray):
            image = image_input.copy()
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif isinstance(image_input, bytes):
            # Decode bytes to numpy array
            nparr = np.frombuffer(image_input, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Не удалось декодировать изображение из bytes")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, BytesIO):
            # Read from BytesIO and decode
            image_bytes = image_input.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Не удалось декодировать изображение из BytesIO")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise TypeError(
                f"Неподдерживаемый тип изображения: {type(image_input)}. "
                "Поддерживаются: np.ndarray, bytes, BytesIO"
            )

        return image

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Изменение размера с сохранением пропорций"""
        h, w = image.shape[:2]

        if max(h, w) > self.config.max_dimension:
            scale = self.config.max_dimension / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        return image

    def _fix_inversion(self, gray: np.ndarray) -> np.ndarray:
        """
        Исправление инверсии цветов.
        """
        h, w = gray.shape
        corner_size = min(h, w) // 10

        corners = [
            gray[:corner_size, :corner_size],
            gray[:corner_size, -corner_size:],
            gray[-corner_size:, :corner_size],
            gray[-corner_size:, -corner_size:],
        ]

        avg_corner_brightness = np.mean([np.mean(corner) for corner in corners])

        if avg_corner_brightness < 127:
            return 255 - gray

        return gray

    def _remove_background(self, gray: np.ndarray) -> np.ndarray:
        """
        Улучшенное удаление фона и выравнивание освещения.

        Использует более мягкий подход для медицинских документов.
        """
        # Оценка фона с помощью сильного размытия
        kernel_size = max(gray.shape) // self.config.background_kernel_divisor
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(kernel_size, 15)

        # Используем Gaussian blur для оценки фона (мягче чем morphology)
        background = cv2.GaussianBlur(gray, (0, 0), kernel_size / 3)

        # Вычитание фона - улучшенная формула
        gray_float = gray.astype(np.float32)
        background_float = background.astype(np.float32)

        # Нормализуем: (original - background) + white_level
        # Это сохраняет текст и делает фон белым
        result = gray_float - (background_float - 255) * self.config.background_weight
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Дополнительная нормализация для выравнивания яркости
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

        return result

    def _bilateral_filter(self, gray: np.ndarray) -> np.ndarray:
        """
        Билатеральная фильтрация - удаляет шум, сохраняя края.

        Идеально для текста: шум удаляется, края букв остаются четкими.
        """
        filtered = cv2.bilateralFilter(
            gray,
            d=self.config.bilateral_d,
            sigmaColor=self.config.bilateral_sigma_color,
            sigmaSpace=self.config.bilateral_sigma_space,
        )
        return filtered

    def _apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """
        CLAHE - Contrast Limited Adaptive Histogram Equalization.

        Лучше обычного увеличения контраста, так как работает адаптивно
        для разных областей изображения.
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_tile_size, self.config.clahe_tile_size),
        )
        enhanced = clahe.apply(gray)
        return enhanced

    def _sharpen_advanced(self, gray: np.ndarray) -> np.ndarray:
        """
        Продвинутое повышение резкости с использованием kernel convolution.
        """
        # Применяем kernel для резкости
        sharpened = cv2.filter2D(gray, -1, self.config.sharpen_kernel)

        # Дополнительный Unsharp Masking для усиления
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
        unsharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

        # Комбинируем оба метода
        result = cv2.addWeighted(sharpened, 0.7, unsharp, 0.3, 0)

        return result

    def _sharpen_light(self, gray: np.ndarray) -> np.ndarray:
        """
        Легкое повышение резкости - более мягкое для сохранения деталей.
        """
        # Unsharp masking с настраиваемыми параметрами
        blurred = cv2.GaussianBlur(gray, (0, 0), self.config.sharpen_blur_sigma)
        # sharpen_amount контролирует силу эффекта
        weight_sharp = 1.0 + (self.config.sharpen_amount - 1.0)
        weight_blur = -(self.config.sharpen_amount - 1.0)
        sharpened = cv2.addWeighted(gray, weight_sharp, blurred, weight_blur, 0)
        return sharpened

    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """
        Улучшение контраста БЕЗ бинаризации - ОПТИМАЛЬНО ДЛЯ LLM!

        Создает высококонтрастное grayscale изображение с четким текстом.
        """
        # Гамма-коррекция (опционально)
        if self.config.gamma_correction != 1.0:
            gray = self._apply_gamma(gray, self.config.gamma_correction)

        # Нормализация гистограммы для максимального контраста
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Улучшенная контрастность через растяжение гистограммы
        # Находим percentiles чтобы избежать влияния выбросов
        p2, p98 = np.percentile(gray, (2, 98))

        # Растягиваем контраст между 2% и 98% перцентилями
        gray = np.clip((gray - p2) * (255 / (p98 - p2)), 0, 255).astype(np.uint8)

        # Дополнительное усиление контраста
        if self.config.contrast_enhancement:
            # Используем адаптивную гамма-коррекцию для усиления текста
            mean_intensity = np.mean(gray)
            if mean_intensity > 127:  # Светлое изображение
                # Делаем темные области темнее
                gamma = 0.8
                gray = self._apply_gamma(gray, gamma)

        return gray

    def _apply_gamma(self, gray: np.ndarray, gamma: float) -> np.ndarray:
        """
        Гамма-коррекция для настройки яркости.
        gamma < 1.0: делает изображение светлее
        gamma > 1.0: делает изображение темнее
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
            np.uint8
        )
        return cv2.LUT(gray, table)

    def _smart_binarize(self, gray: np.ndarray) -> np.ndarray:
        """
        Умная бинаризация БЕЗ черных пятен.

        Использует адаптивную бинаризацию с настраиваемыми параметрами.
        """
        # Адаптивная бинаризация с настраиваемым размером блока
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=self.config.binarization_block_size,
            C=self.config.binarization_c,
        )

        # Проверка на инверсию результата
        if np.mean(binary) < 127:
            binary = 255 - binary

        return binary

    def _adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        """
        Адаптивная бинаризация - лучше работает с неравномерным освещением.

        Использует локальные threshold для каждой области.
        """
        # Адаптивный порог методом Гаусса
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config.adaptive_block_size,
            self.config.adaptive_c,
        )

        return binary

    def _morphological_operations(self, binary: np.ndarray) -> np.ndarray:
        """
        Морфологические операции для улучшения структуры текста.

        - Closing: заполняет небольшие дырки в буквах
        - Opening: удаляет мелкий шум
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.config.morph_kernel_size, self.config.morph_kernel_size),
        )

        # Closing - соединяет близкие пиксели (улучшает связность букв)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Opening - удаляет мелкие точки шума
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

        return opened

    def _deskew(self, binary: np.ndarray) -> np.ndarray:
        """
        Выравнивание наклона изображения (deskewing).

        Находит угол наклона текста и поворачивает изображение.
        """
        # Инвертируем для анализа (OpenCV expects white text on black)
        inverted = 255 - binary

        # Находим координаты всех белых пикселей
        coords = np.column_stack(np.where(inverted > 0))

        if len(coords) < 10:
            return binary  # Недостаточно данных

        # Находим минимальный bounding box с углом
        angle = cv2.minAreaRect(coords)[-1]

        # Корректируем угол
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        # Поворачиваем только если наклон значительный
        if abs(angle) > 0.5:
            h, w = binary.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                binary,
                M,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            return rotated

        return binary

    def _final_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """
        Финальная очистка изображения.
        """
        # Удаление очень мелких компонентов (шум)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            255 - binary, connectivity=8
        )

        # Минимальный размер компонента (пикселей)
        min_size = 15

        # Создаем маску для больших компонентов
        mask = np.zeros_like(binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                mask[labels == i] = 255

        # Инвертируем обратно (белый фон)
        result = 255 - mask

        # Финальная нормализация контраста
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

        return result

    def _save_image(self, image: np.ndarray, path: str):
        """Сохранение в высоком качестве"""
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.config.output_format.upper() == "PNG":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
        else:
            params = [cv2.IMWRITE_JPEG_QUALITY, self.config.output_quality]

        cv2.imwrite(path, image_bgr, params)
