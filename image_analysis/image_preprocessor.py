import cv2
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
from typing import Optional, Tuple, Union
import logging
from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Конфигурация параметров предобработки"""
    # Размер изображения
    target_size: Optional[Tuple[int, int]] = None  # None = не изменять размер
    max_dimension: int = 2048  # Максимальный размер стороны
    
    # Параметры улучшения качества
    denoise_strength: int = 3  # 1-9, нечетное число
    clahe_clip_limit: float = 2.0  # 1.0-4.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    
    # Бинаризация
    use_binarization: bool = False  # Для очень плохих сканов
    adaptive_block_size: int = 11  # Должно быть нечетным
    adaptive_c: int = 2
    
    # Коррекция перспективы
    auto_deskew: bool = True
    deskew_angle_threshold: float = 0.5  # градусы
    
    # Коррекция яркости/контраста
    auto_brightness: bool = True
    auto_contrast: bool = True
    contrast_factor: float = 1.2  # 1.0-2.0
    brightness_factor: float = 1.0  # 0.5-1.5
    
    # Удаление шума
    remove_shadows: bool = False
    remove_artifacts: bool = True
    
    # Увеличение резкости
    sharpen: bool = True
    sharpen_strength: float = 1.5
    
    # Выходной формат
    output_format: str = 'PNG'  # PNG для лучшего качества
    output_quality: int = 100


class MedicalImagePreprocessor:
    """
    Профессиональный препроцессор для медицинских отчетов.
    
    Выполняет комплексную обработку изображений для максимального
    улучшения читаемости текста моделями Vision LLM.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Args:
            config: Конфигурация предобработки. Если None, используются настройки по умолчанию.
        """
        self.config = config or PreprocessingConfig()
        self.logger = logging.getLogger(__name__)
        
    def preprocess(self, 
                   image_input: Union[str, Path, np.ndarray, Image.Image],
                   save_path: Optional[str] = None) -> np.ndarray:
        """
        Полная предобработка медицинского отчета.
        
        Args:
            image_input: Путь к изображению, numpy array или PIL Image
            save_path: Путь для сохранения результата (опционально)
            
        Returns:
            Обработанное изображение как numpy array (RGB)
        """
        try:
            # 1. Загрузка изображения
            image = self._load_image(image_input)
            original_shape = image.shape
            self.logger.info(f"Загружено изображение: {original_shape}")
            
            # 2. Изменение размера (если нужно)
            image = self._resize_image(image)
            
            # 3. Удаление теней и артефактов
            if self.config.remove_shadows:
                image = self._remove_shadows(image)
            
            if self.config.remove_artifacts:
                image = self._remove_artifacts(image)
            
            # 4. Коррекция перспективы (выравнивание)
            if self.config.auto_deskew:
                image = self._deskew_image(image)
            
            # 5. Улучшение яркости и контраста
            if self.config.auto_brightness or self.config.auto_contrast:
                image = self._enhance_brightness_contrast(image)
            
            # 6. Удаление шума
            image = self._denoise(image)
            
            # 7. Увеличение резкости
            if self.config.sharpen:
                image = self._sharpen_image(image)
            
            # 8. Улучшение контраста (CLAHE)
            image = self._apply_clahe(image)
            
            # 9. Бинаризация (опционально, для очень плохого качества)
            if self.config.use_binarization:
                image = self._binarize(image)
            else:
                # Легкая бинаризация для улучшения читаемости
                image = self._soft_binarization(image)
            
            # 10. Финальная очистка
            image = self._final_cleanup(image)
            
            # 11. Сохранение (если указан путь)
            if save_path:
                self._save_image(image, save_path)
                self.logger.info(f"Сохранено в: {save_path}")
            
            self.logger.info("Предобработка завершена успешно")
            return image
            
        except Exception as e:
            self.logger.error(f"Ошибка предобработки: {e}")
            raise
    
    def _load_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Загрузка изображения из различных источников"""
        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input))
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_input}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            image = np.array(image_input.convert('RGB'))
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            raise TypeError("Неподдерживаемый тип входного изображения")
        
        return image
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Изменение размера с сохранением пропорций"""
        h, w = image.shape[:2]
        
        # Проверка максимального размера
        if max(h, w) > self.config.max_dimension:
            scale = self.config.max_dimension / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            self.logger.info(f"Изменен размер: {w}x{h} -> {new_w}x{new_h}")
        
        # Целевой размер (если указан)
        if self.config.target_size:
            image = cv2.resize(image, self.config.target_size, interpolation=cv2.INTER_LANCZOS4)
        
        return image
    
    def _remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """Удаление теней с документа"""
        # Конвертация в LAB для работы с яркостью
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Применение морфологической операции для выделения фона
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        background = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)
        
        # Вычитание фона
        l_normalized = cv2.subtract(background, l)
        l_normalized = cv2.normalize(l_normalized, None, 0, 255, cv2.NORM_MINMAX)
        
        # Инвертирование
        l_final = 255 - l_normalized
        
        # Объединение обратно
        lab_final = cv2.merge([l_final, a, b])
        result = cv2.cvtColor(lab_final, cv2.COLOR_LAB2RGB)
        
        return result
    
    def _remove_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Удаление артефактов и шума"""
        # Билатеральная фильтрация (сохраняет края)
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        return filtered
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Автоматическое выравнивание перекошенного изображения"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Бинаризация для определения угла
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Определение угла наклона
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        # Корректировка угла
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Поворот только если угол значительный
        if abs(angle) > self.config.deskew_angle_threshold:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE
            )
            self.logger.info(f"Скорректирован угол: {angle:.2f}°")
            return rotated
        
        return image
    
    def _enhance_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Автоматическое улучшение яркости и контраста"""
        # Конвертация в PIL для использования ImageEnhance
        pil_image = Image.fromarray(image)
        
        # Улучшение контраста
        if self.config.auto_contrast:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(self.config.contrast_factor)
        
        # Улучшение яркости
        if self.config.auto_brightness:
            # Автоматическое определение необходимой яркости
            gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
            mean_brightness = np.mean(gray)
            
            # Корректировка если слишком темное или светлое
            if mean_brightness < 100:
                brightness_factor = 1.3
            elif mean_brightness > 180:
                brightness_factor = 0.9
            else:
                brightness_factor = self.config.brightness_factor
            
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness_factor)
        
        return np.array(pil_image)
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Удаление шума с сохранением деталей"""
        # Медианная фильтрация для удаления импульсного шума
        denoised = cv2.medianBlur(image, self.config.denoise_strength)
        
        # Дополнительная фильтрация для цветных изображений
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                denoised,
                None,
                h=10,
                hColor=10,
                templateWindowSize=7,
                searchWindowSize=21
            )
        
        return denoised
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Увеличение резкости для улучшения читаемости текста"""
        # Ядро для повышения резкости
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ]) * self.config.sharpen_strength / 9
        
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Ограничение значений
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Применение CLAHE для улучшения локального контраста"""
        # Конвертация в LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Применение CLAHE к каналу яркости
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size
        )
        l_clahe = clahe.apply(l)
        
        # Объединение обратно
        lab_clahe = cv2.merge([l_clahe, a, b])
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        return result
    
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Агрессивная бинаризация (для очень плохого качества)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Адаптивная пороговая обработка
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config.adaptive_block_size,
            self.config.adaptive_c
        )
        
        # Конвертация обратно в RGB
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        return result
    
    def _soft_binarization(self, image: np.ndarray) -> np.ndarray:
        """Мягкая бинаризация для улучшения читаемости без потери деталей"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Метод Otsu для автоматического определения порога
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Смешивание с оригиналом (70% обработанное, 30% оригинал)
        gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        binary_3channel = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        result = cv2.addWeighted(binary_3channel, 0.7, gray_3channel, 0.3, 0)
        
        return result
    
    def _final_cleanup(self, image: np.ndarray) -> np.ndarray:
        """Финальная очистка изображения"""
        # Морфологические операции для удаления мелких артефактов
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Удаление мелкого шума
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Конвертация обратно в RGB
        result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
        
        return result
    
    def _save_image(self, image: np.ndarray, path: str):
        """Сохранение изображения в высоком качестве"""
        # Конвертация RGB -> BGR для OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Параметры сохранения для максимального качества
        if self.config.output_format.upper() == 'PNG':
            params = [cv2.IMWRITE_PNG_COMPRESSION, 0]  # Без сжатия
        else:
            params = [cv2.IMWRITE_JPEG_QUALITY, self.config.output_quality]
        
        cv2.imwrite(path, image_bgr, params)
    
    def batch_preprocess(self, 
                        input_dir: Union[str, Path],
                        output_dir: Union[str, Path],
                        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')) -> int:
        """
        Пакетная обработка изображений из директории.
        
        Args:
            input_dir: Директория с исходными изображениями
            output_dir: Директория для сохранения результатов
            extensions: Поддерживаемые расширения файлов
            
        Returns:
            Количество обработанных изображений
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Поиск всех изображений
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        
        self.logger.info(f"Найдено {len(image_files)} изображений для обработки")
        
        processed_count = 0
        for img_file in image_files:
            try:
                # Формирование пути для сохранения
                relative_path = img_file.relative_to(input_path)
                save_path = output_path / relative_path.with_suffix(f'.{self.config.output_format.lower()}')
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Обработка
                self.preprocess(img_file, str(save_path))
                processed_count += 1
                
                self.logger.info(f"Обработано: {img_file.name} ({processed_count}/{len(image_files)})")
                
            except Exception as e:
                self.logger.error(f"Ошибка обработки {img_file}: {e}")
                continue
        
        self.logger.info(f"Пакетная обработка завершена: {processed_count}/{len(image_files)}")
        return processed_count


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    IMAGE_DIR = Path(__file__).parent.parent / "images"
    SAVE_PATH = Path(__file__).parent / "processed_images"

    preprocessor = MedicalImagePreprocessor()
    
    preprocessor.batch_preprocess(input_dir=IMAGE_DIR, output_dir=SAVE_PATH) 