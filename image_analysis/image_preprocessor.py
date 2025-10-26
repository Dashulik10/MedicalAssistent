import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import logging
from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Конфигурация параметров предобработки"""
    max_dimension: int = 2048  # Максимальный размер стороны
    denoise_strength: int = 3  # Сила удаления шума (нечетное число)
    contrast_alpha: float = 1.5  # Коэффициент контраста (1.0-3.0)
    brightness_beta: int = 10  # Коррекция яркости (-50 to 50)
    sharpen_amount: float = 1.5  # Сила резкости (1.0-3.0)
    output_format: str = 'PNG'
    output_quality: int = 100


class MedicalImagePreprocessor:
    """
    Упрощенный препроцессор медицинских отчетов.
    
    Фокус: четкий черный текст на белом фоне, без шумов.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.logger = logging.getLogger(__name__)
        
    def preprocess(self, 
                   image_input: Union[str, Path, np.ndarray],
                   save_path: Optional[str] = None) -> np.ndarray:
        """
        Предобработка медицинского отчета.
        
        Args:
            image_input: Путь к изображению или numpy array
            save_path: Путь для сохранения (опционально)
            
        Returns:
            Обработанное изображение (RGB numpy array)
        """
        try:
            # 1. Загрузка
            image = self._load_image(image_input)
            
            # 2. Изменение размера
            image = self._resize_image(image)
            
            # 3. Конвертация в grayscale для обработки
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 4. КРИТИЧНО: Проверка инверсии
            gray = self._fix_inversion(gray)
            
            # 5. Удаление шума
            gray = self._denoise(gray)
            
            # 6. Увеличение контраста и яркости
            gray = self._enhance_contrast(gray)
            
            # 7. Повышение резкости текста
            gray = self._sharpen(gray)
            
            # 8. Нормализация (белый фон, черный текст)
            gray = self._normalize(gray)
            
            # 9. Конвертация обратно в RGB
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
            # 10. Сохранение
            if save_path:
                self._save_image(result, save_path)
                self.logger.info(f"✓ Сохранено: {save_path}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка предобработки: {e}")
            raise
    
    def _load_image(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Загрузка изображения"""
        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input))
            if image is None:
                raise ValueError(f"Не удалось загрузить: {image_input}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            raise TypeError("Неподдерживаемый тип изображения")
        
        return image
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Изменение размера с сохранением пропорций"""
        h, w = image.shape[:2]
        
        if max(h, w) > self.config.max_dimension:
            scale = self.config.max_dimension / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            self.logger.info(f"Размер: {w}x{h} → {new_w}x{new_h}")
        
        return image
    
    def _fix_inversion(self, gray: np.ndarray) -> np.ndarray:
        """
        Исправление инверсии цветов.
        
        Проверяет углы изображения (обычно там фон).
        Если фон темный - инвертирует изображение.
        """
        h, w = gray.shape
        corner_size = min(h, w) // 10
        
        # Средняя яркость из 4 углов
        corners = [
            gray[:corner_size, :corner_size],
            gray[:corner_size, -corner_size:],
            gray[-corner_size:, :corner_size],
            gray[-corner_size:, -corner_size:]
        ]
        
        avg_corner_brightness = np.mean([np.mean(corner) for corner in corners])
        
        # Если углы (фон) темные - инвертируем
        if avg_corner_brightness < 127:
            self.logger.info(f"Инверсия обнаружена (яркость фона: {avg_corner_brightness:.0f}), исправляем")
            return 255 - gray
        
        return gray
    
    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        """
        Удаление шума с сохранением деталей текста.
        
        Использует Non-local Means Denoising - лучший метод для документов.
        """
        # Медианный фильтр для импульсного шума
        denoised = cv2.medianBlur(gray, self.config.denoise_strength)
        
        # Non-local means для гауссовского шума
        denoised = cv2.fastNlMeansDenoising(
            denoised,
            None,
            h=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        return denoised
    
    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """
        Увеличение контраста между текстом и фоном.
        
        Формула: output = alpha * input + beta
        """
        enhanced = cv2.convertScaleAbs(
            gray,
            alpha=self.config.contrast_alpha,
            beta=self.config.brightness_beta
        )
        
        return enhanced
    
    def _sharpen(self, gray: np.ndarray) -> np.ndarray:
        """
        Повышение резкости для четкости текста.
        
        Использует Unsharp Masking - стандарт для текстовых документов.
        """
        # Размытие
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        
        # Unsharp mask
        sharpened = cv2.addWeighted(
            gray, 1.0 + self.config.sharpen_amount,
            blurred, -self.config.sharpen_amount,
            0
        )
        
        return sharpened
    
    def _normalize(self, gray: np.ndarray) -> np.ndarray:
        """
        Нормализация для максимального контраста.
        
        Растягивает диапазон яркости на весь спектр 0-255.
        """
        # Метод Otsu для адаптивной бинаризации
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Проверка на случайную инверсию после Otsu
        if np.mean(binary) < 127:
            binary = 255 - binary
        
        # Смешиваем: 80% обработанное, 20% оригинал (для сохранения деталей)
        result = cv2.addWeighted(binary, 0.8, gray, 0.2, 0)
        
        # Финальная нормализация
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        
        return result
    
    def _save_image(self, image: np.ndarray, path: str):
        """Сохранение в высоком качестве"""
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if self.config.output_format.upper() == 'PNG':
            params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
        else:
            params = [cv2.IMWRITE_JPEG_QUALITY, self.config.output_quality]
        
        cv2.imwrite(path, image_bgr, params)
    
    def batch_preprocess(self, 
                        input_dir: Union[str, Path],
                        output_dir: Union[str, Path],
                        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')) -> int:
        """
        Пакетная обработка директории.
        
        Args:
            input_dir: Папка с исходниками
            output_dir: Папка для результатов
            extensions: Расширения файлов
            
        Returns:
            Количество обработанных файлов
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Поиск всех изображений
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        
        total = len(image_files)
        self.logger.info(f"Найдено изображений: {total}")
        
        processed = 0
        failed = 0
        
        for img_file in image_files:
            try:
                # Путь для сохранения
                relative = img_file.relative_to(input_path)
                save_path = output_path / relative.with_suffix(f'.{self.config.output_format.lower()}')
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Обработка
                self.preprocess(img_file, str(save_path))
                processed += 1
                
                if processed % 10 == 0:
                    self.logger.info(f"Прогресс: {processed}/{total}")
                
            except Exception as e:
                failed += 1
                self.logger.error(f"✗ Ошибка {img_file.name}: {e}")
        
        self.logger.info(f"Готово: {processed} успешно, {failed} ошибок")
        return processed


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    IMAGE_DIR = Path(__file__).parent.parent / "bad_images"
    SAVE_PATH = Path(__file__).parent / "processed_images"

    config = PreprocessingConfig(
        denoise_strength=1,     # 3-7
        contrast_alpha=1,     # 1.0-2.5
        brightness_beta=-30,     # -20 до +30
        sharpen_amount=3      # 1.0-3.0
    )
    preprocessor = MedicalImagePreprocessor(config=config)
    
    preprocessor.batch_preprocess(input_dir=IMAGE_DIR, output_dir=SAVE_PATH) 