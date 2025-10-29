"""
Пакетная обработка всех медицинских изображений
================================================

Обрабатывает все изображения из директории data/raw_images/
Результаты сохраняются в data/preprocessed_images/

Использование:
    python scripts/process_dataset.py
    
    # Или с другим пресетом:
    python scripts/process_dataset.py --preset medical_llm_sharp
    
    # Или с кастомными директориями:
    python scripts/process_dataset.py --input data/raw_images --output data/processed
"""

import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime

from config.settings import PROJECT_ROOT
from src.preprocessing.preprocessor import MedicalImagePreprocessor, PreprocessingConfig
from src.preprocessing.config import get_preset


def main():
    parser = argparse.ArgumentParser(
        description='Пакетная обработка медицинских изображений',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--preset',
        type=str,
        default='medical_llm',
        choices=['medical_llm', 'medical_llm_sharp', 'medical_ocr', 'aggressive', 'balanced', 'gentle'],
        help='Пресет обработки (по умолчанию: medical_llm)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw_images',
        help='Входная директория с изображениями (относительно корня проекта)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/preprocessed_images',
        help='Выходная директория для результатов (относительно корня проекта)'
    )
    
    args = parser.parse_args()
    
    # Настройка логирования в директорию logs/
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = logs_dir / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Пути относительно корня проекта
    input_dir = PROJECT_ROOT / args.input
    output_dir = PROJECT_ROOT / args.output
    
    logger.info("=" * 80)
    logger.info("ПАКЕТНАЯ ОБРАБОТКА ВСЕХ МЕДИЦИНСКИХ ИЗОБРАЖЕНИЙ")
    logger.info("=" * 80)
    logger.info(f"Корневая директория проекта: {PROJECT_ROOT}")
    logger.info(f"Пресет обработки: {args.preset}")
    logger.info(f"Входная директория: {input_dir}")
    logger.info(f"Выходная директория: {output_dir}")
    logger.info(f"Лог файл: {log_file}")
    logger.info("")
    
    # Проверка существования входной директории
    if not input_dir.exists():
        logger.error(f"Входная директория не найдена: {input_dir}")
        sys.exit(1)
    
    # Создание выходной директории
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Загрузка конфигурации
    config = get_preset(args.preset)
    config = PreprocessingConfig(**config)
    
    logger.info("Параметры обработки:")
    for key, value in config.__dict__.items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    
    # Создание препроцессора
    preprocessor = MedicalImagePreprocessor(config=config)
    
    # Запуск обработки
    start_time = datetime.now()
    logger.info(f"Начало обработки: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    processed_count = preprocessor.batch_preprocess(
        input_dir=input_dir,
        output_dir=output_dir
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("ОБРАБОТКА ЗАВЕРШЕНА!")
    logger.info("=" * 80)
    logger.info(f"Обработано изображений: {processed_count}")
    logger.info(f"Время обработки: {duration}")
    logger.info(f"Среднее время на изображение: {duration / processed_count if processed_count > 0 else 0}")
    logger.info(f"Результаты сохранены в: {output_dir}")
    logger.info(f"Лог сохранен в: {log_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

