"""Главная точка входа FastAPI приложения."""

import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router


def setup_logging() -> logging.Logger:
    """Настраивает логирование в файл и консоль."""
    # Создаем директорию для логов
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Формат логов
    log_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Очищаем существующие обработчики
    root_logger.handlers.clear()

    # Обработчик для файла
    file_handler = logging.FileHandler(
        log_dir / "app.log",
        mode="a",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)

    # Обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)

    # Добавляем обработчики
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Снижаем уровень логирования для сторонних библиотек
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return root_logger


# Инициализация логирования
logger = setup_logging()

# Создание приложения
app = FastAPI(
    title="Medical Image Processing API",
    description="""
API для обработки медицинских изображений и генерации отчётов.

## Возможности

* **Загрузка изображений** - извлечение данных из медицинских документов
* **Генерация отчётов** - создание PDF отчётов по данным пациента
* **Семантический поиск** - поиск по содержимому документов
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутеров
app.include_router(router, prefix="/api/v1", tags=["Medical"])


@app.get("/", tags=["Health"])
async def root():
    """Проверка работоспособности API."""
    return {
        "status": "ok",
        "message": "Medical Image Processing API is running",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Детальная проверка здоровья системы."""
    return {
        "status": "healthy",
        "components": {
            "api": "ok",
            "database": "ok",  # TODO: добавить реальную проверку
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main_route:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
