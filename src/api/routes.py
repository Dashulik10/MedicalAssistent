"""FastAPI роутеры для обработки медицинских изображений и генерации отчётов."""

import logging
import uuid
from datetime import datetime
from io import BytesIO
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from api.schemas import (
    ContextSourcesResponse,
    ErrorResponse,
    GenerateReportRequest,
    QueryPatientRequest,
    QueryPatientResponse,
    UploadImageResponse,
)
from extraction.pipeline import ExtractionPipeline
from generation.context_builder import ContextBuilder
from generation.llm_client import GroqLLMClient
from generation.prompts import REPORT_GENERATION_SYSTEM_PROMPT
from preprocessing.preprocessor import MedicalImagePreprocessor, PreprocessingConfig
from preprocessing.presets import get_preset
from reporting.pdf_converter import PDFConverter
from reporting.report_generator import ReportGenerator
from repository.new_db.new_repo_adapter import SQLiteMedicalAdapter
from repository.old_db.old_repo_service import ReportService, flatten_report
from vectorization.embeddings import SBERTAdapter
from vectorization.services import EmbeddingAndStoreService
from vectorization.vectorstore import ChromaAdapter

logger = logging.getLogger(__name__)

router = APIRouter()

# === Инициализация сервисов ===

# Препроцессор изображений
preprocessor = MedicalImagePreprocessor(
    config=PreprocessingConfig(**get_preset("medical_llm"))
)

# Пайплайн извлечения данных
extraction_pipeline = ExtractionPipeline()

# MongoDB сервис
report_service = ReportService()

# SQLite адаптер
sqlite_adapter = SQLiteMedicalAdapter()

# Векторное хранилище
embedder = SBERTAdapter()
vector_store = ChromaAdapter(
    persist_directory="./chroma_medical_db",
    collection_name="medical_reports",
)
embed_store_service = EmbeddingAndStoreService(
    embedder=embedder,
    store=vector_store,
    batch_size=32,
)

# Генератор отчётов
report_generator = ReportGenerator()
pdf_converter = PDFConverter()

# RAG-генерация
llm_client = GroqLLMClient()
context_builder = ContextBuilder(
    sqlite_adapter=sqlite_adapter,
    report_service=report_service,
    embed_store_service=embed_store_service,
)


# === Вспомогательные функции ===


def validate_image_format(filename: str) -> bool:
    """Проверяет допустимость формата файла."""
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return f".{ext}" in allowed_extensions


# === Эндпоинты ===


@router.post(
    "/upload_image",
    response_model=UploadImageResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Невалидные входные данные"},
        500: {"model": ErrorResponse, "description": "Ошибка обработки"},
    },
    summary="Загрузка и обработка медицинского изображения",
    description="""
    Загружает изображение медицинского документа, извлекает данные и сохраняет в базу.
    
    Пайплайн обработки:
    1. Валидация изображения (формат JPG/PNG)
    2. Препроцессинг (улучшение качества для OCR)
    3. Извлечение данных через Vision LLM
    4. Сохранение в MongoDB
    5. Индексация в векторной БД для семантического поиска
    """,
)
async def upload_image(
    image: UploadFile = File(..., description="Изображение медицинского документа"),
    patient_identifier: Optional[str] = Form(
        None, description="ID пациента или имя (опционально)"
    ),
):
    """Обрабатывает загруженное медицинское изображение."""
    logger.info(f"Получен запрос на загрузку изображения: {image.filename}")

    # 1. Валидация формата
    if not image.filename or not validate_image_format(image.filename):
        logger.warning(f"Невалидный формат файла: {image.filename}")
        raise HTTPException(
            status_code=400,
            detail="Неподдерживаемый формат файла. Допустимы: JPG, PNG",
        )

    try:
        # 2. Чтение содержимого
        content = await image.read()
        logger.info(f"Прочитано {len(content)} байт")

        # 3. Препроцессинг
        logger.info("Запуск препроцессинга...")
        processed_image = preprocessor.preprocess(content)

        # 4. Извлечение данных через Vision LLM
        logger.info("Извлечение данных через Vision LLM...")
        extraction_result = extraction_pipeline.process(processed_image)

        if extraction_result["status"] != "success":
            error_msg = extraction_result.get("error", "Ошибка извлечения данных")
            logger.error(f"Ошибка извлечения: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

        medical_record = extraction_result["data"]
        record_dict = medical_record.model_dump()

        # 5. Определение patient_id
        # Приоритет: переданный identifier > извлечённый из документа
        patient_id = None
        patient_name = record_dict.get("patient_name")

        if patient_identifier:
            try:
                patient_id = int(patient_identifier)
            except ValueError:
                # Если не число, пробуем найти пациента по имени
                found = sqlite_adapter.find_patient(patient_identifier)
                if found:
                    patient_id = found["id"]
                    patient_name = f"{found.get('surname', '')} {found.get('firstname', '')}".strip()

        # Если patient_id не определён, используем извлечённый из документа
        if patient_id is None and record_dict.get("patient_id"):
            try:
                patient_id = int(record_dict["patient_id"])
            except (ValueError, TypeError):
                pass

        # Если всё ещё нет patient_id, используем временный
        if patient_id is None:
            patient_id = 0  # Неопределённый пациент
            logger.warning("patient_id не определён, используется 0")

        # 6. Генерация raw_text для векторизации
        raw_text = flatten_report(record_dict)

        # 7. Сохранение в MongoDB
        logger.info(f"Сохранение в MongoDB для patient_id={patient_id}...")
        record_id = report_service.save_report(
            report=record_dict,
            patient_id=patient_id,
            raw_text=raw_text,
            patient_name=patient_name,
        )

        # 8. Индексация в векторной БД
        logger.info("Индексация в векторной БД...")
        doc_id = str(uuid.uuid4())

        # ChromaDB не принимает None в метаданных - заменяем на пустые строки
        chroma_metadata = {
            "patient_id": str(patient_id),
            "patient_name": patient_name or "",
            "record_id": record_id,
            "test_type": record_dict.get("test_type") or "",
            "test_date": record_dict.get("test_date") or "",
        }

        embed_store_service.add_document(
            doc_id=doc_id,
            text=raw_text,
            metadata=chroma_metadata,
        )

        logger.info(f"Обработка завершена успешно. record_id={record_id}")

        return UploadImageResponse(
            status="success",
            message="Изображение успешно обработано и данные сохранены",
            record_id=record_id,
            patient_id=patient_id,
            patient_name=patient_name,
            test_type=record_dict.get("test_type"),
            test_date=record_dict.get("test_date"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ошибка при обработке изображения")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки изображения: {str(e)}",
        )


@router.post(
    "/generate_report",
    response_class=StreamingResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Пациент не найден"},
        500: {"model": ErrorResponse, "description": "Ошибка генерации отчёта"},
    },
    summary="Генерация PDF отчёта по пациенту (RAG)",
    description="""
    Генерирует PDF отчёт с медицинскими данными пациента используя RAG-подход.
    
    Пайплайн:
    1. Поиск пациента в SQLite (по ID или имени)
    2. Построение контекста (SQLite + семантический поиск + MongoDB)
    3. Генерация ответа через LLM на основе запроса пользователя
    4. Форматирование отчёта с названием поликлиники и датой
    5. Конвертация в PDF
    
    Если query не указан, будет сгенерирован общий отчёт о состоянии здоровья.
    """,
)
async def generate_report(request: GenerateReportRequest):
    """Генерирует PDF отчёт по данным пациента с использованием RAG."""
    logger.info(
        f"Запрос на генерацию отчёта: patient={request.patient_identifier}, "
        f"query={request.query[:50] if request.query else 'None'}..."
    )

    try:
        # 1. Поиск пациента в SQLite
        patient = sqlite_adapter.find_patient(request.patient_identifier)

        if not patient:
            logger.warning(f"Пациент не найден: {request.patient_identifier}")
            raise HTTPException(
                status_code=404,
                detail=f"Пациент не найден: {request.patient_identifier}",
            )

        patient_id = patient["id"]
        patient_name = (
            f"{patient.get('surname', '')} {patient.get('firstname', '')}".strip()
        )
        logger.info(f"Найден пациент: id={patient_id}, name={patient_name}")

        # 2. Формируем запрос для LLM (если не указан, используем общий)
        user_query = (
            request.query
            if request.query
            else "Создай общий медицинский отчёт о состоянии здоровья пациента, включая основные показатели анализов и общую оценку."
        )

        # 3. Построение контекста (SQLite + Vector Search + MongoDB)
        logger.info("Построение контекста для RAG...")
        patient_context = context_builder.build_context(
            patient_id=patient_id,
            query=user_query,
            vector_top_k=5,
            max_mongo_records=10,
        )

        # 4. Генерация ответа через LLM
        logger.info("Генерация ответа через LLM для PDF отчёта...")
        generation_result = llm_client.generate(
            query=user_query,
            context=patient_context.text,
            system_prompt=REPORT_GENERATION_SYSTEM_PROMPT,
        )

        logger.info(
            f"Ответ LLM сгенерирован: {len(generation_result.text)} символов, "
            f"{generation_result.tokens_used} токенов"
        )

        # 5. Форматирование markdown отчёта
        logger.info("Форматирование markdown отчёта...")
        markdown_report = report_generator.generate_from_llm(
            llm_response=generation_result.text,
            patient_name=patient_name or f"ID: {patient_id}",
            patient_id=patient_id,
        )

        # 6. Конвертация в PDF
        logger.info("Конвертация в PDF...")
        pdf_bytes = pdf_converter.convert_to_pdf(
            markdown_text=markdown_report,
            title=f"Медицинский отчёт - {patient_name or patient_id}",
        )

        # 7. Формирование имени файла
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{patient_id}_{date_str}.pdf"

        logger.info(f"Отчёт сгенерирован: {filename}, размер: {len(pdf_bytes)} байт")

        # 8. Возврат PDF
        return StreamingResponse(
            BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(pdf_bytes)),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ошибка генерации отчёта")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка генерации отчёта: {str(e)}",
        )


@router.post(
    "/query_patient",
    response_model=QueryPatientResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Пациент не найден"},
        500: {"model": ErrorResponse, "description": "Ошибка генерации ответа"},
    },
    summary="RAG-запрос по данным пациента",
    description="""
    Отвечает на вопросы о пациенте, используя RAG-подход:
    
    1. Поиск пациента в SQLite (базовая информация)
    2. Семантический поиск в векторной БД (релевантные записи)
    3. Загрузка дополнительных данных из MongoDB
    4. Генерация ответа через LLM (openai/gpt-oss-120b)
    
    Примеры запросов:
    - "Какие анализы были сданы?"
    - "Есть ли отклонения в показателях крови?"
    - "Краткое резюме состояния здоровья"
    """,
)
async def query_patient(request: QueryPatientRequest):
    """Отвечает на вопросы о пациенте с использованием RAG."""
    logger.info(
        f"RAG-запрос: patient={request.patient_identifier}, query={request.query[:50]}..."
    )

    try:
        # 1. Поиск пациента в SQLite
        patient = sqlite_adapter.find_patient(request.patient_identifier)

        if not patient:
            logger.warning(f"Пациент не найден: {request.patient_identifier}")
            raise HTTPException(
                status_code=404,
                detail=f"Пациент не найден: {request.patient_identifier}",
            )

        patient_id = patient["id"]
        logger.info(f"Найден пациент: id={patient_id}")

        # 2. Построение контекста (SQLite + Vector Search + MongoDB)
        logger.info("Построение контекста для RAG...")
        patient_context = context_builder.build_context(
            patient_id=patient_id,
            query=request.query,
            vector_top_k=5,
            max_mongo_records=10,
        )

        # 3. Генерация ответа через LLM
        logger.info("Генерация ответа через LLM...")
        generation_result = llm_client.generate(
            query=request.query,
            context=patient_context.text,
        )

        logger.info(
            f"Ответ сгенерирован: {len(generation_result.text)} символов, "
            f"{generation_result.tokens_used} токенов"
        )

        # 4. Формирование ответа
        return QueryPatientResponse(
            status="success",
            answer=generation_result.text,
            patient_id=patient_id,
            patient_name=patient_context.patient_name,
            sources=ContextSourcesResponse(
                sqlite_patient=patient_context.sources.sqlite_patient,
                mongodb_records_count=patient_context.sources.mongodb_records_count,
                vector_search_results=patient_context.sources.vector_search_results,
            ),
            model=generation_result.model,
            tokens_used=generation_result.tokens_used,
            generation_time=generation_result.generation_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ошибка RAG-генерации")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка генерации ответа: {str(e)}",
        )
