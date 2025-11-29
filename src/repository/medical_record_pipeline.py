import json
from typing import Dict, Any, List
import logging

from chunker.service import chunk_text
from extraction.schemas import MedicalRecord
from repository.new_db.new_repo_service import PatientMedicalReportService
from repository.old_db.old_repo_service import ReportService, flatten_report
from vectorization.services import EmbeddingAndStoreService

logger = logging.getLogger(__name__)


class MedicalRecordPipeline:
    def __init__(
        self,
        report_service: ReportService,
        patient_service_factory,
        embed_store_service: EmbeddingAndStoreService,
    ):
        self.report_service = report_service
        self.patient_service_factory = patient_service_factory
        self.embed_store_service = embed_store_service

    def process(self, record: MedicalRecord) -> Dict[str, Any]:
        patient_id = record.patient_id
        if not patient_id:
            raise ValueError("MedicalRecord must contain patient_id")

        record_json: Dict[str, Any] = record.model_dump()

        # Save to MongoDB
        report_id = self.report_service.save_report(record_json, int(patient_id))

        # Search in SQLite
        patient_service: PatientMedicalReportService = self.patient_service_factory(
            int(patient_id)
        )
        medical_data = patient_service.execute()

        full_json = {
            "patient_id": patient_id,
            "medical_data": medical_data,
            "old_reports": {"uploaded_record": record_json},
        }

        # Create chunks
        report = flatten_report(full_json)
        chunks = chunk_text(report, max_length=500, overlap=50)

        metadatas = [{"patient_id": patient_id} for _ in chunks]
        self.embed_store_service.index_chunks(chunks, metadatas)
        logger.info(f"Загружено {len(chunks)} чанков в векторную БД.")

        return {
            "status": "ok",
            "patient_id": patient_id,
            "saved_report_id": str(report_id),
            "chunks_count": len(chunks),
        }
