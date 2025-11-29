"""Модуль генерации медицинских отчётов."""

from .pdf_converter import PDFConverter
from .report_generator import ReportGenerator

__all__ = ["ReportGenerator", "PDFConverter"]
