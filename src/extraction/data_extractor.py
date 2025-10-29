"""
Pydantic схемы для структурирования медицинских данных.

Используются для валидации данных, извлеченных из изображений.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import date, datetime
from enum import Enum


class TestStatus(str, Enum):
    """Статус результата теста относительно нормы"""
    NORMAL = "normal"
    HIGH = "high"
    LOW = "low"
    ABNORMAL = "abnormal"
    UNKNOWN = "unknown"


class LabTest(BaseModel):
    """
    Модель для одного результата лабораторного теста.
    
    Пример:
        {
            "name": "Гемоглобин",
            "value": "145",
            "unit": "г/л",
            "reference_range": "130-160",
            "status": "normal"
        }
    """
    name: str = Field(
        description="Название теста/показателя",
        examples=["Гемоглобин", "Лейкоциты", "Глюкоза"]
    )
    
    value: str = Field(
        description="Значение результата теста",
        examples=["145", "5.2", "Отрицательный"]
    )
    
    unit: Optional[str] = Field(
        default=None,
        description="Единица измерения",
        examples=["г/л", "×10⁹/л", "ммоль/л", "мг/дл"]
    )
    
    reference_range: Optional[str] = Field(
        default=None,
        description="Референсный диапазон (норма)",
        examples=["130-160", "4.0-9.0", "< 5.5"]
    )
    
    status: TestStatus = Field(
        default=TestStatus.UNKNOWN,
        description="Статус результата относительно нормы"
    )
    
    notes: Optional[str] = Field(
        default=None,
        description="Дополнительные примечания или флаги"
    )
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Валидация названия теста"""
        if not v or not v.strip():
            raise ValueError("Название теста не может быть пустым")
        return v.strip()
    
    @field_validator('value')
    @classmethod
    def validate_value(cls, v: str) -> str:
        """Валидация значения"""
        if not v or not v.strip():
            raise ValueError("Значение теста не может быть пустым")
        return v.strip()


class PatientInfo(BaseModel):
    """
    Информация о пациенте.
    
    Пример:
        {
            "full_name": "Иванов Иван Иванович",
            "age": 45,
            "gender": "М",
            "patient_id": "12345"
        }
    """
    full_name: str = Field(
        description="ФИО пациента",
        examples=["Иванов Иван Иванович", "Петрова Мария Сергеевна"]
    )
    
    age: Optional[int] = Field(
        default=None,
        ge=0,
        le=150,
        description="Возраст пациента"
    )
    
    gender: Optional[str] = Field(
        default=None,
        description="Пол пациента",
        examples=["М", "Ж", "M", "F"]
    )
    
    patient_id: Optional[str] = Field(
        default=None,
        description="ID пациента в лаборатории"
    )
    
    date_of_birth: Optional[date] = Field(
        default=None,
        description="Дата рождения"
    )
    
    @field_validator('full_name')
    @classmethod
    def validate_full_name(cls, v: str) -> str:
        """Валидация ФИО"""
        if not v or not v.strip():
            raise ValueError("ФИО не может быть пустым")
        return v.strip()
    
    @field_validator('gender')
    @classmethod
    def normalize_gender(cls, v: Optional[str]) -> Optional[str]:
        """Нормализация пола"""
        if v is None:
            return None
        
        gender_map = {
            'м': 'М', 'м.': 'М', 'male': 'М', 'm': 'М',
            'ж': 'Ж', 'ж.': 'Ж', 'female': 'Ж', 'f': 'Ж'
        }
        
        return gender_map.get(v.lower().strip(), v.upper().strip())


class MedicalReport(BaseModel):
    """
    Полная модель медицинского отчета.
    
    Содержит всю информацию, извлеченную из изображения лабораторного отчета.
    
    Пример:
        {
            "patient": {...},
            "report_date": "2024-10-26",
            "report_type": "Общий анализ крови",
            "tests": [...],
            "lab_name": "Лаборатория Инвитро",
            "doctor_name": "Доктор Иванов И.И.",
            "report_id": "LAB-2024-001"
        }
    """
    # Основная информация
    patient: PatientInfo = Field(
        description="Информация о пациенте"
    )
    
    report_date: date = Field(
        description="Дата проведения анализа или создания отчета"
    )
    
    report_type: str = Field(
        description="Тип анализа/отчета",
        examples=[
            "Общий анализ крови",
            "Биохимический анализ крови",
            "Анализ мочи",
            "Гормональный профиль"
        ]
    )
    
    # Результаты тестов
    tests: List[LabTest] = Field(
        description="Список всех результатов тестов",
        min_length=1
    )
    
    # Дополнительная информация
    lab_name: Optional[str] = Field(
        default=None,
        description="Название лаборатории"
    )
    
    doctor_name: Optional[str] = Field(
        default=None,
        description="Имя врача, подписавшего отчет"
    )

    description_name: 
    
    report_id: Optional[str] = Field(
        default=None,
        description="Уникальный идентификатор отчета"
    )
    
    collection_date: Optional[date] = Field(
        default=None,
        description="Дата забора биоматериала (если отличается от даты отчета)"
    )
    
    clinical_info: Optional[str] = Field(
        default=None,
        description="Клиническая информация или диагноз"
    )
    
    conclusions: Optional[str] = Field(
        default=None,
        description="Заключение врача или комментарии"
    )
    
    # Метаданные
    extraction_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Время извлечения данных"
    )
    
    source_image_path: Optional[str] = Field(
        default=None,
        description="Путь к исходному изображению"
    )
    
    @field_validator('report_type')
    @classmethod
    def validate_report_type(cls, v: str) -> str:
        """Валидация типа отчета"""
        if not v or not v.strip():
            raise ValueError("Тип отчета не может быть пустым")
        return v.strip()
    
    @field_validator('tests')
    @classmethod
    def validate_tests(cls, v: List[LabTest]) -> List[LabTest]:
        """Валидация списка тестов"""
        if not v:
            raise ValueError("Отчет должен содержать хотя бы один тест")
        return v
    
    def get_abnormal_tests(self) -> List[LabTest]:
        """
        Возвращает список тестов с отклонениями от нормы.
        
        Returns:
            Список тестов со статусом HIGH, LOW или ABNORMAL
        """
        return [
            test for test in self.tests
            if test.status in [TestStatus.HIGH, TestStatus.LOW, TestStatus.ABNORMAL]
        ]
    
    def get_normal_tests(self) -> List[LabTest]:
        """
        Возвращает список тестов в пределах нормы.
        
        Returns:
            Список тестов со статусом NORMAL
        """
        return [
            test for test in self.tests
            if test.status == TestStatus.NORMAL
        ]
    
    def summary(self) -> dict:
        """
        Создает краткую сводку по отчету.
        
        Returns:
            Словарь с основными статистиками
        """
        total_tests = len(self.tests)
        abnormal_tests = len(self.get_abnormal_tests())
        normal_tests = len(self.get_normal_tests())
        
        return {
            "patient_name": self.patient.full_name,
            "report_date": self.report_date.isoformat(),
            "report_type": self.report_type,
            "total_tests": total_tests,
            "normal_tests": normal_tests,
            "abnormal_tests": abnormal_tests,
            "abnormal_percentage": round(abnormal_tests / total_tests * 100, 1) if total_tests > 0 else 0
        }


class ExtractionResult(BaseModel):
    """
    Результат процесса извлечения данных.
    
    Включает сам отчет и метаинформацию о процессе извлечения.
    """
    success: bool = Field(
        description="Успешно ли прошло извлечение"
    )
    
    report: Optional[MedicalReport] = Field(
        default=None,
        description="Извлеченный медицинский отчет (если успешно)"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Сообщение об ошибке (если неуспешно)"
    )
    
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Оценка уверенности модели (0-1)"
    )
    
    processing_time_seconds: Optional[float] = Field(
        default=None,
        description="Время обработки в секундах"
    )
    
    model_used: Optional[str] = Field(
        default=None,
        description="Название использованной модели"
    )


# ============================================================================
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    import json
    
    print("="*70)
    print("ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ PYDANTIC СХЕМ")
    print("="*70 + "\n")
    
    # === Пример 1: Создание одного теста ===
    print("Пример 1: Создание LabTest\n")
    
    test = LabTest(
        name="Гемоглобин",
        value="145",
        unit="г/л",
        reference_range="130-160",
        status=TestStatus.NORMAL
    )
    
    print(test.model_dump_json(indent=2, ensure_ascii=False))
    print()
    
    
    # === Пример 2: Создание информации о пациенте ===
    print("\nПример 2: Создание PatientInfo\n")
    
    patient = PatientInfo(
        full_name="Иванов Иван Иванович",
        age=45,
        gender="М",
        patient_id="PAT-2024-001"
    )
    
    print(patient.model_dump_json(indent=2, ensure_ascii=False))
    print()
    
    
    # === Пример 3: Создание полного отчета ===
    print("\nПример 3: Создание полного MedicalReport\n")
    
    from datetime import date
    
    report = MedicalReport(
        patient=PatientInfo(
            full_name="Петрова Мария Сергеевна",
            age=32,
            gender="Ж"
        ),
        report_date=date(2024, 10, 26),
        report_type="Общий анализ крови",
        tests=[
            LabTest(
                name="Гемоглобин",
                value="135",
                unit="г/л",
                reference_range="120-140",
                status=TestStatus.NORMAL
            ),
            LabTest(
                name="Лейкоциты",
                value="12.5",
                unit="×10⁹/л",
                reference_range="4.0-9.0",
                status=TestStatus.HIGH,
                notes="Повышены"
            ),
            LabTest(
                name="Эритроциты",
                value="4.2",
                unit="×10¹²/л",
                reference_range="3.7-4.7",
                status=TestStatus.NORMAL
            )
        ],
        lab_name="Медицинская лаборатория Инвитро",
        doctor_name="Доктор Сидоров С.С.",
        report_id="LAB-2024-12345"
    )
    
    print(report.model_dump_json(indent=2, ensure_ascii=False))
    print()
    
    
    # === Пример 4: Использование методов отчета ===
    print("\nПример 4: Анализ отчета\n")
    
    print("Краткая сводка:")
    print(json.dumps(report.summary(), indent=2, ensure_ascii=False))
    print()
    
    print(f"Количество тестов с отклонениями: {len(report.get_abnormal_tests())}")
    print(f"Количество нормальных тестов: {len(report.get_normal_tests())}")
    print()
    
    print("Тесты с отклонениями:")
    for test in report.get_abnormal_tests():
        print(f"  - {test.name}: {test.value} {test.unit or ''} (статус: {test.status.value})")
    print()
    
    
    # === Пример 5: Валидация (покажет ошибки) ===
    print("\nПример 5: Валидация данных\n")
    
    try:
        # Попытка создать тест без названия (вызовет ошибку)
        invalid_test = LabTest(
            name="",
            value="100"
        )
    except Exception as e:
        print(f"Ошибка валидации: {e}")
    
    try:
        # Попытка создать отчет без тестов (вызовет ошибку)
        invalid_report = MedicalReport(
            patient=patient,
            report_date=date.today(),
            report_type="Анализ",
            tests=[]  # Пустой список
        )
    except Exception as e:
        print(f"Ошибка валидации: {e}")
    
    
    # === Пример 6: ExtractionResult ===
    print("\n\nПример 6: Результат извлечения\n")
    
    result = ExtractionResult(
        success=True,
        report=report,
        confidence_score=0.95,
        processing_time_seconds=2.3,
        model_used="llama3.2-vision"
    )
    
    print(f"Успешно: {result.success}")
    print(f"Уверенность: {result.confidence_score}")
    print(f"Время обработки: {result.processing_time_seconds}s")
    print(f"Модель: {result.model_used}")
    
    
    print("\n" + "="*70)
    print("СХЕМЫ ГОТОВЫ К ИСПОЛЬЗОВАНИЮ")
    print("="*70)