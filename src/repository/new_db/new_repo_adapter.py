import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional

from configuration.settings import settings
from repository.new_db.new_repo_interface import PatientMedicalPort

logger = logging.getLogger(__name__)


class SQLiteMedicalAdapter(PatientMedicalPort):

    DB_PATH = settings.SQLITE_PATH

    def _fetch_one(self, cursor, query, params=()):
        cursor.execute(query, params)
        row = cursor.fetchone()
        if not row:
            return None
        return {col[0]: row[i] for i, col in enumerate(cursor.description)}

    def _fetch_all(self, cursor, query, params=()):
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [
            {col[0]: row[i] for i, col in enumerate(cursor.description)} for row in rows
        ]

    def get_basic_info(self, patient_id: int):
        conn = sqlite3.connect(self.DB_PATH)
        cur = conn.cursor()

        patient = self._fetch_one(
            cur,
            """
            SELECT id, user_id, surname, firstname, dob
            FROM patient
            WHERE id = ?
            """,
            (patient_id,),
        )

        conn.close()
        return patient

    def get_medical_card(self, patient_id: int):
        conn = sqlite3.connect(self.DB_PATH)
        cur = conn.cursor()

        medical = self._fetch_one(
            cur,
            """
                                  SELECT gender,
                                         blood_group,
                                         rhesus,
                                         allergic_history,
                                         medication_intolerance,
                                         surgical_intervention,
                                         previous_infectious_diseases
                                  FROM medical_card
                                  WHERE patient_id = ?
                                  """,
            (patient_id,),
        )

        conn.close()
        return medical

    def get_appointments(self, patient_id: int):
        conn = sqlite3.connect(self.DB_PATH)
        cur = conn.cursor()

        appointments = self._fetch_all(
            cur,
            """
                                       SELECT appointment_date_time,
                                              doctor_id,
                                              appointment_details
                                       FROM appointment
                                       WHERE patient_id = ?
                                       ORDER BY appointment_date_time
                                       """,
            (patient_id,),
        )

        conn.close()
        return appointments

    def _parse_json_field(self, value: Any) -> Any:
        if not value:
            return None

        if isinstance(value, (dict, list)):
            return value

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    def get_diagnostics(self, user_id: int) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.DB_PATH)

        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        query = """
                SELECT
                    date, selected AS test_type, result
                FROM diagnostic
                WHERE user_id = ?
                ORDER BY date \
                """

        raw_rows = self._fetch_all(cur, query, (user_id,))

        clean_data = []

        for row in raw_rows:
            row_dict = dict(row)

            row_dict["test_type"] = self._parse_json_field(row_dict["test_type"])
            row_dict["result"] = self._parse_json_field(row_dict["result"])

            clean_data.append(row_dict)

        conn.close()
        return clean_data

    def find_patient_by_name(self, name: str) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.DB_PATH)
        cur = conn.cursor()

        search_pattern = f"%{name}%"
        patients = self._fetch_all(
            cur,
            """
            SELECT id, user_id, surname, firstname, dob
            FROM patient
            WHERE surname LIKE ? OR firstname LIKE ?
            ORDER BY surname, firstname
            """,
            (search_pattern, search_pattern),
        )

        conn.close()
        logger.info(f"SQLite: найдено {len(patients)} пациентов по запросу '{name}'")
        return patients

    def find_patient(self, identifier: str) -> Optional[Dict[str, Any]]:
        try:
            patient_id = int(identifier)
            patient = self.get_basic_info(patient_id)
            if patient:
                logger.info(f"SQLite: найден пациент по ID={patient_id}")
                return patient
        except ValueError:
            pass

        patients = self.find_patient_by_name(identifier)
        if patients:
            logger.info(f"SQLite: найден пациент по имени '{identifier}'")
            return patients[0]

        logger.warning(f"SQLite: пациент не найден по identifier='{identifier}'")
        return None

    def get_full_patient_data(self, patient_id: int) -> Optional[Dict[str, Any]]:
        basic = self.get_basic_info(patient_id)
        if not basic:
            return None

        user_id = basic.get("user_id")

        return {
            "basic_info": basic,
            "medical_card": self.get_medical_card(patient_id),
            "appointments": self.get_appointments(patient_id),
            "diagnostics": self.get_diagnostics(user_id) if user_id else [],
        }
