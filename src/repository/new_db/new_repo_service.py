from typing import Dict, Any

from repository.new_db.new_repo_interface import PatientMedicalPort


class PatientMedicalReportService:
    def __init__(self, patient_id: int, adapter: PatientMedicalPort):
        self.patient_id = patient_id
        self.adapter = adapter

    def execute(self) -> Dict[str, Any]:
        basic = self.adapter.get_basic_info(self.patient_id)

        if not basic:
            return {"error": "Пациент не найден"}

        user_id = basic["user_id"]

        medical_card = self.adapter.get_medical_card(self.patient_id)
        appointments = self.adapter.get_appointments(self.patient_id)
        diagnostics = self.adapter.get_diagnostics(user_id)

        return {
            "patient_basic": {
                "id": basic["id"],
                "surname": basic["surname"],
                "firstname": basic["firstname"],
                "dob": basic["dob"]
            },
            "medical_card": medical_card,
            "appointments": appointments,
            "diagnostics": diagnostics
        }
