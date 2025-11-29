from pydantic_settings import BaseSettings
import dotenv
import os
from pathlib import Path

dotenv.load_dotenv()


class Settings(BaseSettings):
    groq_api_key: str
    model_name: str
    temperature: int
    max_tokens: int

    # image constraints
    max_image_size: int
    max_file_size_mb: int

    # RAG generation settings
    generation_model_name: str
    generation_temperature: float
    generation_max_tokens: int

    # Report settings
    clinic_name: str

    mongo_initdb_root_username: str
    mongo_initdb_root_password: str
    mongo_initdb_database: str

    mongo_app_user: str
    mongo_app_password: str

    mongo_host: str
    mongo_port: int
    mongo_collection: str

    sqlite_db_url: str = "sqlite:///./database.db"

    SRC_DIR: str = str(Path(__file__).parent.parent)
    SQLITE_PATH: str = str(Path(SRC_DIR) / "repository" / "new_db" / "site.db")
    # SQLITE_PATH: str = os.path.join(BASE_DIR, "repository", "new_db", "site.db")

    @property
    def mongo_url(self) -> str:
        return (
            f"mongodb://{self.mongo_app_user}:{self.mongo_app_password}"
            f"@{self.mongo_host}:{self.mongo_port}/{self.mongo_initdb_database}"
            f"?authSource={self.mongo_initdb_database}"
        )


settings = Settings()
