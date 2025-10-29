from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Пути
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_IMAGES_DIR: Path = DATA_DIR / "raw_images"
    PREPROCESSED_IMAGES_DIR: Path = DATA_DIR / "preprocessed_images"
    EXTRACTED_DATA_DIR: Path = DATA_DIR / "extracted_data"
    
    # # Ollama
    # OLLAMA_HOST: str = "http://localhost:11434"
    # VISION_MODEL: str = "llama3.2-vision"
    # TEXT_MODEL: str = "llama3.1:8b"
    
    # # База данных
    # DATABASE_URL: str = "sqlite:///./medical_reports.db"
    
    # # Векторная БД
    # VECTOR_DB_PATH: Path = DATA_DIR / "vector_db"
    # EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    

settings = Settings()