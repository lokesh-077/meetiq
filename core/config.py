import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    gemini_api_key: str = ""
    email_sender: str = ""
    email_password: str = ""
    email_to: str = ""
    
    port: int = 8000
    host: str = "0.0.0.0"
    bot_name: str = "MeetIQ Notetaker"
    whisper_model: str = "small"
    gemini_model: str = "gemini-3.5-flash"
    
    upload_dir: Path = Path("uploads")
    reports_dir: Path = Path("reports")
    samples_dir: Path = Path("samples")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
settings.upload_dir.mkdir(exist_ok=True)
settings.reports_dir.mkdir(exist_ok=True)
settings.samples_dir.mkdir(exist_ok=True)
