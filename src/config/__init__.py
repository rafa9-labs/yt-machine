"""
Centralized Configuration Loader
All secrets, environment variables and app configuration
are loaded exclusively from this module.
"""
import os
from dotenv import load_dotenv
from typing import Optional


# Load environment variables once on module import
load_dotenv()


class Config:
    # Application
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # LLM / Ollama
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    # API Keys
    PEXELS_API_KEY: Optional[str] = os.getenv("PEXELS_API_KEY")
    TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    ELEVENLABS_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")

    # Video Generation
    OUTPUT_DIRECTORY: str = os.getenv("OUTPUT_DIRECTORY", "./output")

    @classmethod
    def validate(cls) -> None:
        """Validate all required configuration values are present and valid"""
        required = [
            ("OLLAMA_HOST", cls.OLLAMA_HOST),
        ]

        missing = []
        for name, value in required:
            if not value:
                missing.append(name)

        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Please check your .env file."
            )


# Validate on import
Config.validate()
