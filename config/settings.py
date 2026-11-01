"""Environment configuration for the experiment."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class OpenRouterConfig:
    """Credentials for OpenRouter requests."""

    api_key: str


def load_openrouter_config() -> OpenRouterConfig:
    """Load and validate OpenRouter settings from the project's .env file."""

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing required environment variable: OPENROUTER_API_KEY")

    return OpenRouterConfig(api_key=api_key)
