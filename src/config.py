"""
Configuration management for RAG Chatbot.

Handles environment variables, API keys, and application settings.
Supports reading from .env files, environment variables, and Streamlit secrets.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application configuration with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys (optional, can be provided via UI)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, alias="HUGGINGFACE_API_KEY")
    cohere_api_key: Optional[str] = Field(default=None, alias="COHERE_API_KEY")
    
    # Model defaults
    default_openai_model: str = "gpt-3.5-turbo-0125"
    default_google_model: str = "gemini-pro"
    default_hf_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Text splitting defaults
    chunk_size: int = 1600
    chunk_overlap: int = 200
    
    # Retriever defaults
    base_retriever_k: int = 16
    compression_retriever_k: int = 20
    compression_chunk_size: int = 500
    cohere_top_n: int = 10
    cohere_model: str = "rerank-multilingual-v2.0"
    
    # LLM defaults
    default_temperature: float = 0.5
    default_top_p: float = 0.95
    
    # Memory defaults
    memory_max_token: int = 1024
    
    # Paths (relative to project root)
    tmp_dir: str = "data/tmp"
    vector_store_dir: str = "data/vector_stores"
    
    def get_tmp_dir(self, base_path: Optional[Path] = None) -> Path:
        """Get absolute path to temporary directory."""
        if base_path is None:
            base_path = Path.cwd()
        return base_path / self.tmp_dir
    
    def get_vector_store_dir(self, base_path: Optional[Path] = None) -> Path:
        """Get absolute path to vector store directory."""
        if base_path is None:
            base_path = Path.cwd()
        return base_path / self.vector_store_dir


# Singleton instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the configuration singleton."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def load_streamlit_secrets():
    """
    Load API keys from Streamlit secrets if available.
    This is called when running on Streamlit Cloud.
    """
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            config = get_config()
            # Update config with Streamlit secrets if they exist
            if 'OPENAI_API_KEY' in st.secrets:
                config.openai_api_key = st.secrets['OPENAI_API_KEY']
            if 'GOOGLE_API_KEY' in st.secrets:
                config.google_api_key = st.secrets['GOOGLE_API_KEY']
            if 'HUGGINGFACE_API_KEY' in st.secrets:
                config.huggingface_api_key = st.secrets['HUGGINGFACE_API_KEY']
            if 'COHERE_API_KEY' in st.secrets:
                config.cohere_api_key = st.secrets['COHERE_API_KEY']
    except ImportError:
        # Streamlit not available, skip
        pass
    except Exception:
        # Secrets not configured, skip
        pass


# Available LLM providers
LLM_PROVIDERS = {
    "OpenAI": {
        "display_name": ":rainbow[**OpenAI**]",
        "models": ["gpt-3.5-turbo-0125", "gpt-3.5-turbo", "gpt-4-turbo-preview"],
        "api_key_link": "https://platform.openai.com/account/api-keys",
        "pricing_link": "https://openai.com/pricing"
    },
    "Google": {
        "display_name": "**Google Generative AI**",
        "models": ["gemini-pro"],
        "api_key_link": "https://makersuite.google.com/app/apikey",
        "rate_limit": "60 requests per minute"
    },
    "HuggingFace": {
        "display_name": ":hugging_face: **HuggingFace**",
        "models": ["mistralai/Mistral-7B-Instruct-v0.2"],
        "api_key_link": "https://huggingface.co/settings/tokens",
        "note": "**Free access.**"
    }
}

# Retriever types
RETRIEVER_TYPES = [
    "Cohere reranker",
    "Contextual compression",
    "Vectorstore backed retriever"
]

# Welcome messages in different languages
WELCOME_MESSAGES = {
    "english": "How can I assist you today?",
    "french": "Comment puis-je vous aider aujourd'hui ?",
    "spanish": "¿Cómo puedo ayudarle hoy?",
    "german": "Wie kann ich Ihnen heute helfen?",
    "russian": "Чем я могу помочь вам сегодня?",
    "chinese": "我今天能帮你什么？",
    "arabic": "كيف يمكنني مساعدتك اليوم؟",
    "portuguese": "Como posso ajudá-lo hoje?",
    "italian": "Come posso assistervi oggi?",
    "japanese": "今日はどのようなご用件でしょうか?"
}

# Supported file types
SUPPORTED_FILE_TYPES = ["pdf", "txt", "docx", "csv"]
