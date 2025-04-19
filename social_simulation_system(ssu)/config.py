# src/casevo/config.py
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Config:
    """Centralized configuration management for CASEVO framework."""
    
    # Memory configuration
    DEFAULT_MEMORY_SIZE: int = int(os.getenv('CASEVO_MEMORY_SIZE', '10'))
    MEMORY_CHUNK_SIZE: int = int(os.getenv('CASEVO_CHUNK_SIZE', '512'))
    MEMORY_RETENTION_DAYS: int = int(os.getenv('CASEVO_RETENTION_DAYS', '30'))
    
    # Thread and concurrency settings
    THREAD_POOL_SIZE: int = int(os.getenv('CASEVO_THREAD_POOL_SIZE', '8'))
    THREAD_TIMEOUT_SECONDS: int = int(os.getenv('CASEVO_THREAD_TIMEOUT', '60'))
    MAX_RETRIES: int = int(os.getenv('CASEVO_MAX_RETRIES', '3'))
    
    # LLM configuration
    LLM_MAX_TOKENS: int = int(os.getenv('CASEVO_LLM_MAX_TOKENS', '2048'))
    LLM_TEMPERATURE: float = float(os.getenv('CASEVO_LLM_TEMPERATURE', '0.7'))
    LLM_REQUEST_TIMEOUT: int = int(os.getenv('CASEVO_LLM_TIMEOUT', '30'))
    
    # Cache settings
    CACHE_SIZE: int = int(os.getenv('CASEVO_CACHE_SIZE', '1000'))
    CACHE_TTL_SECONDS: int = int(os.getenv('CASEVO_CACHE_TTL', '3600'))
    
    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and k.isupper()}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(cls, key) and key.isupper():
                setattr(cls, key, value)

config = Config()