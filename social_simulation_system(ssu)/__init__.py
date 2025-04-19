# src/casevo/__init__.py
"""CASEVO - Cognitive Agents and Social Evolution Simulator."""

from .config import Config, config
from .exceptions import (
    CasevoError, ComponentError, ConfigurationError, 
    MemoryError, LLMError, ThreadingError, 
    ValidationError, TimeoutError, ResourceExhaustedError
)
from .base_component import BaseComponent
from .agent_base import AgentBase
from .model_base import ModelBase
from .memory import Memory, MemoryFactory, MemoryItem
from .chain import ThoughtChain, BaseStep, ChainPool
from .prompt import Prompt, PromptFactory
from .llm_interface import LLM_INTERFACE

__version__ = "1.0.0"
__all__ = [
    'Config',
    'config',
    'CasevoError',
    'ComponentError',
    'ConfigurationError',
    'MemoryError',
    'LLMError',
    'ThreadingError',
    'ValidationError',
    'TimeoutError',
    'ResourceExhaustedError',
    'BaseComponent',
    'AgentBase',
    'ModelBase',
    'Memory',
    'MemoryFactory',
    'MemoryItem',
    'ThoughtChain',
    'BaseStep',
    'ChainPool',
    'Prompt',
    'PromptFactory',
    'LLM_INTERFACE'
]