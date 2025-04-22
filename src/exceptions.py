# src/ssu/exceptions.py
class SsuError(Exception):
    """Base exception class for SSU framework."""
    pass

class ComponentError(SsuError):
    """Base exception for component-related errors."""
    pass

class ConfigurationError(SsuError):
    """Raised when there's a configuration-related error."""
    pass

class MemoryError(ComponentError):
    """Raised when there's an error in memory operations."""
    pass

class LLMError(ComponentError):
    """Raised when there's an error in LLM operations."""
    pass

class ThreadingError(ComponentError):
    """Raised when there's an error in threaded operations."""
    pass

class ValidationError(SsuError):
    """Raised when input validation fails."""
    pass

class TimeoutError(SsuError):
    """Raised when an operation times out."""
    pass

class ResourceExhaustedError(SsuError):
    """Raised when system resources are exhausted."""
    pass