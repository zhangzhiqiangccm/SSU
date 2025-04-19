# src/casevo/exceptions.py
class CasevoError(Exception):
    """Base exception class for CASEVO framework."""
    pass

class ComponentError(CasevoError):
    """Base exception for component-related errors."""
    pass

class ConfigurationError(CasevoError):
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

class ValidationError(CasevoError):
    """Raised when input validation fails."""
    pass

class TimeoutError(CasevoError):
    """Raised when an operation times out."""
    pass

class ResourceExhaustedError(CasevoError):
    """Raised when system resources are exhausted."""
    pass