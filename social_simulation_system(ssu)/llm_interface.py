from abc import ABC, abstractmethod
from typing import Any, List, Union, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import time

from casevo.exceptions import LLMError
from casevo.config import config

logger = logging.getLogger(__name__)

def retry_on_error(max_retries: int = None, delay: float = None):
    """Decorator to retry LLM operations on failure."""
    max_retries = max_retries or config.MAX_RETRIES
    delay = delay or 1.0
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"LLM operation failed (attempt {attempt + 1}/{max_retries}): {str(e)}"
                        )
                        time.sleep(delay)
                    continue
            raise LLMError(f"Operation failed after {max_retries} attempts: {str(last_error)}")
        return wrapper
    return decorator

class LLMInterface(ABC):
    """Abstract base class for Language Model interfaces.
    
    This interface defines the contract for interacting with various Language 
    Model implementations, providing methods for text generation and embedding
    computation.
    """

    def __init__(self):
        """Initialize LLM interface with thread pool for concurrent operations."""
        self._executor = ThreadPoolExecutor(
            max_workers=config.THREAD_POOL_SIZE,
            thread_name_prefix="llm_worker"
        )

    @abstractmethod
    @retry_on_error()
    def send_message(
        self, 
        prompt: str, 
        json_mode: bool = False,
        **kwargs: Any
    ) -> Union[str, Dict[str, Any]]:
        """Send a prompt to the language model and get the response.

        Args:
            prompt: The input prompt text to send to the model
            json_mode: If True, expect and validate JSON response
            **kwargs: Additional model-specific parameters

        Returns:
            Model response as string or parsed JSON object if json_mode=True

        Raises:
            LLMError: If the model request fails or returns invalid response
            ValueError: If the prompt is empty or invalid
        """
        pass

    @abstractmethod
    @retry_on_error()
    def send_embedding(
        self, 
        texts: List[str],
        **kwargs: Any
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of input texts to embed
            **kwargs: Additional model-specific parameters

        Returns:
            List of embedding vectors (as float lists)

        Raises:
            LLMError: If embedding generation fails
            ValueError: If input texts are empty or invalid
        """
        pass

    @abstractmethod
    def get_lang_embedding(self) -> Any:
        """Get the embedding function interface for language models.

        This method should return a callable that matches the interface
        expected by the language model library being used (e.g., LangChain).

        Returns:
            Callable that generates embeddings in the required format

        Raises:
            LLMError: If embedding interface initialization fails
        """
        pass

    def cleanup(self) -> None:
        """Clean up resources used by the LLM interface.
        
        Should be called when the interface is no longer needed to ensure
        proper resource cleanup.
        """
        try:
            self._executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Failed to cleanup LLM interface: {str(e)}")

    def __del__(self) -> None:
        """Ensure cleanup on object deletion."""
        self.cleanup()