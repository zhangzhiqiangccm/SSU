# src/casevo/base_component.py
import logging
from typing import Any, Optional, Dict
from abc import ABC, abstractmethod
from .config import config
from .exceptions import ComponentError, ValidationError

logger = logging.getLogger(__name__)

class BaseComponent(ABC):
    """Base class for all CASEVO components with improved error handling and logging."""
    
    def __init__(self, component_id: str):
        """Initialize the base component.
        
        Args:
            component_id: Unique identifier for the component
        """
        self.component_id = component_id
        self.is_initialized = False
        self._metadata: Dict[str, Any] = {}
        
    def initialize(self) -> None:
        """Initialize the component with proper error handling."""
        try:
            self._validate_initialization()
            self._initialize_component()
            self.is_initialized = True
            logger.info(f"Component {self.component_id} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize component {self.component_id}: {str(e)}")
            raise ComponentError(f"Initialization failed: {str(e)}") from e

    @abstractmethod
    def _initialize_component(self) -> None:
        """Component-specific initialization logic."""
        pass

    def _validate_initialization(self) -> None:
        """Validate component initialization requirements."""
        if not self.component_id:
            raise ValidationError("Component ID is required")
        if self.is_initialized:
            raise ComponentError("Component is already initialized")

    def get_metadata(self) -> Dict[str, Any]:
        """Get component metadata."""
        return self._metadata.copy()

    def set_metadata(self, key: str, value: Any) -> None:
        """Set component metadata with validation."""
        if not isinstance(key, str):
            raise ValidationError("Metadata key must be a string")
        self._metadata[key] = value

    def validate_state(self) -> bool:
        """Validate component state.
        
        Returns:
            bool: True if the component is in a valid state
        """
        try:
            if not self.is_initialized:
                logger.warning(f"Component {self.component_id} is not initialized")
                return False
            return self._validate_component_state()
        except Exception as e:
            logger.error(f"State validation failed for {self.component_id}: {str(e)}")
            return False

    @abstractmethod
    def _validate_component_state(self) -> bool:
        """Component-specific state validation logic."""
        pass

    def cleanup(self) -> None:
        """Clean up component resources safely."""
        try:
            if self.is_initialized:
                self._cleanup_component()
                self.is_initialized = False
                logger.info(f"Component {self.component_id} cleaned up successfully")
        except Exception as e:
            logger.error(f"Failed to clean up component {self.component_id}: {str(e)}")
            raise ComponentError(f"Cleanup failed: {str(e)}") from e

    @abstractmethod
    def _cleanup_component(self) -> None:
        """Component-specific cleanup logic."""
        pass

    def __repr__(self) -> str:
        """Return string representation of the component."""
        return f"{self.__class__.__name__}(id={self.component_id}, initialized={self.is_initialized})"