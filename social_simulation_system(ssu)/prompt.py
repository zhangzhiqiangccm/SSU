from functools import wraps
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from jinja2 import Environment, FileSystemLoader, Template
from jinja2.exceptions import TemplateError, TemplateNotFound

from casevo.base_component import BaseAgentComponent
from casevo.config import config 
from casevo.exceptions import ValidationError
from casevo.llm_interface import LLM_INTERFACE

logger = logging.getLogger(__name__)

def validate_template(func):
    """Decorator to validate template existence and format."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.template:
            raise ValidationError("Template not initialized")
        return func(self, *args, **kwargs)
    return wrapper

class PromptError(Exception):
    """Base exception for prompt-related errors."""
    pass

class TemplateValidationError(PromptError):
    """Raised when template validation fails."""
    pass

class RenderError(PromptError):
    """Raised when template rendering fails."""
    pass

class Prompt:
    """Handles prompt template rendering and message sending."""

    def __init__(self, template: Template, factory: 'PromptFactory'):
        """Initialize prompt with template and factory.
        
        Args:
            template: Jinja2 template for rendering prompts
            factory: Factory instance for template management and message sending
        
        Raises:
            ValidationError: If template or factory is invalid
        """
        if not template:
            raise ValidationError("Template is required")
        if not factory:
            raise ValidationError("Factory is required")
            
        self.template = template
        self.factory = factory
        self._validate_template()

    def _validate_template(self) -> None:
        """Validate template structure and variables."""
        try:
            # Validate template can render with empty context
            self.template.render(agent={}, model={}, extra=None)
        except TemplateError as e:
            raise TemplateValidationError(f"Invalid template structure: {str(e)}")

    @validate_template
    def get_rendered_prompt(self, context: Dict[str, Any]) -> str:
        """Render prompt template with context.
        
        Args:
            context: Dictionary containing template variables
            
        Returns:
            Rendered prompt string
            
        Raises:
            RenderError: If template rendering fails
        """
        try:
            return self.template.render(**context)
        except Exception as e:
            raise RenderError(f"Failed to render template: {str(e)}")

    def send_prompt(
        self,
        extra: Optional[Any] = None,
        agent: Optional[BaseAgentComponent] = None,
        model: Optional[Any] = None
    ) -> str:
        """Send prompt and get response.
        
        Args:
            extra: Additional context data
            agent: Agent component for context
            model: Model component for context
            
        Returns:
            Response from language model
            
        Raises:
            PromptError: If prompt sending fails
        """
        try:
            # Build context dictionary
            agent_context = {
                "description": getattr(agent, "description", ""),
                "context": getattr(agent, "context", {})
            } if agent else {}
            
            model_context = {
                "context": getattr(model, "context", {})
            } if model else {}

            context = {
                "agent": agent_context,
                "model": model_context,
                "extra": extra
            }
            
            prompt_text = self.get_rendered_prompt(context)
            return self.factory.send_message(prompt_text)
            
        except Exception as e:
            logger.error(f"Failed to send prompt: {str(e)}")
            raise PromptError(f"Failed to send prompt: {str(e)}")

class PromptFactory:
    """Factory for creating and managing prompts."""
    
    def __init__(self, template_dir: Union[str, Path], llm: LLM_INTERFACE):
        """Initialize prompt factory.
        
        Args:
            template_dir: Directory containing template files
            llm: Language model interface
            
        Raises:
            ValidationError: If template directory doesn't exist
            PromptError: If environment setup fails
        """
        self.template_dir = Path(template_dir)
        self.llm = llm
        
        self._validate_directory()
        self._setup_environment()

    def _validate_directory(self) -> None:
        """Validate template directory exists."""
        if not self.template_dir.exists():
            raise ValidationError(f"Template directory not found: {self.template_dir}")
        if not self.template_dir.is_dir():
            raise ValidationError(f"Template path is not a directory: {self.template_dir}")

    def _setup_environment(self) -> None:
        """Set up Jinja environment."""
        try:
            self.env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=True,
                trim_blocks=True,
                lstrip_blocks=True
            )
        except Exception as e:
            raise PromptError(f"Failed to setup template environment: {str(e)}")

    def get_template(self, template_name: str) -> Prompt:
        """Get prompt instance for template.
        
        Args:
            template_name: Name of template file
            
        Returns:
            Initialized prompt instance
            
        Raises:
            TemplateNotFound: If template file doesn't exist
            PromptError: If template loading fails
        """
        try:
            template_path = self.template_dir / template_name
            if not template_path.exists():
                raise TemplateNotFound(f"Template not found: {template_name}")
                
            template = self.env.get_template(template_name)
            return Prompt(template, self)
            
        except TemplateNotFound as e:
            raise e
        except Exception as e:
            raise PromptError(f"Failed to load template: {str(e)}")

    def send_message(self, prompt_text: str) -> str:
        """Send message to language model.
        
        Args:
            prompt_text: Rendered prompt text
            
        Returns:
            Model response
            
        Raises:
            PromptError: If message sending fails
        """
        try:
            return self.llm.send_message(prompt_text)
        except Exception as e:
            raise PromptError(f"Failed to send message: {str(e)}")
            
    def cleanup(self) -> None:
        """Clean up factory resources."""
        # Currently no cleanup needed, but interface provided for future use
        pass