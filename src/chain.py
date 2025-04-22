from dataclasses import dataclass
from enum import Enum, auto
import json
import logging
import queue
import re
import threading
import time
from typing import Any, Dict, List, Optional, Pattern, Union
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from ssu.base_component import BaseAgentComponent, BaseModelComponent
from ssu.config import config
from ssu.exceptions import ValidationError, ThreadingError

logger = logging.getLogger(__name__)

class ChainState(Enum):
    """Enumeration of possible chain states."""
    INIT = auto()
    READY = auto() 
    RUNNING = auto()
    FINISHED = auto()
    ERROR = auto()

class StepError(Exception):
    """Base exception for step-related errors."""
    pass

class StepValidationError(StepError):
    """Raised when step validation fails."""
    pass

class StepExecutionError(StepError):
    """Raised when step execution fails."""
    pass

@dataclass
class StepResult:
    """Container for step execution results."""
    step_id: str
    input_data: Any
    output_data: Any
    metadata: Dict[str, Any] = None

class BaseStep:
    """Base class for all chain steps."""
    
    def __init__(self, step_id: str, prompt: Any):
        """Initialize step with ID and prompt."""
        self.step_id = step_id
        self.prompt = prompt
        self._validate_init()

    def _validate_init(self) -> None:
        """Validate step initialization."""
        if not self.step_id:
            raise StepValidationError("Step ID is required")
        if not self.prompt:
            raise StepValidationError("Prompt is required")

    def pre_process(self, input_data: Any, agent: Optional[Any] = None, model: Optional[Any] = None) -> Any:
        """Pre-process input data before main action."""
        return input_data

    def action(self, input_data: Any, agent: Optional[Any] = None, model: Optional[Any] = None) -> Any:
        """Execute main step action."""
        try:
            return self.prompt.send_prompt(input_data, agent, model)
        except Exception as e:
            logger.error(f"Step {self.step_id} action failed: {str(e)}")
            raise StepExecutionError(f"Action failed: {str(e)}") from e

    def after_process(self, input_data: Any, response: Any, agent: Optional[Any] = None, model: Optional[Any] = None) -> Dict[str, Any]:
        """Process action response."""
        return {
            'input': input_data,
            'last_response': response
        }

    def get_id(self) -> str:
        """Get step ID."""
        return self.step_id

class ChoiceStep(BaseStep):
    """Step for handling choice-based responses."""
    
    def __init__(self, step_id: str, prompt: Any, choice_pattern: Optional[Pattern] = None):
        super().__init__(step_id, prompt)
        self.answer_pattern = choice_pattern or re.compile(r"[A-Z]")

    def after_process(self, input_data: Any, response: str, agent: Optional[Any] = None, model: Optional[Any] = None) -> Dict[str, Any]:
        """Extract choice from response."""
        match = self.answer_pattern.search(response)
        if not match:
            raise StepValidationError("No valid choice found in response")
        return {
            'input': input_data,
            'choice': match.group()
        }

class ScoreStep(BaseStep):
    """Step for handling numeric score responses."""
    
    def __init__(self, step_id: str, prompt: Any, score_pattern: Optional[Pattern] = None):
        super().__init__(step_id, prompt)
        self.answer_pattern = score_pattern or re.compile(r"(-?\d+)(\.\d+)?")

    def after_process(self, input_data: Any, response: str, agent: Optional[Any] = None, model: Optional[Any] = None) -> Dict[str, Any]:
        """Extract score from response."""
        match = self.answer_pattern.search(response)
        if not match:
            raise StepValidationError("No valid score found in response")
        return {
            'input': input_data,
            'score': float(match.group())
        }

class JsonStep(BaseStep):
    """Step for handling JSON responses."""
    
    def __init__(self, step_id: str, prompt: Any, json_pattern: Optional[Pattern] = None):
        super().__init__(step_id, prompt)
        self.answer_pattern = json_pattern or re.compile(r"\{[\s\S]*\}")

    def after_process(self, input_data: Any, response: str, agent: Optional[Any] = None, model: Optional[Any] = None) -> Dict[str, Any]:
        """Extract and parse JSON from response."""
        match = self.answer_pattern.search(response)
        if not match:
            raise StepValidationError("No valid JSON found in response")
        try:
            parsed_json = json.loads(match.group())
            return {
                'input': input_data,
                'json': parsed_json
            }
        except json.JSONDecodeError as e:
            raise StepValidationError(f"Invalid JSON format: {str(e)}")

class ThoughtChain(BaseAgentComponent):
    """Chain of thought processing implementation."""
    
    def __init__(self, agent: Any, steps: List[BaseStep]):
        super().__init__(f"{agent.component_id}_chain", 'chain', agent)
        self.steps = steps
        self.state = ChainState.INIT
        self.input_content = None
        self.step_history: List[StepResult] = []
        self.output_content = None
        self._lock = threading.RLock()

    def set_input(self, input_data: Any) -> None:
        """Set input data for chain processing."""
        with self._lock:
            if self.state not in {ChainState.INIT, ChainState.FINISHED}:
                raise ValidationError("Cannot set input in current state")
            self.input_content = input_data
            self.step_history = []
            self.state = ChainState.READY

    def run_step(self) -> None:
        """Execute chain steps with retry mechanism."""
        with self._lock:
            if self.state != ChainState.READY:
                raise ValidationError("Chain not ready for execution")
            
            self.state = ChainState.RUNNING
            current_input = self.input_content

            try:
                for step in self.steps:
                    result = self._execute_step_with_retry(step, current_input)
                    self.step_history.append(result)
                    current_input = result.output_data

                self.output_content = self.step_history[-1].output_data
                self.state = ChainState.FINISHED

            except Exception as e:
                self.state = ChainState.ERROR
                logger.error(f"Chain execution failed: {str(e)}")
                raise

    def _execute_step_with_retry(self, step: BaseStep, input_data: Any) -> StepResult:
        """Execute a single step with configurable retries."""
        last_error = None
        for attempt in range(config.MAX_RETRIES):
            try:
                processed_input = step.pre_process(input_data, self.agent, self.agent.model)
                response = step.action(processed_input, self.agent, self.agent.model)
                output = step.after_process(processed_input, response, self.agent, self.agent.model)
                
                return StepResult(
                    step_id=step.get_id(),
                    input_data=processed_input,
                    output_data=output
                )

            except Exception as e:
                last_error = e
                logger.warning(f"Step {step.get_id()} failed (attempt {attempt + 1}/{config.MAX_RETRIES}): {str(e)}")
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(1)  # Add delay between retries
                continue

        raise StepExecutionError(f"Step failed after {config.MAX_RETRIES} attempts: {str(last_error)}")

    def get_output(self) -> Any:
        """Get chain output."""
        with self._lock:
            if self.state != ChainState.FINISHED:
                raise ValidationError("Chain execution not finished")
            return self.output_content

    def get_history(self) -> List[StepResult]:
        """Get chain execution history."""
        with self._lock:
            if self.state != ChainState.FINISHED:
                raise ValidationError("Chain execution not finished")
            return self.step_history.copy()

class ChainPool:
    """Thread pool for parallel chain execution."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or config.THREAD_POOL_SIZE
        self.chain_queue = queue.Queue()
        self.state = ChainState.INIT
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="chain_worker"
        )
        self._shutdown = threading.Event()

    def add_chains(self, chains: List[ThoughtChain]) -> None:
        """Add chains to execution queue."""
        if self.state != ChainState.INIT:
            raise ValidationError("Cannot add chains after pool has started")
        
        for chain in chains:
            self.chain_queue.put(chain)
        self.state = ChainState.READY

    def _worker(self) -> None:
        """Chain execution worker."""
        while not self._shutdown.is_set():
            try:
                chain = self.chain_queue.get_nowait()
                try:
                    chain.run_step()
                except Exception as e:
                    logger.error(f"Chain execution failed: {str(e)}")
                finally:
                    self.chain_queue.task_done()
                    
            except queue.Empty:
                break

            time.sleep(config.THREAD_TIMEOUT_SECONDS)

    def start_pool(self) -> None:
        """Start chain execution pool."""
        if self.state != ChainState.READY:
            raise ValidationError("Pool not ready to start")

        try:
            workers = [
                self._executor.submit(self._worker)
                for _ in range(self.max_workers)
            ]
            self.chain_queue.join()

        except Exception as e:
            self._shutdown.set()
            raise ThreadingError(f"Pool execution failed: {str(e)}") from e

        finally:
            self._executor.shutdown(wait=True)

    def cleanup(self) -> None:
        """Clean up pool resources."""
        self._shutdown.set()
        self._executor.shutdown(wait=True)