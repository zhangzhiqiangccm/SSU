from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import threading
from contextlib import contextmanager
import logging
import chromadb
from chromadb.api.types import QueryResult

from ssu.base_component import BaseAgentComponent, BaseModelComponent
from ssu.llm_interface import LLM_INTERFACE
from ssu.config import config
from ssu.exceptions import (
    MemoryError, ValidationError, ThreadingError, 
    TimeoutError, ResourceExhaustedError
)

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class MemoryItem:
    """Immutable memory item with validation."""
    
    timestamp: float
    source: str  
    target: str
    action: str
    content: str
    item_id: int = -1

    def __post_init__(self) -> None:
        """Validate memory item fields."""
        if not isinstance(self.timestamp, (int, float)):
            raise ValidationError("Timestamp must be a number")
        if not all(isinstance(x, str) for x in [self.source, self.target, self.action, self.content]):
            raise ValidationError("Source, target, action and content must be strings")
        if not self.source or not self.target:
            raise ValidationError("Source and target cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory item to dictionary format."""
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "target": self.target, 
            "action": self.action,
            "content": self.content,
            "id": self.item_id
        }

    @staticmethod
    def to_lists(memory_items: List["MemoryItem"], start_id: int = 0) -> tuple[List[str], List[Dict], List[str]]:
        """Convert memory items to format required by ChromaDB."""
        content_list = []
        metadata_list = []
        id_list = []
        
        for idx, item in enumerate(memory_items, start=start_id):
            item_dict = item.to_dict()
            item_dict["id"] = idx
            content_list.append(item_dict["content"])
            metadata_list.append(item_dict)
            id_list.append(str(idx))
            
        return content_list, metadata_list, id_list


class Memory(BaseAgentComponent):
    """Agent memory component with short and long-term memory management."""

    def __init__(self, component_id: str, agent: Any, memory_factory: "MemoryFactory"):
        """Initialize memory component."""
        super().__init__(component_id, "memory", agent)
        self.memory_factory = memory_factory
        self.long_term_memory: Optional[str] = None
        self.last_reflection_id: int = -1
        self._lock = threading.RLock()

    def add_short_term_memory(
        self, 
        source: str, 
        target: str, 
        action: str, 
        content: str, 
        timestamp: Optional[float] = None
    ) -> bool:
        """Add a memory item to short-term memory."""
        try:
            if timestamp is None:
                timestamp = self.agent.model.schedule.time
                
            memory_item = MemoryItem(
                timestamp=timestamp,
                source=source,
                target=target,
                action=action,
                content=content
            )
            
            return self.memory_factory.add_short_term_memory([memory_item])
            
        except Exception as e:
            logger.error(f"Failed to add short-term memory: {str(e)}")
            raise MemoryError(f"Memory addition failed: {str(e)}") from e

    def search_short_term_memory(self, query_contents: List[str]) -> QueryResult:
        """Search short-term memory based on content."""
        try:
            return self.memory_factory.search_short_term_memory(
                query_contents, 
                self.agent.component_id
            )
        except Exception as e:
            logger.error(f"Memory search failed: {str(e)}")
            raise MemoryError(f"Memory search failed: {str(e)}") from e

    def reflect_on_memory(self) -> None:
        """Update long-term memory through reflection."""
        try:
            with self._lock:
                response, last_id = self.memory_factory.reflect_on_memory(
                    self.agent,
                    self.last_reflection_id,
                    self.long_term_memory
                )
                self.last_reflection_id = last_id
                self.long_term_memory = response
                
        except Exception as e:
            logger.error(f"Memory reflection failed: {str(e)}")
            raise MemoryError(f"Memory reflection failed: {str(e)}") from e

    def get_long_term_memory(self) -> Optional[str]:
        """Get current long-term memory state."""
        return self.long_term_memory


class MemoryFactory(BaseModelComponent):
    """Factory for creating and managing agent memories."""

    def __init__(
        self,
        llm_interface: LLM_INTERFACE,
        max_results: int,
        reflection_prompt: Any,
        model: Any,
        persistence_path: Optional[str] = None
    ):
        """Initialize memory factory."""
        super().__init__("memory_factory", "memory_factory", model)
        
        self.llm = llm_interface
        self.max_results = max_results
        self.reflection_prompt = reflection_prompt
        
        try:
            self.client = (
                chromadb.PersistentClient(path=persistence_path)
                if persistence_path
                else chromadb.Client()
            )
            
            self.memory_collection = self.client.get_or_create_collection(
                name="memory",
                embedding_function=self.llm.get_lang_embedding()
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise ResourceExhaustedError(f"Database initialization failed: {str(e)}") from e
            
        self._locks = {
            "collection": threading.RLock(),
            "reflection": threading.RLock()
        }

    @contextmanager
    def _acquire_locks(self, *lock_names: str, timeout: float = config.THREAD_TIMEOUT_SECONDS):
        """Acquire multiple locks with timeout."""
        locks = [self._locks[name] for name in lock_names]
        acquired = []
        
        try:
            for lock in locks:
                if not lock.acquire(timeout=timeout):
                    raise TimeoutError("Failed to acquire lock within timeout")
                acquired.append(lock)
            yield
            
        finally:
            for lock in reversed(acquired):
                lock.release()

    def create_memory(self, agent: Any) -> Memory:
        """Create a new memory component for an agent."""
        return Memory(f"{agent.component_id}_memory", agent, self)

    def add_short_term_memory(self, memory_items: List[MemoryItem]) -> bool:
        """Add memory items to short-term memory."""
        try:
            with self._acquire_locks("collection"):
                start_pos = self.memory_collection.count()
                content_list, metadata_list, id_list = MemoryItem.to_lists(memory_items, start_pos)
                
                self.memory_collection.add(
                    documents=content_list,
                    metadatas=metadata_list,
                    ids=id_list
                )
                return True
                
        except TimeoutError as e:
            raise ThreadingError("Memory addition timed out") from e
        except Exception as e:
            logger.error(f"Failed to add memories: {str(e)}")
            raise MemoryError(f"Memory addition failed: {str(e)}") from e

    def search_short_term_memory(
        self, 
        query_contents: List[str], 
        agent_id: str
    ) -> QueryResult:
        """Search short-term memory for relevant content."""
        try:
            with self._acquire_locks("collection"):
                return self.memory_collection.query(
                    query_texts=query_contents,
                    n_results=self.max_results,
                    where={"$or": [{"source": agent_id}, {"target": agent_id}]}
                )
                
        except TimeoutError as e:
            raise ThreadingError("Memory search timed out") from e
        except Exception as e:
            logger.error(f"Memory search failed: {str(e)}")
            raise MemoryError(f"Memory search failed: {str(e)}") from e

    def reflect_on_memory(
        self, 
        agent: Any, 
        last_reflection_id: int,
        long_term_memory: Optional[str]
    ) -> tuple[str, int]:
        """Process memory reflection for an agent."""
        try:
            with self._acquire_locks("collection", "reflection"):
                memory_items = self.memory_collection.get(
                    where={
                        "$and": [
                            {"id": {"$gt": last_reflection_id}},
                            {"$or": [
                                {"source": agent.component_id},
                                {"target": agent.component_id}
                            ]}
                        ]
                    }
                )

                memory_data = {
                    "long_memory": long_term_memory,
                    "short_memory": memory_items["metadatas"]
                }

                response = self.reflection_prompt.send_prompt(memory_data, agent, self.model)
                
                last_id = max(
                    (int(item_id) for item_id in memory_items["ids"]),
                    default=last_reflection_id
                )

                return response, last_id

        except TimeoutError as e:
            raise ThreadingError("Memory reflection timed out") from e
        except Exception as e:
            logger.error(f"Memory reflection failed: {str(e)}")
            raise MemoryError(f"Memory reflection failed: {str(e)}") from e

    def cleanup(self) -> None:
        """Clean up resources on shutdown."""
        try:
            self.client.reset()
        except Exception as e:
            logger.error(f"Failed to cleanup memory factory: {str(e)}")