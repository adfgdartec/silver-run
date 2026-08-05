from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional


class RunState(Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class RunEvent:
    kind: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Checkpoint:
    id: str
    created_at: float
    payload: Any


@dataclass
class TrainingContext:
    emit: Callable[[Dict[str, Any]], RunEvent]
    should_stop: Callable[[], bool]
    checkpoint: Callable[[Any, Optional[str]], Any]


@dataclass
class TrainingRunOptions:
    checkpoint_store: Optional[Any] = None
    clock: Optional[Callable[[], float]] = None
