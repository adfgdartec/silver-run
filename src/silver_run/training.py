import asyncio
import time
import json
from typing import Any, Callable, Dict, List, Optional

from .backend import TrainingBackend
from .checkpoints import MemoryCheckpointStore
from .models import Checkpoint, RunEvent, RunState, TrainingContext, TrainingRunOptions


class TrainingRun:
    def __init__(self, options: Optional[TrainingRunOptions] = None):
        self._current_state = RunState.CREATED
        self._event_log: List[RunEvent] = []
        self._checkpoint_store = (
            options.checkpoint_store if options else MemoryCheckpointStore()
        )
        self._clock = options.clock if options and options.clock else time.time
        self._checkpoint_number = 0
        self._subscribers: List[Callable[[RunEvent], None]] = []

    def subscribe(self, callback: Callable[[RunEvent], None]) -> Callable[[], None]:
        """Subscribe to emitted events and return an unsubscribe function."""
        self._subscribers.append(callback)
        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)
        return unsubscribe

    def summary(self) -> Dict[str, Any]:
        """Return a compact, serializable run summary for dashboards."""
        return {"state": self.state.value, "events": len(self._event_log),
                "checkpoints": self._checkpoint_number,
                "last_event": self._event_log[-1].kind if self._event_log else None}

    @property
    def state(self) -> RunState:
        return self._current_state

    def events(self, kind: Optional[str] = None) -> List[RunEvent]:
        events = [
            RunEvent(kind=event.kind, timestamp=event.timestamp, data=dict(event.data))
            for event in self._event_log
        ]
        return [event for event in events if kind is None or event.kind == kind]

    @property
    def duration(self) -> Optional[float]:
        if len(self._event_log) < 2:
            return None
        return max(0.0, self._event_log[-1].timestamp - self._event_log[0].timestamp)

    def to_json(self) -> str:
        return json.dumps({"summary": self.summary(), "events": [
            {"kind": event.kind, "timestamp": event.timestamp, "data": event.data}
            for event in self._event_log
        ]}, sort_keys=True)

    def start(self) -> None:
        if self._current_state not in [RunState.CREATED, RunState.PAUSED]:
            raise ValueError(f"cannot start a {self._current_state.value} run")
        self._current_state = RunState.RUNNING
        self._emit({"kind": "run_started"})

    def pause(self) -> None:
        if self._current_state != RunState.RUNNING:
            raise ValueError(f"cannot pause a {self._current_state.value} run")
        self._current_state = RunState.PAUSED
        self._emit({"kind": "run_paused"})

    def resume(self) -> None:
        if self._current_state != RunState.PAUSED:
            raise ValueError(f"cannot resume a {self._current_state.value} run")
        self._current_state = RunState.RUNNING
        self._emit({"kind": "run_resumed"})

    def stop(self, reason: str = "stopped by user") -> None:
        if self._current_state not in [
            RunState.RUNNING,
            RunState.PAUSED,
            RunState.CREATED,
        ]:
            return
        self._current_state = RunState.STOPPED
        self._emit({"kind": "run_stopped", "reason": reason})

    def cancel(self, reason: str = "cancelled by user") -> None:
        if self._current_state in [
            RunState.COMPLETED,
            RunState.FAILED,
            RunState.CANCELLED,
        ]:
            return
        self._current_state = RunState.CANCELLED
        self._emit({"kind": "run_cancelled", "reason": reason})

    def complete(self) -> None:
        if self._current_state not in [RunState.RUNNING, RunState.PAUSED]:
            raise ValueError(f"cannot complete a {self._current_state.value} run")
        self._current_state = RunState.COMPLETED
        self._emit({"kind": "run_completed"})

    def fail(self, error: Exception) -> None:
        self._current_state = RunState.FAILED
        self._emit({"kind": "run_failed", "error": str(error)})

    def _emit(self, event_data: dict[str, Any]) -> RunEvent:
        event = RunEvent(
            kind=event_data["kind"],
            timestamp=self._clock(),
            data={key: value for key, value in event_data.items() if key != "kind"},
        )
        self._event_log.append(event)
        for subscriber in tuple(self._subscribers):
            subscriber(event)
        if self._current_state == RunState.CREATED and event.kind != "run_started":
            self._current_state = RunState.RUNNING
        return event

    async def checkpoint(self, payload: Any, id: Optional[str] = None) -> Checkpoint:
        if id is None:
            self._checkpoint_number += 1
            id = f"checkpoint-{self._checkpoint_number}"
        checkpoint = Checkpoint(id=id, created_at=self._clock(), payload=payload)
        await self._checkpoint_store.save(checkpoint)
        self._emit({"kind": "checkpoint_saved", "checkpoint_id": id})
        return checkpoint

    async def latest_checkpoint(self) -> Optional[Checkpoint]:
        return await self._checkpoint_store.latest()

    async def execute(self, backend: TrainingBackend) -> RunState:
        self.start()
        try:
            context = TrainingContext(
                emit=lambda event: self._emit(event),
                should_stop=lambda: self._current_state
                in [
                    RunState.STOPPED,
                    RunState.CANCELLED,
                ],
                checkpoint=lambda payload, id=None: self.checkpoint(payload, id),
            )
            async for event in backend.run(context):
                if self._current_state == RunState.PAUSED:
                    await self._wait_until_active()
                if self._current_state in [RunState.STOPPED, RunState.CANCELLED]:
                    break
                self._emit(event)
            if self._current_state == RunState.RUNNING:
                self.complete()
        except Exception as error:
            self.fail(error)
            raise
        return self._current_state

    async def _wait_until_active(self) -> None:
        while self._current_state == RunState.PAUSED:
            await asyncio.sleep(0.001)
