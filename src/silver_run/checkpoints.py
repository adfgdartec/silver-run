from typing import Dict, Optional

from .models import Checkpoint


class CheckpointStore:
    async def save(self, checkpoint: Checkpoint) -> None:
        raise NotImplementedError

    async def latest(self) -> Optional[Checkpoint]:
        raise NotImplementedError

    async def get(self, id: str) -> Optional[Checkpoint]:
        raise NotImplementedError


class MemoryCheckpointStore(CheckpointStore):
    def __init__(self):
        self._values: Dict[str, Checkpoint] = {}

    async def save(self, checkpoint: Checkpoint) -> None:
        self._values[checkpoint.id] = checkpoint

    async def latest(self) -> Optional[Checkpoint]:
        if not self._values:
            return None
        return list(self._values.values())[-1]

    async def get(self, id: str) -> Optional[Checkpoint]:
        return self._values.get(id)
