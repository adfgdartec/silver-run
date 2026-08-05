from typing import Any, AsyncIterator, Dict

from .models import TrainingContext


class TrainingBackend:
    async def run(self, context: TrainingContext) -> AsyncIterator[Dict[str, Any]]:
        raise NotImplementedError
