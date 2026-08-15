"""Explicit GPU allocation and load-balancing primitives for Silver runs."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class GPUDevice:
    id: str
    memory_mb: int
    capacity_percent: float = 100.0


@dataclass(frozen=True)
class TaskResourceRequest:
    task_id: str
    gpu_percent: float = 100.0
    memory_mb: int = 0
    preferred_gpus: Tuple[str, ...] = ()
    priority: int = 0


@dataclass(frozen=True)
class GPUAllocation:
    task_id: str
    gpu_id: str
    gpu_percent: float
    memory_mb: int

    def environment(self) -> Dict[str, str]:
        return {"CUDA_VISIBLE_DEVICES": self.gpu_id,
                "SILVER_GPU_PERCENT": str(self.gpu_percent),
                "SILVER_GPU_MEMORY_MB": str(self.memory_mb)}


@dataclass(frozen=True)
class ResourcePlan:
    allocations: Tuple[GPUAllocation, ...]
    rejected: Tuple[str, ...] = ()

    def for_task(self, task_id: str) -> Optional[GPUAllocation]:
        return next((item for item in self.allocations if item.task_id == task_id), None)

    def to_dict(self) -> Dict[str, object]:
        return {"allocations": [{"task_id": item.task_id, "gpu_id": item.gpu_id,
                                  "gpu_percent": item.gpu_percent, "memory_mb": item.memory_mb}
                                 for item in self.allocations],
                "rejected": list(self.rejected)}


class GPUResourceScheduler:
    """Priority-aware greedy scheduler that balances tasks across GPUs."""

    def __init__(self, devices: Iterable[GPUDevice]):
        self.devices = tuple(devices)
        if not self.devices or len({device.id for device in self.devices}) != len(self.devices):
            raise ValueError("at least one uniquely named GPU is required")

    def plan(self, requests: Iterable[TaskResourceRequest]) -> ResourcePlan:
        usage = {device.id: [0.0, 0] for device in self.devices}
        allocations: List[GPUAllocation] = []
        rejected: List[str] = []
        for request in sorted(requests, key=lambda item: (-item.priority, item.task_id)):
            if not 0 < request.gpu_percent <= 100 or request.memory_mb < 0:
                raise ValueError("GPU percentage must be in (0, 100] and memory non-negative")
            candidates = [device for device in self.devices if not request.preferred_gpus or device.id in request.preferred_gpus]
            candidates = [device for device in candidates if usage[device.id][0] + request.gpu_percent <= device.capacity_percent
                          and usage[device.id][1] + request.memory_mb <= device.memory_mb]
            if not candidates:
                rejected.append(request.task_id)
                continue
            device = min(candidates, key=lambda item: (usage[item.id][0] / item.capacity_percent,
                                                       usage[item.id][1] / max(item.memory_mb, 1), item.id))
            usage[device.id][0] += request.gpu_percent
            usage[device.id][1] += request.memory_mb
            allocations.append(GPUAllocation(request.task_id, device.id, request.gpu_percent, request.memory_mb))
        return ResourcePlan(tuple(allocations), tuple(rejected))
