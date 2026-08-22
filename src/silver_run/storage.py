"""Dependency-free, durable local storage for Silver experiments."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

from .checkpoints import CheckpointStore
from .models import Checkpoint, RunEvent


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _validate_id(value: str, kind: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(
            "%s must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_', or '-'" % kind
        )
    return value


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=".silver-", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, str(path))
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class FileCheckpointStore(CheckpointStore):
    """Atomically persist JSON-safe checkpoints in a local directory."""

    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    async def save(self, checkpoint: Checkpoint) -> None:
        checkpoint_id = _validate_id(checkpoint.id, "checkpoint id")
        _atomic_json(self.directory / (checkpoint_id + ".json"), {
            "schema": "silver.run/checkpoint-1",
            "id": checkpoint_id,
            "created_at": checkpoint.created_at,
            "payload": checkpoint.payload,
        })

    async def latest(self) -> Optional[Checkpoint]:
        values = [self._load(path) for path in self.directory.glob("*.json")]
        return max(values, key=lambda item: item.created_at) if values else None

    async def get(self, id: str) -> Optional[Checkpoint]:
        path = self.directory / (_validate_id(id, "checkpoint id") + ".json")
        return self._load(path) if path.exists() else None

    @staticmethod
    def _load(path: Path) -> Checkpoint:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "silver.run/checkpoint-1":
            raise ValueError("unsupported checkpoint schema in %s" % path)
        return Checkpoint(
            id=str(payload["id"]),
            created_at=float(payload["created_at"]),
            payload=payload.get("payload"),
        )


@dataclass(frozen=True)
class StoredRun:
    id: str
    state: str
    metadata: Dict[str, Any]
    events: Tuple[RunEvent, ...]

    @property
    def duration(self) -> Optional[float]:
        if len(self.events) < 2:
            return None
        return max(0.0, self.events[-1].timestamp - self.events[0].timestamp)

    @property
    def metrics(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(dict(event.data.get("metrics", {})) for event in self.events
                     if isinstance(event.data.get("metrics"), dict))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "silver.run/stored-run-1",
            "id": self.id,
            "state": self.state,
            "metadata": dict(self.metadata),
            "duration": self.duration,
            "events": [
                {"kind": event.kind, "timestamp": event.timestamp, "data": dict(event.data)}
                for event in self.events
            ],
        }


class LocalRunStore:
    """A transparent local experiment tracker backed by JSON and JSONL files."""

    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def start(self, run_id: str, metadata: Dict[str, Any]) -> None:
        run_id = _validate_id(run_id, "run id")
        run_directory = self.directory / run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        manifest = run_directory / "run.json"
        if manifest.exists():
            raise ValueError("run %r already exists" % run_id)
        _atomic_json(manifest, {
            "schema": "silver.run/manifest-1",
            "id": run_id,
            "state": "created",
            "metadata": dict(metadata),
        })

    def append(self, run_id: str, event: RunEvent, state: str) -> None:
        run_id = _validate_id(run_id, "run id")
        run_directory = self.directory / run_id
        manifest_path = run_directory / "run.json"
        if not manifest_path.exists():
            raise KeyError("unknown run %r" % run_id)
        line = json.dumps({
            "kind": event.kind,
            "timestamp": event.timestamp,
            "data": event.data,
        }, sort_keys=True, allow_nan=False) + "\n"
        with self._lock:
            with (run_directory / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["state"] = state
            _atomic_json(manifest_path, manifest)

    def load(self, run_id: str) -> StoredRun:
        run_id = _validate_id(run_id, "run id")
        run_directory = self.directory / run_id
        manifest_path = run_directory / "run.json"
        if not manifest_path.exists():
            raise KeyError("unknown run %r" % run_id)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "silver.run/manifest-1":
            raise ValueError("unsupported run manifest schema")
        events: List[RunEvent] = []
        events_path = run_directory / "events.jsonl"
        if events_path.exists():
            for line_number, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    value = json.loads(line)
                    events.append(RunEvent(
                        kind=str(value["kind"]),
                        timestamp=float(value["timestamp"]),
                        data=dict(value.get("data", {})),
                    ))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        "invalid run event at line %d" % line_number
                    ) from error
        return StoredRun(
            id=run_id,
            state=str(manifest["state"]),
            metadata=dict(manifest.get("metadata", {})),
            events=tuple(events),
        )

    def list_runs(self) -> Tuple[StoredRun, ...]:
        values = []
        for path in self.directory.iterdir():
            if path.is_dir() and (path / "run.json").exists():
                values.append(self.load(path.name))
        return tuple(sorted(values, key=lambda item: item.id))
