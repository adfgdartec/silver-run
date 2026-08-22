import json

import pytest

from silver_run import (
    Checkpoint,
    FileCheckpointStore,
    LocalRunStore,
    TrainingRun,
    TrainingRunOptions,
)


@pytest.mark.asyncio
async def test_file_checkpoint_store_roundtrip_and_latest(tmp_path):
    store = FileCheckpointStore(str(tmp_path / "checkpoints"))
    first = Checkpoint("first", 1.0, {"epoch": 1})
    second = Checkpoint("second", 2.0, {"epoch": 2})
    await store.save(first)
    await store.save(second)

    assert await store.get("first") == first
    assert await store.latest() == second
    payload = json.loads((tmp_path / "checkpoints" / "second.json").read_text())
    assert payload["schema"] == "silver.run/checkpoint-1"


@pytest.mark.asyncio
async def test_file_checkpoint_store_rejects_unsafe_ids_and_payloads(tmp_path):
    store = FileCheckpointStore(str(tmp_path))
    with pytest.raises(ValueError, match="checkpoint id"):
        await store.save(Checkpoint("../escape", 1.0, {}))
    with pytest.raises(ValueError):
        await store.save(Checkpoint("nan", 1.0, {"value": float("nan")}))


@pytest.mark.asyncio
async def test_training_run_persists_replayable_events_and_metadata(tmp_path):
    store = LocalRunStore(str(tmp_path / "runs"))
    run = TrainingRun(TrainingRunOptions(
        run_id="experiment-001",
        metadata={"model": "tiny", "dataset": "demo"},
        run_store=store,
    ))
    run.start()
    run._emit({"kind": "epoch", "epoch": 1, "metrics": {"loss": 0.5}})
    await run.checkpoint({"epoch": 1})
    run.complete()

    restored = store.load("experiment-001")
    assert restored.state == "completed"
    assert restored.metadata["model"] == "tiny"
    assert restored.metrics == ({"loss": 0.5},)
    assert [event.kind for event in restored.events] == [
        "run_started", "epoch", "checkpoint_saved", "run_completed",
    ]
    assert store.list_runs() == (restored,)


def test_run_store_rejects_duplicate_and_unsafe_run_ids(tmp_path):
    store = LocalRunStore(str(tmp_path))
    TrainingRun(TrainingRunOptions(run_id="safe-run", run_store=store))
    with pytest.raises(ValueError, match="already exists"):
        TrainingRun(TrainingRunOptions(run_id="safe-run", run_store=store))
    with pytest.raises(ValueError, match="run id"):
        TrainingRun(TrainingRunOptions(run_id="../escape", run_store=store))
