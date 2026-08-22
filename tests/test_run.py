import pytest
import asyncio
from silver_run import (
    TrainingRun,
    TrainingBackend,
    TrainingContext,
    TrainingRunOptions,
    RunState,
    RunEvent,
    Checkpoint,
    CheckpointStore,
    MemoryCheckpointStore,
)


class MockBackend(TrainingBackend):
    """Mock training backend for testing."""

    def __init__(self, events=None):
        self.events = events or [
            {"kind": "epoch", "epoch": 1, "loss": 0.5},
            {"kind": "epoch", "epoch": 2, "loss": 0.3},
            {"kind": "metrics", "accuracy": 0.9},
        ]

    async def run(self, context: TrainingContext):
        for event in self.events:
            if context.should_stop():
                break
            yield event


class TestTrainingRunCreation:
    def test_default_initialization(self):
        run = TrainingRun()
        assert run.state == RunState.CREATED
        assert len(run.events()) == 0

    def test_custom_checkpoint_store(self):
        store = MemoryCheckpointStore()
        run = TrainingRun(TrainingRunOptions(checkpoint_store=store))
        assert run.state == RunState.CREATED

    def test_custom_clock(self):
        custom_time = lambda: 1000.0
        run = TrainingRun(TrainingRunOptions(clock=custom_time))
        run.start()
        assert run.events()[0].timestamp == 1000.0


class TestTrainingRunLifecycle:
    def test_start_from_created(self):
        run = TrainingRun()
        run.start()
        assert run.state == RunState.RUNNING
        assert run.events()[-1].kind == "run_started"

    def test_start_from_paused(self):
        run = TrainingRun()
        run.start()
        run.pause()
        run.resume()
        assert run.state == RunState.RUNNING

    def test_start_from_invalid_state_raises_error(self):
        run = TrainingRun()
        run.start()
        run.complete()

        with pytest.raises(ValueError, match="cannot start a completed run"):
            run.start()

    def test_pause(self):
        run = TrainingRun()
        run.start()
        run.pause()
        assert run.state == RunState.PAUSED
        assert run.events()[-1].kind == "run_paused"

    def test_pause_from_invalid_state_raises_error(self):
        run = TrainingRun()
        with pytest.raises(ValueError, match="cannot pause a created run"):
            run.pause()

    def test_resume(self):
        run = TrainingRun()
        run.start()
        run.pause()
        run.resume()
        assert run.state == RunState.RUNNING
        assert run.events()[-1].kind == "run_resumed"

    def test_resume_from_invalid_state_raises_error(self):
        run = TrainingRun()
        with pytest.raises(ValueError, match="cannot resume a created run"):
            run.resume()

    def test_stop(self):
        run = TrainingRun()
        run.start()
        run.stop("test reason")
        assert run.state == RunState.STOPPED
        assert run.events()[-1].kind == "run_stopped"

    def test_stop_from_created(self):
        run = TrainingRun()
        run.stop()
        assert run.state == RunState.STOPPED

    def test_stop_from_terminal_state_no_op(self):
        run = TrainingRun()
        run.start()
        run.complete()
        run.stop()  # Should not change state
        assert run.state == RunState.COMPLETED

    def test_cancel(self):
        run = TrainingRun()
        run.start()
        run.cancel("test reason")
        assert run.state == RunState.CANCELLED
        assert run.events()[-1].kind == "run_cancelled"

    def test_cancel_from_terminal_state_no_op(self):
        run = TrainingRun()
        run.start()
        run.complete()
        run.cancel()  # Should not change state
        assert run.state == RunState.COMPLETED

    def test_complete(self):
        run = TrainingRun()
        run.start()
        run.complete()
        assert run.state == RunState.COMPLETED
        assert run.events()[-1].kind == "run_completed"

    def test_complete_from_invalid_state_raises_error(self):
        run = TrainingRun()
        with pytest.raises(ValueError, match="cannot complete a created run"):
            run.complete()

    def test_fail(self):
        run = TrainingRun()
        run.start()
        error = Exception("test error")
        run.fail(error)
        assert run.state == RunState.FAILED
        assert run.events()[-1].kind == "run_failed"
        assert "test error" in run.events()[-1].data["error"]


class TestTrainingRunEvents:
    def test_event_filter_duration_and_json(self):
        run = TrainingRun(TrainingRunOptions(clock=iter([1.0, 2.0]).__next__))
        run.start()
        run.complete()
        assert len(run.events("run_started")) == 1
        assert run.duration == 1.0
        assert '"events"' in run.to_json()

    def test_subscriber_receives_events_and_can_unsubscribe(self):
        run = TrainingRun()
        received = []
        unsubscribe = run.subscribe(received.append)
        run.start()
        unsubscribe()
        run.complete()
        assert [event.kind for event in received] == ["run_started"]
        assert run.summary()["state"] == "completed"

    def test_events_returns_copy(self):
        run = TrainingRun()
        run.start()
        events = run.events()
        events.clear()

        assert len(run.events()) > 0

    def test_emit_in_created_auto_transitions(self):
        run = TrainingRun()
        run._emit({"kind": "custom_event"})
        assert run.state == RunState.RUNNING


class TestTrainingRunCheckpoints:
    @pytest.mark.asyncio
    async def test_options_without_store_use_default_memory_store(self):
        run = TrainingRun(TrainingRunOptions())
        checkpoint = await run.checkpoint({"step": 1})
        assert await run.latest_checkpoint() == checkpoint

    @pytest.mark.asyncio
    async def test_checkpoint(self):
        run = TrainingRun()
        checkpoint = await run.checkpoint({"model": "state"})

        assert checkpoint.id == "checkpoint-1"
        assert checkpoint.payload == {"model": "state"}
        assert run.events()[-1].kind == "checkpoint_saved"

    @pytest.mark.asyncio
    async def test_checkpoint_custom_id(self):
        run = TrainingRun()
        checkpoint = await run.checkpoint({"model": "state"}, "custom-id")

        assert checkpoint.id == "custom-id"

    @pytest.mark.asyncio
    async def test_latest_checkpoint(self):
        run = TrainingRun()
        await run.checkpoint({"step": 1})
        await run.checkpoint({"step": 2})

        latest = await run.latest_checkpoint()
        assert latest.payload == {"step": 2}

    @pytest.mark.asyncio
    async def test_latest_checkpoint_empty(self):
        run = TrainingRun()
        latest = await run.latest_checkpoint()
        assert latest is None


class TestTrainingRunExecution:
    @pytest.mark.asyncio
    async def test_execute_success(self):
        run = TrainingRun()
        backend = MockBackend()

        final_state = await run.execute(backend)
        assert final_state == RunState.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_with_backend_error(self):
        class FailingBackend(TrainingBackend):
            async def run(self, context: TrainingContext):
                yield {"kind": "epoch"}
                raise Exception("backend error")

        run = TrainingRun()
        backend = FailingBackend()

        with pytest.raises(Exception, match="backend error"):
            await run.execute(backend)

        assert run.state == RunState.FAILED

    @pytest.mark.asyncio
    async def test_execute_with_stop(self):
        run = TrainingRun()

        class SlowBackend(MockBackend):
            async def run(self, context):
                for event in self.events:
                    await asyncio.sleep(0.01)
                    if context.should_stop():
                        break
                    yield event

        backend = SlowBackend()

        async def delayed_stop():
            await asyncio.sleep(0.01)
            run.stop()

        asyncio.create_task(delayed_stop())
        final_state = await run.execute(backend)
        assert final_state == RunState.STOPPED

    @pytest.mark.asyncio
    async def test_execute_records_events(self):
        run = TrainingRun()
        backend = MockBackend()

        await run.execute(backend)

        event_kinds = [e.kind for e in run.events()]
        assert "run_started" in event_kinds
        assert "epoch" in event_kinds
        assert "metrics" in event_kinds
        assert "run_completed" in event_kinds


class TestMemoryCheckpointStore:
    @pytest.mark.asyncio
    async def test_save_and_retrieve(self):
        store = MemoryCheckpointStore()
        checkpoint = Checkpoint(id="test", created_at=1000, payload={"data": "value"})

        await store.save(checkpoint)
        retrieved = await store.get("test")

        assert retrieved == checkpoint

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        store = MemoryCheckpointStore()
        retrieved = await store.get("nonexistent")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_latest(self):
        store = MemoryCheckpointStore()
        await store.save(Checkpoint(id="1", created_at=1000, payload={}))
        await store.save(Checkpoint(id="2", created_at=2000, payload={}))

        latest = await store.latest()
        assert latest.id == "2"

    @pytest.mark.asyncio
    async def test_latest_empty(self):
        store = MemoryCheckpointStore()
        latest = await store.latest()
        assert latest is None


class TestTrainingContext:
    def test_training_context_creation(self):
        def emit(event):
            return RunEvent(kind=event["kind"], timestamp=0, data={})

        def should_stop():
            return False

        async def checkpoint(payload, id=None):
            return Checkpoint(id=id or "test", created_at=0, payload=payload)

        context = TrainingContext(emit, should_stop, checkpoint)
        assert callable(context.emit)
        assert callable(context.should_stop)
        assert callable(context.checkpoint)
