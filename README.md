# silver-run

<p align="center"><img src="https://raw.githubusercontent.com/adfgdartec/silver-run/main/docs/assets/silver-hero.png" alt="Silver local experiment tracking" width="100%"></p>

**Local experiment tracking that is transparent enough to debug with a text editor.**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/adfgdartec/silver-run/actions/workflows/ci.yml/badge.svg)](https://github.com/adfgdartec/silver-run/actions/workflows/ci.yml)
[![Code Style](https://img.shields.io/badge/code%20style-flake8-blue.svg)](https://flake8.pycqa.org/)

Backend-neutral ML run lifecycle, events, and checkpoints for Silver. A Python package designed for ML researchers who need flexible training orchestration across different frameworks.

## Installation

```bash
pip install silver-run
```

## Quick Start

### Durable tracking without a server

```python
from silver_run import (
    FileCheckpointStore, LocalRunStore, TrainingRun, TrainingRunOptions,
)

store = LocalRunStore(".silver/runs")
run = TrainingRun(TrainingRunOptions(
    metadata={"model": "tabular-v1", "dataset": "customers-2026"},
    run_store=store,
    checkpoint_store=FileCheckpointStore(".silver/checkpoints"),
))

run.start()
run.emit("epoch", {"epoch": 1, "metrics": {"loss": 0.42}})
run.complete()

restored = store.load(run.id)
print(restored.state, restored.metrics, restored.duration)
```

Run manifests are atomic JSON; events are append-only JSONL; checkpoint IDs are
path-safe; checkpoint payloads are explicit JSON instead of unsafe pickle.

```python
from silver_run import TrainingRun, TrainingBackend, TrainingContext
import asyncio

class MyBackend(TrainingBackend):
    async def run(self, context: TrainingContext):
        for epoch in range(10):
            if context.should_stop():
                break
            # Your training logic here
            context.emit({
                "kind": "epoch",
                "epoch": epoch,
                "metrics": {"loss": 0.5 - epoch * 0.05}
            })
            await asyncio.sleep(0.1)

async def main():
    run = TrainingRun()
    backend = MyBackend()
    final_state = await run.execute(backend)
    print(f"Run finished with state: {final_state.value}")

asyncio.run(main())
```

## Features

- **Training Lifecycle Management**: Full state machine (created, running, paused, stopped, cancelled, completed, failed)
- **Event Logging**: Comprehensive event tracking with timestamps for training observability
- **Checkpoint Management**: Pluggable storage backends for model checkpointing
- **Backend-Agnostic**: Works with PyTorch, TensorFlow, JAX, or any custom training framework
- **Async/Await Support**: Modern Python async patterns for concurrent training
- **Pause/Resume**: Control long-running training jobs with pause and resume functionality
- **Type Safety**: Full type hints for better IDE support and fewer bugs

## Use Cases

### PyTorch Training Integration

```python
from silver_run import TrainingRun, TrainingBackend, TrainingContext
import torch
import asyncio

class PyTorchBackend(TrainingBackend):
    def __init__(self, model, optimizer, train_loader):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
    
    async def run(self, context: TrainingContext):
        for epoch in range(10):
            if context.should_stop():
                break
            
            self.model.train()
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            # Emit epoch completion event
            context.emit({
                "kind": "epoch",
                "epoch": epoch,
                "metrics": {"loss": total_loss / len(self.train_loader)}
            })
            
            # Checkpoint every 5 epochs
            if epoch % 5 == 0:
                await context.checkpoint({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()
                })

async def main():
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters())
    train_loader = [...]  # Your data loader
    
    run = TrainingRun()
    backend = PyTorchBackend(model, optimizer, train_loader)
    final_state = await run.execute(backend)
    
    # Review events
    for event in run.events():
        print(f"{event.kind}: {event.data}")

asyncio.run(main())
```

### Training with Pause/Resume

```python
from silver_run import TrainingRun, TrainingBackend
import asyncio

class LongRunningBackend(TrainingBackend):
    async def run(self, context: TrainingContext):
        for step in range(1000):
            if context.should_stop():
                break
            
            # Simulate training step
            await asyncio.sleep(0.01)
            
            # Emit progress
            if step % 100 == 0:
                context.emit({
                    "kind": "progress",
                    "step": step,
                    "total": 1000
                })

async def main():
    run = TrainingRun()
    backend = LongRunningBackend()
    
    # Start training in background
    training_task = asyncio.create_task(run.execute(backend))
    
    # Pause after some time
    await asyncio.sleep(0.5)
    run.pause()
    print("Training paused")
    
    # Resume after some time
    await asyncio.sleep(0.5)
    run.resume()
    print("Training resumed")
    
    # Wait for completion
    final_state = await training_task
    print(f"Training finished: {final_state.value}")

asyncio.run(main())
```

### Custom Checkpoint Storage

```python
from silver_run import TrainingRun, CheckpointStore, Checkpoint
import asyncio

class S3CheckpointStore(CheckpointStore):
    def __init__(self, bucket, prefix):
        self.bucket = bucket
        self.prefix = prefix
        self.checkpoints = {}
    
    async def save(self, checkpoint: Checkpoint):
        # Save to S3
        key = f"{self.prefix}/{checkpoint.id}"
        print(f"Saving checkpoint to S3: {key}")
        self.checkpoints[checkpoint.id] = checkpoint
    
    async def latest(self):
        if not self.checkpoints:
            return None
        return list(self.checkpoints.values())[-1]
    
    async def get(self, id: str):
        return self.checkpoints.get(id)

async def main():
    store = S3CheckpointStore("my-bucket", "checkpoints")
    run = TrainingRun(options=TrainingRunOptions(checkpoint_store=store))
    
    # Use custom checkpoint store
    await run.checkpoint({"model": "state"}, "checkpoint-1")
    latest = await run.latest_checkpoint()
    print(f"Latest checkpoint: {latest.id}")

asyncio.run(main())
```

## Advanced Usage

### Event Filtering and Analysis

```python
from silver_run import TrainingRun

# Filter events by type
def get_epoch_events(run):
    return [e for e in run.events() if e.kind == "epoch"]

def get_error_events(run):
    return [e for e in run.events() if e.kind == "error"]

# Analyze training progression
def analyze_training(run):
    epoch_events = get_epoch_events(run)
    losses = [e.data.get("metrics", {}).get("loss") for e in epoch_events]
    
    if losses:
        print(f"Initial loss: {losses[0]}")
        print(f"Final loss: {losses[-1]}")
        print(f"Loss reduction: {losses[0] - losses[-1]}")
```

### Multi-Run Experiments

```python
from silver_run import TrainingRun
import asyncio

async def run_experiment(config):
    run = TrainingRun()
    backend = MyBackend(config)
    return await run.execute(backend)

async def main():
    configs = [
        {"learning_rate": 0.001},
        {"learning_rate": 0.01},
        {"learning_rate": 0.1}
    ]
    
    results = await asyncio.gather(*[
        run_experiment(config) for config in configs
    ])
    
    for config, result in zip(configs, results):
        print(f"LR {config['learning_rate']}: {result.value}")

asyncio.run(main())
```

## Requirements

- Python 3.10+

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=silver_run --cov-report=html

# Run linting
flake8 src/ tests/
mypy src/
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache-2.0 - see [LICENSE](LICENSE) file for details.

## Related Packages

- [silver-data](https://github.com/adfgdartec/silver-data) - Dataset handling
- [silver-diagnostics](https://github.com/adfgdartec/silver-diagnostics) - ML diagnostics
- [silver-adapters](https://github.com/adfgdartec/silver-adapters) - Framework adapters
