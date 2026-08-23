# Replay an experiment as a visual timeline

`silver-run` renders persisted run evidence rather than a live-only dashboard.

```python
stored = store.load(run.id)
open("run-timeline.svg", "w", encoding="utf-8").write(stored.to_svg())

# Or render directly from the store:
store.visualize(run.id, "run-timeline.svg")
```

The SVG lays lifecycle events on their real timestamps and plots loss and
validation loss from recorded epoch events. A failed run stays visibly failed;
missing metrics produce an empty series rather than fabricated points. The
source remains the append-only JSONL journal and atomic run manifest.
