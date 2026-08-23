# Changelog

## 1.3.0 - 2026-08-23

- Coordinated ecosystem release for validated, decision-ready experiment artifacts.

## 1.2.0 - 2026-08-23

- Added documented replay/verification workflow for timeline and checkpoint evidence in decision plans.

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-08-22

### Added
- Added deterministic SVG run timelines generated from persisted lifecycle events,
  timestamps, and training/validation metric history.
- Added `StoredRun.to_svg()` and `LocalRunStore.visualize()` convenience APIs.

## [1.0.0] - 2026-08-22

### Added
- Added atomic, JSON-safe `FileCheckpointStore` persistence with path traversal
  protection and latest-checkpoint recovery.
- Added `LocalRunStore`, a dependency-free local experiment tracker with durable
  manifests, append-only JSONL events, replay, run discovery, and metric history.
- Added stable run IDs, metadata, public custom event emission, and richer
  serializable summaries.
- Added PEP 561 package markers and declared the stable public API.

## [0.4.0] - 2026-08-22

### Fixed
- Restored the default in-memory checkpoint store when an options object does not
  provide a custom store.
- Restored Python 3.8 import compatibility in the event-emission annotation.

### Added
- Hardened CI and releases with supported-Python testing and tag/version checks.

## [0.3.0] - 2026-08-15

### Changed
- Coordinated the package release with explicit model and GPU workflow examples.

## [0.2.0] - 2026-08-15

### Changed
- Added event subscriptions for live dashboards and a serializable run `summary()`.
- Added event filtering, duration tracking, and JSON run export.
- Added explicit GPU percentage/memory requests and balanced multi-GPU resource plans.

## [0.1.0] - 2024-08-04

### Added
- Initial release of silver-run
- Backend-neutral training lifecycle management
- Event logging and inspection for training runs
- Checkpoint management with pluggable storage backends
- Async/await support for modern Python training workflows
- In-memory checkpoint store for local development
- Full training state machine (created, running, paused, stopped, cancelled, completed, failed)
- Comprehensive test suite with async test coverage
- Support for Python 3.8-3.12

### Features
- `TrainingRun` - Main class for managing training lifecycle
- `TrainingBackend` - Abstract interface for training implementations
- `CheckpointStore` - Pluggable checkpoint storage interface
- `MemoryCheckpointStore` - In-memory checkpoint implementation
- `TrainingContext` - Context object passed to training backends
- State management (start, pause, resume, stop, cancel, complete, fail)
- Event emission and logging with timestamps
- Checkpoint creation and retrieval
- Backend execution with automatic state management
- Pause/resume functionality for long-running training

## [Unreleased]
