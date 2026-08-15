# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
