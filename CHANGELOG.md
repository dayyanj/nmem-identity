# Changelog

All notable changes to nmem-identity are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — 2026-09-08

**Config unified; wired into a live agent.** Still pre-1.0 — the API may change.

### Added

- Configuration guide + a model-introspecting reference generator.

### Changed

- Migrated to **pydantic-settings** (`NMEM_IDENTITY_*`): typed config, no ad-hoc env reads.

## [0.1.0] — initial

Self-supervised multi-modal (voice + face) person identity as an nmem extension: the
matcher service + identity store (`identity_*` pgvector tables), install- and
agent-agnostic (no runtime dependency on nmem-core). Phase 0 (voice) wired into a live
agent.
