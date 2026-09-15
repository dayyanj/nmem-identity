"""Central configuration for nmem-identity.

Everything host-specific is funnelled through here so the package itself is **install- and
agent-agnostic**: it names no particular host agent, database, or machine. A deployment supplies its
own values via environment variables (or an optional TOML file); the defaults are generic localhost
placeholders suitable for a fresh install, not any one deployment.

Precedence (low -> high): built-in defaults  <  TOML file  <  environment variables.
The TOML file is looked up at ``$NMEM_IDENTITY_CONFIG`` or ``./nmem-identity.toml`` if present
(a flat table or an ``[nmem_identity]`` section). Each flag has ONE canonical name: the env var is
always ``NMEM_IDENTITY_ + FIELD.upper()`` (no aliases). Access via the ``settings`` singleton or the
thin ``get()``/``dsn()``/``blob_dir()`` accessors.

What is NOT configurable here, by design:
  * **Embedder model ids** (``campplus_onnx_v1``, ``arcface_buffalo_l``, ...) are schema-bound
    constants, not runtime knobs. A partial HNSW index is defined with the model_id as a *literal*
    in the DDL, and every KNN predicate must inline the *same* literal for the index to be used
    (see store/templates.py ``_model_literal``). A model swap is therefore a migration (register a
    new id, re-embed, re-index), never a config flip. Model ids live in store/templates.py.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


def _toml_data() -> dict[str, Any]:
    path = os.environ.get("NMEM_IDENTITY_CONFIG")
    p = Path(path) if path else Path.cwd() / "nmem-identity.toml"
    if not p.is_file():
        return {}
    with p.open("rb") as fh:
        data = tomllib.load(fh)
    # Accept either a flat table or an [nmem_identity] section.
    return data.get("nmem_identity", data)


class _TomlSource(PydanticBaseSettingsSource):
    """Settings source for the optional nmem-identity.toml (sits below env, above defaults)."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._data = _toml_data()

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:  # noqa: ARG002
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        fields = self.settings_cls.model_fields
        return {k: v for k, v in self._data.items() if k in fields}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NMEM_IDENTITY_", extra="ignore")

    # ── Store ──
    dsn: str = Field(
        "postgresql://nmem:nmem@localhost:5432/nmem_identity",
        description="asyncpg DSN (plain postgres:// scheme, NOT postgresql+asyncpg://). Override per deployment.")
    blob_dir: str = Field(
        "~/.local/share/nmem-identity/blobs",
        description="Blob store for raw crops / audio snippets (deletable; re-embedded on a model swap).")

    # ── Service ports ──
    matcher_port: int = Field(9404, description="Live matcher service port.")
    face_embed_port: int = Field(9405, description="Face embedding service port.")
    voice_embed_port: int = Field(9402, description="Voice embedding service port.")
    text_embed_port: int = Field(9406, description="Text-style (LUAR authorship) embedding service port.")

    # ── Face model pack (schema-bound: a change is a model-swap migration, not a runtime knob) ──
    face_model_pack: str = Field(
        "buffalo_s",
        description="Face model pack (LIGHT pack shares one GPU). Must match the store's face model_id + index.")
    face_det_size: int = Field(
        640, description="SCRFD detector input size (square). 640 accurate; 320 ~4x cheaper.")

    # ── Text-style embedder (LUAR authorship). The store model_id ('luar_v1') is schema-bound; the HF
    #    CHECKPOINT NAME + episode shape below are runtime knobs for the embed sidecar. A checkpoint or
    #    preprocessing change that moves cosine distances is a re-embed migration (bump embed_version /
    #    model_id), not just an env swap. ──
    text_model_name: str = Field(
        "rrivera1849/LUAR-MUD",
        description="HuggingFace LUAR checkpoint for the author embedder (Apache-2.0; MUD = Reddit-trained).")
    text_episode_len: int = Field(
        16, description="Max short documents (chat turns) aggregated into one LUAR author episode.")
    text_doc_max_tokens: int = Field(
        32, description="Max tokens per document in a LUAR episode (LUAR's native short-doc window).")

    # ── Live matcher policy — TEXT-STYLE (LUAR) channel; quality = 0..1 authored-text sufficiency.
    #    UNCALIBRATED placeholders: text is a WEAK vote and these are conservative until
    #    eval/text_benchmark.py fits them on a length-stratified EER curve. Do NOT reuse voice/face. ──
    text_tau_match: float = Field(
        0.30, description="Text-style match cosine threshold (PLACEHOLDER — calibrate via eval).")
    text_margin: float = Field(
        0.12, description="Text-style match margin (PLACEHOLDER — calibrate via eval).")
    text_min_quality: float = Field(
        0.50, description="Text-style minimum quality (0..1 authored-text sufficiency).")

    # ── Live matcher policy — VOICE (CAM++) channel; quality = utterance seconds ──
    tau_match: float = Field(0.50, description="Voice match cosine threshold.")
    margin: float = Field(0.10, description="Voice match margin.")
    min_quality: float = Field(1.5, description="Voice minimum quality (utterance duration, seconds).")

    # ── Live matcher policy — FACE (ArcFace) channel; quality = 0..1 score (do NOT reuse voice values) ──
    face_tau_match: float = Field(
        0.30, description="Face match cosine threshold (ArcFace cosines run lower than CAM++).")
    face_margin: float = Field(0.10, description="Face match margin.")
    face_min_quality: float = Field(0.45, description="Face minimum quality (0..1 score).")

    # ── Fusion policy (Phase 5) — how the /resolve endpoint combines channel votes + context priors.
    #    Log-odds thresholds (posterior = sigmoid): a deployment tunes these without touching fuse.py.
    #    text is a STRUCTURALLY weak vote — its LLR is clamped to fusion_text_llr_cap AND it can never
    #    reach the confident grounding without non-weak corroboration (enforced in fuse.py, not here). ──
    fusion_text_llr_cap: float = Field(
        1.5, description="Max |LLR| a weak (text_style) channel contributes to fusion (nats).")
    fusion_speculative_logodds: float = Field(
        0.5, description="Fused log-odds for a 'speculative' identity (posterior ~0.62).")
    fusion_confident_logodds: float = Field(
        2.2, description="Fused log-odds for a 'confident' identity (posterior ~0.90); also needs corroboration.")
    fusion_margin_logodds: float = Field(
        1.0, description="Min fused log-odds the winner must beat the runner-up by, else held at speculative.")
    resolve_k: int = Field(
        3, description="KNN breadth per channel when /resolve gathers fusion candidates.")

    # ── Misc ──
    ask_recurrence: int = Field(
        3, description="An unbound cluster seen/heard this many times is worth asking who it is.")
    viz_url: str | None = Field(None, description="Optional live-viz sink (no-op unless set).")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority high→low: init kwargs > env > TOML > field defaults.
        return (init_settings, env_settings, _TomlSource(settings_cls))


settings = Settings()


# ── Thin accessors over the singleton (single source of truth; NOT env aliases) ──
def get(key: str) -> Any:
    """Return a single resolved config value (defaults < TOML < env)."""
    return getattr(settings, key)


def dsn() -> str:
    return settings.dsn


def blob_dir() -> Path:
    return Path(settings.blob_dir).expanduser()


def all_values() -> dict[str, Any]:
    """A copy of every resolved value (for --show / diagnostics)."""
    return settings.model_dump()
