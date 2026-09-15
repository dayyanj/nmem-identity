# Configuration — nmem-identity

nmem-identity is an **install- and agent-agnostic** package: it names no particular host,
database, or machine. A deployment supplies its own values; the defaults are generic
localhost placeholders for a fresh install. Every runtime knob is below, plus recipes for
common setups.

## How configuration works

All settings live on one `pydantic-settings` model, `Settings`, in
[src/nmem_identity/config.py](../src/nmem_identity/config.py), exposed as a singleton (with a
few thin accessors for convenience):

```python
from nmem_identity import config
dsn = config.settings.dsn          # or config.dsn()
blob = config.blob_dir()           # Path, ~ expanded
config.get("face_tau_match")       # single value by name
config.all_values()                # dict of every resolved value (for --show / diagnostics)
```

- **One canonical name per flag.** The env var is always `NMEM_IDENTITY_` + the field name
  upper-cased — e.g. `face_tau_match` ⇄ `NMEM_IDENTITY_FACE_TAU_MATCH`. There are **no
  aliases**; the accessors above read the singleton, they are not alternate env names.
- **Precedence (low → high): built-in defaults  <  TOML file  <  environment variables.**
  This is the one repo in the family with a config file layer.
- **The TOML file** is looked up at `$NMEM_IDENTITY_CONFIG`, or `./nmem-identity.toml` in the
  working directory if that env var is unset. It may be a flat table or an `[nmem_identity]`
  section. Keys are the canonical field names (lowercase). Unknown keys are ignored.
- **Everything is optional.** Every field has a default, so a fresh install runs unconfigured
  (against localhost placeholders).
- **Booleans** (none today, but the rule holds for future flags) accept `1/0`, `true/false`,
  `yes/no`, `on/off`; invalid values raise at startup.
- **Not configurable here, by design:** embedder **model ids** (`campplus_onnx_v1`,
  `arcface_buffalo_l`, …) are schema-bound constants in `store/templates.py`, not runtime
  knobs — a partial HNSW index inlines the model_id as a literal, so a model swap is a
  *migration* (register a new id, re-embed, re-index), never a config flip.
- **`FACE_MODEL_PACK` is a special case:** it *is* an env field (so the sidecar can select the
  pack), but it must match the store's active face `model_id` + index. Changing it is a model
  swap/migration, not a live tuning knob (the embed sidecar fails fast on a mismatch).

## Recipes

### Point identity at your database

```bash
# plain postgres:// scheme — NOT postgresql+asyncpg://
export NMEM_IDENTITY_DSN="postgresql://user:pass@host:5432/nmem_identity"
export NMEM_IDENTITY_BLOB_DIR="/var/lib/nmem-identity/blobs"
```

### Configure via a TOML file instead of env vars

Write `nmem-identity.toml` (or point `NMEM_IDENTITY_CONFIG` at it). Env still wins over TOML,
so you can commit a base file and override per-host with env:

```toml
# nmem-identity.toml  (flat table, or wrap these in an [nmem_identity] section)
dsn = "postgresql://user:pass@host:5432/nmem_identity"
matcher_port = 9404
face_model_pack = "buffalo_s"
tau_match = 0.50
```

```bash
export NMEM_IDENTITY_CONFIG=/etc/nmem-identity/prod.toml   # else ./nmem-identity.toml
```

### Set the service ports

The three sidecars each have a port; set them if the defaults collide with other services:

```bash
export NMEM_IDENTITY_MATCHER_PORT=9404
export NMEM_IDENTITY_VOICE_EMBED_PORT=9402
export NMEM_IDENTITY_FACE_EMBED_PORT=9405
```

### Tune the VOICE (CAM++) matcher policy

Voice quality is **utterance duration in seconds**. Loosen to accept shorter/less-certain
utterances; tighten to reduce false accepts:

```bash
export NMEM_IDENTITY_TAU_MATCH=0.50     # cosine threshold
export NMEM_IDENTITY_MARGIN=0.10        # runner-up margin
export NMEM_IDENTITY_MIN_QUALITY=1.5    # seconds of speech required
```

### Tune the FACE (ArcFace) matcher policy

> ArcFace cosines run **lower** than CAM++ — do **not** reuse the voice thresholds here. Face
> quality is a `0..1` score, not seconds.

```bash
export NMEM_IDENTITY_FACE_TAU_MATCH=0.30   # face cosine threshold (lower than voice)
export NMEM_IDENTITY_FACE_MARGIN=0.10
export NMEM_IDENTITY_FACE_MIN_QUALITY=0.45 # 0..1 quality score
```

### Share one GPU: run the light face pack / cheaper detection

The light pack (`buffalo_s`) is the default so the face channel shares a single GPU with the
rest of the host stack. Detection size is the dominant per-frame cost:

```bash
export NMEM_IDENTITY_FACE_DET_SIZE=320   # ~4x cheaper than 640; fine for close-range single subject
# a heavier pack is a migration, not a flip — see the model-id note above
```

### Set how quickly the agent asks "who is this?"

```bash
export NMEM_IDENTITY_ASK_RECURRENCE=3    # an unbound cluster seen/heard 3x is worth asking about
```

### Send live events to a viz sink

```bash
export NMEM_IDENTITY_VIZ_URL="ws://viz-host:5174/ingest"   # no-op unless set
```

## Complete environment variable reference

<!-- 15 env fields — generated by scripts/gen_config_reference.py -->

### Store

| Env var | Default | Type | Purpose |
|---|---|---|---|
| `NMEM_IDENTITY_DSN` | `postgresql://nmem:nmem@localhost:5432/nmem_identity` | str | asyncpg DSN (plain postgres:// scheme, NOT postgresql+asyncpg://). Override per deployment. |
| `NMEM_IDENTITY_BLOB_DIR` | `~/.local/share/nmem-identity/blobs` | str | Blob store for raw crops / audio snippets (deletable; re-embedded on a model swap). |

### Service ports

| Env var | Default | Type | Purpose |
|---|---|---|---|
| `NMEM_IDENTITY_MATCHER_PORT` | `9404` | int | Live matcher service port. |
| `NMEM_IDENTITY_FACE_EMBED_PORT` | `9405` | int | Face embedding service port. |
| `NMEM_IDENTITY_VOICE_EMBED_PORT` | `9402` | int | Voice embedding service port. |

### Face model pack (schema-bound: a change is a model-swap migration, not a runtime knob)

| Env var | Default | Type | Purpose |
|---|---|---|---|
| `NMEM_IDENTITY_FACE_MODEL_PACK` | `buffalo_s` | str | Face model pack (LIGHT pack shares one GPU). Must match the store's face model_id + index. |
| `NMEM_IDENTITY_FACE_DET_SIZE` | `640` | int | SCRFD detector input size (square). 640 accurate; 320 ~4x cheaper. |

### Live matcher policy — VOICE (CAM++) channel; quality = utterance seconds

| Env var | Default | Type | Purpose |
|---|---|---|---|
| `NMEM_IDENTITY_TAU_MATCH` | `0.5` | float | Voice match cosine threshold. |
| `NMEM_IDENTITY_MARGIN` | `0.1` | float | Voice match margin. |
| `NMEM_IDENTITY_MIN_QUALITY` | `1.5` | float | Voice minimum quality (utterance duration, seconds). |

### Live matcher policy — FACE (ArcFace) channel; quality = 0..1 score (do NOT reuse voice values)

| Env var | Default | Type | Purpose |
|---|---|---|---|
| `NMEM_IDENTITY_FACE_TAU_MATCH` | `0.3` | float | Face match cosine threshold (ArcFace cosines run lower than CAM++). |
| `NMEM_IDENTITY_FACE_MARGIN` | `0.1` | float | Face match margin. |
| `NMEM_IDENTITY_FACE_MIN_QUALITY` | `0.45` | float | Face minimum quality (0..1 score). |

### Misc

| Env var | Default | Type | Purpose |
|---|---|---|---|
| `NMEM_IDENTITY_ASK_RECURRENCE` | `3` | int | An unbound cluster seen/heard this many times is worth asking who it is. |
| `NMEM_IDENTITY_VIZ_URL` | `(unset)` | str | Optional live-viz sink (no-op unless set). |

