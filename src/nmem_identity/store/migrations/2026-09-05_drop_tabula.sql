-- Migration: drop the parked TABULA2 columns/constraints/index.
--
-- schema.sql is create-only (IF NOT EXISTS), so editing it does NOT alter a database that already ran
-- the original schema. Apply this once against such a database to bring it in line with the current
-- schema.sql. Safe: the TABULA2 embedder was dropped and never run, so these columns hold no data.
--
--   psql "$NMEM_IDENTITY_DSN" -f src/nmem_identity/store/migrations/2026-09-05_drop_tabula.sql
--
-- Idempotent (IF EXISTS everywhere) and transactional — a fresh DB created from the current schema.sql
-- (which no longer has these objects) is unaffected.

BEGIN;

-- Index first.
DROP INDEX IF EXISTS idx_cluster_tabula;

-- identity_cluster: drop the triple's CHECK, then its columns.
ALTER TABLE identity_cluster DROP CONSTRAINT IF EXISTS ck_cluster_tabula;
ALTER TABLE identity_cluster
    DROP COLUMN IF EXISTS centroid_tabula,
    DROP COLUMN IF EXISTS tabula_model_id,
    DROP COLUMN IF EXISTS tabula_embed_version;

-- identity_template: the "at least one embedder" CHECK references tabula_emb, so recreate it BEFORE
-- dropping the column; then drop the tabula triple CHECK and columns.
ALTER TABLE identity_template DROP CONSTRAINT IF EXISTS ck_tmpl_any;
ALTER TABLE identity_template
    ADD CONSTRAINT ck_tmpl_any CHECK (campp_emb IS NOT NULL OR face_emb IS NOT NULL);
ALTER TABLE identity_template DROP CONSTRAINT IF EXISTS ck_tmpl_tabula;
ALTER TABLE identity_template
    DROP COLUMN IF EXISTS tabula_emb,
    DROP COLUMN IF EXISTS tabula_model_id,
    DROP COLUMN IF EXISTS tabula_embed_version;

-- identity_calibration: tighten the channel CHECK (delete any tabula rows first — there should be none).
DELETE FROM identity_calibration WHERE channel = 'tabula';
ALTER TABLE identity_calibration DROP CONSTRAINT IF EXISTS ck_calib_channel;
ALTER TABLE identity_calibration
    ADD CONSTRAINT ck_calib_channel CHECK (channel IN ('campp','context','face'));

COMMIT;
