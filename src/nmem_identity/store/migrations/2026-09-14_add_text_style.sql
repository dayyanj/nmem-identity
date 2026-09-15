-- Migration: add the text_style (LUAR authorship / chat-writing-style) embedder columns.
--
-- The mirror-reverse of 2026-09-05_drop_tabula.sql: it ADDS the third embedder's self-describing
-- triple (centroid/emb, model_id, embed_version) to identity_cluster + identity_template, the triple
-- CHECKs, widens the "at least one embedder" CHECK, adds the partial HNSW index scoped to the
-- 'luar_v1' model_id literal, and widens the calibration channel CHECK to admit 'text_style'.
--
-- schema.sql is create-only (IF NOT EXISTS), so a FRESH DB built from the current schema.sql already
-- has all of this and is unaffected. Apply this once against a database that ran an EARLIER schema.sql
-- (voice+face only) to bring it in line:
--
--   psql "$NMEM_IDENTITY_DSN" -f src/nmem_identity/store/migrations/2026-09-14_add_text_style.sql
--
-- Idempotent (IF NOT EXISTS on columns/index; DROP-then-ADD on the CHECKs) and transactional. The new
-- columns are nullable and hold no data until the chat-text channel writes, so this is a safe,
-- online, additive change: voice and face rows are untouched.

BEGIN;

-- identity_cluster: add the text centroid triple, then its all-or-none CHECK.
ALTER TABLE identity_cluster
    ADD COLUMN IF NOT EXISTS centroid_text      vector(512),
    ADD COLUMN IF NOT EXISTS text_model_id      TEXT,
    ADD COLUMN IF NOT EXISTS text_embed_version INT;
ALTER TABLE identity_cluster DROP CONSTRAINT IF EXISTS ck_cluster_text;
ALTER TABLE identity_cluster
    ADD CONSTRAINT ck_cluster_text CHECK ((centroid_text IS NULL) = (text_model_id IS NULL) AND (text_model_id IS NULL) = (text_embed_version IS NULL));

-- identity_template: add the text triple + its CHECK, then WIDEN the "at least one embedder" CHECK to
-- admit a text-only observation (order matters: recreate ck_tmpl_any AFTER the column exists).
ALTER TABLE identity_template
    ADD COLUMN IF NOT EXISTS text_emb           vector(512),
    ADD COLUMN IF NOT EXISTS text_model_id      TEXT,
    ADD COLUMN IF NOT EXISTS text_embed_version INT;
ALTER TABLE identity_template DROP CONSTRAINT IF EXISTS ck_tmpl_text;
ALTER TABLE identity_template
    ADD CONSTRAINT ck_tmpl_text CHECK ((text_emb IS NULL) = (text_model_id IS NULL) AND (text_model_id IS NULL) = (text_embed_version IS NULL));
ALTER TABLE identity_template DROP CONSTRAINT IF EXISTS ck_tmpl_any;
ALTER TABLE identity_template
    ADD CONSTRAINT ck_tmpl_any CHECK (campp_emb IS NOT NULL OR face_emb IS NOT NULL OR text_emb IS NOT NULL);

-- identity_calibration: widen the channel CHECK to admit the text_style channel.
ALTER TABLE identity_calibration DROP CONSTRAINT IF EXISTS ck_calib_channel;
ALTER TABLE identity_calibration
    ADD CONSTRAINT ck_calib_channel CHECK (channel IN ('campp','context','face','text_style'));

-- Partial HNSW index over the text centroid, scoped to the 'luar_v1' model_id literal — must match
-- the KNN predicate store/templates.py inlines, or the index is silently skipped (seqscan).
CREATE INDEX IF NOT EXISTS idx_cluster_text ON identity_cluster USING hnsw (centroid_text vector_cosine_ops) WHERE text_model_id = 'luar_v1';

COMMIT;
