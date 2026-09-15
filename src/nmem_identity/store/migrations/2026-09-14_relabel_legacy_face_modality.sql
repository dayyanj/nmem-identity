-- Migration: relabel legacy face-only clusters/templates from modality='voice' to 'face'.
--
-- WHY: before the matcher passed an explicit `modality` per embedder, cluster.match_or_create
-- defaulted modality='voice' for EVERY embedder — so any face observation made through the old
-- /match created a cluster (and templates) labelled 'voice'. Now that /match passes modality='face'
-- for face observations, knn_clusters(modality='face') filters those legacy rows OUT: previously
-- enrolled faces would read as unknown and spawn duplicate clusters (codex). This relabels them.
--
-- SAFETY: only a row that is PURELY face (has a face centroid/emb and NO campp one) is touched, so a
-- genuine voice or multi cluster is never mislabelled. Person bindings live on the cluster row and are
-- untouched. Idempotent (re-running matches nothing) and transactional. On the common case — a DB with
-- no face data yet (the face channel goes live once a camera exists) — this is a zero-row no-op.
--
--   psql "$NMEM_IDENTITY_DSN" -f src/nmem_identity/store/migrations/2026-09-14_relabel_legacy_face_modality.sql

BEGIN;

UPDATE identity_cluster
    SET modality = 'face'
    WHERE modality = 'voice' AND centroid_face IS NOT NULL AND centroid_campp IS NULL;

UPDATE identity_template
    SET modality = 'face'
    WHERE modality = 'voice' AND face_emb IS NOT NULL AND campp_emb IS NULL;

COMMIT;
