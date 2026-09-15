-- nmem-identity store schema — voice-first, face-ready, multi-embedder, model-versioned.
-- Design: nmem repo docs/design/multi-modal-identity{,-plan}.md.
--
-- Home: any pgvector-enabled Postgres (see config.py / NMEM_IDENTITY_DSN). Colocating it in the same
-- database as an nmem deployment makes the matcher's id-join to nmem's `nmem_entity_memory` a plain
-- local join, but that is a deployment choice. Self-contained `identity_*` tables; never touches
-- nmem's 384-locked SQLAlchemy models.
--
-- INVARIANTS (do not violate):
--  * Each embedder's vector is a SELF-DESCRIBING triple: (vector, <e>_model_id, <e>_embed_version).
--    There is NO single ambiguous `model_id` per row — CAM++ voice (192-d) and ArcFace face (512-d)
--    live in their own columns, each carrying its own model_id + version. A preprocessing change bumps
--    that embedder's embed_version (and/or a new model_id); every KNN filters on the matching
--    *_model_id. Cross-model cosine is thus structurally impossible.
--  * Model ids: 'campplus_onnx_v1' (CAM++ voice, 192-d), 'arcface_buffalo_s' (face, light pack, 512-d),
--    'luar_v1' (LUAR authorship / chat-writing style, 512-d — a SEPARATE column from face, never
--    cosine-comparable to it).
--  * Face + text-style columns present now; populated once a camera / chat-text channel is wired.
--  * Deletability is enforced by the application `delete_person(person_id)`, NOT by DB cascade alone
--    (a person also owns raw-PCM blobs + nmem entity rows + curiosity rows that only app code can
--    clear). `delete_person` deletes the person's clusters (which CASCADE their templates), deletes
--    the raw blobs, and deletes/nulls the nmem entity + curiosity rows. The cluster->person FK is
--    ON DELETE SET NULL only so a raw `DELETE FROM identity_person` degrades a cluster to unbound
--    rather than corrupting it — it is NOT the deletion path.

CREATE EXTENSION IF NOT EXISTS vector;

-- Named, enrolled identities. Linked to nmem entity memory by id (Option-B).
CREATE TABLE IF NOT EXISTS identity_person (
    person_id       BIGSERIAL PRIMARY KEY,
    display_name    TEXT,
    nmem_entity_id  TEXT,                       -- -> nmem_entity_memory (entity_type='person', entity_id); cross-schema, not an enforced FK
    status          TEXT NOT NULL DEFAULT 'active',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- A voice/appearance identity, named or not. One cluster == one speaker; it carries a per-embedder
-- centroid (each a self-describing triple), so the cluster itself is model-agnostic.
CREATE TABLE IF NOT EXISTS identity_cluster (
    cluster_id             BIGSERIAL PRIMARY KEY,
    modality               TEXT NOT NULL,                       -- 'voice' now; 'face' | 'multi' ready
    person_id              BIGINT REFERENCES identity_person(person_id) ON DELETE SET NULL,
    grounding_status       TEXT NOT NULL DEFAULT 'ungrounded',  -- ungrounded | speculative | confident | disputed
    -- CAM++ centroid triple
    centroid_campp         vector(192),
    campp_model_id         TEXT,
    campp_embed_version    INT,
    -- Face centroid triple (READY; null until a camera exists)
    centroid_face          vector(512),
    face_model_id          TEXT,
    face_embed_version     INT,
    -- Text-style (LUAR authorship) centroid triple (READY; null until the chat-text channel is wired)
    centroid_text          vector(512),
    text_model_id          TEXT,
    text_embed_version     INT,
    member_count           INT  NOT NULL DEFAULT 0,
    coherence              REAL NOT NULL DEFAULT 0,
    coherence_at_formation REAL,
    centroid_snapshot      TEXT,                                -- rechallenge drift baseline
    grounded_at            TIMESTAMPTZ,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_cluster_status CHECK (grounding_status IN ('ungrounded','speculative','confident','disputed')),
    -- vector present <=> its model_id + embed_version present (self-describing triples)
    CONSTRAINT ck_cluster_campp  CHECK ((centroid_campp  IS NULL) = (campp_model_id  IS NULL) AND (campp_model_id  IS NULL) = (campp_embed_version  IS NULL)),
    CONSTRAINT ck_cluster_face   CHECK ((centroid_face   IS NULL) = (face_model_id   IS NULL) AND (face_model_id   IS NULL) = (face_embed_version   IS NULL)),
    CONSTRAINT ck_cluster_text   CHECK ((centroid_text   IS NULL) = (text_model_id   IS NULL) AND (text_model_id   IS NULL) = (text_embed_version   IS NULL))
);

-- One observation (one turn / one detection). Each fired embedder writes its own self-describing
-- triple on the row (linked to its raw_ref blob). Templates CASCADE with their cluster.
CREATE TABLE IF NOT EXISTS identity_template (
    id                   BIGSERIAL PRIMARY KEY,
    cluster_id           BIGINT NOT NULL REFERENCES identity_cluster(cluster_id) ON DELETE CASCADE,
    modality             TEXT NOT NULL,
    campp_emb            vector(192),
    campp_model_id       TEXT,
    campp_embed_version  INT,
    face_emb             vector(512),
    face_model_id        TEXT,
    face_embed_version   INT,
    text_emb             vector(512),
    text_model_id        TEXT,
    text_embed_version   INT,
    quality              REAL,                  -- utterance duration / SNR / VAD-frame count -> gates centroid update
    raw_ref              TEXT,                  -- blob-store key for the raw PCM (deletable; re-embed on model swap)
    session_id           TEXT,
    captured_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    regime               TEXT,                  -- eval label: clean | noisy | short | crosstalk
    CONSTRAINT ck_tmpl_campp  CHECK ((campp_emb  IS NULL) = (campp_model_id  IS NULL) AND (campp_model_id  IS NULL) = (campp_embed_version  IS NULL)),
    CONSTRAINT ck_tmpl_face   CHECK ((face_emb   IS NULL) = (face_model_id   IS NULL) AND (face_model_id   IS NULL) = (face_embed_version   IS NULL)),
    CONSTRAINT ck_tmpl_text   CHECK ((text_emb   IS NULL) = (text_model_id   IS NULL) AND (text_model_id   IS NULL) = (text_embed_version   IS NULL)),
    CONSTRAINT ck_tmpl_any    CHECK (campp_emb IS NOT NULL OR face_emb IS NOT NULL OR text_emb IS NOT NULL)
);

-- Persisted per-channel cosine->LLR calibration curves. Versioned by (model_id, embed_version) so a
-- preprocessing change re-fits rather than silently reusing a stale curve.
CREATE TABLE IF NOT EXISTS identity_calibration (
    model_id      TEXT NOT NULL,
    embed_version INT  NOT NULL,
    channel       TEXT NOT NULL,               -- 'campp' | 'context' | 'face' | 'text_style'
    curve         JSONB NOT NULL,
    n_genuine     INT,
    n_impostor    INT,
    fitted_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_id, embed_version, channel),
    CONSTRAINT ck_calib_channel CHECK (channel IN ('campp','context','face','text_style'))
);

-- Context priors (§2.6): temporal arrival histogram + (person, feature) co-occurrence counts.
CREATE TABLE IF NOT EXISTS identity_prior_arrival (
    person_id   BIGINT NOT NULL REFERENCES identity_person(person_id) ON DELETE CASCADE,
    weekday     INT NOT NULL,
    hour_bucket INT NOT NULL,
    weight      REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (person_id, weekday, hour_bucket)
);
CREATE TABLE IF NOT EXISTS identity_prior_cooc (
    person_id   BIGINT NOT NULL REFERENCES identity_person(person_id) ON DELETE CASCADE,
    feature     TEXT NOT NULL,
    count       REAL NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (person_id, feature)
);

-- Per-embedder ANN indexes over cluster centroids, PARTIAL on that embedder's model_id so a KNN can
-- only ever compare centroids from the same model generation. (The live matcher KNNs centroids, not
-- templates; the benchmark reads templates in bulk offline, so no template ANN index is needed.)
CREATE INDEX IF NOT EXISTS idx_cluster_campp  ON identity_cluster USING hnsw (centroid_campp  vector_cosine_ops) WHERE campp_model_id  = 'campplus_onnx_v1';
-- Face channel: partial HNSW on the ArcFace centroid, scoped to its model_id literal (must match the
-- KNN predicate inlined by store/templates.py _model_literal, else the index is silently skipped).
-- Default is the LIGHT pack (buffalo_s) since the face channel shares a GPU with the rest of the host
-- stack; swapping to a heavier pack is a migration that adds a new index for the new model_id literal.
CREATE INDEX IF NOT EXISTS idx_cluster_face   ON identity_cluster USING hnsw (centroid_face   vector_cosine_ops) WHERE face_model_id   = 'arcface_buffalo_s';
-- Text-style channel: partial HNSW on the LUAR centroid, scoped to its model_id literal 'luar_v1'
-- (must match the KNN predicate inlined by store/templates.py _model_literal, else the index is
-- silently skipped). A LUAR weights/preprocessing change is a migration that adds a new index for the
-- new model_id literal + re-embeds — never a config flip.
CREATE INDEX IF NOT EXISTS idx_cluster_text   ON identity_cluster USING hnsw (centroid_text   vector_cosine_ops) WHERE text_model_id   = 'luar_v1';
CREATE INDEX IF NOT EXISTS idx_cluster_person ON identity_cluster (person_id);
CREATE INDEX IF NOT EXISTS idx_tmpl_cluster   ON identity_template (cluster_id);
CREATE INDEX IF NOT EXISTS idx_tmpl_session   ON identity_template (session_id);
