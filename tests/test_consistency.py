"""Guard the versioning invariant: the schema's partial-index model_id literals MUST equal the
model_ids the store inlines into KNN predicates. If these drift, the ANN index is silently skipped
(seqscan) or — worse — a swap compares across generations. Cheap to check, catastrophic to miss."""
import pathlib

from nmem_identity import models
from nmem_identity.store import templates as T

_SCHEMA = pathlib.Path(__file__).resolve().parents[1] / "src/nmem_identity/store/schema.sql"


def test_partial_index_literals_match_active_model_ids():
    sql = _SCHEMA.read_text()
    for emb in ("campp", "face", "text_style"):
        mid = models.model_id(emb)
        assert f"= '{mid}'" in sql, f"schema has no partial-index literal for {emb} (={mid})"
        assert T._model_literal(emb) == f"'{mid}'"   # the exact literal the KNN inlines


def test_face_pack_maps_to_active_face_model_id():
    assert models.FACE_PACK_MODEL_ID[models.DEFAULT_FACE_PACK] == models.model_id("face")


def test_embedder_dims_are_consistent():
    assert models.dim("campp") == 192 and models.dim("face") == 512
    assert T._EMBEDDERS["campp"]["dim"] == 192 and T._EMBEDDERS["face"]["dim"] == 512
    # text_style (LUAR) is 512-d like face, but a SEPARATE column + model_id (never cosine-comparable).
    assert models.dim("text_style") == 512 and models.model_id("text_style") == "luar_v1"
    assert T._EMBEDDERS["text_style"]["dim"] == 512
    assert T._COLUMNS["text_style"]["centroid"] == "centroid_text"
    assert T._MODALITY_EMBEDDERS["text_style"] == ("text_style",)


def test_text_style_triple_columns_present_in_schema():
    sql = _SCHEMA.read_text()
    # cluster + template triples, both CHECKs widened, calibration channel admitted.
    for col in ("centroid_text", "text_emb", "text_model_id", "text_embed_version"):
        assert col in sql, f"schema missing text_style column {col}"
    assert "ck_cluster_text" in sql and "ck_tmpl_text" in sql
    assert "text_emb IS NOT NULL" in sql               # ck_tmpl_any widened
    assert "'text_style'" in sql                        # ck_calib_channel widened


def test_tabula_is_fully_gone_from_registry_and_schema():
    assert "tabula" not in models.EMBEDDERS and "tabula" not in T._EMBEDDERS
    assert "tabula" not in _SCHEMA.read_text()
