"""Text-style embedder — the parts that DON'T need torch/transformers (import safety, doc splitting,
checkpoint→model_id self-labelling, template shape). The actual LUAR forward is covered by the
length-stratified eval benchmark, not a unit test."""
import numpy as np
import pytest

from nmem_identity import models
from nmem_identity.perception import text_style as ts


def test_module_imports_without_ml_deps():
    # Mirrors face.py: the module imports in a store-only install; ML deps are lazy in the class.
    assert ts.EMBED_DIM == 512
    assert ts.TEXT_MODEL_ID == models.model_id("text_style") == "luar_v1"


def test_default_checkpoint_maps_to_store_model_id():
    # The default MUD checkpoint is the one the store's partial index (luar_v1) is built on.
    assert ts.LUAR_MODEL_ID[ts.DEFAULT_TEXT_MODEL] == models.model_id("text_style")


def test_unknown_checkpoint_self_labels_safely():
    # An unknown checkpoint gets a derived [a-z0-9_]+ id (store's _SAFE_MODEL_ID guard) that DIFFERS
    # from luar_v1 — so the sidecar's desync guard rejects it rather than mixing generations.
    suffix = ts._safe_suffix("some-org/My.Weird_Checkpoint-v2")
    assert suffix == "my_weird_checkpoint_v2"
    import re
    assert re.fullmatch(r"[a-z0-9_]+", "luar_" + suffix)


def test_split_documents():
    # Sentence + newline chunking for good LUAR episode coverage (a multi-sentence paragraph becomes
    # several short documents, not one 32-token doc). Formatting-based paste detection does NOT rely on
    # this split — the sidecar scores contamination on the original input.
    docs = ts.split_documents("First one. Second one!\nThird line?  ")
    assert docs == ["First one.", "Second one!", "Third line?"]
    assert ts.split_documents("no boundary here") == ["no boundary here"]
    assert ts.split_documents("   ") == []


def test_authored_run_sentence_splits_for_embedding():
    # codex: a fully-typed multi-sentence message coalesces to ONE authored run, then sentence-splits
    # into multiple episode documents (like the text path) — not one 32-token doc. Pure composition of
    # text_spans.authored_documents + text_style.split_documents (the observe_spans / sidecar pipeline).
    from nmem_identity.perception import text_spans as sp
    spans = sp.parse_spans([{"text": "First sentence. Second sentence! Third?", "source": "typed"}])
    runs = sp.authored_documents(spans)
    assert runs == ["First sentence. Second sentence! Third?"]          # coalesced to one run
    docs = [d for run in runs for d in ts.split_documents(run)]
    assert docs == ["First sentence.", "Second sentence!", "Third?"]    # split for embedding


def test_observation_template_shape():
    vec = np.ones(512, dtype=np.float32)
    o = ts.TextObservation(embedding=vec, n_documents=3, authored_chars=140, quality=0.7)
    t = o.as_template()
    assert t["modality"] == "text_style"
    assert list(t["embeddings"].keys()) == ["text_style"]
    assert len(t["embeddings"]["text_style"]) == 512
    assert t["embed_versions"] == {"text_style": 1}


def test_l2norm_unit_and_zero_safe():
    v = ts._l2norm(np.array([3.0, 4.0], dtype=np.float32))
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-6
    z = ts._l2norm(np.zeros(4, dtype=np.float32))
    assert float(np.linalg.norm(z)) == 0.0            # zero vector stays zero (no div-by-zero)


def test_constructing_embedder_without_ml_deps_raises_clearly():
    # Without the 'text' extra, construction must raise a clear install hint, not a bare ImportError.
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        with pytest.raises(RuntimeError, match="text.*extra"):
            ts.TextStyleEmbedder()
