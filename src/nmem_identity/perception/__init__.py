"""Perception embedders — turn raw signal into identity vectors.

Kept import-light on purpose: importing `nmem_identity.perception` must NOT pull in heavy ML deps
(insightface / onnxruntime / opencv). The store + matcher path imports none of this. Import the
concrete embedder (`from nmem_identity.perception.face import FaceEmbedder`) only where you actually
run a model — those modules lazy-import their ML deps inside the class so a store-only install stays
dependency-free.
"""
