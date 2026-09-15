"""Authored-vs-pasted span contract (Phase 4) — the guarantee that relayed text never enters the
author embedding. Pure, no ML deps."""
from nmem_identity.perception import text_spans as sp


def test_parse_spans_tolerant():
    spans = sp.parse_spans([
        {"text": "hi there", "source": "typed"},
        {"text": "PASTED BLOCK", "source": "Pasted"},     # case-insensitive source
        {"text": "", "source": "typed"},                   # empty text dropped
        {"source": "typed"},                               # no text dropped
        "not a dict",                                      # non-dict dropped
        {"text": "no source given"},                       # missing source → unknown (pasted)
    ])
    assert [(s.text, s.source) for s in spans] == [
        ("hi there", "typed"), ("PASTED BLOCK", "pasted"), ("no source given", "unknown")]
    assert sp.parse_spans(None) == [] and sp.parse_spans("nope") == []


def test_authored_documents_excludes_pasted_and_unknown():
    spans = sp.parse_spans([
        {"text": "  my own words  ", "source": "typed"},
        {"text": "relayed from the CEO", "source": "pasted"},
        {"text": "dictated bit", "source": "dictated"},
        {"text": "who knows", "source": "unknown"},        # conservative → excluded
    ])
    # Only authored (typed/dictated) spans survive, stripped; pasted + unknown are gone. The two
    # authored spans are separated by a pasted span, so they stay distinct documents.
    assert sp.authored_documents(spans) == ["my own words", "dictated bit"]


def test_adjacent_authored_spans_coalesce():
    # codex: keystroke/edit fragments must join into ONE document (verbatim, no inserted separators),
    # not many — else the episode limit truncates them and mid-word splits corrupt the signature.
    frags = [sp.Span("Hel", "typed"), sp.Span("lo the", "typed"), sp.Span("re", "dictated")]
    assert sp.authored_documents(frags) == ["Hello there"]
    # a pasted span BREAKS the authored run into separate documents (relayed text never bridges them).
    broken = [sp.Span("my intro ", "typed"), sp.Span("RELAYED", "pasted"), sp.Span("my outro", "typed")]
    assert sp.authored_documents(broken) == ["my intro", "my outro"]


def test_pasted_ratio_is_char_fraction():
    spans = [sp.Span("aaaa", "typed"), sp.Span("bbbbbb", "pasted")]   # 4 authored, 6 pasted
    assert sp.pasted_ratio(spans) == 6 / 10
    assert sp.pasted_ratio([]) == 0.0
    assert sp.pasted_ratio([sp.Span("all mine", "typed")]) == 0.0
    assert sp.pasted_ratio([sp.Span("all theirs", "pasted")]) == 1.0


def test_fully_pasted_turn_has_no_authored_documents():
    # The core Phase-4 guarantee at the data layer: a fully-relayed turn yields nothing to embed.
    spans = sp.parse_spans([{"text": "someone else entirely", "source": "pasted"}])
    assert sp.authored_documents(spans) == []
    assert sp.pasted_ratio(spans) == 1.0
