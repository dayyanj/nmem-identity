"""Text-style quality gate — the pure authorship-quality heuristic (no ML deps)."""
from nmem_identity.perception import text_quality as tq

# A genuinely sufficient episode (>= GOOD_AUTHORED_CHARS of diverse prose) — the accumulated
# authored text of several chat turns, which is when the learning gate should open.
_LONG = ("I have been thinking about the migration plan and I reckon the safest path is to stage the "
         "additive columns first, then backfill in a separate pass so nothing blocks writes. We can "
         "always tighten the constraints afterwards once the data looks right. Honestly the part that "
         "worries me is the index rebuild on the big table during peak hours, so let us schedule that "
         "for the quiet window overnight and keep a rollback script handy just in case something goes "
         "sideways with the partial predicate.")
# Borderline: a single medium turn — past the hard floor, but too little to move a centroid.
_MEDIUM = ("Could you take a look at the deploy script when you get a chance, and let me know whether the "
           "rollback path still works after the last change? I want to be sure before we ship it today.")


def test_short_text_is_insufficient():
    q, parts = tq.text_quality("ok sounds good")
    assert parts["authored_chars"] < tq.MIN_AUTHORED_CHARS
    assert tq.regime(parts) == "text_insufficient"
    assert not tq.passes_gate(q)


def test_sufficient_diverse_text_passes():
    q, parts = tq.text_quality(_LONG)
    assert q >= tq.TEXT_QUALITY_GATE
    assert tq.regime(parts) == "text_authentic"
    assert tq.passes_gate(q)


def test_borderline_medium_turn_is_degraded_not_authentic():
    # Enough to be recognised from, not enough to learn from — the short-text guard in action.
    q, parts = tq.text_quality(_MEDIUM)
    assert parts["authored_chars"] >= tq.MIN_AUTHORED_CHARS   # past the hard floor
    assert not tq.passes_gate(q)
    assert tq.regime(parts) == "text_degraded"


def test_degenerate_repetition_is_docked():
    # Long enough by chars, but almost no lexical diversity → diversity credit drags quality down.
    q, parts = tq.text_quality("ok " * 80)
    assert parts["authored_chars"] >= tq.MIN_AUTHORED_CHARS
    assert parts["diversity_credit"] < 0.5
    assert q < tq.TEXT_QUALITY_GATE


def test_pasted_ratio_discounts_quality_without_hard_labelling():
    base, _ = tq.text_quality(_LONG, pasted_ratio=0.0)
    disc, parts = tq.text_quality(_LONG, pasted_ratio=0.6)
    assert parts["authored_fraction"] == 0.4
    assert disc < base                                # relayed text binds a weaker authorship signal
    # pasted_ratio never changes the length-based regime split (it discounts, not hard-labels).
    assert tq.regime(parts) in ("text_authentic", "text_degraded")


def test_quality_is_product_of_credits():
    q, p = tq.text_quality(_LONG, pasted_ratio=0.25)
    assert abs(q - p["sufficiency"] * p["authored_fraction"] * p["diversity_credit"]) < 1e-9


def test_empty_text():
    q, parts = tq.text_quality("")
    assert q == 0.0 and tq.regime(parts) == "text_insufficient"


# ── Phase 4: sufficiency gate + heuristic contamination ──────────────────────────

def test_sufficient_length_gate():
    assert not tq.sufficient_length("ok")
    assert not tq.sufficient_length("   ")
    assert not tq.sufficient_length("x" * (int(tq.MIN_AUTHORED_CHARS) - 1))
    assert tq.sufficient_length("x" * int(tq.MIN_AUTHORED_CHARS))


def test_heuristic_contamination_flags_relayed_formatting():
    clean = "just some ordinary sentences that a person would type out themselves in a chat window"
    r_clean, _ = tq.heuristic_contamination(clean)
    assert r_clean == 0.0
    pasted = "here you go: ```def f():\n    return 1``` and see https://example.com/very/long/path for more"
    r_pasted, parts = tq.heuristic_contamination(pasted)
    assert 0.0 < r_pasted <= 1.0
    assert parts["flagged_chars"] > 0                     # code fence + URL flagged


def test_heuristic_contamination_ratio_never_exceeds_one():
    # Overlapping artifacts (a URL inside a quoted line) must union, not sum past 1.0.
    text = "> see https://example.com/a/b/c\n> https://example.com/x/y/z"
    r, _ = tq.heuristic_contamination(text)
    assert 0.0 <= r <= 1.0


def test_heuristic_contamination_empty():
    r, parts = tq.heuristic_contamination("")
    assert r == 0.0 and parts["total_chars"] == 0.0


def test_heuristic_quote_detection_needs_line_boundaries():
    # codex: quote detection is line-anchored, so the embed server must feed contamination the text
    # WITH newlines (not a space-join of split documents), or a quoted line goes undetected.
    with_lines, _ = tq.heuristic_contamination("My reply.\n> Quoted material from elsewhere")
    without_lines, _ = tq.heuristic_contamination("My reply. > Quoted material from elsewhere")
    assert with_lines > 0.0            # the quoted line is flagged when boundaries survive
    assert without_lines == 0.0        # space-joined: the '>' is mid-line, not a quote line


def test_heuristic_quote_regex_is_line_bounded():
    # codex: leading indent is [ \t]* (not \s*), so blank-line-heavy input stays linear and a quote
    # after blank lines / with leading tab is still detected.
    r, _ = tq.heuristic_contamination("hi\n\n\n\t> indented quote\n\n> another")
    assert r > 0.0
    r_blank, _ = tq.heuristic_contamination("\n" * 20000)   # must not hang / backtrack
    assert r_blank == 0.0


def test_heuristic_flags_unclosed_code_fence():
    # codex: a code block whose closing ``` was truncated away (odd fence count) must still count as
    # relayed from the opener to the end, not earn authored credit.
    closed, _ = tq.heuristic_contamination("intro\n```\ncode here\n```\ndone")
    unclosed, parts = tq.heuristic_contamination("intro\n```\ncode here\nmore code that got cut off")
    assert closed > 0.0 and unclosed > 0.0
    # the unclosed fence flags everything from ``` onward
    assert parts["flagged_chars"] >= len("```\ncode here\nmore code that got cut off")


def test_heuristic_contamination_docs_scopes_fence_per_document():
    # codex: an unclosed fence in one turn must NOT consume later independent turns. Per-document
    # aggregation flags only the first turn; the clean turns contribute zero.
    docs = ["```\nsome code cut off", "an ordinary authored reply here", "another normal turn of text"]
    ratio, parts = tq.heuristic_contamination_docs(docs)
    assert ratio < 1.0                                   # clean turns not consumed by the fence
    assert parts["flagged_chars"] == float(len(docs[0]))  # exactly (only) the first turn is flagged


def test_fence_across_lines_scored_whole_for_one_turn():
    # A code block is one turn's content: scored as a WHOLE (the sidecar's `text` path scans the
    # original text) the spanning fence catches the enclosed code; scored per independent turn it would
    # miss the middle code lines. So the `text` path uses heuristic_contamination(whole), the
    # `documents` path uses heuristic_contamination_docs(per turn).
    block = "```python\nx = compute_all(data)\nreturn x\n```"
    whole, _ = tq.heuristic_contamination(block)                          # text path: one turn
    per_turn, _ = tq.heuristic_contamination_docs(block.splitlines())     # as if independent turns
    assert whole > per_turn                                               # spanning fence catches code
