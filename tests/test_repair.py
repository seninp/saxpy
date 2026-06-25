"""Testing the RePair grammar inference implementation.

Validated against golden output captured from the sibling jMotif C++/R project.
Because RePair's tie-resolution among equal-frequency digrams is not portable
across implementations (it depends on hash iteration order), the tests assert
numbering-INDEPENDENT invariants -- decompression round-trip, the RePair
termination guarantee (no repeated digram in R0), and structure on tie-free
inputs -- rather than exact rule-id equality.
"""

import os

import pytest

from saxpy.repair import RepairRule, str_to_repair_grammar

GOLDEN_SAX = os.path.join(os.path.dirname(__file__), "ecg_sax_string.txt")


def _decompress(grammar):
    """Recursively expand R0 back to the original terminal string."""
    s = grammar[0].rule_string
    while "R" in s:
        out = []
        for tok in s.split(" "):
            if tok.startswith("R") and tok[1:].isdigit() and int(tok[1:]) in grammar:
                rid = int(tok[1:])
                if rid != 0:
                    out.append(grammar[rid].expanded_rule_string)
                    continue
            out.append(tok)
        new_s = " ".join(out)
        if new_s == s:
            break
        s = new_s
    return s


def _r0_has_no_repeated_digram(grammar):
    toks = grammar[0].rule_string.split(" ")
    seen = set()
    for i in range(len(toks) - 1):
        d = (toks[i], toks[i + 1])
        if d in seen:
            return False
        # RePair only guarantees no digram occurs *more than once*; track all.
    # Count occurrences properly:
    from collections import Counter

    counts = Counter((toks[i], toks[i + 1]) for i in range(len(toks) - 1))
    return all(c <= 1 for c in counts.values())


def test_paper_grammar():
    """The documented golden grammar from the RePair paper / jmotif test.

    Input has an anomalous 'xxx' that no rule should absorb, so R0 surfaces it
    between two copies of the longest repeated pattern (R4)."""
    grammar = str_to_repair_grammar("abc abc cba cba bac xxx abc abc cba cba bac")
    assert grammar[0].rule_string == "R4 xxx R4"
    assert grammar[4].expanded_rule_string == "abc abc cba cba bac"


def test_decompression_roundtrip_paper():
    """Expanding R0 must reproduce the exact input (order-invariant oracle)."""
    s = "abc abc cba cba bac xxx abc abc cba cba bac"
    assert _decompress(str_to_repair_grammar(s)) == s


def test_decompression_roundtrip_ecg():
    """Round-trip on a long, realistic SAX string (2140 tokens, ~130 rules)."""
    with open(GOLDEN_SAX) as f:
        sax_string = f.read().strip()
    grammar = str_to_repair_grammar(sax_string)
    assert _decompress(grammar) == sax_string


def test_repair_termination_no_repeated_digram():
    """RePair's defining guarantee: no digram occurs more than once in R0."""
    with open(GOLDEN_SAX) as f:
        sax_string = f.read().strip()
    grammar = str_to_repair_grammar(sax_string)
    assert _r0_has_no_repeated_digram(grammar)


def test_every_rule_expands_to_at_least_two_tokens():
    """Every non-R0 rule comes from a digram, so it expands to >= 2 tokens."""
    with open(GOLDEN_SAX) as f:
        sax_string = f.read().strip()
    grammar = str_to_repair_grammar(sax_string)
    for rid, rule in grammar.items():
        if rid == 0:
            continue
        assert len(rule.expanded_rule_string.split(" ")) >= 2


def test_rule_intervals_match_expansion_length():
    """interval end - start equals the number of spaces in the expansion."""
    with open(GOLDEN_SAX) as f:
        sax_string = f.read().strip()
    grammar = str_to_repair_grammar(sax_string)
    for rid, rule in grammar.items():
        if rid == 0:
            continue
        spaces = rule.expanded_rule_string.count(" ")
        for start, end in rule.intervals:
            assert end - start == spaces


def test_grammar_is_deterministic():
    """The same input must yield the same grammar across runs (no hash-order
    dependence -- the reason new_digrams is insertion-ordered)."""
    with open(GOLDEN_SAX) as f:
        sax_string = f.read().strip()
    g1 = str_to_repair_grammar(sax_string)
    g2 = str_to_repair_grammar(sax_string)
    assert g1[0].rule_string == g2[0].rule_string
    sig1 = sorted(r.expanded_rule_string for i, r in g1.items() if i != 0)
    sig2 = sorted(r.expanded_rule_string for i, r in g2.items() if i != 0)
    assert sig1 == sig2


@pytest.mark.parametrize(
    "text,expected_r0",
    [
        ("a b a b", "R1 R1"),
        ("a a a a", "R1 R1"),
        ("a b c a b c", "R2 R2"),
        ("a b c d", "a b c d"),  # no repeats -> grammar is just R0
        ("a a a a a a a a", "R2 R2"),
    ],
)
def test_small_handcrafted_grammars(text, expected_r0):
    """Tie-free / minimal inputs whose top-level R0 structure is unambiguous."""
    grammar = str_to_repair_grammar(text)
    assert grammar[0].rule_string == expected_r0
    # Round-trip must hold regardless.
    assert _decompress(grammar) == text


def test_no_repeats_yields_only_r0():
    """A string with no repeated digram compresses to nothing but R0."""
    grammar = str_to_repair_grammar("a b c d e")
    assert len(grammar) == 1
    assert grammar[0].rule_string == "a b c d e"


def test_empty_input():
    """An empty string yields a degenerate single-R0 grammar, not a crash."""
    grammar = str_to_repair_grammar("")
    assert isinstance(grammar[0], RepairRule)
    assert len(grammar) == 1


def test_non_string_input_raises():
    with pytest.raises(TypeError):
        str_to_repair_grammar(["a", "b"])
