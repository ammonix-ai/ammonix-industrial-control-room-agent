"""C2 free-text sanitisation for the runtime choke points (safety plan v0.1;
approval: the builder, 'start c2', 2026-08-21).

World free text -- the supervisor's directive and the crew radio -- is
untrusted input at deployment. On the released (label-only) arm it reaches
the loop at exactly two places: the directive choke point, where prose
becomes one canonical label, and the two radio regexes. This module wraps
those readers with:

  normalise_text     non-string -> "" (never an exception), NFKC folding,
                     control / zero-width characters stripped, whitespace
                     collapsed, hard cap of TEXT_CAP characters.
  canonical_directive
                     the pinned keyword extractor's substring semantics plus
                     (1) negation: a keyword with a negator in the three
                     tokens before it does not count; (2) conflicts: if
                     SAFETY FIRST survives it dominates; an irreconcilable
                     MAXIMISE + CONSERVE pair is neutral (balanced).
  safe_radio_suspects / safe_radio_spill_bunker / radio_hint
                     the pinned regexes on normalised text; a spill message
                     naming a bunker outside 0..2 is ignored.

SEALED NEUTRALITY: every recorded directive (4 distinct) and radio line
(33 distinct) in the frozen corpus is short printable ASCII with a single
keyword and an in-range bunker, so every wrapper returns exactly what the
pinned reader returns on them (tests/test_hardening_c2.py enumerates the
corpus). The pinned extractor in tokenizers/oven.py is not modified.
"""

from __future__ import annotations

import re
import unicodedata

from skills.oven_intake import _HINT_RE, radio_spill_bunker, radio_suspects
from tokenizers.oven import _directive_of

TEXT_CAP = 2000
RAW_FACTOR = 4
N_BUNKERS = 3
_WS = re.compile(r"\s+")
_KEYWORDS = {
    "maximize_output": ("MAXIMISE OUTPUT", "MAXIMIZE OUTPUT"),
    "conserve_budget": ("CONSERVE BUDGET",),
    "safety_first": ("SAFETY FIRST",),
}
_NEGATORS = frozenset({
    "NOT", "NEVER", "NO", "DONT", "DON'T", "STOP", "AVOID", "WITHOUT",
    "CANNOT", "CANT", "CAN'T", "WONT", "WON'T", "NEITHER", "NOR",
})
_NEGATION_WINDOW = 3
_TOKEN = re.compile(r"[A-Z']+")


def normalise_text(text: object, cap: int = TEXT_CAP) -> str:
    """Untrusted text -> bounded printable string. Never raises. Work is
    bounded BEFORE normalisation by a raw pre-cut of RAW_FACTOR x cap (a
    tail padded with strippable characters is dropped, fail-safe)."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text[:RAW_FACTOR * cap])
    t = "".join(ch for ch in t
                if unicodedata.category(ch) not in ("Cc", "Cf") or ch in "\n\t")
    t = _WS.sub(" ", t).strip()
    return t[:cap]


def _negated(upper: str, start: int) -> bool:
    before = upper[max(0, start - 60):start]
    tokens = _TOKEN.findall(before)[-_NEGATION_WINDOW:]
    return any(tok in _NEGATORS for tok in tokens)


def _present(upper: str, phrases: tuple[str, ...]) -> bool:
    """True when at least one occurrence of a phrase is not negated."""
    for phrase in phrases:
        pos = upper.find(phrase)
        while pos != -1:
            if not _negated(upper, pos):
                return True
            pos = upper.find(phrase, pos + 1)
    return False


def canonical_directive(text: object) -> str:
    """Canonical directive type for untrusted supervisor text."""
    upper = normalise_text(text).upper()
    present = {k for k, phrases in _KEYWORDS.items() if _present(upper, phrases)}
    if "safety_first" in present:
        return "safety_first"
    if len(present) == 1:
        return next(iter(present))
    return "balanced"   # none, or the irreconcilable maximise + conserve pair


def safe_radio_suspects(text: object) -> set[str]:
    return radio_suspects(normalise_text(text) or None)


def safe_radio_spill_bunker(text: object) -> int | None:
    b = radio_spill_bunker(normalise_text(text) or None)
    return b if b is not None and 0 <= b < N_BUNKERS else None


def radio_hint(text: object) -> bool:
    t = normalise_text(text)
    return bool(t and _HINT_RE.search(t))


def pinned_parity(text: object) -> bool:
    """True when the hardened extractor agrees with the pinned one -- the
    sealed-neutrality property, asserted over the corpus by the tests."""
    return canonical_directive(text) == _directive_of(text)
