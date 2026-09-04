"""Entity naming rules shared by extraction, review, and graph loading.

Name normalisation lives here rather than in one service because the entity id
has to be computed identically everywhere - a different rule in graph-load than
in extract would silently split a node in two.
"""
from __future__ import annotations

import re
import unicodedata

from . import state

# Rank and honorific prefixes ignored when comparing names.  "SSgt Smith" and
# "Smith" must reach the same normalised form for the merge candidate to be
# proposed at all.
RANKS = {
    "amn", "a1c", "sra", "ssgt", "tsgt", "msgt", "smsgt", "cmsgt", "ccm",
    "2lt", "1lt", "capt", "maj", "ltcol", "col", "briggen", "majgen", "ltgen", "gen",
    "pvt", "pfc", "spc", "cpl", "sgt", "sfc", "1sg", "sgm", "csm",
    "ens", "ltjg", "lcdr", "cdr", "cpt", "cwo", "wo",
    "mr", "mrs", "ms", "miss", "dr", "sir", "madam",
    "sa", "det", "ofc", "officer", "agent", "inv",
}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md", "esq"}

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(name: str) -> str:
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _PUNCT.sub(" ", text).casefold()
    tokens = [t for t in _WS.split(text) if t]
    tokens = [t for t in tokens if t not in RANKS and t not in SUFFIXES]
    return " ".join(tokens)


def entity_id(entity_type: str, name: str) -> str:
    norm = normalize(name) or _WS.sub(" ", name.strip().casefold())
    return f"{entity_type}:{norm}"


def initials(norm_name: str) -> str:
    return "".join(t[0] for t in norm_name.split() if t)


def resolve_canonical(eid: str) -> str:
    """Follow approved merges to the surviving entity.

    Loop-guarded: a merge chain that somehow became cyclic returns the entry
    point rather than hanging a load.
    """
    seen: set[str] = set()
    current = eid
    while True:
        row = state.query_one(
            "SELECT merged_into FROM entities WHERE entity_id=?", (current,))
        if not row or not row["merged_into"] or current in seen:
            return current
        seen.add(current)
        current = row["merged_into"]


def display_name(eid: str) -> str:
    row = state.query_one(
        "SELECT canonical_name FROM entities WHERE entity_id=?", (eid,))
    if row:
        return row["canonical_name"]
    return eid.split(":", 1)[-1]
