"""Compare several independent runs of the same report against each other.

A run of this pipeline is a sample, not a measurement. The text model is asked
at a non-zero temperature and every run draws its own seed, so two runs over the
same corpus are two independent draws from one distribution rather than a copy
and its duplicate - with the seed pinned they were byte-identical and a repeated
run proved nothing. What independent draws buy is a check no single run can
perform on itself: an allegation whose disposition changes between runs was not
decided by the record, it was decided by which token happened to be sampled, and
the operator has to be told that before relying on the label.

Nothing in this module knows what any disposition means. Labels are compared as
opaque strings - grouped by their normalised spelling and counted - so the
comparison is structurally incapable of preferring one outcome to another, and
it stays correct if the set of labels ever changes. The only judgement made here
is whether the runs said the same thing, which is a question about the runs and
not about the case.

The parsers below read a written report, which is markdown the model had a hand
in shaping. A markdown-trained model decorates its own structure freely, so
every anchor tolerates leading heading marks, bullets, numbering and emphasis.
A parser that finds no dispositions at all raises rather than returning an empty
result: a run that cannot be read must be counted against agreement, never
quietly dropped from the denominator so the survivors look unanimous.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter

from . import state
from .log import get_logger

log = get_logger("consensus")

# A ceiling, not a recommendation. Each run is a full report - several model
# passes per allegation on CPU - so the cost is linear in this number and an
# operator who types 50 has made a mistake rather than a choice.
MAX_RUNS = 9

# What a section says where the disposition line is missing or unreadable. It is
# carried as a label of its own rather than dropped, because "this run did not
# say" is a different outcome from any label and has to count as disagreement.
NO_DISPOSITION = "(no disposition written)"

SEED_KEY = "report_seed:"


class UnreadableRun(Exception):
    """A report body from which no disposition could be read at all."""


# Markdown decoration that may sit in front of a line the parsers below anchor
# on. The model writes '**Disposition:**', '#### Allegation 2:', '> - Findings'
# and '1. CONFLICTS:' interchangeably, so an anchor tied to a bare line start
# silently reads zero rows. At most three decoration marks are allowed and each
# is bounded, which keeps the alternation from backtracking over a long run of
# hashes in a way that would cost more than the line is worth.
_DECOR = (r"[ \t]*(?:(?:[#>]{1,6}|[-*+•]|\d{1,3}[.)])[ \t]*){0,3}"
          r"[*_~`]{0,3}[ \t]*")

_ALLEGATION_HEAD = re.compile(
    rf"(?im)^{_DECOR}allegation[ \t]+(\d+)[ \t]*[*_~`]{{0,3}}[ \t]*[:.–—-]")

_FINDINGS_HEAD = re.compile(rf"(?im)^{_DECOR}findings[ \t]+by[ \t]+allegation\b")

_DISPOSITION = re.compile(
    rf"(?im)^{_DECOR}disposition[ \t]*[*_~`]{{0,3}}[ \t]*[:\-–—][ \t]*(.+)$")

# Deliberately a copy of the citation shapes the report module uses rather than
# an import of its private names: this module has to keep reading reports that
# were written by an older version of that one.
_CITE = re.compile(r"\[[^\]\n]{1,160}\]")
_PAGE_CITE = re.compile(r"\bp{1,2}\.?[ \t]*\d", re.I)


def _clean_label(raw: str) -> str:
    """A disposition as written, with decoration and trailing punctuation off."""
    text = " ".join((raw or "").split())
    text = text.strip("*_`~ ").strip("*_`~ .,;:")
    return text[:120].strip()


def _key(label: str) -> str:
    """The form two spellings of the same label have to share to be one group."""
    return " ".join((label or "").split()).strip(" .,:;*_`~-").casefold()


def _citations(text: str) -> set[str]:
    """The bracketed sources in a section, normalised for comparison.

    Only brackets carrying a page are kept. A bracket without one is prose or a
    reference to the claim under test, and counting it would make two runs look
    like they rested on different evidence when they differed only in aside.
    """
    found = set()
    for cite in _CITE.findall(text):
        inner = cite[1:-1]
        if not _PAGE_CITE.search(inner):
            continue
        found.add(" ".join(inner.split()).strip(" .,;").casefold())
    return found


def _table_dispositions(body: str) -> dict[int, str]:
    """The dispositions table, which is assembled mechanically by the writer."""
    found: dict[int, str] = {}
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 3 or not cells[0].isdigit():
            continue
        label = _clean_label(cells[-1])
        if label:
            found.setdefault(int(cells[0]), label)
    return found


def _sections(body: str) -> dict[int, str]:
    """One text block per allegation, taken from the findings half of the body.

    The summary at the top of a report also names allegations by number, so the
    search starts after the findings heading where one is present. Where it is
    not, a later block wins over an earlier one with the same number, because
    the findings are written after the summary and are the authority.
    """
    region = body
    head = _FINDINGS_HEAD.search(body)
    if head:
        region = body[head.end():]
    marks = list(_ALLEGATION_HEAD.finditer(region))
    found: dict[int, str] = {}
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(region)
        found[int(mark.group(1))] = region[mark.start():end]
    return found


def read_run(body: str) -> dict[int, dict]:
    """One report body as {allegation number: disposition and its citations}.

    Raises UnreadableRun when not one disposition can be read. A report whose
    structure the parser cannot see is not a report that agreed with the others;
    treating it as one would let a parsing failure be reported as consensus.
    """
    table = _table_dispositions(body)
    sections = _sections(body)
    rows: dict[int, dict] = {}
    for index in sorted(set(table) | set(sections)):
        section = sections.get(index, "")
        match = _DISPOSITION.search(section) if section else None
        written = _clean_label(match.group(1)) if match else ""
        tabled = table.get(index, "")
        rows[index] = {
            "disposition": written or tabled or NO_DISPOSITION,
            # The writer states that its table cannot disagree with the finding
            # blocks. If it does, that is worth saying out loud rather than
            # resolving silently in favour of either one.
            "mismatch": bool(written and tabled and _key(written) != _key(tabled)),
            "citations": _citations(section),
        }
    if not any(r["disposition"] != NO_DISPOSITION for r in rows.values()):
        raise UnreadableRun(
            f"no disposition could be read from the report body "
            f"({len(rows)} allegation block(s) found)")
    return rows


def compare(runs: list[dict]) -> dict:
    """Agreement across runs, per allegation.

    runs carries one dict per requested run: n, seed, report_id, error and
    allegations (the output of read_run, or None where the run failed or could
    not be read). The denominator everywhere below is the number of runs that
    were requested, not the number that came back readable, so a run that broke
    lowers agreement instead of disappearing from it.
    """
    total = len(runs)
    readable = [r for r in runs if r.get("allegations")]
    indexes = sorted({i for r in readable for i in r["allegations"]})

    allegations = []
    for index in indexes:
        counts: Counter = Counter()
        first: dict[str, str] = {}          # normalised key -> as first written
        cite_sets: list[set[str]] = []
        mismatched: list[int] = []
        for run in readable:
            row = run["allegations"].get(index)
            if not row:
                continue
            key = _key(row["disposition"])
            counts[key] += 1
            first.setdefault(key, row["disposition"])
            cite_sets.append(row["citations"])
            if row["mismatch"]:
                mismatched.append(run["n"])

        # Ordered by how many runs said it, then by which was said first. There
        # is no preferred label and no alphabetical fallback: either would be a
        # thumb on the scale in exactly the case this comparison exists to catch.
        order = list(first)
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], order.index(kv[0])))
        top = ordered[0][1] if ordered else 0
        tied = [k for k, c in ordered if c == top]
        # A tie is reported as a tie. Breaking it would invent a verdict the runs
        # did not reach.
        modal = first[tied[0]] if len(tied) == 1 else None
        unanimous = modal is not None and top == total

        union: set[str] = set().union(*cite_sets) if cite_sets else set()
        shared: set[str] = set.intersection(*cite_sets) if cite_sets else set()

        allegations.append({
            "index": index,
            "modal": modal,
            "split": modal is None,
            "agreement": top,
            "of": total,
            "present": len(cite_sets),
            "unanimous": unanimous,
            "unstable": not unanimous,
            "counts": [{"disposition": first[k], "runs": c} for k, c in ordered],
            "mismatch_runs": mismatched,
            "evidence": {
                "shared": len(shared),
                "union": len(union),
                "stable": bool(union) and shared == union,
                "varying": sorted(union - shared),
            },
        })

    return {
        "runs": total,
        "readable": [r["n"] for r in readable],
        "unreadable": [r["n"] for r in runs if not r.get("allegations")],
        "allegations": allegations,
        "unstable": [a["index"] for a in allegations if a["unstable"]],
        "evidence_unstable": [a["index"] for a in allegations
                              if a["unanimous"] and not a["evidence"]["stable"]],
    }


def _ratio(item: dict) -> str:
    return f"{item['agreement']}/{item['of']}"


def render(summary: dict, members: list[dict], created: str, model: str) -> str:
    """The comparison as a markdown document, in the shape of a report."""
    lines = [
        f"# Consensus across {summary['runs']} runs",
        "",
        f"Generated {created} using {model or 'an unrecorded model'}. Each run "
        f"below answered the same objective over the same corpus and drew its "
        f"own random seed, so the runs are independent samples rather than "
        f"copies of one another. Where they disagree, the disagreement is the "
        f"finding: a disposition that changes between runs of an unchanged "
        f"record was not determined by that record.",
        "",
        "## Runs",
        "",
        "| Run | Seed | Report | Outcome |",
        "|---|---|---|---|",
    ]
    for member in members:
        outcome = member.get("error") or "read"
        rid = member.get("report_id") or "—"
        lines.append(f"| {member['n']} | {member.get('seed', '—')} | `{rid}` | "
                     f"{outcome} |")
    lines += ["",
              "The seed is recorded against each run so any single run above can "
              "be reproduced exactly by generating one report with that seed.",
              ""]

    if summary["unreadable"]:
        names = ", ".join(f"run {n}" for n in summary["unreadable"])
        lines += [
            f"> **{len(summary['unreadable'])} of {summary['runs']} runs "
            f"produced nothing readable ({names}).** A run whose dispositions "
            f"cannot be read did not agree with anything, so it is counted "
            f"against agreement below rather than removed from the denominator.",
            ""]

    if not summary["allegations"]:
        lines += ["No allegation could be read from any run, so there is nothing "
                  "to compare. Treat every run above as suspect.", ""]
        return "\n".join(lines) + "\n"

    lines += ["## Agreement by allegation", "",
              "| # | Agreement | Modal disposition | Cited evidence |",
              "|---|---|---|---|"]
    for item in summary["allegations"]:
        modal = item["modal"] or "**split — no modal disposition**"
        if item["unstable"] and item["modal"]:
            modal = f"{modal} **(unstable)**"
        ev = item["evidence"]
        if not ev["union"]:
            cited = "no page citations found"
        elif ev["stable"]:
            cited = f"same {ev['shared']} citation(s) in every run"
        else:
            cited = (f"{ev['shared']} of {ev['union']} citation(s) common to "
                     f"every run")
        lines.append(f"| {item['index']} | {_ratio(item)} | {modal} | {cited} |")
    lines.append("")

    lines += ["## Stability findings", ""]
    if not summary["unstable"] and not summary["evidence_unstable"]:
        lines += [f"Every allegation drew the same disposition in all "
                  f"{summary['runs']} runs, from the same cited evidence each "
                  f"time. That is agreement between samples, not proof that the "
                  f"disposition is correct.", ""]
    for item in summary["allegations"]:
        if not item["unstable"] and item["evidence"]["stable"]:
            continue
        lines.append(f"### Allegation {item['index']}")
        lines.append("")
        if item["unstable"]:
            split = "; ".join(f"{c['runs']} run(s): {c['disposition']}"
                              for c in item["counts"])
            missing = item["of"] - sum(c["runs"] for c in item["counts"])
            if missing > 0:
                split += f"; {missing} run(s): the allegation was not answered"
            lines += [
                f"**UNSTABLE — {_ratio(item)} agreement.** The runs split "
                f"{split}. This allegation is not determined by the current "
                f"record: repeating the same analysis over the same documents "
                f"returns a different answer. No disposition below unanimous "
                f"agreement should be carried into a report of investigation "
                f"without saying so.",
                ""]
        if not item["evidence"]["stable"]:
            ev = item["evidence"]
            varying = ", ".join(f"`{c}`" for c in ev["varying"][:8])
            more = ("" if len(ev["varying"]) <= 8
                    else f" and {len(ev['varying']) - 8} more")
            if item["unanimous"]:
                lines.append(
                    f"**The disposition held but the evidence did not.** All "
                    f"{item['of']} runs agreed on the label while resting it on "
                    f"different material: {ev['shared']} of {ev['union']} "
                    f"citation(s) appeared in every run. The same answer reached "
                    f"from different evidence each time is agreement about the "
                    f"conclusion, not about the record.")
            else:
                lines.append(
                    f"The cited evidence also moved: {ev['shared']} of "
                    f"{ev['union']} citation(s) appeared in every run.")
            if varying:
                lines.append("")
                lines.append(f"Citations not present in every run: {varying}{more}.")
            lines.append("")
        if item["mismatch_runs"]:
            names = ", ".join(f"run {n}" for n in item["mismatch_runs"])
            lines += [f"In {names} the dispositions table disagreed with the "
                      f"disposition written in the finding block. The finding "
                      f"block was taken as authoritative; the report(s) named "
                      f"should be read directly.", ""]

    lines += ["---", "",
              "This comparison measures reproducibility, not correctness. Runs "
              "that agree can agree on an error, and this document says nothing "
              "about which disposition the evidence supports. What it does "
              "establish is which of the dispositions above survive being asked "
              "again.", ""]
    return "\n".join(lines)


def record_seed(report_id: str, seed: int) -> None:
    """Tie a generated report to the seed that produced it.

    Kept in the settings store rather than a column on reports so that a report
    written before this existed is simply absent from it rather than wrong.
    """
    state.set_setting(SEED_KEY + report_id, str(seed))


def seeds() -> dict[str, str]:
    """Every recorded seed, keyed by report id."""
    rows = state.query("SELECT key, value FROM settings WHERE key LIKE ?",
                       (SEED_KEY + "%",))
    return {row["key"][len(SEED_KEY):]: row["value"] for row in rows}


def save(member_ids: list[str], body: str, created: str) -> str:
    """Store the comparison as a report, so it can be re-read and downloaded.

    It goes in the reports table beside the runs it compares - the same storage,
    the same download route - with an id that says what it is, because a
    consensus document listed as though it were an ordinary report would be read
    as one more opinion rather than as a check on the others.
    """
    source = state.query_one(
        "SELECT objective, model, documents, assertions FROM reports "
        "WHERE report_id=?", (member_ids[0],)) if member_ids else None
    report_id = "consensus-" + hashlib.sha256(
        "|".join([created] + member_ids).encode()).hexdigest()[:16]
    with state.tx() as conn:
        conn.execute(
            """INSERT INTO reports (report_id, objective, body, model, documents,
                                    assertions, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (report_id, source["objective"] if source else "", body,
             source["model"] if source else None,
             source["documents"] if source else None,
             source["assertions"] if source else None, created))
    log.info("consensus %s written over %d run(s)", report_id, len(member_ids))
    return report_id
