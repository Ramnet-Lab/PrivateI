# Evaluation

Two layers, deliberately separate.

## 1. Mechanics suite — `run_mechanics.py`

Ten tiny fixtures, each isolating one mechanic that accurate investigation
depends on. They are case-independent by construction: the names are throwaway,
and every assertion is a *property* rather than an answer.

```bash
python3 eval/run_mechanics.py              # all
python3 eval/run_mechanics.py negation     # one mechanic
```

Exit code is the number of failures, so it gates CI.

| Mechanic | What fails if it breaks |
|---|---|
| `negation` | A negated statement rendered positive reverses a finding and survives a skim |
| `coreference` | "I"/"you" become entities, or one person's actions land on another |
| `grounding` | A person is created who exists in no document |
| `date_precision` | An approximation is asserted as a specific day |
| `hearsay` | Secondhand knowledge is recorded as direct observation |

Add a mechanic by appending to `build_fixtures.py` and regenerating
`mechanics.yaml` — never hand-edit the YAML, the quoting is fragile.

**A fixture must be answerable from its own text alone.** If it needs knowledge
of a particular case, it belongs in the case scorer instead.

## 2. Case scorer — `score_case.py`

Grades a fully processed case against a ground-truth key.

```bash
python3 eval/score_case.py eval/keys/cdi_2026-04_ground_truth.json
python3 eval/score_case.py <key> --report path/to/report.md
```

The key schema is the contract, not the case: entity roster with aliases, gold
facts with primary and corroborating documents, expected conflicts, a trap
inventory grouped by mechanic, and pass thresholds. Any investigation with a key
in that shape scores identically.

Scored categories: entity resolution (precision must be 1.0 — inventing a person
is a hard fail), negation accuracy, date accuracy (no precision upgrades),
sourcing discipline (documentary facts cite their custodian, not a restatement),
fact recall, and — with `--report` — disposition accuracy plus summary/findings
consistency.

## Reading results honestly

The scorer is code and can be wrong. Its first run reported three date failures
that were entirely its own: the hedge-word regex matched "about a foot from my
face" and read a distance as an approximate date. **When a category fails, check
the scorer before changing the pipeline.** A measurement you have not audited is
worse than no measurement, because it manufactures confident nonsense.

## Negative controls

```bash
python3 eval/run_mechanics.py --self-test
```

Feeds every assertion the exact defect it exists to catch and requires it to
fire. Run this whenever a check is added or edited: a suite that has never
failed proves nothing, and a green run only means something once you know the
checks can go red.

## A worked example of why you audit the scorer

The sourcing check was wrong three times before it was right, and each time it
was confidently wrong in a different direction:

1. It read the **last** two-digit number in the trap text as the primary
   document. That is the *secondary* document, so the check passed by testing
   the very thing it exists to catch.
2. Fixed to parse the named document, it then matched facts using content words
   taken from the trap's own prose - `primary`, `source`, `restatement` - so it
   matched any quote containing the word "source" and reported a failure while
   testing nothing at all.
3. Only when it resolved the gold fact IDs the trap names (`F15/F16`) and
   matched on *their* text did it measure the pipeline's actual behaviour -
   which turned out to be correct all along.

Between (1) and (3) the same check reported PASS, then FAIL, then PASS, against
a pipeline whose sourcing behaviour never changed. Believe a red light only
after you have read the check that produced it.
