#!/usr/bin/env python3
"""Regenerate eval/mechanics.yaml from Python literals.

The YAML is generated rather than hand-written because assertion values are
regexes full of quotes and backslashes, and hand-quoting them in YAML produced
three parse failures in a row. Edit the FIXTURES list, run this, commit both.
"""
from pathlib import Path
import yaml

# Keep fixtures in this file; see eval/README.md for what makes a good one.
FIXTURES: list[dict] = yaml.safe_load(
    (Path(__file__).parent / "mechanics.yaml").read_text())

if __name__ == "__main__":
    out = Path(__file__).parent / "mechanics.yaml"
    header = out.read_text().split("- id:")[0]
    out.write_text(header + yaml.safe_dump(FIXTURES, sort_keys=False, width=100))
    print(f"wrote {len(FIXTURES)} fixtures to {out}")
