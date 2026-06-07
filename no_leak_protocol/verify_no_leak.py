"""Audit the DFME internal-validation splits for selection leakage.

For every split file the audit asserts the invariants of the no-leak protocol:

    * fit and validation subjects are disjoint;
    * the training pool (fit and validation) is disjoint from test A and test B;
    * test A and test B are disjoint;
    * the validation partition covers every class.

The script exits non-zero on the first violation so it can run as a check.

Example:
    python no_leak_protocol/verify_no_leak.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

SPLIT_DIR = Path(__file__).resolve().parent / "splits"


def _subjects(partition: dict) -> set:
    """Return the subject set of a split partition."""
    return set(partition["subjects"])


def audit_split(path: Path) -> List[str]:
    """Return a list of invariant violations for one split file (empty if clean)."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    split = doc["split"]
    fit = _subjects(split["fit"])
    val = _subjects(split["val"])
    test_a = _subjects(split["testA"])
    test_b = _subjects(split["testB"])
    train_pool = fit | val

    violations: List[str] = []
    if fit & val:
        violations.append("fit and validation subjects overlap")
    if train_pool & test_a:
        violations.append("training pool overlaps test A")
    if train_pool & test_b:
        violations.append("training pool overlaps test B")
    if test_a & test_b:
        violations.append("test A and test B overlap")
    if not split["val"].get("class_complete", False):
        violations.append("validation partition does not cover every class")
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DFME splits for leakage")
    parser.add_argument("--split-dir", default=str(SPLIT_DIR))
    args = parser.parse_args()

    split_paths = sorted(Path(args.split_dir).glob("dfme_*class_internalval.json"))
    if not split_paths:
        print(f"no split files found in {args.split_dir}", file=sys.stderr)
        sys.exit(2)

    clean = True
    for path in split_paths:
        violations = audit_split(path)
        if violations:
            clean = False
            print(f"FAIL {path.name}")
            for violation in violations:
                print(f"  - {violation}")
        else:
            doc = json.loads(path.read_text(encoding="utf-8"))["split"]
            print(
                f"OK   {path.name}  fit={doc['fit']['n_subjects']} "
                f"val={doc['val']['n_subjects']} testA={doc['testA']['n_subjects']} "
                f"testB={doc['testB']['n_subjects']} subjects"
            )

    if not clean:
        sys.exit(1)
    print("all splits satisfy the no-leak invariants")


if __name__ == "__main__":
    main()
