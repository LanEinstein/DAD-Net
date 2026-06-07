# No-Leak DFME Protocol

This folder documents and verifies the **leakage-free selection protocol** behind
the DFME results. Prior micro-expression work often selects the best epoch or
hyper-parameters on the test set, which inflates reported scores into oracle
upper bounds. Here every selection signal is confined to an internal validation
partition; test A and test B are held out and scored once.

It is a self-contained companion to the main `dadnet` package: the protocol is
*specified* here and *enforced* by the package training script
[`scripts/train_dfme.py`](../scripts/train_dfme.py). No model weights and no
dataset are included.

## Contents

| Path | Description |
|---|---|
| [`protocol.md`](protocol.md) | The frozen protocol: data partitions, selection rule, training recipe, seeds, invariants |
| [`splits/`](splits/) | The actual internal-validation carves for the 3-, 4-, and 7-class settings (subject lists, per-class counts, source checksums) |
| [`monitor.py`](monitor.py) | Selection-source guard: held-out partitions are not valid monitor values |
| [`verify_no_leak.py`](verify_no_leak.py) | Standalone audit asserting the disjointness and class-coverage invariants |
| [`tests/test_no_leak.py`](tests/test_no_leak.py) | The protocol guarantees as checks |

## What the splits show

Each split file records the exact subject-disjoint carve used in the paper. For
the three-class setting, the official training subjects are partitioned into 141
fit and 35 validation subjects (the validation partition covers all classes),
with 42 test-A and 41 test-B subjects held out. The carve is deterministic from
a fixed split seed.

## Verify it

```bash
# Audit every shipped split for leakage (non-zero exit on any violation).
python no_leak_protocol/verify_no_leak.py

# Run the protocol checks.
pytest no_leak_protocol/tests/
```

## Why a separate module

The exploration that produced these splits and audits lived in a research
harness with many dataset- and cluster-specific dependencies. This folder
distills the part that is reusable and verifiable: the protocol specification,
the real split artifacts, the selection-source guard, and the leakage audit.
Training itself is performed by the main package under the same protocol.
