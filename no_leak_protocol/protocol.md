# No-Leak DFME Protocol

This document specifies the leakage-free selection protocol used for all DFME
experiments. It exists because a common practice in prior micro-expression work
is to select the best epoch (or hyper-parameters) on the test set itself, which
turns the reported score into an oracle upper bound rather than a generalization
estimate. The protocol below removes every selection signal from the held-out
test partitions, so the reported numbers reflect what a practitioner could
obtain without ever seeing the test labels during model development.

## Data partitions

DFME ships an official `train` / `test A` / `test B` split. We derive a fourth,
internal partition for model selection:

- **fit** — the official training subjects minus the internal validation subjects.
- **internal validation** — about 20% of the official training subjects, drawn
  so that the partition is subject-disjoint from `fit` and covers every class.
  The carve is deterministic (a fixed split seed).
- **test A / test B** — the official held-out partitions, scored once on the
  selected model. Test B probes out-of-distribution generalization.

The exact carves used in the paper are stored in
[`splits/dfme_{3,4,7}class_internalval.json`](splits/), including the subject
lists, per-class clip counts, and the source-CSV checksums.

## Selection rule

The **only** signal permitted for selection is the internal validation
partition. This applies to early stopping, best-epoch selection, and every
hyper-parameter choice (alignment stages, alignment weight, student width,
augmentation).

- **Monitor** = `inner_val`. Test A and test B are never used as a monitor.
- **Selection metric** = UF1, with UAR as the tiebreaker. A pure-recall
  criterion can be inflated by an early degenerate checkpoint that over-predicts
  the minority class; UF1 includes precision and matches the headline metric.
- The monitor source is constrained at the type level: the held-out partitions
  are not valid monitor values, so they cannot be wired into selection by
  mistake (see [`monitor.py`](monitor.py)).

## Training recipe

The student is trained with AdamW (learning rate 3e-5, weight decay 1e-4, batch
size 32), a cosine schedule decaying to 1e-7, for at most 100 epochs with early
stopping (patience 30) on the internal validation UF1. The alignment uses GSWD
with 100 projections and a label-smoothing factor of 0.05. The alignment weight
lambda is selected on internal validation and may be 0, which switches the
teacher off and reduces the model to the single-stream student.

## Seeds and reporting

Because the validation carve and optimization are stochastic, each configuration
is trained over five seeds, and test A and test B are each reported as the mean
and standard deviation over those seeds. Handcrafted baselines, which are
deterministic, are run once.

## Macro teacher

The frozen teacher is pretrained on the posed CK+ macro-expression dataset,
whose subjects are distinct from the spontaneous DFME cohort. We verify zero
subject overlap between the teacher's training data and the DFME official split,
so no test identity is seen during teacher training.

## Invariants

1. The monitor is always the internal validation partition; test A and test B
   never enter selection.
2. `fit` and validation subjects are disjoint, the training pool is disjoint
   from both test partitions, and the validation partition covers every class.
3. Test A and test B are scored once, with the model selected on internal
   validation.

## Verifying the protocol

```bash
# Audit every shipped split for the disjointness and class-coverage invariants.
python no_leak_protocol/verify_no_leak.py

# Run the protocol checks (monitor guard + split invariants).
pytest no_leak_protocol/tests/
```

To train under this protocol, use the package script, which carves the internal
validation partition, monitors it for selection, and scores test A and test B
once per seed:

```bash
python scripts/train_dfme.py --benchmark dfme_3class \
    --train-index data/DFME/index_dfme_train_3class.csv \
    --testa-index data/DFME/index_dfme_testA_3class.csv \
    --testb-index data/DFME/index_dfme_testB_3class.csv \
    --output outputs/dfme_3class.json
```
