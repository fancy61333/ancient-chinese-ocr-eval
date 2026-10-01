# Companion code, configuration and de-identified statistics

Companion resources for the manuscript

> **Graphemic variation and the limits of shape-based identification in oracle-bone and bronze inscriptions**

This repository is the addressable artefact referenced by the *Data availability statement* of
that manuscript. It contains **only** what can be released: the inference script of the
submitted system, the configuration actually used for the submitted run, a validation
utility for submission files, the scripts used to produce the distribution statistics
reported in the paper, and the de-identified outputs of those scripts.

Nothing here allows the reported scores to be recomputed end to end, and the repository is
not a reproduction package. See [What is *not* here](#what-is-not-here).

---

## Contents

| Path | What it is | Where it is used in the manuscript |
|---|---|---|
| `scripts/run_inference_yolo.py` | Two-stage inference (YOLOv8n detector + ResNet18 classifier) as submitted | Appendix A (system configuration); Section 3 (pipeline) |
| `config/inference_config.yaml` | Parameter values of the submitted run, transcribed from the configuration record | Appendix A; Section 4.3 (metric definitions) |
| `scripts/validate_prediction_format.py` | Checks that a prediction file conforms to the submission schema | Section 4.2 |
| `scripts/per_class_stats.py` | Per-category sample-count and per-category error statistics | Sections 4.1, 6.1, 7 |
| `results/glyph_distribution_summary.txt` | Distribution of images per category (auxiliary glyph set) | Section 4.1 |
| `results/glyph_distribution_histogram.csv` | The same distribution as a binned histogram (shareable table) | Section 4.1, supplementary material |

---

## Reproducing the statistics

The two statistics scripts run on the standard library alone (Python 3.8+). They take either a
directory laid out one subdirectory per category, a label list, or a JSON Lines metadata file.

```bash
# Images per category, from a directory of per-category subdirectories
python scripts/per_class_stats.py train --root /path/to/train_root --out out/prefix

# ... or from a JSON Lines file with one record per image
python scripts/per_class_stats.py train \
    --jsonl metadata.jsonl --key char --out out/prefix

# Per-category accuracy and confusion pairs, from a ground-truth/prediction list
python scripts/per_class_stats.py errors \
    --preds predictions.csv --train-counts out/prefix_train_counts.csv --out out/prefix
```

`--preds` expects two columns, ground-truth label and predicted label, one row per instance.
Outputs are aggregates only: counts per category, quantiles, share of categories below a
given sample size, accuracy bucketed by training frequency, and the most frequent confusion
pairs. No images, character forms or raw predictions are emitted.

The files in `results/` were produced with the second command above, on the auxiliary glyph
set described in Section 4.1 of the manuscript (575 categories, 100,554 images).

---

## Configuration caveats

Two points are stated in the manuscript and are repeated here because they are easy to
misread:

1. **The organiser's scorer is not public.** Detection F1, matched-box recognition accuracy
   and end-to-end F1 are returned by the shared-task platform. The box-matching rule and its
   intersection-over-union threshold were never published, so the numbers in the paper cannot
   be recomputed from this repository, and no value for that threshold should be inferred
   from anything here.
2. **The non-maximum-suppression threshold is 0.45 in the submitted configuration record,
   but the inference call does not pass `iou` explicitly**, so the value actually in effect
   at run time is whatever the detector library defaults to. The recorded value and the
   library default are both documented in `config/inference_config.yaml`. The distinction
   matters only for reproduction, not for the argument of the paper.

---

## What is *not* here

Deliberately excluded, with the reason:

| Excluded | Why |
|---|---|
| Model weights (detector and classifier checkpoints) | Derived from competition data released under a shared-task licence that does not permit redistribution |
| Competition training and evaluation images | Same reason |
| Character dictionaries and id-to-label mappings | Derived from the competition inventory; redistributable only with the organiser's permission |
| Raw prediction files and per-instance error lists | Contain competition image identifiers and unaggregated label information |
| The organiser's scorer | Never released by the organiser |

The per-image data needed to reproduce the reported scores therefore has to be requested from
the shared-task organiser. What this repository does provide is the configuration, the
measurement code and the de-identified aggregates sufficient to check the claims of
Sections 4.1, 6.1 and Appendix A.

---

## Anonymity

Author identity is deliberately not recorded in this repository while the manuscript is under
double-blind review. If the manuscript is accepted, the author details will be added here and
the repository will be archived with a persistent identifier.

---

## Licence

MIT, see `LICENSE`. The licence covers the code in `scripts/` and the configuration in
`config/`. The aggregate statistics in `results/` are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); they contain no image data and no
identifying information.
