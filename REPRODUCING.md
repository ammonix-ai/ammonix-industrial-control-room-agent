# Reproducing the paper

Paper: *The Ammonix Industrial Control Room Agent: Learning to Operate Industrial Plants from
Logged Experience* (https://doi.org/10.5281/zenodo.22871228). Everything below is regenerated or audited from
artefacts shipped in this repository. All data is synthetic — there is no dataset to obtain
and no credentialing step.

> Verified 2026-09-21 on a fresh clone in a clean venv (Python 3.13): `scripts/check_release.py`
> — 323 pinned artefacts present with their hashes, the writer prompt hashes to the value the
> sealed report pins, the rebuilt neighbour index matches the pinned matrix hash, and all 200
> sealed shifts replay through the plant engine to their recorded outcome (178 successes,
> 0 hard-safety violations); `scripts/bench_summary.py` — the 570-shift crisis table recomputed
> from the shipped per-shift rows; test suite 17 pass.

## Environment

- Python ≥ 3.12 (built and verified on 3.13); `pip install -r requirements.txt`. scikit-learn
  and xgboost are pinned exactly: the basis ships pickled calibrators, scalers and swarm
  classifiers.
- The plant engine is deterministic: a shift specification is a pure function of the cohort
  seed and shift index, and applying the recorded actions to a fresh environment rebuilds
  every plant state the agent saw. This is what the theatre and `scripts/check_release.py` do.
- Every shipped artefact is content-hashed in `runs/manifests/release_files.json`; the
  neighbour index (rebuilt on first start, see `ui/nn_index.py`) is pinned by the hash of the
  matrix it must contain.
- Models. The writer (`runs/manifests/writer_pin.json`) operated the recorded shifts and
  replays from its traces here; no result in this document depends on the dialog model
  (`runs/manifests/explainer_pin.json`), which only explains recorded decisions to the
  operator. A hosted dialog backend is configured with `DEMO_DIALOG_BASE`, `DEMO_DIALOG_MODEL`,
  `DEMO_DIALOG_KEY` and, for gateways whose control fields differ, `DEMO_DIALOG_EXTRA` (JSON
  merged into the request body); off by default, disclosed in the widget when remote.

## Result classes

- **exact** — regenerates bit-for-bit (or visually identically) from shipped artefacts.
- **statistical** — a rerun draws fresh samples (a live language model, GPU training); expect
  the paper's conclusions and intervals, not identical digits.
- **evidence-only** — cannot be rerun from this drop (the producing pipeline is not part of
  it); the producing run's full report ships, and verification means auditing it.

## The sealed evaluation (the run this UI serves)

Cohort: seed 99944001, namespace `g`, shifts 0–199, world v0.4, contract tolerance ±15%.
The agent never trained on it. Arm `block10b_v5clean_judgement_v` on basis
`oven_c5x_v5_clean-starter` (225,644 recorded states), writer prompt
`harness/prompts/m1_oven_v6_clean.txt` (SHA-256 `aa4ab48c…`).

| Result | Value | Source artefact | Regenerate / audit | Class |
|---|---|---|---|---|
| Agent, fine-tuned writer | 178/200 = 89.0%, 0 hard, mean score 0.5109 | `runs/reports/r2_sealed_g.json`, `runs/loops/r2_sealed_g/`, `runs/loops/r2_sealed_g_checkpoint.jsonl` | `python scripts/check_release.py` replays every shift and confirms each recorded outcome and the tally | exact (replay) |
| Agent, frozen writer | 177/200 = 88.5%, 0 hard, mean score 0.5057 | `runs/reports/headline_a_v5clean.json` | audit | evidence-only (traces not in this drop) |
| Fine-tuned vs frozen writer | McNemar p = 1.0, score Δ +0.0053 (4 vs 3 discordant shifts) | `runs/reports/r2_sealed_g.json` → `vs_base_sealed` | audit | evidence-only |
| Teacher personas (cautious / aggressive / sloppy) | 85.0% / 81.5% / 79.5% with 5 / 28 / 31 hard violations | `runs/reports/headline_a_v5clean.json` → `baselines` | audit | evidence-only |
| Oracle ceiling | 197/200 winnable (98.5%); unwinnable shifts 17, 57, 97 | `runs/reports/oracle_ceiling_a_v5clean.json` | audit | evidence-only |
| Universe ablation (episodic memory off) | 84.0% vs 88.5% full; Wilcoxon p = 0.00014 on score, McNemar p = 0.20 on success | `runs/reports/ablation_b1_g.json` | audit | evidence-only |
| Behaviour-cloning arm | 47.0% (94/200), holdout accuracy 0.897 | `runs/reports/imitation_g.json` | audit | evidence-only |

Rerunning the agent on the cohort needs the factory's evaluation runner and a local vLLM
serving the pinned base Ornith-1.5-9B with the r2 adapter (or without it for the frozen
row); with temperature 0 and schema-constrained decoding the rerun is expected to reproduce
the digits up to serving nondeterminism — statistical.

## The crisis benchmarks (570 shifts, frozen writer vs fine-tuned writer)

Four strata, each run with both writers on identical shifts: ordinary dev A/B (60), single
crises (300, rare-event families), crisis pairs (150, compound onsets), difficult tails (60).

| Stratum | n | success frozen → fine-tuned | hard frozen → fine-tuned |
|---|---:|---:|---:|
| ordinary dev A/B | 60 | 57 → 53 | 0 → 0 |
| single crises | 300 | 202 → 207 | 7 → 6 |
| crisis pairs | 150 | 64 → 60 | 3 → 0 |
| difficult tails | 60 | 54 → 53 | 1 → 0 |
| **all** | **570** | **377 → 373** (66.1% → 65.4%, McNemar p = 0.69) | **11 → 6** |

Source artefacts: `runs/reports/policy_eval_v04_ornith{,_r2}.json`,
`rare_bench_block10b_v5clean_judgement_v_c5x_v5_clean_ornith{,_r2}.json`,
`compound_bench_block10b_v5clean_judgement_v_ornith{,_r2}.json`,
`difficult_slice_block10b_v5clean_judgement_v_ornith{,_r2}.json` (per-shift rows, teacher
comparisons and family breakdowns inside). `python scripts/bench_summary.py` recomputes the
table, the paired flips and the exact McNemar p from the rows — exact. Rerunning the
benchmarks needs the factory and the local models — statistical.

## The Knowledge Universe

| Item | Source artefact | Regenerate / audit | Class |
|---|---|---|---|
| 3-D map (PCA of calibrated score vectors, tribes, Ether surface) | `runs/reports/oven_c5x_v5_clean_universe_map3d.json`, `…_tsne_inv.json` | served at `/universe/`; per-point records read from `basis/oven_c5x_v5_clean/universe.parquet` + `data/working/oven_c5x/examples.parquet` | exact (artefact) |
| Live scoring of replayed states (tick panel, trajectory snapping) | `basis/oven_c5x_v5_clean/` (manifest, qpsi, tribes, skills, artefacts) | computed by the server from the shipped basis; no model call | exact |
| Basis training (swarm, calibration, tribes, skills) | `basis/oven_c5x_v5_clean/manifest.json` (fold metrics, tokenizer, pins), `fold_metrics.json` | builder not in this drop | evidence-only |

## The writer

`runs/manifests/writer_pin.json`: base Ornith-1.5-9B (public weights, config hash pinned) +
the r2 QLoRA adapter (160 training pairs selected by the v2 closed-loop payload reward;
adapter hash pinned), served at temperature 0 with JSON-schema-constrained decoding and
thinking disabled. The weights are on Hugging Face as `Ammonix/AmmonixWtE-Writer-9B` (the sealed
adapter in `adapter/`, merged weights at the root); the training chain is part of the
factory, not of this drop — statistical when rerun (GPU training is not bit-deterministic).

## What the full factory adds

World generation and the working corpus (9,870 shifts, 225,644 states, rare-event
enrichment), tokenizer and swarm training, the harness loop with M1/M2 and the runtime
validator, the evaluation runners (sealed cohorts, teachers, oracle ceiling, ablations, the
crisis benchmarks) and the writer's fine-tuning chain. Its reports ship here under
`runs/reports/` so every number above can be audited now; the code follows as the factory
release.
