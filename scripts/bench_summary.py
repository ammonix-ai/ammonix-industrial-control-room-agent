"""The 570-shift crisis benchmarks, frozen writer vs fine-tuned writer, from
the shipped per-shift rows.

    python scripts/bench_summary.py

Four strata, each run once with the frozen base writer (`*_ornith.json`) and
once with the fine-tuned writer (`*_ornith_r2.json`) on the identical shifts:
ordinary dev A/B (60), single crises (300), crisis pairs (150), difficult
tails (60). Prints successes, hard-safety violations, the paired flips and
the exact McNemar p per stratum and pooled, then one JSON verdict line.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "runs" / "reports"
ARM = "block10b_v5clean_judgement_v"
HARD = {"emissions", "overtemp", "overflow", "directive_safety_floor"}
STRATA = [
    ("ordinary dev A/B", "policy_eval_v04_ornith{tag}.json"),
    ("single crises", f"rare_bench_{ARM}_c5x_v5_clean_ornith{{tag}}.json"),
    ("crisis pairs", f"compound_bench_{ARM}_ornith{{tag}}.json"),
    ("difficult tails", f"difficult_slice_{ARM}_ornith{{tag}}.json"),
]


def rows(name: str) -> dict[int, dict]:
    report = json.loads((REPORTS / name).read_text(encoding="utf-8"))
    shifts = report["shifts"]
    if isinstance(shifts, dict):          # policy_eval keys its rows by arm
        shifts = shifts[ARM]
    out = {int(r["shift_index"]): r for r in shifts}
    if len(out) != len(shifts):
        raise RuntimeError(f"duplicate shift rows in {name}")
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p on the discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> int:
    print(f"{'stratum':18} {'n':>4} {'base':>5} {'r2':>5} {'hard b':>7} {'hard r2':>8} "
          f"{'r2-only':>8} {'base-only':>10} {'p':>8}")
    total = {"n": 0, "base": 0, "r2": 0, "hard_base": 0, "hard_r2": 0, "b": 0, "c": 0}
    per_stratum = {}
    for label, pattern in STRATA:
        base = rows(pattern.format(tag=""))
        r2 = rows(pattern.format(tag="_r2"))
        if base.keys() != r2.keys():
            raise RuntimeError(f"{label}: the two writers were not run on the same shifts")
        n = len(base)
        wins_base = sum(bool(r["success"]) for r in base.values())
        wins_r2 = sum(bool(r["success"]) for r in r2.values())
        hard_base = sum(r["violation"] in HARD for r in base.values())
        hard_r2 = sum(r["violation"] in HARD for r in r2.values())
        r2_only = sum(bool(r2[i]["success"]) and not base[i]["success"] for i in base)
        base_only = sum(bool(base[i]["success"]) and not r2[i]["success"] for i in base)
        p = mcnemar_exact(base_only, r2_only)
        per_stratum[label] = {"n": n, "base": wins_base, "r2": wins_r2, "hard_base": hard_base,
                              "hard_r2": hard_r2, "r2_only": r2_only, "base_only": base_only,
                              "mcnemar_p": round(p, 5)}
        print(f"{label:18} {n:>4} {wins_base:>5} {wins_r2:>5} {hard_base:>7} {hard_r2:>8} "
              f"{r2_only:>8} {base_only:>10} {p:>8.3f}")
        for key, value in (("n", n), ("base", wins_base), ("r2", wins_r2),
                           ("hard_base", hard_base), ("hard_r2", hard_r2),
                           ("b", base_only), ("c", r2_only)):
            total[key] += value
    p_all = mcnemar_exact(total["b"], total["c"])
    print(f"{'all':18} {total['n']:>4} {total['base']:>5} {total['r2']:>5} "
          f"{total['hard_base']:>7} {total['hard_r2']:>8} {total['c']:>8} {total['b']:>10} "
          f"{p_all:>8.3f}")
    verdict = {
        "check": "bench_summary",
        "strata": per_stratum,
        "pooled": {"n": total["n"], "base_success": total["base"], "r2_success": total["r2"],
                   "base_rate": round(total["base"] / total["n"], 4),
                   "r2_rate": round(total["r2"] / total["n"], 4),
                   "hard_base": total["hard_base"], "hard_r2": total["hard_r2"],
                   "r2_only": total["c"], "base_only": total["b"],
                   "mcnemar_p": round(p_all, 5)},
        "verdict": total["n"] == 570,
    }
    print(json.dumps(verdict))
    return 0 if verdict["verdict"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
