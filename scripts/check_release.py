"""Verify the release: shipped artefacts, the rebuilt index, and the replay.

    python scripts/check_release.py

1. every file pinned in runs/manifests/release_files.json is present with
   the pinned SHA-256 (drift is listed);
2. the writer prompt hashes to the value the sealed report pins;
3. the neighbour index exists (built if missing) and its fitted matrix
   hashes to the pinned value;
4. all 200 sealed shifts replay through the deterministic engine to the
   recorded outcome (success flag and violation), and the tally equals the
   sealed report: 178 successes, 0 hard-safety violations.

Prints one JSON verdict line; exit status 0 only when everything holds.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "runs" / "manifests" / "release_files.json"
HARD = {"emissions", "overtemp", "overflow", "directive_safety_floor"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    started = time.time()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    drifted, missing = [], []
    for rel, pin in manifest["files"].items():
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
        elif sha256_file(path) != pin["sha256"]:
            drifted.append(rel)
    files_ok = not drifted and not missing

    from ui import server
    from ui.nn_index import build_nn_index, index_matrix_sha256

    report = server._report()
    prompt_sha = sha256_file(ROOT / "harness" / "prompts" / "m1_oven_v6_clean.txt")
    prompt_ok = prompt_sha == report.get("prompt_sha256")

    build_nn_index(server.BASIS_DIR)
    index_sha = index_matrix_sha256(server.BASIS_DIR)
    index_ok = index_sha == manifest.get("index_matrix_sha256")

    mismatches = []
    successes = hard = 0
    rows = server._summaries()
    for row in rows:
        replay = server._replay(row["shift_index"])
        check = replay["replay_check"]
        if not check["success_matches"] or check["replayed_violation"] != row.get("violation"):
            mismatches.append(row["shift_index"])
        if any("replay_error" in t for t in replay["ticks"]):
            mismatches.append(row["shift_index"])
        successes += int(bool(row.get("success")))
        hard += int(row.get("violation") in HARD)
    arm = report.get("this_arm", {})
    tally_ok = (len(rows) == report.get("n_shifts") == 200
                and successes == round(arm.get("success_rate", 0) * 200)
                and hard == arm.get("hard_violations"))
    replay_ok = not mismatches and tally_ok

    verdict = {
        "check": "release",
        "files": {"n": len(manifest["files"]), "missing": missing, "drifted": drifted},
        "prompt_sha256_ok": prompt_ok,
        "index_matrix_ok": index_ok,
        "replay": {"shifts": len(rows), "successes": successes, "hard_violations": hard,
                   "mismatches": mismatches, "report_success_rate": arm.get("success_rate")},
        "seconds": round(time.time() - started, 1),
        "verdict": bool(files_ok and prompt_ok and index_ok and replay_ok),
    }
    print(json.dumps(verdict))
    return 0 if verdict["verdict"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
