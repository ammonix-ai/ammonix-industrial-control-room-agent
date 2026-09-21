"""Rebuild the basis's exact-neighbour index (artefacts/index_nn.joblib).

    python scripts/build_nn_index.py [--force] [--verify]

The server does this on its own at first start; run it explicitly to
pre-build (about ten seconds) or to check the rebuilt matrix against the
hash pinned in runs/manifests/release_files.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.nn_index import build_nn_index, index_matrix_sha256  # noqa: E402

BASIS_DIR = ROOT / "basis" / "oven_c5x_v5_clean"
MANIFEST = ROOT / "runs" / "manifests" / "release_files.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rebuild even if present")
    parser.add_argument("--verify", action="store_true",
                        help="compare the matrix hash with the release manifest")
    args = parser.parse_args()
    started = time.time()
    path = build_nn_index(BASIS_DIR, force=args.force)
    verdict = {"check": "nn_index", "path": str(path.relative_to(ROOT)),
               "bytes": path.stat().st_size, "seconds": round(time.time() - started, 1)}
    if args.verify:
        expected = json.loads(MANIFEST.read_text(encoding="utf-8")).get("index_matrix_sha256")
        actual = index_matrix_sha256(BASIS_DIR)
        verdict.update({"index_matrix_sha256": actual, "expected": expected,
                        "verdict": actual == expected})
    else:
        verdict["verdict"] = path.exists()
    print(json.dumps(verdict))
    return 0 if verdict["verdict"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
