"""Pin every shipped artefact: runs/manifests/release_files.json.

    python scripts/make_release_manifest.py

Records the SHA-256 and size of the basis, the working corpus, the sealed
traces, checkpoint and reports, the writer prompt and the vendored assets,
plus the hash of the neighbour-index matrix the release rebuilds on first
start. `scripts/check_release.py` re-verifies all of it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.nn_index import build_nn_index, index_matrix_sha256  # noqa: E402

BASIS_DIR = ROOT / "basis" / "oven_c5x_v5_clean"
OUT = ROOT / "runs" / "manifests" / "release_files.json"
GLOBS = (
    "basis/oven_c5x_v5_clean/*.json",
    "basis/oven_c5x_v5_clean/*.parquet",
    "basis/oven_c5x_v5_clean/artefacts/*.joblib",
    "data/working/oven_c5x/*",
    "runs/loops/r2_sealed_g/*.jsonl",
    "runs/loops/r2_sealed_g_checkpoint.jsonl",
    "runs/reports/*.json",
    "runs/manifests/writer_pin.json",
    "runs/manifests/explainer_pin.json",
    "harness/prompts/*.txt",
    "ui/static/vendor/*.js",
)
# rebuilt, never shipped
EXCLUDE = {"basis/oven_c5x_v5_clean/artefacts/index_nn.joblib"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shipped_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in GLOBS:
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    return sorted(p for p in files
                  if p.relative_to(ROOT).as_posix() not in EXCLUDE
                  and not p.name.endswith(".tmp"))


def main() -> int:
    entries = {}
    total = 0
    for path in shipped_files():
        rel = path.relative_to(ROOT).as_posix()
        size = path.stat().st_size
        total += size
        entries[rel] = {"sha256": sha256_file(path), "bytes": size}
    build_nn_index(BASIS_DIR)
    manifest = {
        "release": ("industrial-control-room-agent UI drop: "
                    "fine-tuned writer, sealed 200-shift cohort"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_files": len(entries),
        "total_bytes": total,
        "index_matrix_sha256": index_matrix_sha256(BASIS_DIR),
        "index_note": ("basis/oven_c5x_v5_clean/artefacts/index_nn.joblib is rebuilt on "
                       "first start from index_scaler.joblib + qpsi.parquet (ui/nn_index.py); "
                       "this is the SHA-256 of the fitted matrix it must contain"),
        "files": entries,
    }
    OUT.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"check": "release_manifest", "n_files": len(entries),
                      "total_mb": round(total / 1e6, 1), "path": OUT.relative_to(ROOT).as_posix(),
                      "verdict": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
