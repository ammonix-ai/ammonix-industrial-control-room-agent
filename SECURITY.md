# Security

Please do not report security issues in public GitHub issues.

Report them privately to licensing@ammonix.ai with "SECURITY" in the subject.
We will acknowledge within 5 business days. This is research software (see the
disclaimer in the README); it is not intended for production or clinical deployment.

The trained artefacts under `basis/` are `joblib` pickles. Loading a pickle executes code, so
load only the files shipped here and verified by `python scripts/check_release.py`, which
checks every one of them against the SHA-256 pinned in `runs/manifests/release_files.json`.
