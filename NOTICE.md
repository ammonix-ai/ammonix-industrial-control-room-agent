# Third-party notices

This repository vendors the following third-party software. Everything else is
original work released under the Ammonix Research License (see `LICENSE.md`).

## three.js (r160)

- Files: `ui/static/vendor/three.module.js`, `ui/static/vendor/OrbitControls.js`
  (the latter from three.js `examples/jsm/controls/`)
- Copyright © 2010-2023 Three.js Authors
- License: MIT — full text in `ui/static/vendor/LICENSE.three.txt`
- Used by the Knowledge Universe 3-D explorer page only.

## Model weights (not included in this repository)

- Ornith-1.5-9B (ornith-ai) is the base of the fine-tuned writer; the base
  weights are published by their authors under their own license. The r2
  adapter is released separately (`runs/manifests/writer_pin.json`).
- Qwen (Alibaba Cloud) models used for the operator dialog are served from
  outside this repository under their own license.
