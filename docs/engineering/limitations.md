# VISTA — Known Limitations, Scientific Caveats & Technical Debt

This document records verified boundaries and open limitations transparently so downstream CV researchers and operators never mistake proxy metrics or unverified integrations for guarantees.

---

## 1. Dataset Coverage & Statistical Caveats

1. **Limited Category Coverage (3 of 10 Categories):** Only `cable_gland` (manufactured, 354 samples), `cookie` (organic, 363 samples), and `dowel` (manufactured, 452 samples) — 1,169 samples total — are currently ingested. The remaining 7 categories (`bagel`, `carrot`, `foam`, `peach`, `potato`, `rope`, `tire`) are **not** ingested. Do **not** generalize organic-vs-manufactured contrasts (`n=1` organic vs `n=2` manufactured) to untested categories.
2. **Pseudoreplication / Correlated Acquisitions:** `355 / 1,169` (`30.37%`) samples trigger perceptual near-duplicate `WARN` flags across repeated acquisitions of similar parts. Standard sample-level p-values and bootstrap confidence intervals treat samples as i.i.d. and therefore **overstate effective independent sample size**; treat statistical tests as descriptive summaries.
3. **Acquisition-Session Shortcut Risks (`cable_gland` & `cookie`):** Retrospective diagnostics show measurable divergence between `test/good` and `test/anomalous` brightness and background 3D dropout (`cable_gland` `bg_missing +3.21 SD` on anomalous vs `+0.48 SD` on `test/good`; `cookie` `bg_brightness +4.29 SD` and `fg_brightness +2.93 SD` on `test/good`). Downstream CV anomaly detectors MUST mask background pixels and test photometric robustness to avoid exploiting capture shortcuts.
4. **No Causal Proofs:** Observational correlations and KS distribution shifts expose associations and hypotheses, never proven physical root causes.

---

## 2. Foreground Proxy Limitations (`fg_otsu_plane_v1`)

1. **Heuristic Proxies, Not Ground Truth:** `fg_rgb` (border-referenced Otsu + morphology) and `fg_3d` (border plane residual `> 4 mm`) are strictly unsupervised proxies. `fg_rgb_3d_iou` quantifies **cross-proxy consistency**, NEVER ground-truth segmentation accuracy.
2. **Observed Failure Modes:**
   - **Dark internal textures / shadows:** Dark chocolate chips (`cookie`), shadowed threaded flanges (`cable_gland/train/good/081`, IoU `0.7153`), and recessed ribs/dark contamination (`dowel/test/contamination/008`, IoU `0.7615`) can be under-segmented by `fg_rgb`.
   - **3D background false positives:** Isolated crumbs or table-edge sensor noise (`cookie/test/crack/001`, `plane_inlier=0.9745`) trigger spurious `fg_3d` blobs outside the object.
   - **Hollow interiors & steep bevels:** Hollow openings (`cable_gland/train/good/199`, `fg_missing=4.23%`) and deep cracks (`cookie/test/crack/015`, `fg_missing=0.65%`) cause genuine on-object 3D occlusion.
3. **Controlled Robustness Alternative:** `compare_foreground_strategies()` (`src/vista/quality/foreground.py`) demonstrates that interior hole-filling (`rgb_holefill_lcc`) + largest-connected-component filtering (`plane_3d_lcc`) mitigate internal dark holes and external 3D table noise while keeping `rgb_holefill_lcc` non-circular with 3D sensor validity. We intentionally retained `fg_otsu_plane_v1` for `silver_v1` to preserve exact reproducibility across all 4 committed snapshots; adopting `rgb_holefill_lcc` in future work should be versioned as `silver_v2`.

---

## 3. Infrastructure & Portability Limitations

1. **Google Drive OAuth Scope Boundary:** Application path validation and `--drive-root-folder-id` are **operational safeguards**, NOT Google OAuth permission boundaries. The underlying rclone OAuth token retains its configured Drive scope and must remain private in `~/.config/rclone/rclone.conf`.
2. **Live AWS S3 Cloud Unverified:** S3 REST API portability (including virtual-directory `stat()`, multipart ETag SHA-256 read-back, HTTP 404/403 error semantics, Gold snapshots, and sample recovery) is verified end-to-end against an ephemeral S3-compatible HTTP server (`tests/integration/test_s3_portability.py`). Live AWS S3 cloud buckets, IAM roles, KMS encryption, and cross-region endpoints remain **CONFIGURATION-LEVEL / NOT VERIFIED END-TO-END**.
3. **MLflow Scope:** MLflow currently tracks Data Engineering snapshot provenance (`snapshot_id`, `manifest_sha256`, `train_manifest_hashes`, `code_provenance`). Full CV model training, Model Registry promotion, and serving are intentionally deferred to the CV modeling phase.
4. **Solid `.tar.xz` Bronze Archive Extraction:** Because official MVTec 3D-AD Bronze archives use solid LZMA `.tar.xz` compression, restoring even a single missing RGB/GT file on a completely cold cache requires streaming the category's `.tar.xz` archive once (`~338 MB` for `cable_gland`). VISTA mitigates this by extracting all needed split RGB/GT files in a single pass, skipping bulky `xyz/*.tiff` files during ML loading, and deleting the temporary `.tar.xz` immediately when `keep_downloaded_archives=False`.
5. **Minor Non-Blocking Upstream Warnings:** Older synthetic tests invoke `tifffile.imwrite` without `photometric="rgb"` (deprecation warning only), MLflow 3.x + SQLAlchemy 2.1 emits a `noload` deprecation warning internally, and Prefect 3's ephemeral Windows asyncio server occasionally logs `WinError 10054` socket-close noise on clean shutdown.
