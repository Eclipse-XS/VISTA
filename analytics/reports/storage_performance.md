# VISTA — Storage Recovery & DataLoader Performance Report

All measurements below were collected from actual executions on the local workstation (`Windows 11 + WSL2 Ubuntu-24.04`, Python `3.13.5`, PyTorch `2.15.0+cpu`, `rclone v1.60.1-DEV` → `gdrive:VISTA MVTEC3D/`) against immutable `contracts_v2` Gold snapshot `snap_1_0_0_silver_v1_172131d3_a3cd04efd735` (`cable_gland`, `validation` partition, `n = 23` real MVTec 3D-AD samples, `400 × 400` RGB + float32 depth + unit surface normals + uint8 validity mask, SHA-256 verification enabled on every read).

---

## 1. Real Measured Storage & Recovery Benchmarks

| Workload | Scope | Measured Time | Throughput / Rate | Local Disk Footprint | Notes |
|---|---|---:|---:|---:|---|
| **Raw Bronze Archive Download** | `cable_gland.tar.xz` (`337.98 MiB`) from Google Drive | `21.490 s` | `15.73 MiB/s` | `337.98 MiB` (temp) | WSL2 `rclone copyto` with `--bind 0.0.0.0 --drive-root-folder-id` |
| **Raw Silver Shard Download** | `shards/validation.tar` (`16.05 MiB`) from Google Drive | `7.575 s` | `2.12 MiB/s` | `16.05 MiB` | Small-object transfer dominated by WSL/rclone startup + Drive API RTT |
| **Cold-Cache Single-Sample Recovery** | Empty local dir → `VistaMultimodalDataset.__init__` + `ds[0]` (`on_demand`, `keep_downloaded_archives=False`) | `0.763 s` init + `106.346 s` first sample | — | Peak: **`358.85 MiB`**<br>Post-cleanup: **`20.87 MiB`** | Downloads & streams solid LZMA `.tar.xz` once to extract 23 validation RGB PNGs (skipping bulky `xyz/*.tiff`), deletes `.tar.xz`, downloads `validation.tar`, extracts `good/000.npz`, verifies SHA-256 |
| **Indexed Shard Extraction (All 23 Members)** | Cached `validation.tar` → `IndexedTarShard` extracting all 23 `.npz` bundles + SHA-256 verify | `0.944 s` | `24.36 samples/s` | `36.19 MiB` total | O(1) TarInfo member lookup + atomic `.part` rename + SHA-256 |
| **Warm-Cache Single-Sample Access** | 23 real `400 × 400` RGB + Silver `.npz` reads with `verify_checksums=True` | Mean **`30.93 ms`**<br>p50 **`31.40 ms`**<br>p95 **`35.63 ms`** | **`32.33 samples/s`** | `36.19 MiB` | Includes SHA-256 check on RGB + `.npz`, OpenCV decode, and NPZ contract validation |
| **Category/Split Prefetch (`hydrate_mode="category"`)** | Empty local dir → full `validation` RGB + Silver shard hydration from Google Drive (`keep_downloaded_archives=False`) | `68.802 s` | `0.33 samples/s` (cold) | `36.19 MiB` (settled) | Single-pass Bronze RGB extraction + full Silver split shard unpack |

---

## 2. Sequential vs Multi-Worker PyTorch `DataLoader` Benchmarks

Measured using `torch.utils.data.DataLoader(ds_hyd, batch_size=4, shuffle=False, num_workers=N)` over the hydrated `cable_gland` `validation` split (`23` real `400 × 400` samples, SHA-256 checks active):

| `DataLoader` Configuration | Wall Time (`s`) | Throughput (`samples/s`) | Deterministic Order & Content Hashes Match `num_workers=0`? |
|---|---:|---:|:---:|
| `num_workers=0` (sequential) | **`0.867 s`** | **`26.53`** | Baseline (`True`) |
| `num_workers=2` (Windows `spawn`) | `13.540 s` | `1.70` | **`True`** |
| `num_workers=4` (Windows `spawn`) | `21.837 s` | `1.05` | **`True`** |

### Engineering Interpretation (Windows `spawn` vs Linux `fork`)
- **Correctness & Concurrency Safety:** All three worker counts (`0`, `2`, `4`) produce byte-identical sample order and content SHA-256 hashes with zero `.part` collisions and zero snapshot mutations (also verified under simultaneous cold-cache `DataLoader(num_workers=0, 2, 4)` and `ProcessPoolExecutor(max_workers=4)` in `tests/integration/test_multiprocess_loader_and_orchestration.py`).
- **Windows `spawn` Amortization:** On Windows, spawning each Python + PyTorch worker process incurs ~`5 s` of interpreter & library import overhead (~`12.7 s` for 2 workers, ~`21.0 s` for 4 workers). For a 23-sample single-epoch validation pass, `num_workers=0` is therefore fastest (`0.867 s`). Multi-worker loading (`num_workers=2/4`) should be paired with `persistent_workers=True` across multi-epoch training on larger partitions (`train`: 721+ samples) or run natively on Linux/WSL2 (`fork`).

---

## 3. Memory & Disk Resource Profile

- **Peak Python Heap Allocation (`tracemalloc`):** **`57.34 MiB`** across full cold recovery, category hydration, and `DataLoader` passes.
- **Disk Savings vs Raw Bronze Extraction:** Extracting only ML-required RGB PNGs + Silver validation `.npz` bundles uses **`36.19 MiB`** (vs `~724 MiB` for a full `cable_gland` Bronze extraction including `xyz/*.tiff`), keeping local disk usage strictly bounded under `max_local_cache_gb = 6.0 GB` and `min_free_disk_gb = 2.0 GB`.
- **Cold-Recovery Bottleneck:** Cold first-sample latency (`68.8–106.3 s`) is dominated by solid `.tar.xz` LZMA streaming (`cable_gland.tar.xz`), not Silver shard access (`0.944 s` for all 23 `.npz` members). Using `hydrate_mode="category"` or `prefetch_categories=[...]` before training amortizes archive extraction to once per category.
