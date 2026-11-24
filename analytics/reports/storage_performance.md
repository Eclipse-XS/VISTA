# VISTA — Storage Recovery & DataLoader Performance Report

All measurements below were collected from actual executions on the local workstation (`Windows 11`, `12` logical CPUs, `WSL2 Ubuntu-24.04`, Python `3.13.5`, PyTorch `2.15.0+cpu`, `rclone v1.60.1-DEV` → `gdrive:VISTA MVTEC3D/`) against immutable `contracts_v2` Gold snapshot `snap_1_0_0_silver_v1_172131d3_a3cd04efd735` (`cable_gland`, `400 × 400` RGB + float32 depth + unit surface normals + uint8 validity mask, `verify_checksums=True` and full read-lifecycle worker leases active on every read).

---

## 1. Representative Training-Split Multi-Epoch `DataLoader` Benchmark (`n = 223`)

Benchmarked on the full already-ingested **`cable_gland` `train` partition (`223` real `400 × 400` samples)** with `batch_size=16`, `shuffle=False`, `max_cache_gb=6.0`, `min_free_disk_gb=2.0`, and real 64-bit Windows process working-set RSS (`GetProcessMemoryInfo` with explicit `HANDLE` / `c_void_p` types):

| Scenario | Epochs | `num_workers` | `persistent_workers` | Per-Epoch Wall Time (`s`) | Total Wall (`s`) | Overall Throughput (`samples/s`) | Steady-State Throughput (`samples/s`, Ep 2–3) | Remote Downloads (Calls / MiB) | Peak Main RSS (`MiB`) | `tracemalloc` Peak (`MiB`) | Peak Disk (`MiB`) | Canonical SHA-256 Match |
|---|---:|---:|:---:|---|---:|---:|---:|---|---:|---:|---:|:---:|
| **Cold on-demand** (`hydrate_mode="on_demand"`) | 1 | 0 | No | `[192.958]` | `193.047` | `1.16` | `1.16` | `2` / `493.20 MiB` | `270.86` | `99.87` | `350.37` | **PASS** |
| **Prefetched category** (`hydrate_mode="category"`) | 1 | 0 | No | Hydrate `88.041 s` + Ep1 `11.766 s` | `99.869` (`11.828` iter) | `18.85` (iter) | `18.95` (iter) | `2` hydrate / `0` iter (`493.20 MiB`) | `270.86` | `99.68` | `350.25` | **PASS** |
| **Warm cache (`w0`)** | 3 | 0 | No | `[11.732, 11.311, 11.200]` | `34.424` | `19.43` | `19.81` | `0` (`100%` cache hit) | `270.86` | `99.74` | `350.25` | **PASS** |
| **Warm cache (`w2`, non-persistent)** | 3 | 2 | No | `[20.472, 17.849, 19.778]` | `58.309` | `11.47` | `11.85` | `0` (`100%` cache hit) | `273.80` | `195.58` | `350.25` | **PASS** |
| **Warm cache (`w2`, persistent)** | 3 | 2 | **Yes** | `[15.501, 5.447, 5.669]` | **`26.722`** | **`25.04`** | **`40.12`** | `0` (`100%` cache hit) | `274.86` | `195.46` | `350.25` | **PASS** |
| **Warm cache (`w4`, non-persistent)** | 3 | 4 | No | `[23.511, 22.566, 22.038]` | `68.387` | `9.78` | `10.00` | `0` (`100%` cache hit) | `342.88` | `293.19` | `350.25` | **PASS** |
| **Warm cache (`w4`, persistent)** | 3 | 4 | **Yes** | `[27.491, 5.617, 5.474]` | `38.878` | `17.21` | **`40.21`** | `0` (`100%` cache hit) | `342.88` | `244.33` | `350.25` | **PASS** |

- **Exact Determinism Across All 7 Scenarios & All 17 Epochs:** Every configuration produced the exact same sample ordering and byte-identical canonical SHA-256 digest (`6259efdbb465d7463bad05be28f41f81ac6d81592684cd09d10cb19e48d5815a`) across RGB, float32 depth, and uint8 validity mask tensors.
- **Note on RSS Measurement:** `Peak Main RSS` reflects the main training process `PeakWorkingSetSize` (`270.86–342.88 MiB`). On Windows (`spawn`), each worker subprocess runs in a separate address space importing Python/NumPy/OpenCV/PyTorch, so total system RAM scales linearly with `num_workers`.

### Operational Recommendations (Windows 11 vs WSL2 / Linux)

1. **Always Prefetch Before Training (`hydrate_mode="category"`):** Category hydration completed cold-to-epoch-1 in **`99.869 s`** vs **`193.047 s`** for cold `on_demand` (**`1.93×` faster**) because it unpacks all `223` Silver `.npz` members in one pass instead of reopening `train.tar` 223 times, while keeping settled local disk usage bounded at **`350.25 MiB`** (deleting the `337.98 MiB` `.tar.xz` immediately after extracting RGB PNGs).
2. **Windows 11 (`spawn`) Multi-Epoch Training:**
   - **Recommended:** **`num_workers=2, persistent_workers=True`** (`40.12 samples/s` steady-state = **`2.03×`** speedup over `num_workers=0`, fastest 3-epoch wall time `26.722 s`, modest Peak Main RSS `274.86 MiB`).
   - **Avoid `persistent_workers=False` with `num_workers > 0` on Windows:** Respawning Python + PyTorch workers every epoch costs `~12–17 s/epoch`, making `w2` (`58.31 s`) and `w4` (`68.39 s`) **1.7–2.0× slower** than `num_workers=0` (`34.42 s`).
   - **Avoid `num_workers=4` for `< 500`-sample Windows runs:** Epoch 1 spawn cost (`~22 s`) outweighs marginal steady-state gains (`40.21` vs `40.12 samples/s` — IPC pickling of `400 × 400` multimodal arrays + per-read SHA-256 verification saturates around `~40 samples/s`).
   - **Single-epoch evaluation / validation:** Use **`num_workers=0`** (`11.2–11.8 s` for 223 samples; `0.867 s` for 23 validation samples).
3. **WSL2 / Native Linux (`fork`) Training:** Without Windows `spawn` startup overhead, `num_workers=2` or `4` with `persistent_workers=True` is recommended for multi-epoch training on hydrated local disk (store cache on native ext4 inside WSL2 rather than `/mnt/c` for best metadata I/O latency).

---

## 2. Granular Storage & Validation-Split Micro-Benchmarks (`n = 23`)

Measured on `cable_gland` `validation` (`n = 23` real `400 × 400` samples, `verify_checksums=True`):

| Workload | Scope | Measured Time | Throughput / Rate | Local Disk Footprint | Notes |
|---|---|---:|---:|---:|---|
| **Raw Bronze Archive Download** | `cable_gland.tar.xz` (`337.98 MiB`) from Google Drive | `21.490 s` | `15.73 MiB/s` | `337.98 MiB` (temp) | WSL2 `rclone copyto` with `--bind 0.0.0.0 --drive-root-folder-id` |
| **Raw Silver Shard Download** | `shards/validation.tar` (`16.05 MiB`) from Google Drive | `7.575 s` | `2.12 MiB/s` | `16.05 MiB` | Small-object transfer dominated by WSL/rclone startup + Drive API RTT |
| **Cold-Cache Single-Sample Recovery** | Empty local dir → `VistaMultimodalDataset.__init__` + `ds[0]` (`on_demand`) | `0.763 s` init + `106.346 s` first sample | — | Peak: **`358.85 MiB`**<br>Post-cleanup: **`20.87 MiB`** | Streams `.tar.xz` once for 23 validation RGB PNGs (skips `xyz/*.tiff`), deletes `.tar.xz`, downloads `validation.tar`, verifies SHA-256 |
| **Indexed Shard Extraction (All 23 Members)** | Cached `validation.tar` → `IndexedTarShard` extracting all 23 `.npz` bundles | `0.944 s` | `24.36 samples/s` | `36.19 MiB` total | O(1) TarInfo member lookup + atomic `.part` rename + SHA-256 |
| **Warm-Cache Single-Sample Access** | 23 real `400 × 400` RGB + Silver `.npz` reads (`verify_checksums=True`) | Mean **`30.93 ms`**<br>p50 **`31.40 ms`**<br>p95 **`35.63 ms`** | **`32.33 samples/s`** | `36.19 MiB` | Includes SHA-256 on RGB + `.npz`, OpenCV decode, and NPZ contract validation |
| **Validation 1-Epoch `DataLoader` (`w0` / `w2` / `w4`, non-persistent)** | 23 hydrated samples (`batch_size=4`) | `0.867 s` / `13.540 s` / `21.837 s` | `26.53` / `1.70` / `1.05` | `36.19 MiB` | Confirms `num_workers=0` is optimal for small 1-epoch validation passes on Windows |

---

## 3. Disk Budget & Cold-Recovery Trade-offs

- **Disk Savings vs Raw Bronze Extraction:** Extracting only ML-required RGB PNGs + Silver `.npz` bundles uses **`350.25 MiB`** for the entire 223-sample `cable_gland` training split (`36.19 MiB` for 23 validation samples) vs `~724 MiB` for a full `cable_gland` Bronze extraction including `xyz/*.tiff`.
- **Atomic Disk Budget Enforcement:** `LocalCacheBudgetManager.reserve_space()` atomically reserves peak transfer + extraction bytes before download under `.reservations/*.res.json` (accounting for competing processes and reclaiming dead-PID / expired reservations), while `VistaMultimodalDataset.__getitem__()` holds PID-checked worker leases across the entire RGB/Silver/GT read and decode lifecycle.
- **Cold-Recovery Bottleneck:** Cold first-sample latency (`88–106 s`) is dominated by solid `.tar.xz` LZMA streaming (`cable_gland.tar.xz`), not Silver shard access (`0.944 s` for 23 `.npz` members). Using `hydrate_mode="category"` amortizes archive extraction to once per category.
