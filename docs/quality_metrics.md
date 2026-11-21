# VISTA Data Quality Metrics, Foreground Masks & Leak-Free Normalization

All raw metrics and foreground masks are strictly unsupervised functions of raw RGB and organized XYZ inputs. Benchmark ground-truth defect masks (`test/*/gt/*.png`) are **never** used to compute quality metrics or normalization parameters.

## 1. Global RGB Appearance Metrics

Let $I \in \{0, \dots, 255\}^{H \times W \times 3}$ be the decoded RGB image ($N = H \cdot W$ pixels), and define Rec.709 luminance:
$$Y(u, v) = 0.2126 R(u, v) + 0.7152 G(u, v) + 0.0722 B(u, v)$$

| Metric | Mathematical Definition | Limitations |
|---|---|---|
| `rgb_brightness_mean` | $\mu_Y = \frac{1}{N} \sum_{u,v} Y(u,v)$ | Mixes foreground object and dark background; sensitive to object footprint and session background illumination. |
| `rgb_brightness_std` / `rgb_contrast_rms` | $\sigma_Y$ and $\sigma_Y / 255.0$ | Global dispersion; strongly coupled to foreground-to-background area ratio. |
| `rgb_contrast_p5_p95` | $Q_{0.95}(Y) - Q_{0.05}(Y)$ | Robust dynamic contrast excluding 5% extreme tails. |
| `rgb_laplacian_var` | $\operatorname{Var}\big((\Delta_1 * Y)(u,v)\big)$ using 4-neighbor kernel | Standard sharpness proxy; influenced by background area and sensor noise. |
| `rgb_tenengrad_mean` | $\frac{1}{N}\sum_{u,v}\big((G_x * Y)^2 + (G_y * Y)^2\big)$ via $3\times 3$ Sobel | Edge-energy focus proxy. |
| `rgb_saturation_mean` | $\frac{1}{N}\sum_{u,v} S_{\text{HSV}}(u,v) / 255.0$ | Mean chroma saturation in $[0,1]$. |
| `rgb_underexposed_ratio` / `rgb_overexposed_ratio` | $\mathbb{P}[Y < 15]$ and $\mathbb{P}[Y > 240]$ | Global dark-pixel ratio reflects dark matte table backgrounds rather than part underexposure. |

## 2. Global 3D Point-Cloud Metrics

Let $P(u,v) = (X(u,v), Y(u,v), Z(u,v)) \in \mathbb{R}^3$ be the organized 3D coordinate map (`xyz/*.tiff`).
Pixel validity $V(u,v) = \text{finite}(P) \land (|Z| > \epsilon) \land (\|P\|_\infty \le M)$ ($\epsilon=10^{-6}\text{ m}$, $M=100\text{ m}$).

- `valid_point_ratio` $= |\mathcal{V}|/N$, `missing_point_ratio` $= 1 - |\mathcal{V}|/N$, `nan_point_ratio`, `depth_coverage_ratio`.
- Depth & extent statistics over $\mathcal{V}$: `z_min`, `z_max`, `z_mean`, `z_std`, `z_median`, `z_iqr`, `x_span`, `y_span`, `z_span`, `bbox_volume`, `point_density_proxy`.

## 3. Unsupervised Foreground-Aware Metrics (`fg_otsu_plane_v1`)

To separate object-level quality from dark-background structured-light dropouts **without** equating `valid_3d` with true object foreground and **without** touching test GT masks:

1. **Appearance foreground proxy (`fg_rgb`):** Border-referenced absolute contrast $|Y_{\text{blur}} - \operatorname{median}(Y_{\text{border}})|$ thresholded via Otsu and morphologically cleaned (close + open).
2. **3D height-above-plane proxy (`fg_3d`):** Iteratively trimmed least-squares plane $z_{\text{plane}}(x,y) = ax + by + c$ fitted on valid border pixels; pixels protruding $> 4\text{ mm}$ toward the camera ($z_{\text{plane}} - Z > 0.004\text{ m}$) form `fg_3d`.
3. **Non-circular on-object 3D coverage:** `fg_missing_3d_ratio` is evaluated on `fg_rgb` pixels (not `fg_3d`), preventing circularity when measuring 3D sensor dropout on the inspected object!
4. **Background & framing indicators:** `bg_missing_3d_ratio`, `bg_rgb_brightness_mean`, `fg_area_ratio`, `fg_rgb_3d_iou`, `fg_bbox_fill_ratio`, `fg_center_offset_norm`, `fg_plane_inlier_ratio`, `fg_height_above_plane_max`.

## 4. Three-Tier Feature Separation (Zero Test Leakage)

1. **Raw single-sample quality metrics (`quality_metrics`):** Deterministic per-sample measurements.
2. **Descriptive EDA-only within-split robust z-scores (`z_mean_robust_z`, `missing_ratio_robust_z`, `sharpness_robust_z`):** Retained strictly for retrospective EDA diagnostics; **excluded** from `training_ready_samples.parquet`, `mart_training_ready_samples`, and `VistaMultimodalDataset`.
3. **Frozen train-only ML normalization (`normalization_artifacts` + `ml_normalized_features`):**
   - Fitted **exclusively** on eligible `split == 'train'` samples (`validation_status IN ('PASS', 'WARN')`, `is_cross_split_leak = FALSE`).
   - Persisted with `train_manifest_hash`, `params_sha256`, `norm_version`, `config_hash`, and `train_sample_count`.
   - Applied frozen (`0.67448975 * (x - median_train) / MAD_train`, std fallback when `MAD == 0`) identically to `train`, `validation`, and `test`. Modifying test samples or labels cannot change fitted parameters.
