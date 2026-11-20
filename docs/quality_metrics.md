# VISTA Data Quality Metrics — Mathematical Definitions & Limitations

All metrics are computed strictly from raw RGB and organized XYZ inputs without using benchmark test defect labels.

## 1. RGB Appearance Metrics

Let $I \in \{0, \dots, 255\}^{H \times W \times 3}$ be the decoded RGB image ($N = H \cdot W$ pixels), and define Rec.709 luminance:
$$Y(u, v) = 0.2126 R(u, v) + 0.7152 G(u, v) + 0.0722 B(u, v)$$

| Metric | Mathematical Definition | Interpretation & Limitations |
|---|---|---|
| `rgb_brightness_mean` | $\mu_Y = \frac{1}{N} \sum_{u,v} Y(u,v)$ | Global illumination level (0–255). Mixes foreground object and conveyor/table background; dark backgrounds lower $\mu_Y$ regardless of object exposure. |
| `rgb_brightness_std` | $\sigma_Y = \sqrt{\frac{1}{N}\sum_{u,v} (Y(u,v) - \mu_Y)^2}$ | Global intensity dispersion. Sensitive to object-to-background area ratio. |
| `rgb_contrast_rms` | $\sigma_Y / 255.0$ | Normalized RMS contrast in $[0, 0.5]$. Global statistic; does not isolate local texture contrast on the inspected part. |
| `rgb_contrast_p5_p95` | $Q_{0.95}(Y) - Q_{0.05}(Y)$ | Robust dynamic contrast excluding 5% extreme tails (specular highlights / dead pixels). |
| `rgb_laplacian_var` | $\operatorname{Var}\big((\Delta_3 * Y)(u,v)\big)$ using $3\times 3$ discrete Laplacian | Standard focus/sharpness proxy. Sensitive to sensor noise and high-frequency background textures; values are comparable within a category, not directly across unrelated textures. |
| `rgb_tenengrad_mean` | $\frac{1}{N}\sum_{u,v} \big((G_x * Y)^2 + (G_y * Y)^2\big)$ via $3\times 3$ Sobel | Edge-energy sharpness indicator; complements Laplacian variance with lower sensitivity to single-pixel noise. |
| `rgb_saturation_mean` | $\frac{1}{N}\sum_{u,v} S_{\text{HSV}}(u,v) / 255.0$ | Mean chroma saturation in $[0,1]$. Near-zero for metallic/greyscale industrial parts. |
| `rgb_underexposed_ratio` | $\frac{1}{N}\sum_{u,v} \mathbb{1}[Y(u,v) < 15]$ | Fraction of clipped-dark pixels. Can reflect intentional dark matte backgrounds rather than sensor underexposure. |
| `rgb_overexposed_ratio` | $\frac{1}{N}\sum_{u,v} \mathbb{1}[Y(u,v) > 240]$ | Fraction of near-saturated pixels; flags specular glare on metallic/glossy parts. |
| `rgb_dynamic_range` | $\max(Y) - \min(Y)$ | Full observed luminance span; sensitive to single outlier pixels. |

## 2. 3D Point-Cloud & Depth Metrics

Let $P(u,v) = (X(u,v), Y(u,v), Z(u,v)) \in \mathbb{R}^3$ be the organized 3D coordinate map (`xyz/*.tiff`).
Define pixel validity mask $V(u,v)$ as:
$$V(u,v) = \text{finite}(P(u,v)) \land \big(|Z(u,v)| > \epsilon\big) \land \big(\|P(u,v)\|_\infty \le M\big)$$
with defaults $\epsilon = 10^{-6}\text{ m}$ and $M = 100.0\text{ m}$. Let $\mathcal{V} = \{(u,v) : V(u,v) = 1\}$.

| Metric | Mathematical Definition | Interpretation & Limitations |
|---|---|---|
| `valid_point_ratio` | $|\mathcal{V}| / N$ | Fraction of image pixels carrying valid 3D measurements. |
| `missing_point_ratio` | $1 - |\mathcal{V}| / N$ | Total missingness (combines sensor dropouts, occlusions, and zero-filled background). |
| `nan_point_ratio` | $|\{(u,v) : \text{NaN} \in P(u,v)\}| / N$ | Fraction of explicitly NaN-encoded pixels (distinct from `(0,0,0)` zero-filled holes). |
| `depth_coverage_ratio` | $|\{(u,v) : \text{finite}(P) \land |Z| > \epsilon\}| / N$ | Finite non-zero depth coverage prior to coordinate bounds gating. |
| `z_min`, `z_max`, `z_mean`, `z_std`, `z_median`, `z_iqr` | Summary statistics of $\{Z(u,v) : (u,v) \in \mathcal{V}\}$ | Camera-to-surface depth distribution (meters). Includes table plane unless foreground-segmented. |
| `x_span`, `y_span`, `z_span` | $\max - \min$ along each axis over $\mathcal{V}$ | Axis-aligned bounding extents (meters); sensitive to stray flying pixels. |
| `bbox_volume` | $x_{\text{span}} \cdot y_{\text{span}} \cdot z_{\text{span}}$ | Axis-aligned 3D bounding box volume ($\text{m}^3$). |
| `point_density_proxy` | $|\mathcal{V}| / \max(x_{\text{span}} \cdot y_{\text{span}}, 10^{-9})$ | Approximate valid 3D points per $\text{m}^2$ of projected $XY$ area. |

## 3. Unsupervised Outlier Indicators

Within each `(category_name, split)` group (never stratified by test defect labels), robust modified Z-scores are computed via Median Absolute Deviation (MAD):
$$z_{\text{robust}}(x_i) = 0.67448975 \cdot \frac{x_i - \operatorname{median}(x)}{\operatorname{MAD}(x)}$$
- `rgb_quality_outlier_flag`: $|z_{\text{robust}}(\text{laplacian\_var})| > 3.5$
- `geometry_outlier_flag`: $|z_{\text{robust}}(z_{\text{mean}})| > 3.5$ or $|z_{\text{robust}}(\text{missing\_point\_ratio})| > 3.5$

Limitation: For small sample splits or heavily bimodal distributions, MAD can under- or over-flag tail samples; outlier flags are diagnostic signals for inspection, not automated sample deletion criteria.
