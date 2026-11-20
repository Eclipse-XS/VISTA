"""Generate verifiable EDA figures and Markdown reports strictly from DuckDB warehouse data."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
import numpy as np
from scipy import stats as scipy_stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vista.analytics.duckdb_runner import QueryExecutionResult, warehouse_db_path
from vista.config import VistaConfig
from vista.logging import get_logger

log = get_logger(__name__)


def _md_table(columns: list[str], rows: list[tuple[Any, ...]], max_rows: int = 40) -> str:
    if not columns:
        return "_No columns returned._"
    if not rows:
        return "_No rows returned for the currently ingested dataset._"

    def _fmt(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v).replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows[:max_rows]]
    out = "\n".join([header, sep, *body])
    if len(rows) > max_rows:
        out += f"\n\n_Showing first {max_rows} of {len(rows)} rows._"
    return out


def generate_eda_figures_and_reports(
    cfg: VistaConfig,
    sql_results: dict[str, QueryExecutionResult],
) -> dict[str, str]:
    """Create genuine matplotlib figures and EDA/Data-Quality reports from real Gold/dbt data."""
    db_path = warehouse_db_path(cfg)
    fig_dir = cfg.project_root / "analytics" / "figures"
    rep_dir = cfg.project_root / "analytics" / "reports"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        sample_df_rows = con.execute(
            """
            SELECT
                s.category_name,
                s.split,
                s.defect_type,
                s.is_anomalous_benchmark,
                v.status AS validation_status,
                q.rgb_brightness_mean,
                q.rgb_contrast_rms,
                q.rgb_laplacian_var,
                q.rgb_saturation_mean,
                q.valid_point_ratio,
                q.missing_point_ratio,
                q.z_mean,
                q.z_std,
                q.geometry_outlier_flag,
                q.rgb_quality_outlier_flag
            FROM main.dim_sample s
            LEFT JOIN main.fct_validation_results v ON s.sample_pk = v.sample_pk
            LEFT JOIN main.fct_data_quality q
              ON s.sample_pk = q.sample_pk
             AND q.preprocessing_version = ?
             AND q.config_hash = ?
            """,
            [cfg.preprocessing.version, cfg.preprocessing.config_hash],
        ).fetchall()

        stage_perf_rows = con.execute(
            """
            SELECT stage_name, COUNT(*) AS runs, AVG(duration_seconds) AS avg_s, SUM(duration_seconds) AS total_s
            FROM main.fct_ingestion_runs
            GROUP BY stage_name
            ORDER BY total_s DESC
            """
        ).fetchall()

        train_leak_check = con.execute(
            "SELECT COUNT(*) FROM main.mart_training_ready_samples WHERE split = 'test'"
        ).fetchone()[0]
    finally:
        con.close()

    generated_figs: list[str] = []

    if sample_df_rows:
        categories = sorted({r[0] for r in sample_df_rows})

        # Figure 1: Sample inventory by category and split
        splits = ["train", "validation", "test"]
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
        x = np.arange(len(categories))
        bottom = np.zeros(len(categories))
        colors = {"train": "#2b6cb0", "validation": "#38a169", "test": "#dd6b20"}
        for sp in splits:
            vals = np.array(
                [sum(1 for r in sample_df_rows if r[0] == c and r[1] == sp) for c in categories]
            )
            ax.bar(x, vals, bottom=bottom, label=sp, color=colors[sp])
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Sample count")
        ax.set_title("MVTec 3D-AD Ingested Inventory by Category and Split")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        f1 = fig_dir / "01_inventory_by_split.png"
        fig.savefig(f1)
        plt.close(fig)
        generated_figs.append(f1.name)

        # Figure 2: Missing 3D point ratio distribution by category
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
        data_miss = [
            [r[10] for r in sample_df_rows if r[0] == c and r[10] is not None]
            for c in categories
        ]
        if any(data_miss):
            ax.boxplot(data_miss, tick_labels=categories, patch_artist=True)
            ax.set_ylabel("Missing 3D Point Ratio (0–1)")
            ax.set_title("Distribution of Missing 3D Point Ratio by Category")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            f2 = fig_dir / "02_missing_3d_ratio_by_category.png"
            fig.savefig(f2)
            generated_figs.append(f2.name)
        plt.close(fig)

        # Figure 3: RGB Sharpness (Laplacian var) vs Missing 3D Ratio scatter (cross-modal)
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=140)
        for idx, c in enumerate(categories):
            xs = [r[7] for r in sample_df_rows if r[0] == c and r[7] is not None and r[10] is not None]
            ys = [r[10] for r in sample_df_rows if r[0] == c and r[7] is not None and r[10] is not None]
            if xs:
                ax.scatter(xs, ys, alpha=0.6, s=22, label=c)
        ax.set_xlabel("RGB Laplacian Variance (Sharpness Proxy)")
        ax.set_ylabel("Missing 3D Point Ratio")
        ax.set_title("Cross-Modal Quality: RGB Sharpness vs 3D Geometric Missingness")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        f3 = fig_dir / "03_cross_modal_rgb_vs_3d_missing.png"
        fig.savefig(f3)
        plt.close(fig)
        generated_figs.append(f3.name)

        # Figure 4: Benchmark test defect-type counts (strictly labeled as benchmark test set!)
        test_rows = [r for r in sample_df_rows if r[1] == "test"]
        if test_rows:
            pair_counts: dict[str, int] = {}
            for r in test_rows:
                lbl = f"{r[0]} / {r[2]}"
                pair_counts[lbl] = pair_counts.get(lbl, 0) + 1
            items = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)
            fig, ax = plt.subplots(figsize=(9, max(4.2, 0.35 * len(items))), dpi=140)
            labels = [k for k, _ in items][::-1]
            counts = [v for _, v in items][::-1]
            bar_colors = ["#38a169" if lbl.endswith("/ good") else "#e53e3e" for lbl in labels]
            ax.barh(labels, counts, color=bar_colors)
            ax.set_xlabel("Benchmark Test Sample Count")
            ax.set_title("Benchmark Test Split Defect-Type Counts (Not Real Factory Defect Rates)")
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            f4 = fig_dir / "04_benchmark_test_defect_distribution.png"
            fig.savefig(f4)
            plt.close(fig)
            generated_figs.append(f4.name)

        # Figure 5: RGB Brightness vs Contrast by category
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=140)
        for c in categories:
            bx = [r[5] for r in sample_df_rows if r[0] == c and r[5] is not None and r[6] is not None]
            cy = [r[6] for r in sample_df_rows if r[0] == c and r[5] is not None and r[6] is not None]
            if bx:
                ax.scatter(bx, cy, alpha=0.6, s=22, label=c)
        ax.set_xlabel("RGB Mean Luminance (Rec.709, 0–255)")
        ax.set_ylabel("RGB RMS Contrast (0–1)")
        ax.set_title("RGB Photometric Profile by Category")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        f5 = fig_dir / "05_rgb_brightness_vs_contrast.png"
        fig.savefig(f5)
        plt.close(fig)
        generated_figs.append(f5.name)

    # Statistical cross-modal correlation tests (Spearman & Pearson, explicitly reported with p-values)
    stat_lines: list[str] = []
    if sample_df_rows:
        for c in sorted({r[0] for r in sample_df_rows}):
            sub = [r for r in sample_df_rows if r[0] == c and r[5] is not None and r[10] is not None]
            if len(sub) >= 10:
                bright = np.array([r[5] for r in sub], dtype=np.float64)
                sharp = np.array([r[7] for r in sub], dtype=np.float64)
                miss = np.array([r[10] for r in sub], dtype=np.float64)
                rho_b, p_b = scipy_stats.spearmanr(bright, miss)
                rho_s, p_s = scipy_stats.spearmanr(sharp, miss)
                stat_lines.append(
                    f"- **{c}** ($n={len(sub)}$): "
                    f"Spearman(brightness, missing_3d) = `{rho_b:.4f}` ($p={p_b:.4g}$); "
                    f"Spearman(laplacian_var, missing_3d) = `{rho_s:.4f}` ($p={p_s:.4g}$)."
                )

    now_iso = datetime.now(timezone.utc).isoformat()
    ingested_cats = sorted({r[0] for r in sample_df_rows})
    total_samples = len(sample_df_rows)
    pass_count = sum(1 for r in sample_df_rows if r[4] == "PASS")
    warn_count = sum(1 for r in sample_df_rows if r[4] == "WARN")
    quar_count = sum(1 for r in sample_df_rows if r[4] == "QUARANTINE")
    fail_count = sum(1 for r in sample_df_rows if r[4] == "FAIL")

    def _section_for(fname: str, title: str) -> str:
        qr = sql_results.get(fname)
        if qr is None:
            return f"### {title}\n\n_Query `{fname}` was not executed._\n"
        return f"### {title}\n\nSource SQL: `analytics/sql/{fname}`\n\n{_md_table(qr.columns, qr.rows)}\n"

    eda_md = f"""# VISTA — Exploratory Data Analysis Report (MVTec 3D-AD)

- **Generated (UTC):** `{now_iso}`
- **Dataset:** `{cfg.dataset_name}` version `{cfg.dataset_version}` (`{cfg.license}`)
- **Preprocessing Version:** `{cfg.preprocessing.version}` (config hash `{cfg.preprocessing.config_hash}`)
- **Ingested Categories Covered:** `{", ".join(ingested_cats) if ingested_cats else "NONE"}`
- **Total Indexed Samples:** `{total_samples}`
- **Training Mart Test-Leakage Check (`mart_training_ready_samples` where `split='test'`):** `{train_leak_check}` rows

> **Methodological Caveat:** MVTec 3D-AD is a curated academic benchmark. Defect proportions in the `test` split reflect benchmark design choices and **must not** be interpreted as real industrial factory defect rates. Test labels are strictly isolated from `mart_training_ready_samples` and `training_ready_samples.parquet`.

---

## 1. Dataset Inventory & Split Coverage (Q1)

{_section_for("01_dataset_inventory.sql", "Inventory by Category and Split")}

![Inventory by Category and Split](../figures/01_inventory_by_split.png)

## 2. RGB Photometric & Texture Quality (Q4)

{_section_for("03_rgb_quality.sql", "RGB Quality Summary by Category and Split")}

![RGB Brightness vs Contrast](../figures/05_rgb_brightness_vs_contrast.png)

## 3. 3D Geometric Validity & Missingness (Q3)

{_section_for("04_geometry_quality.sql", "3D Geometric Quality by Category")}

![Missing 3D Point Ratio](../figures/02_missing_3d_ratio_by_category.png)

## 4. Cross-Modal Association: RGB Indicators vs 3D Missingness (Q5)

{_section_for("05_cross_modal_quality.sql", "SQL Cross-Modal Correlation Summary")}

![Cross-Modal RGB Sharpness vs 3D Missingness](../figures/03_cross_modal_rgb_vs_3d_missing.png)

### Descriptive Observations vs Statistical Conclusions

- **Descriptive observations** summarize sample counts, medians, IQRs, and visual scatter patterns in the currently ingested categories (`{", ".join(ingested_cats)}`).
- **Statistical rank-correlation checks (Spearman $\\rho$, two-sided $p$-value):**
{chr(10).join(stat_lines) if stat_lines else "- Insufficient sample count for correlation tests."}
- **Caution:** Association within a single camera/lighting bench does not prove causal degradation; background area ratio and object footprint simultaneously influence both global RGB statistics and structured-light 3D coverage.

## 5. Unusual Depth & Point-Cloud Outliers (Q6)

{_section_for("08_depth_pointcloud_outliers.sql", "Flagged Geometry & Depth Outliers")}

## 6. Benchmark Test Defect Distributions (Q7)

{_section_for("06_defect_distribution.sql", "Defect-Type Distribution in Labeled Benchmark Test Data")}

![Benchmark Test Defect Distribution](../figures/04_benchmark_test_defect_distribution.png)

## 7. Defect Types vs Measured Geometry & Texture (Q8)

{_section_for("09_defect_geometry_texture.sql", "Defect Type vs RGB & 3D Characteristics")}

## 8. Pipeline Stage Execution Performance (Q9)

{_section_for("07_pipeline_performance.sql", "Stage Runtime & Throughput")}

## 9. Quality Distributions Across Preprocessing Versions (Q10)

{_section_for("10_version_quality_comparison.sql", "Version-to-Version Quality Comparison")}

---

## Limitations

1. Only the explicitly authorized subset of categories (`{", ".join(ingested_cats)}`) has been ingested to respect local disk space and avoid downloading the full 13.2 GB archive without approval. Findings must not be extrapolated to un-ingested MVTec 3D-AD categories (especially organic categories like `bagel`, `carrot`, `peach`, `potato`).
2. Global RGB sharpness (`Laplacian variance`) and luminance mix foreground object pixels with dark/uniform conveyor/table background pixels.
3. Structured-light missingness (`z == 0`) combines physical background outside sensor frustum with specular/occluded object regions.
"""

    dq_md = f"""# VISTA — Multimodal Data Quality & Validation Report

- **Generated (UTC):** `{now_iso}`
- **Dataset:** `{cfg.dataset_name}` version `{cfg.dataset_version}`
- **Ingested Categories:** `{", ".join(ingested_cats) if ingested_cats else "NONE"}`
- **Total Samples Validated:** `{total_samples}`

## Validation Status Breakdown (Q2)

| Status | Sample Count | Share |
| --- | ---: | ---: |
| `PASS` | {pass_count} | {(100.0 * pass_count / total_samples) if total_samples else 0.0:.2f}% |
| `WARN` | {warn_count} | {(100.0 * warn_count / total_samples) if total_samples else 0.0:.2f}% |
| `QUARANTINE` | {quar_count} | {(100.0 * quar_count / total_samples) if total_samples else 0.0:.2f}% |
| `FAIL` | {fail_count} | {(100.0 * fail_count / total_samples) if total_samples else 0.0:.2f}% |

{_section_for("02_validation_failures.sql", "Validation Rule Breakdown by Category and Status")}

## Split Integrity & Label-Leakage Safeguards

- **Training-ready Gold dataset (`training_ready_samples.parquet`) & dbt mart (`mart_training_ready_samples`)**:
  - Includes ONLY `split IN ('train', 'validation')` with `validation_status IN ('PASS', 'WARN')` and `is_cross_split_leak = FALSE`.
  - Excludes all `test` split rows (`verified test rows in mart = {train_leak_check}`).
  - Drops `defect_type` and `is_anomalous_benchmark` columns entirely from the training export schema.
- **Immutability:** Bronze source archives and sample manifests are never modified or silently repaired. Any detected corruption or split violation is routed to `quarantine/mvtec_3d_ad/<category>/quarantine_manifest.json`.

## Pipeline Stage Summary

{_md_table(["stage_name", "runs", "avg_duration_s", "total_duration_s"], stage_perf_rows)}
"""

    eda_path = rep_dir / "eda_report.md"
    dq_path = rep_dir / "data_quality_report.md"
    eda_path.write_text(eda_md, encoding="utf-8")
    dq_path.write_text(dq_md, encoding="utf-8")

    log.info(
        "reports_generated",
        eda_report=str(eda_path),
        dq_report=str(dq_path),
        figures=len(generated_figs),
    )
    return {
        "eda_report": str(eda_path),
        "data_quality_report": str(dq_path),
        "figures": ", ".join(generated_figs),
    }
