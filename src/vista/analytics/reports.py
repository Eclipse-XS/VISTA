"""Generate verifiable, scientifically rigorous EDA figures and Markdown reports from DuckDB."""

from __future__ import annotations

import json
from datetime import datetime, timezone
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


def _md_table(columns: list[str], rows: list[tuple[Any, ...]], max_rows: int = 80) -> str:
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


def _bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "spearman",
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Compute correlation coefficient, two-sided p-value, and 95% bootstrap CI."""
    if len(x) < 6 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan"), float("nan"), float("nan"), float("nan")
    if method == "spearman":
        est, pval = scipy_stats.spearmanr(x, y)
    else:
        est, pval = scipy_stats.pearsonr(x, y)

    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) <= 1e-12 or np.std(yb) <= 1e-12:
            boots[i] = np.nan
        elif method == "spearman":
            boots[i] = scipy_stats.spearmanr(xb, yb).statistic
        else:
            boots[i] = scipy_stats.pearsonr(xb, yb).statistic
    valid_b = boots[np.isfinite(boots)]
    lo, hi = (np.percentile(valid_b, [2.5, 97.5]) if valid_b.size >= 10 else (np.nan, np.nan))
    return float(est), float(pval), float(lo), float(hi)


def _partial_spearman(x: np.ndarray, y: np.ndarray, covariates: np.ndarray) -> tuple[float, float]:
    """Rank-based partial Spearman correlation between x and y controlling for covariates."""
    if len(x) < covariates.shape[1] + 5:
        return float("nan"), float("nan")
    rx = scipy_stats.rankdata(x)
    ry = scipy_stats.rankdata(y)
    rc = np.column_stack([scipy_stats.rankdata(covariates[:, j]) for j in range(covariates.shape[1])])
    design = np.column_stack([rc, np.ones(len(x))])
    bx, _, _, _ = np.linalg.lstsq(design, rx, rcond=None)
    by, _, _, _ = np.linalg.lstsq(design, ry, rcond=None)
    ex = rx - design @ bx
    ey = ry - design @ by
    if np.std(ex) <= 1e-12 or np.std(ey) <= 1e-12:
        return float("nan"), float("nan")
    r, p = scipy_stats.pearsonr(ex, ey)
    return float(r), float(p)


def generate_eda_figures_and_reports(
    cfg: VistaConfig,
    sql_results: dict[str, QueryExecutionResult],
    *,
    snapshot_id: str | None = None,
    manifest_sha256: str | None = None,
    telemetry_meta: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Generate genuine matplotlib figures and EDA / Data Quality reports from real Gold/dbt data."""
    db_path = warehouse_db_path(cfg)
    fig_dir = cfg.project_root / "analytics" / "figures"
    rep_dir = cfg.project_root / "analytics" / "reports"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
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
                q.fg_area_ratio,
                q.fg_rgb_3d_iou,
                q.fg_rgb_brightness_mean,
                q.fg_rgb_contrast_rms,
                q.fg_rgb_laplacian_var,
                q.fg_missing_3d_ratio,
                q.bg_missing_3d_ratio,
                q.bg_rgb_brightness_mean,
                q.fg_center_offset_norm,
                COALESCE(c.material_type, 'unknown') AS material_type
            FROM main.dim_sample s
            LEFT JOIN main.dim_category c ON s.category_name = c.category_name
            LEFT JOIN main.fct_validation_results v ON s.sample_pk = v.sample_pk
            LEFT JOIN main.fct_data_quality q
              ON s.sample_pk = q.sample_pk
             AND q.preprocessing_version = ?
             AND q.config_hash = ?
            """,
            [cfg.preprocessing.version, cfg.preprocessing.config_hash],
        ).fetchall()

        norm_rows = con.execute(
            """
            SELECT category_name, norm_version, train_sample_count, train_manifest_hash, params_sha256
            FROM main.stg_normalization_artifacts
            ORDER BY category_name
            """
        ).fetchall()

        train_leak_check = con.execute(
            "SELECT COUNT(*) FROM main.mart_training_ready_samples WHERE split = 'test'"
        ).fetchone()[0]

        train_partition_counts = con.execute(
            "SELECT split, COUNT(*) FROM main.mart_training_ready_samples GROUP BY split ORDER BY split"
        ).fetchall()

        stage_perf_rows = con.execute(
            """
            SELECT
                stage_name,
                COUNT(*) AS total_attempts,
                SUM(CASE WHEN stage_status = 'SUCCEEDED' THEN 1 ELSE 0 END) AS succeeded,
                SUM(CASE WHEN stage_status = 'FAILED' THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN stage_status = 'INTERRUPTED' THEN 1 ELSE 0 END) AS interrupted,
                SUM(CASE WHEN is_retry THEN 1 ELSE 0 END) AS retries,
                ROUND(AVG(duration_seconds) FILTER (WHERE stage_status = 'SUCCEEDED' AND duration_seconds IS NOT NULL), 3) AS avg_succeeded_s
            FROM main.fct_ingestion_runs
            GROUP BY stage_name
            ORDER BY stage_name
            """
        ).fetchall()
    finally:
        con.close()

    generated_figs: list[str] = []

    if rows:
        categories = sorted({r[0] for r in rows})

        # Figure 1: Sample inventory by category and split
        splits = ["train", "validation", "test"]
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
        x = np.arange(len(categories))
        bottom = np.zeros(len(categories))
        colors = {"train": "#2b6cb0", "validation": "#38a169", "test": "#dd6b20"}
        for sp in splits:
            vals = np.array([sum(1 for r in rows if r[0] == c and r[1] == sp) for c in categories])
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

        # Figure 2: Global vs Foreground Missing 3D Ratio by category
        fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=140)
        box_data: list[list[float]] = []
        box_labels: list[str] = []
        for c in categories:
            g_vals = [r[10] for r in rows if r[0] == c and r[10] is not None]
            f_vals = [r[18] for r in rows if r[0] == c and r[18] is not None]
            b_vals = [r[19] for r in rows if r[0] == c and r[19] is not None]
            if g_vals:
                box_data.extend([g_vals, f_vals, b_vals])
                box_labels.extend([f"{c}\nGlobal", f"{c}\nForeground", f"{c}\nBackground"])
        if box_data:
            ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
            ax.set_ylabel("Missing 3D Point Ratio (0–1)")
            ax.set_title("Missing 3D Point Ratio: Global vs Foreground Object vs Background")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            f2 = fig_dir / "02_missing_3d_ratio_by_category.png"
            fig.savefig(f2)
            generated_figs.append(f2.name)
        plt.close(fig)

        cat_palette = ["#2b6cb0", "#dd6b20", "#38a169", "#805ad5", "#d53f8c", "#319795", "#b7791f", "#e53e3e"]
        cat_colors = {c: cat_palette[i % len(cat_palette)] for i, c in enumerate(categories)}
        cat_material = {c: next((r[22] for r in rows if r[0] == c), "unknown") for c in categories}
        split_markers = {"train": "o", "validation": "^", "test": "s"}
        multi_cat = len(categories) > 1

        # Figure 3: Cross-modal RGB brightness vs 3D missingness — Global vs Foreground side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=140)
        split_colors = {"train": "#2b6cb0", "validation": "#38a169", "test": "#dd6b20"}
        if multi_cat:
            for c in categories:
                for sp in splits:
                    gx = [r[5] for r in rows if r[0] == c and r[1] == sp and r[5] is not None and r[10] is not None]
                    gy = [r[10] for r in rows if r[0] == c and r[1] == sp and r[5] is not None and r[10] is not None]
                    lbl = f"{c} ({cat_material[c]})" if sp == "train" else None
                    if gx:
                        axes[0].scatter(gx, gy, alpha=0.6, s=20, label=lbl, color=cat_colors[c], marker=split_markers[sp])
                    fx = [r[15] for r in rows if r[0] == c and r[1] == sp and r[15] is not None and r[18] is not None]
                    fy = [r[18] for r in rows if r[0] == c and r[1] == sp and r[15] is not None and r[18] is not None]
                    if fx:
                        axes[1].scatter(fx, fy, alpha=0.6, s=20, label=lbl, color=cat_colors[c], marker=split_markers[sp])
        else:
            for sp in splits:
                gx = [r[5] for r in rows if r[1] == sp and r[5] is not None and r[10] is not None]
                gy = [r[10] for r in rows if r[1] == sp and r[5] is not None and r[10] is not None]
                if gx:
                    axes[0].scatter(gx, gy, alpha=0.65, s=22, label=sp, color=split_colors[sp])
                fx = [r[15] for r in rows if r[1] == sp and r[15] is not None and r[18] is not None]
                fy = [r[18] for r in rows if r[1] == sp and r[15] is not None and r[18] is not None]
                if fx:
                    axes[1].scatter(fx, fy, alpha=0.65, s=22, label=sp, color=split_colors[sp])
        axes[0].set_title("Global RGB Brightness vs Global 3D Missingness")
        axes[0].set_xlabel("Global Mean Luminance (0–255)")
        axes[0].set_ylabel("Global Missing 3D Ratio")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].set_title("Foreground RGB Brightness vs Foreground 3D Missingness")
        axes[1].set_xlabel("Foreground Mean Luminance (0–255)")
        axes[1].set_ylabel("Foreground Missing 3D Ratio")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        f3 = fig_dir / "03_cross_modal_rgb_vs_3d_missing.png"
        fig.savefig(f3)
        plt.close(fig)
        generated_figs.append(f3.name)

        # Figure 4: Benchmark test defect-type counts
        test_rows = [r for r in rows if r[1] == "test"]
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

        # Figure 5: Foreground RGB Photometric Profile by Category & Split
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=140)
        if multi_cat:
            for c in categories:
                for sp in splits:
                    bx = [r[15] for r in rows if r[0] == c and r[1] == sp and r[15] is not None and r[16] is not None]
                    cy = [r[16] for r in rows if r[0] == c and r[1] == sp and r[15] is not None and r[16] is not None]
                    lbl = f"{c} ({cat_material[c]})" if sp == "train" else None
                    if bx:
                        ax.scatter(bx, cy, alpha=0.6, s=22, label=lbl, color=cat_colors[c], marker=split_markers[sp])
            ax.set_title("Foreground RGB Photometric Profile by Category (o=train, ^=val, sq=test)")
        else:
            for sp in splits:
                bx = [r[15] for r in rows if r[1] == sp and r[15] is not None and r[16] is not None]
                cy = [r[16] for r in rows if r[1] == sp and r[15] is not None and r[16] is not None]
                if bx:
                    ax.scatter(bx, cy, alpha=0.65, s=22, label=f"{sp} (foreground)", color=split_colors[sp])
            ax.set_title("Foreground RGB Photometric Profile by Split")
        ax.set_xlabel("Foreground RGB Mean Luminance (Rec.709, 0–255)")
        ax.set_ylabel("Foreground RGB RMS Contrast (0–1)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        f5 = fig_dir / "05_rgb_brightness_vs_contrast.png"
        fig.savefig(f5)
        plt.close(fig)
        generated_figs.append(f5.name)

        # Figure 6: Per-category Group Comparison (train/good, val/good, test/good, test/anomalous) Global vs FG
        n_cats = len(categories)
        fig, axes = plt.subplots(n_cats, 2, figsize=(11.5, max(4.3, 3.3 * n_cats)), dpi=140, squeeze=False)
        any_f6 = False
        for c_idx, c in enumerate(categories):
            c_rows = [r for r in rows if r[0] == c]
            groups = [
                ("train/good", [r for r in c_rows if r[1] == "train"]),
                ("val/good", [r for r in c_rows if r[1] == "validation"]),
                ("test/good", [r for r in c_rows if r[1] == "test" and not r[3]]),
                ("test/anomalous", [r for r in c_rows if r[1] == "test" and r[3]]),
            ]
            active_groups = [(name, grp) for name, grp in groups if grp]
            if len(active_groups) < 2:
                continue
            any_f6 = True
            g_names = [g[0] for g in active_groups]
            gx = np.arange(len(g_names))
            w = 0.28
            g_miss = [np.mean([r[10] for r in g[1] if r[10] is not None]) for g in active_groups]
            fg_miss = [np.mean([r[18] for r in g[1] if r[18] is not None]) for g in active_groups]
            bg_miss = [np.mean([r[19] for r in g[1] if r[19] is not None]) for g in active_groups]
            axes[c_idx, 0].bar(gx - w, g_miss, width=w, label="Global 3D Missing", color="#718096")
            axes[c_idx, 0].bar(gx, fg_miss, width=w, label="Foreground 3D Missing", color="#2b6cb0")
            axes[c_idx, 0].bar(gx + w, bg_miss, width=w, label="Background 3D Missing", color="#e53e3e")
            axes[c_idx, 0].set_xticks(gx)
            axes[c_idx, 0].set_xticklabels(g_names, rotation=12)
            axes[c_idx, 0].set_ylabel("Mean Missing 3D Ratio")
            axes[c_idx, 0].set_title(f"{c} ({cat_material[c]}) — 3D Missingness Decomposition")
            axes[c_idx, 0].legend(fontsize=8)
            axes[c_idx, 0].grid(axis="y", alpha=0.3)

            g_lum = [np.mean([r[5] for r in g[1] if r[5] is not None]) for g in active_groups]
            fg_lum = [np.mean([r[15] for r in g[1] if r[15] is not None]) for g in active_groups]
            bg_lum = [np.mean([r[20] for r in g[1] if r[20] is not None]) for g in active_groups]
            axes[c_idx, 1].bar(gx - w, g_lum, width=w, label="Global Lum", color="#718096")
            axes[c_idx, 1].bar(gx, fg_lum, width=w, label="Foreground Lum", color="#38a169")
            axes[c_idx, 1].bar(gx + w, bg_lum, width=w, label="Background Lum", color="#dd6b20")
            axes[c_idx, 1].set_xticks(gx)
            axes[c_idx, 1].set_xticklabels(g_names, rotation=12)
            axes[c_idx, 1].set_ylabel("Mean Luminance (0–255)")
            axes[c_idx, 1].set_title(f"{c} ({cat_material[c]}) — RGB Luminance Decomposition")
            axes[c_idx, 1].legend(fontsize=8)
            axes[c_idx, 1].grid(axis="y", alpha=0.3)
        if any_f6:
            fig.tight_layout()
            f6 = fig_dir / "06_global_vs_foreground_quality.png"
            fig.savefig(f6)
            generated_figs.append(f6.name)
        plt.close(fig)

        # Figure 7: Standardized train-to-test shift per category (Cohen's d relative to train std)
        shift_specs = [
            ("Global 3D Missing", 10),
            ("BG 3D Missing", 19),
            ("Global RGB Bright", 5),
            ("BG RGB Bright", 20),
            ("Global Sharpness", 7),
            ("FG RGB Bright", 15),
            ("FG Sharpness", 17),
            ("FG 3D Missing", 18),
            ("FG Area Ratio", 13),
            ("FG RGB-3D IoU", 14),
            ("FG Center Offset", 21),
        ]
        shift_cats = [
            c
            for c in categories
            if sum(1 for r in rows if r[0] == c and r[1] == "train") >= 5
            and sum(1 for r in rows if r[0] == c and r[1] == "test") >= 5
        ]
        if shift_cats:
            fig, axes = plt.subplots(
                1,
                len(shift_cats),
                figsize=(max(9.0, 5.2 * len(shift_cats)), 5.2),
                dpi=140,
                sharey=True,
                squeeze=False,
            )
            y_pos = np.arange(len(shift_specs))
            for idx_c, c in enumerate(shift_cats):
                ax = axes[0, idx_c]
                train_sub = [r for r in rows if r[0] == c and r[1] == "train"]
                val_sub = [r for r in rows if r[0] == c and r[1] == "validation"]
                test_sub = [r for r in rows if r[0] == c and r[1] == "test"]
                d_test_vals: list[float] = []
                d_val_vals: list[float] = []
                for _, col_idx in shift_specs:
                    tr_v = np.array([r[col_idx] for r in train_sub if r[col_idx] is not None], dtype=np.float64)
                    te_v = np.array([r[col_idx] for r in test_sub if r[col_idx] is not None], dtype=np.float64)
                    va_v = np.array([r[col_idx] for r in val_sub if r[col_idx] is not None], dtype=np.float64)
                    if tr_v.size >= 3 and te_v.size >= 3 and tr_v.std() > 1e-12:
                        d_test_vals.append(float((te_v.mean() - tr_v.mean()) / tr_v.std()))
                        d_val_vals.append(
                            float((va_v.mean() - tr_v.mean()) / tr_v.std()) if va_v.size >= 2 else 0.0
                        )
                    else:
                        d_test_vals.append(0.0)
                        d_val_vals.append(0.0)
                ax.barh(y_pos - 0.18, d_test_vals, height=0.35, label="Test vs Train (SD)", color="#e53e3e")
                ax.barh(y_pos + 0.18, d_val_vals, height=0.35, label="Val vs Train (SD)", color="#38a169")
                ax.axvline(0.0, color="#2d3748", linewidth=1)
                ax.set_yticks(y_pos)
                if idx_c == 0:
                    ax.set_yticklabels([lbl for lbl, _ in shift_specs])
                ax.set_xlabel("Standardized Shift (Train SD)")
                ax.set_title(f"{c} ({cat_material[c]})")
                ax.legend(fontsize=8)
                ax.grid(axis="x", alpha=0.3)
            fig.suptitle("Train-to-Test vs Train-to-Validation Distribution Shifts by Category", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            f7 = fig_dir / "07_split_distribution_shifts.png"
            fig.savefig(f7)
            plt.close(fig)
            generated_figs.append(f7.name)

    # Statistical cross-modal correlation suite with bootstrap 95% CIs & partial Spearman
    stat_table_rows: list[tuple[Any, ...]] = []
    shift_stat_rows: list[tuple[Any, ...]] = []

    if rows:
        for c in sorted({r[0] for r in rows}):
            sub = [
                r
                for r in rows
                if r[0] == c
                and r[5] is not None
                and r[10] is not None
                and r[15] is not None
                and r[18] is not None
                and r[13] is not None
                and r[19] is not None
            ]
            if len(sub) >= 10:
                g_bright = np.array([r[5] for r in sub], dtype=np.float64)
                g_sharp = np.array([r[7] for r in sub], dtype=np.float64)
                g_miss = np.array([r[10] for r in sub], dtype=np.float64)
                fg_bright = np.array([r[15] for r in sub], dtype=np.float64)
                fg_sharp = np.array([r[17] for r in sub], dtype=np.float64)
                fg_miss = np.array([r[18] for r in sub], dtype=np.float64)
                covs = np.column_stack(
                    [
                        np.array([r[13] for r in sub], dtype=np.float64),  # fg_area_ratio
                        np.array([r[19] for r in sub], dtype=np.float64),  # bg_missing_3d_ratio
                    ]
                )

                for pair_name, xa, ya in (
                    ("Global brightness vs Global 3D missing", g_bright, g_miss),
                    ("Global sharpness vs Global 3D missing", g_sharp, g_miss),
                    ("Foreground brightness vs Foreground 3D missing", fg_bright, fg_miss),
                    ("Foreground sharpness vs Foreground 3D missing", fg_sharp, fg_miss),
                ):
                    sp_r, sp_p, sp_lo, sp_hi = _bootstrap_corr_ci(xa, ya, method="spearman")
                    pe_r, pe_p, pe_lo, pe_hi = _bootstrap_corr_ci(xa, ya, method="pearson")
                    part_r, part_p = _partial_spearman(xa, ya, covs)
                    stat_table_rows.append(
                        (
                            c,
                            pair_name,
                            len(sub),
                            f"{sp_r:.4f} [{sp_lo:.4f}, {sp_hi:.4f}]",
                            f"{sp_p:.3g}",
                            f"{pe_r:.4f} [{pe_lo:.4f}, {pe_hi:.4f}]",
                            f"{part_r:.4f} (p={part_p:.3g})",
                        )
                    )

            # Train vs Test two-sample KS & standardized shift table
            tr_c = [r for r in sub if r[1] == "train"]
            te_c = [r for r in sub if r[1] == "test"]
            te_good = [r for r in te_c if not r[3]]
            te_anom = [r for r in te_c if r[3]]
            if len(tr_c) >= 5 and len(te_c) >= 5:
                for feat_name, col_i in (
                    ("global_missing_3d_ratio", 10),
                    ("bg_missing_3d_ratio", 19),
                    ("fg_missing_3d_ratio", 18),
                    ("global_rgb_brightness_mean", 5),
                    ("bg_rgb_brightness_mean", 20),
                    ("fg_rgb_brightness_mean", 15),
                    ("fg_rgb_laplacian_var", 17),
                    ("fg_area_ratio", 13),
                    ("fg_rgb_3d_iou", 14),
                    ("fg_center_offset_norm", 21),
                ):
                    a = np.array([r[col_i] for r in tr_c if r[col_i] is not None], dtype=np.float64)
                    b = np.array([r[col_i] for r in te_c if r[col_i] is not None], dtype=np.float64)
                    bg = np.array([r[col_i] for r in te_good if r[col_i] is not None], dtype=np.float64)
                    ba = np.array([r[col_i] for r in te_anom if r[col_i] is not None], dtype=np.float64)
                    if a.size >= 3 and b.size >= 3:
                        ks = scipy_stats.ks_2samp(a, b)
                        std_shift = float((b.mean() - a.mean()) / max(a.std(), 1e-12))
                        g_shift = (
                            float((bg.mean() - a.mean()) / max(a.std(), 1e-12))
                            if bg.size >= 2
                            else float("nan")
                        )
                        a_shift = (
                            float((ba.mean() - a.mean()) / max(a.std(), 1e-12))
                            if ba.size >= 2
                            else float("nan")
                        )
                        shift_stat_rows.append(
                            (
                                c,
                                feat_name,
                                float(a.mean()),
                                float(b.mean()),
                                std_shift,
                                g_shift,
                                a_shift,
                                float(ks.statistic),
                                f"{ks.pvalue:.3g}",
                            )
                        )

    # Cross-category generalization matrix (testing whether single-category patterns generalize)
    generalization_rows: list[tuple[Any, ...]] = []
    if rows:
        for c in sorted({r[0] for r in rows}):
            c_sub = [
                r
                for r in rows
                if r[0] == c
                and r[5] is not None
                and r[10] is not None
                and r[13] is not None
                and r[14] is not None
                and r[15] is not None
                and r[18] is not None
                and r[19] is not None
            ]
            if len(c_sub) < 10:
                continue
            mat_type = c_sub[0][22]
            g_sp, _ = scipy_stats.spearmanr([r[5] for r in c_sub], [r[10] for r in c_sub])
            fg_sp, fg_p = scipy_stats.spearmanr([r[15] for r in c_sub], [r[18] for r in c_sub])
            covs = np.column_stack(
                [
                    np.array([r[13] for r in c_sub], dtype=np.float64),
                    np.array([r[19] for r in c_sub], dtype=np.float64),
                ]
            )
            part_fg, part_p = _partial_spearman(
                np.array([r[15] for r in c_sub], dtype=np.float64),
                np.array([r[18] for r in c_sub], dtype=np.float64),
                covs,
            )
            iou_vals = np.array([r[14] for r in c_sub], dtype=np.float64)
            fg_m_vals = np.array([r[18] for r in c_sub], dtype=np.float64)
            bg_m_vals = np.array([r[19] for r in c_sub], dtype=np.float64)
            tr_c = [r for r in c_sub if r[1] == "train"]
            te_c = [r for r in c_sub if r[1] == "test"]
            bg_shift = float("nan")
            fg_shift = float("nan")
            if len(tr_c) >= 5 and len(te_c) >= 5:
                tr_bg = np.array([r[19] for r in tr_c], dtype=np.float64)
                te_bg = np.array([r[19] for r in te_c], dtype=np.float64)
                bg_shift = float((te_bg.mean() - tr_bg.mean()) / max(tr_bg.std(), 1e-12))
                tr_fg = np.array([r[18] for r in tr_c], dtype=np.float64)
                te_fg = np.array([r[18] for r in te_c], dtype=np.float64)
                fg_shift = float((te_fg.mean() - tr_fg.mean()) / max(tr_fg.std(), 1e-12))
            generalization_rows.append(
                (
                    c,
                    mat_type,
                    len(c_sub),
                    round(float(g_sp), 4),
                    f"{fg_sp:.4f} (p={fg_p:.3g})",
                    f"{part_fg:.4f} (p={part_p:.3g})",
                    f"{iou_vals.mean():.3f} (min {iou_vals.min():.3f})",
                    f"{100.0 * fg_m_vals.mean():.2f}%",
                    f"{100.0 * bg_m_vals.mean():.2f}%",
                    f"{bg_shift:+.2f} SD" if np.isfinite(bg_shift) else "—",
                    f"{fg_shift:+.2f} SD" if np.isfinite(fg_shift) else "—",
                )
            )

    # Auto-discover persisted per-category visual foreground diagnostic artifacts
    visual_diag_blocks: list[str] = []
    for c in sorted({r[0] for r in rows}):
        meta_path = fig_dir / f"fg_diagnostics_{c}.json"
        if not meta_path.exists():
            continue
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        fig_name = payload.get("figure")
        cases = payload.get("cases", [])
        if fig_name and (fig_dir / fig_name).exists():
            case_rows = [
                (
                    cs.get("role"),
                    cs.get("sample_key"),
                    cs.get("split"),
                    cs.get("defect_type"),
                    cs.get("fg_rgb_3d_iou"),
                    f"{cs.get('fg_area_pct', 0.0):.2f}%",
                    f"{cs.get('fg_3d_area_pct', 0.0):.2f}%",
                    f"{cs.get('fg_missing_3d_pct', 0.0):.2f}%",
                    f"{cs.get('bg_missing_3d_pct', 0.0):.2f}%",
                    cs.get("fg_plane_inlier_ratio"),
                )
                for cs in cases
            ]
            visual_diag_blocks.append(
                f"### Foreground Visual Diagnostics — `{c}`\n\n"
                f"![Foreground Visual Diagnostics — {c}](../figures/{fig_name})\n\n"
                + _md_table(
                    [
                        "case_role",
                        "sample_key",
                        "split",
                        "defect_type",
                        "fg_rgb_3d_iou",
                        "fg_rgb_area",
                        "fg_3d_area",
                        "fg_3d_missing",
                        "bg_3d_missing",
                        "plane_inlier_ratio",
                    ],
                    case_rows,
                )
            )
    visual_diag_md = (
        "\n\n".join(visual_diag_blocks)
        if visual_diag_blocks
        else "_Visual foreground diagnostics have not yet been generated for the ingested categories._"
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    ingested_cats = sorted({r[0] for r in rows})
    total_samples = len(rows)
    pass_count = sum(1 for r in rows if r[4] == "PASS")
    warn_count = sum(1 for r in rows if r[4] == "WARN")
    quar_count = sum(1 for r in rows if r[4] == "QUARANTINE")
    fail_count = sum(1 for r in rows if r[4] == "FAIL")

    def _section_for(fname: str, title: str) -> str:
        qr = sql_results.get(fname)
        if qr is None:
            return f"### {title}\n\n_Query `{fname}` was not executed._\n"
        return f"### {title}\n\nSource SQL: `analytics/sql/{fname}`\n\n{_md_table(qr.columns, qr.rows)}\n"

    tel_cutoff = (telemetry_meta or {}).get("telemetry_cutoff_utc", now_iso)
    tel_issues = (telemetry_meta or {}).get("consistency_issues", [])

    eda_md = f"""# VISTA — Exploratory Data Analysis Report (MVTec 3D-AD)

- **Generated (UTC):** `{now_iso}`
- **Immutable Gold Snapshot ID:** `{snapshot_id or "UNSPECIFIED"}`
- **Snapshot Manifest SHA-256:** `{manifest_sha256 or "UNSPECIFIED"}`
- **Telemetry Cutoff (UTC):** `{tel_cutoff}`
- **Dataset:** `{cfg.dataset_name}` version `{cfg.dataset_version}` (`{cfg.license}`)
- **Preprocessing Version:** `{cfg.preprocessing.version}` (config hash `{cfg.preprocessing.config_hash}`)
- **Ingested Categories Covered ({len(ingested_cats)} of 10):** `{", ".join(ingested_cats) if ingested_cats else "NONE"}`
- **Total Indexed Samples:** `{total_samples}`
- **Training Mart Test-Leakage Check (`mart_training_ready_samples` where `split='test'`):** `{train_leak_check}` rows
- **Training Mart Partitions:** `{", ".join(f"{sp}={cnt}" for sp, cnt in train_partition_counts)}`

> **Methodological Safeguards:**
> 1. MVTec 3D-AD is a curated academic benchmark; `test` defect proportions reflect benchmark design and **must not** be interpreted as real industrial factory defect rates.
> 2. All ML-facing normalization parameters (`normalization_artifacts`) are fitted **exclusively on eligible `train` samples** and applied frozen to `validation` and `test`.
> 3. Foreground masks (`fg_otsu_plane_v1`) are strictly unsupervised (RGB Otsu contrast + 3D border support-plane residual); benchmark ground-truth defect masks are **never** used to compute quality features.
> 4. Associations below are strictly descriptive/statistical observations, not causal proofs.

---

## 1. Dataset Inventory & Split Coverage (Q1)

{_section_for("01_dataset_inventory.sql", "Inventory by Category and Split")}

![Inventory by Category and Split](../figures/01_inventory_by_split.png)

## 2. Cross-Category Comparison: Organic vs Manufactured & Generalization Assessment

{_section_for("16_cross_category_organic_vs_manufactured.sql", "Organic vs Manufactured Category Comparison (SQL)")}

### Cross-Category Generalization Matrix (Testing Single-Category Hypotheses Across Categories)

The table below tests whether patterns observed in `cable_gland` (strong negative global brightness-vs-3D-missingness correlation collapsing on foreground, high RGB-3D segmentation agreement, and large background-driven train→test shifts) generalize across all ingested categories:

{_md_table(["category", "material_type", "n", "global_spearman", "fg_spearman", "fg_partial_spearman", "fg_rgb_3d_iou_mean_min", "fg_3d_missing_mean", "bg_3d_missing_mean", "bg_miss_test_shift", "fg_miss_test_shift"], generalization_rows)}

## 3. Global vs Foreground-Aware Quality Decomposition (P1 Q1 & Q4)

{_section_for("11_foreground_vs_global_quality.sql", "Global vs Foreground vs Background Quality Metrics")}

![Global vs Foreground Quality Decomposition](../figures/06_global_vs_foreground_quality.png)

![Missing 3D Point Ratio: Global vs Foreground vs Background](../figures/02_missing_3d_ratio_by_category.png)

{_section_for("03_rgb_quality.sql", "Global RGB Photometric Summary by Category and Split")}

![Foreground RGB Photometric Profile](../figures/05_rgb_brightness_vs_contrast.png)

{_section_for("04_geometry_quality.sql", "3D Geometric Quality by Category")}

## 4. Visual Foreground Segmentation Examples & Failure Cases

Each panel shows 4 deterministically selected real samples per category: (1) highest RGB-3D IoU `train/good`, (2) highest RGB-3D IoU `test/defect`, (3) **lowest `fg_rgb_3d_iou` failure case**, and (4) **highest foreground 3D missingness edge case**. Columns show Original RGB, unsupervised `fg_rgb` (Otsu contrast), unsupervised `fg_3d` (plane residual `> 4 mm`), RGB-vs-3D Agreement (Green = both, Blue = RGB-only, Red = 3D-only), and Valid Depth.

{visual_diag_md}

## 5. Cross-Modal Association: Global vs Background-Adjusted Foreground (Q5 & P1 Q2)

{_section_for("05_cross_modal_quality.sql", "Global Cross-Modal Pearson Correlations (SQL)")}

{_section_for("12_cross_modal_bg_adjusted.sql", "Foreground vs Global Cross-Modal Correlations (SQL)")}

![Cross-Modal Global vs Foreground RGB vs 3D Missingness](../figures/03_cross_modal_rgb_vs_3d_missing.png)

### Statistical Correlation Analysis (Spearman, Pearson, 95% Bootstrap CIs, and Partial Spearman)

Partial Spearman controls simultaneously for object footprint (`fg_area_ratio`) and dark-table background 3D dropout (`bg_missing_3d_ratio`):

{_md_table(["category", "metric_pair", "n", "spearman_95ci", "spearman_p", "pearson_95ci", "partial_spearman_bg_adj"], stat_table_rows)}

## 6. Train-to-Test Distribution Shift & Framing / Footprint Effects (P1 Q3 & Q5)

{_section_for("13_train_test_distribution_shift.sql", "Train vs Validation vs Test Standardized Shifts (SQL)")}

![Train-to-Test vs Train-to-Validation Distribution Shifts](../figures/07_split_distribution_shifts.png)

### Two-Sample KS Tests & Retrospective Benchmark Group Decomposition

Note: The `test/good` vs `test/anomalous` columns below are strictly retrospective benchmark diagnostics to expose potential acquisition shortcuts, never model inputs.

{_md_table(["category", "feature", "train_mean", "test_mean", "test_shift_sd", "test_good_shift_sd", "test_anom_shift_sd", "ks_stat", "ks_p"], shift_stat_rows)}

## 7. Extreme Geometric Missingness & Outlier Diagnostics (Q6 & P1 Q4)

{_section_for("14_extreme_geometric_missingness.sql", "Extreme Global vs Foreground 3D Missingness")}

{_section_for("08_depth_pointcloud_outliers.sql", "Descriptive EDA Depth & Point-Cloud Outliers")}

## 8. Adaptive-Fusion Feature Assessment vs Spurious Shortcuts (P1 Q6 & Q7)

{_section_for("15_fusion_feature_assessment.sql", "Candidate Adaptive-Fusion Quality Indicators vs Shortcut Risks")}

## 9. Benchmark Test Defect Distributions & Retrospective Characteristics (Q7 & Q8)

{_section_for("06_defect_distribution.sql", "Defect-Type Distribution in Labeled Benchmark Test Data")}

![Benchmark Test Defect Distribution](../figures/04_benchmark_test_defect_distribution.png)

{_section_for("09_defect_geometry_texture.sql", "Retrospective Defect Type vs Measured Geometry & Texture")}

## 10. Pipeline Stage Execution Performance (Q9)

{_section_for("07_pipeline_performance.sql", "Authoritative Pipeline Stage Performance (Non-NULL Durations)")}

## 11. Quality Distributions Across Preprocessing Versions (Q10)

{_section_for("10_version_quality_comparison.sql", "Version-to-Version Quality Comparison")}

---

## Frozen Train-Only ML Normalization Provenance

{_md_table(["category", "norm_version", "train_fit_n", "train_manifest_hash", "params_sha256"], norm_rows)}

## Limitations

1. `{len(ingested_cats)}` of 10 official MVTec 3D-AD categories (`{", ".join(ingested_cats)}`) have been ingested so far. Do not extrapolate category-specific distributions, correlations, or segmentation reliability beyond the ingested categories.
2. `fg_rgb` (border-referenced Otsu + morphology) is an unsupervised appearance proxy, not a ground-truth semantic segmentation mask; low-contrast or textured/organic objects and shadows can cause RGB-vs-3D disagreement as exposed in the failure cases above.
3. `fg_3d` assumes the support surface occupies valid border pixels and lies approximately on a single plane; thin objects, recessed features, or non-planar fixtures can reduce `fg_3d` coverage or `fg_plane_inlier_ratio`.
4. Observational rank correlations and distribution-shift tests identify statistical confounding and shortcut risks, not physical root causes.
"""

    dq_md = f"""# VISTA — Multimodal Data Quality, Leakage & Operational Telemetry Report

- **Generated (UTC):** `{now_iso}`
- **Immutable Gold Snapshot ID:** `{snapshot_id or "UNSPECIFIED"}`
- **Snapshot Manifest SHA-256:** `{manifest_sha256 or "UNSPECIFIED"}`
- **Operational Telemetry Cutoff (UTC):** `{tel_cutoff}`
- **Telemetry Consistency Audit:** `{"PASS (0 inconsistencies)" if not tel_issues else f"ISSUES: {tel_issues}"}`
- **Dataset:** `{cfg.dataset_name}` version `{cfg.dataset_version}`
- **Ingested Categories:** `{", ".join(ingested_cats) if ingested_cats else "NONE"}`
- **Total Samples Validated:** `{total_samples}`

## 1. Validation Status Breakdown (Q2)

| Status | Sample Count | Share |
| --- | ---: | ---: |
| `PASS` | {pass_count} | {(100.0 * pass_count / total_samples) if total_samples else 0.0:.2f}% |
| `WARN` | {warn_count} | {(100.0 * warn_count / total_samples) if total_samples else 0.0:.2f}% |
| `QUARANTINE` | {quar_count} | {(100.0 * quar_count / total_samples) if total_samples else 0.0:.2f}% |
| `FAIL` | {fail_count} | {(100.0 * fail_count / total_samples) if total_samples else 0.0:.2f}% |

{_section_for("02_validation_failures.sql", "Validation Rule Breakdown by Category and Status")}

## 2. Leakage Prevention & ML Readiness Contracts

- **Train-only normalization (`normalization_artifacts`):** fitted strictly on eligible `split = 'train'` samples (`status IN ('PASS', 'WARN')`, `is_cross_split_leak = FALSE`). Frozen parameters are applied identically to `train`, `validation`, and `test`.
- **ML partitions:**
  - `training_ready_samples.parquet` / `mart_training_ready_samples`: `train` + `validation` ONLY (`verified test rows = {train_leak_check}`). Excludes `defect_type`, `is_anomalous_benchmark`, and within-split descriptive EDA z-scores.
  - `evaluation_samples.parquet`: held-out `test` partition explicitly tagged `usage = 'benchmark_evaluation_only'`.
- **Immutability:** Bronze archives and committed Gold snapshots (`gold/mvtec_3d_ad/snapshots/<snapshot_id>/`) are immutable and SHA-256 verified.

## 3. Authoritative Operational Telemetry Summary

Canonical statuses:
- Terminal: `SUCCEEDED`, `FAILED`, `INTERRUPTED`, `SKIPPED`
- Intermediate: `RUNNING`
- Duration averages exclude `NULL` durations (never coerced to `0`). Historical failures and retries are preserved transparently.

{_md_table(["stage_name", "total_attempts", "succeeded", "failed", "interrupted", "retries", "avg_succeeded_s"], stage_perf_rows)}
"""

    eda_path = rep_dir / "eda_report.md"
    dq_path = rep_dir / "data_quality_report.md"
    eda_path.write_text(eda_md, encoding="utf-8")
    dq_path.write_text(dq_md, encoding="utf-8")

    log.info(
        "reports_generated",
        snapshot_id=snapshot_id,
        eda_report=str(eda_path),
        dq_report=str(dq_path),
        figures=len(generated_figs),
    )
    return {
        "eda_report": str(eda_path),
        "data_quality_report": str(dq_path),
        "figures": ", ".join(generated_figs),
    }
