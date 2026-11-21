"""Command-line interface for the VISTA Data Engineering platform."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from vista.analytics.duckdb_runner import execute_analytical_sql_suite, run_dbt_build
from vista.analytics.gold_publisher import (
    export_operational_telemetry,
    publish_gold_datasets,
    resolve_latest_snapshot_dir,
)
from vista.analytics.reports import generate_eda_figures_and_reports
from vista.config import VistaConfig
from vista.contracts.schemas import verify_snapshot_directory
from vista.ingestion.catalog import list_catalog_categories, probe_category_source
from vista.logging import configure_logging
from vista.metadata.database import DatabaseSessionManager, create_db_engine, init_db
from vista.orchestration.flows import run_category_pipeline
from vista.storage import build_storage_backend, cleanup_verified_category_cache


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vista",
        description="VISTA — MVTec 3D-AD Data Engineering Platform CLI",
    )
    parser.add_argument("--json-logs", action="store_true", help="Emit JSON-formatted logs.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-storage", help="Verify Google Drive / S3 workspace confinement and reachability.")
    sub.add_parser("init-db", help="Initialize or migrate PostgreSQL operational metadata tables.")

    p_cat = sub.add_parser("catalog", help="List official MVTec 3D-AD categories (optionally probe URLs).")
    p_cat.add_argument("--probe", action="store_true", help="Send HTTP HEAD requests to verify archives.")

    p_run = sub.add_parser(
        "run-category",
        help="Run the full Prefect pipeline for one category: discover->ingest->validate->transform->quality->gold->analytics.",
    )
    p_run.add_argument("category", help="MVTec 3D-AD category name (e.g., cable_gland, dowel).")
    p_run.add_argument("--archive", type=Path, default=None, help="Optional local .tar.xz path override.")
    p_run.add_argument("--no-upload", action="store_true", help="Skip uploading Bronze/Silver/Gold to remote storage.")
    p_run.add_argument("--skip-analytics", action="store_true", help="Skip dbt build & EDA report generation.")
    p_run.add_argument("--cleanup-cache", action="store_true", help="Safely reclaim local Bronze/Silver/archive cache after verified remote commit.")

    p_clean = sub.add_parser("cleanup-cache", help="Verify remote Bronze/Silver/Gold artifacts and safely reclaim local cache for one category.")
    p_clean.add_argument("category", help="Category name to reclaim local cache for.")
    p_clean.add_argument("--dry-run", action="store_true", help="Verify remote artifacts and report reclaimable size without deleting local files.")

    p_gold = sub.add_parser("publish-gold", help="Publish an immutable Gold snapshot from PostgreSQL.")
    p_gold.add_argument("--no-upload", action="store_true")

    p_ver = sub.add_parser("verify-snapshot", help="Independently verify a committed Gold snapshot.")
    p_ver.add_argument("snapshot_dir", type=Path, nargs="?", default=None)

    sub.add_parser("analytics", help="Run dbt build, execute SQL suite, and regenerate EDA reports & charts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging(json_logs=args.json_logs)
    cfg = VistaConfig.load()

    if args.command == "check-storage":
        storage = build_storage_backend(cfg.storage)
        print(json.dumps(storage.check_workspace(), indent=2))
        return 0

    if args.command == "init-db":
        engine = create_db_engine(cfg.database)
        init_db(engine)
        print(f"Initialized metadata tables in {cfg.database.host}:{cfg.database.port}/{cfg.database.database}")
        return 0

    if args.command == "catalog":
        rows = []
        for spec in list_catalog_categories(cfg):
            entry = {
                "category": spec.name,
                "material_type": spec.material_type,
                "approx_size_mb": spec.approx_size_mb,
                "archive_filename": spec.archive_filename,
            }
            if args.probe:
                chk = probe_category_source(spec)
                entry["reachable"] = chk.reachable
                entry["status_code"] = chk.status_code
                entry["content_length"] = chk.content_length
                entry["accept_ranges"] = chk.accept_ranges
            rows.append(entry)
        print(json.dumps(rows, indent=2))
        return 0

    if args.command == "run-category":
        result = run_category_pipeline(
            args.category,
            cfg=cfg,
            local_archive_override=args.archive,
            upload_to_remote=not args.no_upload,
            run_analytics=not args.skip_analytics,
            cleanup_cache_after_commit=args.cleanup_cache,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.command == "cleanup-cache":
        engine = create_db_engine(cfg.database)
        init_db(engine)
        db = DatabaseSessionManager(engine)
        storage = build_storage_backend(cfg.storage)
        res = cleanup_verified_category_cache(
            cfg=cfg,
            storage=storage,
            db=db,
            category=args.category,
            dry_run=args.dry_run,
        )
        print(json.dumps(asdict(res), indent=2))
        return 0

    if args.command == "publish-gold":
        from vista.metadata.repository import MetadataRepository

        engine = create_db_engine(cfg.database)
        init_db(engine)
        db = DatabaseSessionManager(engine)
        storage = build_storage_backend(cfg.storage)
        with db.session() as sess:
            repo = MetadataRepository(sess)
            run = repo.start_pipeline_run(
                flow_name="manual-publish-gold",
                category_name="all",
                dataset_version=cfg.dataset_version,
                preprocessing_version=cfg.preprocessing.version,
                config=cfg.preprocessing.canonical_dict(),
            )
            run_id = run.run_id
            stage = repo.start_stage_run(run_id, "publish_gold", "all")
            stage_id = stage.stage_run_id
        try:
            snapshot = publish_gold_datasets(
                cfg=cfg,
                storage=storage,
                db=db,
                run_id=run_id,
                upload_to_remote=not args.no_upload,
                exclude_in_flight_stage_run_id=stage_id,
            )
        except Exception as exc:
            with db.session() as sess:
                repo = MetadataRepository(sess)
                repo.finish_stage_run(stage_id, "FAILED", error_message=str(exc))
                repo.finish_pipeline_run(run_id, "FAILED", error_message=str(exc))
            raise
        with db.session() as sess:
            repo = MetadataRepository(sess)
            repo.finish_stage_run(stage_id, "SUCCEEDED", items_processed=len(snapshot.tables))
            repo.finish_pipeline_run(run_id, "SUCCEEDED")
        print(json.dumps(asdict(snapshot), indent=2, default=str))
        return 0

    if args.command == "verify-snapshot":
        snap_dir = args.snapshot_dir or resolve_latest_snapshot_dir(cfg)
        res = verify_snapshot_directory(snap_dir)
        print(json.dumps(asdict(res), indent=2))
        return 0

    if args.command == "analytics":
        engine = create_db_engine(cfg.database)
        init_db(engine)
        db = DatabaseSessionManager(engine)
        tel = export_operational_telemetry(cfg=cfg, db=db)
        dbt_out = run_dbt_build(cfg, telemetry_dir=Path(tel["telemetry_dir"]))
        sql_res = execute_analytical_sql_suite(cfg)
        rep_out = generate_eda_figures_and_reports(
            cfg,
            sql_res,
            snapshot_id=dbt_out["snapshot_id"],
            manifest_sha256=dbt_out["manifest_sha256"],
            telemetry_meta=tel,
        )
        print(
            json.dumps(
                {
                    "dbt": dbt_out,
                    "telemetry": tel,
                    "sql_files": {k: len(v.rows) for k, v in sql_res.items()},
                    "reports": rep_out,
                },
                indent=2,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
