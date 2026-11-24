SELECT
    norm_artifact_id,
    artifact_key,
    dataset_version,
    category_name,
    preprocessing_version,
    config_hash,
    norm_version,
    train_manifest_hash,
    train_sample_count,
    eligible_statuses_json,
    feature_names_json,
    params_json,
    params_sha256,
    run_id,
    CAST(fitted_at AS TIMESTAMPTZ) AS fitted_at
FROM read_parquet('{{ env_var("VISTA_GOLD_DIR") }}/normalization_artifacts.parquet')
