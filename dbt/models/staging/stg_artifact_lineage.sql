SELECT
    lineage_id,
    sample_pk,
    source_layer,
    source_uri,
    source_sha256,
    target_layer,
    target_uri,
    target_sha256,
    transformation_name,
    transformation_version,
    config_hash,
    run_id,
    CAST(created_at AS TIMESTAMPTZ) AS created_at
FROM read_parquet('{{ env_var("VISTA_GOLD_DIR") }}/artifact_lineage.parquet')
