SELECT
    category_id,
    name AS category_name,
    material_type,
    source_url,
    archive_filename,
    approx_size_mb,
    CAST(created_at AS TIMESTAMPTZ) AS created_at
FROM read_parquet('{{ env_var("VISTA_GOLD_DIR") }}/categories.parquet')
