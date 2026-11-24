-- Ensures every SUCCEEDED Silver artifact has a traceable Bronze->Silver lineage edge.
SELECT
    sa.artifact_id,
    sa.sample_pk,
    sa.depth_remote_uri
FROM {{ ref('stg_silver_artifacts') }} sa
LEFT JOIN {{ ref('stg_artifact_lineage') }} al
    ON sa.sample_pk = al.sample_pk
   AND al.target_layer = 'silver'
   AND al.transformation_version = sa.preprocessing_version
   AND al.config_hash = sa.config_hash
WHERE sa.artifact_status = 'SUCCEEDED'
  AND al.lineage_id IS NULL
