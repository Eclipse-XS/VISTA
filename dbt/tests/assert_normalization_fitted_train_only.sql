-- Fails if any train-only normalization artifact lacks valid training provenance.
SELECT
    norm_artifact_id,
    artifact_key,
    train_sample_count,
    train_manifest_hash,
    params_sha256
FROM {{ ref('stg_normalization_artifacts') }}
WHERE train_sample_count <= 0
   OR train_manifest_hash IS NULL
   OR LENGTH(train_manifest_hash) < 32
   OR params_sha256 IS NULL
   OR LENGTH(params_sha256) < 32
