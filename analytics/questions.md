# VISTA Analytical Questions Catalog

Each question below maps directly to an executable DuckDB SQL query in `analytics/sql/` targeting the dbt warehouse (`dim_*`, `fct_*`, `mart_*`) built over committed immutable Gold snapshots.

| # | Analytical Question | SQL File | Primary Models |
|---|---|---|---|
| 1 | How many samples exist per category and dataset split? | `01_dataset_inventory.sql` | `mart_dataset_inventory` |
| 2 | What proportion of samples pass all validation rules? | `02_validation_failures.sql` | `fct_validation_results` |
| 3 | Which categories have the highest missing 3D-point ratios? | `04_geometry_quality.sql` | `mart_category_quality`, `fct_data_quality` |
| 4 | How do RGB brightness, sharpness, and contrast vary between categories? | `03_rgb_quality.sql` | `fct_data_quality`, `dim_category` |
| 5 | Are RGB quality indicators associated with geometric missingness? | `05_cross_modal_quality.sql` | `fct_data_quality` |
| 6 | Which samples exhibit unusual depth or point-cloud statistics? | `08_depth_pointcloud_outliers.sql` | `fct_data_quality`, `dim_sample` |
| 7 | What are the defect-type distributions in labeled benchmark test data? | `06_defect_distribution.sql` | `mart_defect_distribution` |
| 8 | How do different defect types relate to measured geometry and texture characteristics? | `09_defect_geometry_texture.sql` | `mart_defect_distribution`, `fct_data_quality`, `dim_sample` |
| 9 | Which ingestion or transformation stages consume the most time? | `07_pipeline_performance.sql` | `fct_ingestion_runs` |
| 10 | How do quality distributions change between dataset or preprocessing versions? | `10_version_quality_comparison.sql` | `fct_data_quality`, `dim_sample` |
| 11 | How much do foreground-aware metrics differ from global metrics? | `11_foreground_vs_global_quality.sql` | `mart_foreground_vs_global_quality` |
| 12 | Does RGB–3D quality correlation persist after background adjustment? | `12_cross_modal_bg_adjusted.sql` | `fct_data_quality` |
| 13 | Which modality exhibits stronger train-to-test shifts and framing/footprint effects? | `13_train_test_distribution_shift.sql` | `mart_split_distribution_shift` |
| 14 | Which samples have extreme global vs foreground geometric missingness? | `14_extreme_geometric_missingness.sql` | `fct_data_quality`, `fct_ml_normalized_features` |
| 15 | Which quality indicators are plausible adaptive-fusion candidates vs shortcut risks? | `15_fusion_feature_assessment.sql` | `mart_split_distribution_shift` |
