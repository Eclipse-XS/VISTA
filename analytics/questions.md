# VISTA Analytical Questions Catalog

Each question below maps directly to an executable DuckDB SQL query in `analytics/sql/` targeting the dbt warehouse (`dim_*`, `fct_*`, `mart_*`) built over Gold Parquet datasets.

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
