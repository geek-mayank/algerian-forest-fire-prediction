### Algerian Forest Fires — EDA ➜ Modeling (End-to-End)

- **Dataset**: 244 daily records from two Algerian regions (Bejaia, Sidi Bel-abbes), Jun–Sep 2012, 11 weather/FWI features + class.
- **Cleaning**:
  - Fixed headers/whitespace and types; added `Region` flag.
  - Dropped NaNs and a separator row → **243 rows** retained.
  - Exported `Algerian_forest_fires_dataset_CLEANED.csv`.

- **EDA Highlights**:
  - **Distributions**: Weather and FWI components show meaningful spread; `FWI` is right-skewed with outliers.
  - **Correlations**: Strong relationships among FWI components; `FWI` aligns highly with `FFMC`, `ISI`, `BUI`, `DMC`.
  - **Class view**: Converted `Classes` → 0 (not fire), 1 (fire) for EDA; seasonal/region patterns visible in monthly counts.
  - Visuals: Histograms, correlation heatmap, boxplots (FWI), region-wise monthly fire counts.

- **Feature Prep for Modeling**:
  - Dropped `day`, `month`, `year`.
  - Removed highly correlated predictors (|r| > 0.85) to reduce multicollinearity.
  - Standardized features with `StandardScaler`.

- **Task**: Predict **Fire Weather Index (FWI)** as a regression problem.

- **Models & Test Performance**:
  | Model            | MAE | R²    |
  |------------------|-----|-------|
  | Linear Regression| 0.547 | 0.9848 |
  | Ridge            | 0.564 | 0.9843 |
  | Lasso            | 1.133 | 0.9492 |
  | LassoCV          | 0.620 | 0.9821 |
  | ElasticNet       | 1.882 | 0.8753 |
  | ElasticNetCV     | 0.658 | 0.9814 |

- **Outcome**: Simple linear models (Linear/Ridge) performed best with very high explanatory power on test data.
- **Artifacts**: Saved `scaler.pkl` and `ridge.pkl` for deployment.

FWI helps quantify fire risk; with lightweight preprocessing and regularized linear models, we achieved strong, stable performance suitable for operationalization.

#MachineLearning #DataScience #EDA #Regression #MLOps #Algeria #ForestFires #FWI