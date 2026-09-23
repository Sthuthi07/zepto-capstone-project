# Analytics Pipeline

## Run order

```bash
python 01_eda.py
python 02_modeling.py
python verify_model.py
```

`01_eda.py` is the only script that calls `sns.load_dataset("titanic")`. It immediately writes `titanic.csv`; `02_modeling.py` reads that saved CSV.

## Missing-value decisions

The script prints the exact missing percentages. Columns below 5% are row-dropped. Columns from 5% through 30% are imputed. A column above 30% is retained with `"Missing"` as its explicit category. Modeling independently uses training-only imputers inside a `ColumnTransformer`.

## EDA interpretations

- **Age histogram:** Age is concentrated among young and middle-aged passengers, with fewer observations at the extremes.
- **Fare histogram:** Fare has a long upper tail; its mean/median/mode ordering is used in the generated report to describe the distribution.
- **Survival by sex:** Survival rates differ substantially by sex, making sex informative for the target.
- **Survival by class:** Survival varies across passenger classes, indicating class is associated with the outcome.
- **Fare/class/survival chart:** Fare varies strongly by class, and survival patterns differ across those groups.
- **Correlation heatmap:** Only `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare` are used. `adult_male` and `alone` are intentionally excluded. The script ranks the two largest absolute off-diagonal correlations and writes them to `output/eda_report.txt`.

The exploratory z-score check standardizes `age` and `fare` on the cleaned full DataFrame. It is not used by the modeling pipeline.

## Modeling

The split is stratified before preprocessing to preserve the observed class proportions. Numeric features are median-imputed and standardized; categorical features are most-frequent imputed and one-hot encoded. The entire preprocessing stack is inside a Pipeline/ColumnTransformer so it is fitted only on training data.

Three classifiers use the identical split: Logistic Regression, Decision Tree and Random Forest. Metrics include confusion matrices, accuracy, precision, recall, F1 and ROC/AUC.

Imbalance is compared using baseline, `class_weight="balanced"`, and SMOTE. SMOTE is applied only to training data through an imbalanced-learn pipeline.

Random Forest tuning uses GridSearchCV over `n_estimators`, `max_depth`, and `max_features`; the estimator is constructed with `oob_score=True`, so the tuned model's OOB score is available.

The regression side-task predicts fare from the other available features and reports MAE, RMSE, R² and adjusted R². The residual plot is inspected and a spread-based heteroscedasticity indication is recorded.

The complete fitted Random Forest pipeline, including preprocessing and estimator, is saved as `models/best_random_forest_pipeline.joblib` and reloaded by `verify_model.py`.

## Final verification

The three project modules were executed locally after dependency installation:

- Data pipeline: scraping, cleaning, SQLite loading, SQL queries, and pandas/SQL join validation.
- Analytics: EDA, classification, imbalance comparison, regression, evaluation metrics, and saved model pipeline.
- Support assistant: document ingestion, retrieval, deterministic mock responses, FastAPI endpoint, and Docker configuration.

The analytics module also includes `analytics/titanic.csv` as the committed offline fallback dataset.
