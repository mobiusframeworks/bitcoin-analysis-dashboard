# ML Model Selection Pipeline

A comprehensive, reusable Python module for machine learning model selection with nested cross-validation, hyperparameter tuning, overfitting detection, and comprehensive reporting.

## Features

- **Handles both Classification and Regression tasks**
- **Supports Time Series and Non-Time Series data**
- **Nested Cross-Validation** to prevent overfitting
- **Adaptive CV Fold Selection** based on diminishing returns
- **Hyperparameter Tuning** with GridSearchCV and RandomizedSearchCV
- **Overfitting Detection** with train/validation/test comparisons
- **Learning and Validation Curves** for model diagnostics
- **Comprehensive Reporting** with text and visualizations

## Quick Start

### Basic Usage

```python
from ml_pipeline.model_selection_pipeline import run_model_selection

results, best_model, X_test, y_test, X_train_val, y_train_val = run_model_selection(
    data_path="path/to/data.csv",
    target_col="target_column_name",
    task_type="REGRESSION",  # or "CLASSIFICATION"
    time_col="Date",  # optional, for time series
    is_timeseries=True,  # or False
    output_dir="ml_results"
)
```

### For Bitcoin Price Prediction

```bash
python ml_pipeline/run_bitcoin_prediction.py
```

This will:
1. Load Bitcoin data
2. Run model selection with nested CV
3. Train best model
4. Make prediction on most recent data
5. Save results to `ml_pipeline/bitcoin_results/`

### Generate Comprehensive PDF Report

```bash
python ml_pipeline/generate_comprehensive_pdf_report.py
```

This creates `Bitcoin_Comprehensive_Analysis_Report.pdf` with:
- PCA Analysis
- Feature Selection
- FRED Analysis
- SMA Analysis
- ML Model Selection
- Overfitting Analysis
- Current Price and Future Projections
- All Charts and Visualizations

## Pipeline Components

### 1. Data Loading & Preparation
- Automatically detects numeric vs categorical columns
- Builds preprocessing pipeline with:
  - Numeric: median imputation + StandardScaler
  - Categorical: most frequent imputation + OneHotEncoder

### 2. Train/Validation/Test Split
- **Non-Time Series**: Random split (60/20/20)
- **Time Series**: Chronological split (60/20/20)

### 3. Model Candidates

**For Classification:**
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier (if available)

**For Regression:**
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor (if available)

### 4. Nested Cross-Validation
- Outer CV: Evaluates model performance
- Inner CV: Tunes hyperparameters
- Prevents data leakage and overfitting

### 5. Adaptive CV Fold Selection
- Tests k values: [3, 5, 7, 10]
- Chooses k where improvement < 1% (diminishing returns)
- Plots CV score vs number of folds

### 6. Overfitting Detection
- Compares train vs validation scores
- Before and after tuning analysis
- Flags overfitting if gap > threshold (default 5%)

### 7. Learning & Validation Curves
- Learning curve: training set size vs performance
- Validation curve: hyperparameter value vs performance
- Helps diagnose bias/variance tradeoff

## Output Files

### Results Directory Structure
```
ml_results/
├── results.json              # Complete results
├── latest_prediction.json    # Most recent prediction
└── plots/
    ├── cv_folds_vs_score.png
    ├── learning_curve_best_model.png
    └── validation_curve_best_model.png
```

### results.json Contents
- Task type and configuration
- Optimal CV folds and reasoning
- Model performance (nested CV scores)
- Best model and hyperparameters
- Overfitting analysis (before/after tuning)
- Test set metrics

## Example: Bitcoin Price Prediction

```python
from ml_pipeline.model_selection_pipeline import run_model_selection

# Run on Bitcoin data
results, model, X_test, y_test, X_train, y_train = run_model_selection(
    data_path="data/btc-ohlc.csv",
    target_col="Close",
    task_type="REGRESSION",
    time_col="Date",
    is_timeseries=True,
    output_dir="bitcoin_results"
)

# Make prediction
import pandas as pd
latest_data = pd.read_csv("data/btc-ohlc.csv").iloc[-1:]
X_latest = latest_data.drop(columns=["Close", "Date"])
prediction = model.predict(X_latest)[0]

print(f"Predicted Price: ${prediction:,.2f}")
```

## Configuration Options

### Task Type
- `"CLASSIFICATION"`: For classification problems
- `"REGRESSION"`: For regression problems

### Time Series
- `is_timeseries=True`: Uses TimeSeriesSplit, chronological splits
- `is_timeseries=False`: Uses KFold, random splits

### Hyperparameter Grids
Default grids are provided, but you can modify `get_candidate_models()` to customize.

## Model Selection Criteria

The pipeline selects the best model based on:
1. **Mean nested CV score** (primary)
2. **Model simplicity** (tie-breaker if scores within 1%)
   - Simpler = fewer parameters
   - Linear > Tree > Boosted

## Overfitting Detection

Overfitting is detected if:
```
|train_score - val_score| / |val_score| > threshold (default 0.05)
```

The pipeline reports:
- Before tuning: train vs validation gap
- After tuning: train+val vs test gap
- Whether tuning reduced overfitting

## Notes

- **XGBoost**: Optional dependency. Pipeline works without it.
- **Time Series**: Always use `is_timeseries=True` for time-ordered data
- **Large Datasets**: Consider using `RandomizedSearchCV` for large hyperparameter grids
- **Memory**: Nested CV can be memory-intensive for large datasets

## Troubleshooting

### XGBoost Import Error
If you see XGBoost errors, the pipeline will continue without it. To fix:
```bash
brew install libomp  # macOS
pip install xgboost --upgrade
```

### Memory Issues
- Reduce CV folds
- Use smaller hyperparameter grids
- Process data in chunks

### Slow Execution
- Use `RandomizedSearchCV` instead of `GridSearchCV`
- Reduce number of CV folds
- Use fewer hyperparameter combinations

## Citation

If you use this pipeline in your research, please cite:
```
ML Model Selection Pipeline
Comprehensive framework for model selection with nested CV and overfitting detection
```

## License

This code is provided as-is for research and educational purposes.



