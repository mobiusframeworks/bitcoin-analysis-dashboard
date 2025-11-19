#!/usr/bin/env python3
"""
Reusable ML Model Selection Pipeline

Features:
- Handles classification and regression tasks
- Supports time series and non-time series data
- Nested cross-validation with hyperparameter tuning
- Adaptive CV fold selection
- Overfitting detection
- Learning and validation curves
- Comprehensive reporting
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    KFold, TimeSeriesSplit, train_test_split,
    RandomizedSearchCV, GridSearchCV,
    learning_curve, validation_curve
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, accuracy_score, log_loss,
    mean_squared_error, mean_absolute_error, r2_score,
    make_scorer
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBOOST_AVAILABLE = False


def load_and_prepare_data(data_path, target_col, time_col=None, is_timeseries=False):
    """
    Load CSV and prepare features/target.
    
    Returns:
        X: features DataFrame
        y: target Series
        time_index: time column if time series, else None
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"  Original shape: {df.shape}")
    
    # Handle time column - try case variations
    time_index = None
    if is_timeseries and time_col:
        time_col_actual = None
        for col in df.columns:
            if col.lower() == time_col.lower():
                time_col_actual = col
                break
        
        if time_col_actual:
            df[time_col_actual] = pd.to_datetime(df[time_col_actual], format='mixed', errors='coerce')
            df = df.sort_values(time_col_actual)
            time_index = df[time_col_actual]
        else:
            print(f"  Warning: Time column '{time_col}' not found. Available: {list(df.columns)}. Treating as non-time-series.")
            is_timeseries = False
    
    # Extract target - try case variations
    target_col_actual = None
    for col in df.columns:
        if col.lower() == target_col.lower():
            target_col_actual = col
            break
    
    if target_col_actual is None:
        raise ValueError(f"Target column '{target_col}' not found in data. Available columns: {list(df.columns)}")
    
    y = df[target_col_actual].copy()
    X = df.drop(columns=[target_col_actual])
    
    # Remove time column from features if present
    if time_col_actual and time_col_actual in X.columns:
        X = X.drop(columns=[time_col_actual])
    
    # Remove rows with missing target
    mask = ~y.isna()
    X = X[mask].copy()
    y = y[mask].copy()
    if time_index is not None:
        time_index = time_index[mask].reset_index(drop=True)
    
    print(f"  After removing missing targets: {X.shape}")
    print(f"  Target distribution:\n{y.describe()}")
    
    return X, y, time_index, is_timeseries


def build_preprocessing_pipeline(X):
    """
    Build preprocessing pipeline using ColumnTransformer.
    
    Returns:
        preprocessor: ColumnTransformer
        numeric_features: list of numeric column names
        categorical_features: list of categorical column names
    """
    # Detect numeric vs categorical
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    
    # Build transformers
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor, numeric_features, categorical_features


def get_candidate_models(task_type, preprocessor):
    """
    Get candidate models with hyperparameter grids.
    
    Returns:
        dict: {model_name: (pipeline, param_grid)}
    """
    models = {}
    
    if task_type.upper() == 'CLASSIFICATION':
        # Logistic Regression
        lr_pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', LogisticRegression(random_state=42, max_iter=1000))
        ])
        models['LogisticRegression'] = (
            lr_pipeline,
            {
                'model__C': [0.01, 0.1, 1, 10, 100],
                'model__penalty': ['l2'],
                'model__solver': ['lbfgs', 'liblinear']
            }
        )
        
        # Random Forest
        rf_pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
        ])
        models['RandomForest'] = (
            rf_pipeline,
            {
                'model__n_estimators': [100, 300, 500],
                'model__max_depth': [None, 5, 10, 20],
                'model__max_features': ['sqrt', 'log2']
            }
        )
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            xgb_pipeline = Pipeline([
                ('preprocess', preprocessor),
                ('model', xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'))
            ])
            models['XGBoost'] = (
                xgb_pipeline,
                {
                    'model__n_estimators': [100, 300, 500],
                    'model__max_depth': [3, 5, 7],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__subsample': [0.7, 1.0]
                }
            )
    
    else:  # REGRESSION
        # Ridge Regression with stronger regularization
        ridge_pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', Ridge(random_state=42))
        ])
        models['Ridge'] = (
            ridge_pipeline,
            {
                'model__alpha': [1, 10, 100, 1000, 10000]  # Stronger regularization
            }
        )
        
        # Lasso Regression (L1 regularization for feature selection)
        from sklearn.linear_model import Lasso
        lasso_pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', Lasso(random_state=42, max_iter=2000))
        ])
        models['Lasso'] = (
            lasso_pipeline,
            {
                'model__alpha': [0.1, 1, 10, 100, 1000]
            }
        )
        
        # Random Forest
        rf_pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
        ])
        models['RandomForest'] = (
            rf_pipeline,
            {
                'model__n_estimators': [100, 300, 500],
                'model__max_depth': [None, 5, 10, 20],
                'model__max_features': ['sqrt', 'log2']
            }
        )
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            xgb_pipeline = Pipeline([
                ('preprocess', preprocessor),
                ('model', xgb.XGBRegressor(random_state=42, n_jobs=-1))
            ])
            models['XGBoost'] = (
                xgb_pipeline,
                {
                    'model__n_estimators': [100, 300, 500],
                    'model__max_depth': [3, 5, 7],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__subsample': [0.7, 1.0]
                }
            )
    
    return models


def choose_optimal_cv_folds(X_train, y_train, task_type, is_timeseries, 
                            preprocessor, scoring, k_values=[3, 5, 7, 10]):
    """
    Choose optimal number of CV folds by testing different k values.
    
    Returns:
        optimal_k: chosen number of folds
        k_scores: dict of {k: (mean_score, std_score)}
    """
    print("\n" + "="*80)
    print("CHOOSING OPTIMAL NUMBER OF CV FOLDS")
    print("="*80)
    
    # Use RandomForest as representative model
    if task_type.upper() == 'CLASSIFICATION':
        base_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        base_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', base_model)
    ])
    
    k_scores = {}
    
    for k in k_values:
        if is_timeseries:
            cv = TimeSeriesSplit(n_splits=k)
        else:
            cv = KFold(n_splits=k, shuffle=True, random_state=42)
        
        # Fit preprocessor
        X_train_processed = preprocessor.fit_transform(X_train)
        
        # Cross-validate
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        mean_score = scores.mean()
        std_score = scores.std()
        k_scores[k] = (mean_score, std_score)
        
        print(f"  k={k}: mean={mean_score:.4f}, std={std_score:.4f}")
    
    # Choose optimal k (diminishing returns)
    k_list = sorted(k_scores.keys())
    means = [k_scores[k][0] for k in k_list]
    
    # Find k where improvement < 1%
    optimal_k = k_list[0]
    for i in range(1, len(k_list)):
        improvement = (means[i] - means[i-1]) / abs(means[i-1]) if means[i-1] != 0 else 0
        if improvement < 0.01:  # Less than 1% improvement
            optimal_k = k_list[i-1]
            break
        optimal_k = k_list[i]
    
    print(f"\n  Chosen k={optimal_k} (diminishing returns threshold: 1%)")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    means = [k_scores[k][0] for k in k_list]
    stds = [k_scores[k][1] for k in k_list]
    
    ax.errorbar(k_list, means, yerr=stds, marker='o', capsize=5, capthick=2, linewidth=2)
    ax.axvline(x=optimal_k, color='red', linestyle='--', label=f'Chosen k={optimal_k}')
    ax.set_xlabel('Number of CV Folds (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('CV Score', fontsize=12, fontweight='bold')
    ax.set_title('CV Score vs Number of Folds', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    return optimal_k, k_scores


def detect_overfitting(train_score, val_score, threshold=0.05):
    """
    Detect overfitting by comparing train and validation scores.
    
    Returns:
        is_overfitting: bool
        gap: relative difference
    """
    if val_score == 0:
        return True, float('inf')
    
    gap = abs(train_score - val_score) / abs(val_score)
    is_overfitting = gap > threshold
    
    return is_overfitting, gap


def nested_cv_search(X_train, y_train, models, task_type, is_timeseries, 
                     optimal_k, scoring, n_iter=50):
    """
    Perform nested cross-validation for each model.
    
    Returns:
        results: dict with model performance
    """
    print("\n" + "="*80)
    print("NESTED CROSS-VALIDATION")
    print("="*80)
    
    # Setup CV
    if is_timeseries:
        outer_cv = TimeSeriesSplit(n_splits=optimal_k)
        inner_cv = TimeSeriesSplit(n_splits=min(3, optimal_k))
    else:
        outer_cv = KFold(n_splits=optimal_k, shuffle=True, random_state=42)
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    results = {}
    
    for model_name, (pipeline, param_grid) in models.items():
        print(f"\n{model_name}:")
        print("-" * 80)
        
        # Use RandomizedSearchCV for large grids, GridSearchCV for small ones
        n_params = np.prod([len(v) if isinstance(v, list) else 1 for v in param_grid.values()])
        
        if n_params > 50:
            search = RandomizedSearchCV(
                pipeline, param_grid, cv=inner_cv, scoring=scoring,
                n_iter=n_iter, n_jobs=-1, random_state=42, verbose=0
            )
        else:
            search = GridSearchCV(
                pipeline, param_grid, cv=inner_cv, scoring=scoring,
                n_jobs=-1, verbose=0
            )
        
        # Outer CV loop
        outer_scores = []
        best_params_list = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train)):
            X_fold_train = X_train.iloc[train_idx] if isinstance(X_train, pd.DataFrame) else X_train[train_idx]
            y_fold_train = y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]
            X_fold_val = X_train.iloc[val_idx] if isinstance(X_train, pd.DataFrame) else X_train[val_idx]
            y_fold_val = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]
            
            # Inner search
            search.fit(X_fold_train, y_fold_train)
            best_params_list.append(search.best_params_)
            
            # Evaluate on outer validation set
            y_pred = search.predict(X_fold_val)
            
            if task_type.upper() == 'CLASSIFICATION':
                if len(np.unique(y_train)) == 2:  # Binary
                    try:
                        y_pred_proba = search.predict_proba(X_fold_val)[:, 1]
                        score = roc_auc_score(y_fold_val, y_pred_proba)
                    except:
                        score = accuracy_score(y_fold_val, y_pred)
                else:
                    score = accuracy_score(y_fold_val, y_pred)
            else:  # Regression
                score = -mean_squared_error(y_fold_val, y_pred)  # Negative for consistency
            
            outer_scores.append(score)
        
        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        
        # Get most common best params
        from collections import Counter
        param_strs = [str(p) for p in best_params_list]
        most_common_params = eval(Counter(param_strs).most_common(1)[0][0])
        
        results[model_name] = {
            'mean_cv_score': mean_score,
            'std_cv_score': std_score,
            'best_params': most_common_params,
            'outer_scores': outer_scores
        }
        
        print(f"  Mean CV Score: {mean_score:.4f} (+/- {std_score:.4f})")
        print(f"  Best Params: {most_common_params}")
    
    return results


def plot_learning_curve(model, X, y, scoring, cv, task_type, save_path):
    """Plot learning curve for the best model."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring=scoring, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score', linewidth=2)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Learning Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_validation_curve(model, X, y, param_name, param_range, scoring, cv, save_path):
    """Plot validation curve for a key hyperparameter."""
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(param_range, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.plot(param_range, val_mean, 'o-', color='red', label='Validation Score', linewidth=2)
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    ax.set_xlabel(param_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Validation Curve: {param_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_model_selection(data_path, target_col, task_type='REGRESSION', 
                       time_col=None, is_timeseries=False, output_dir='ml_results'):
    """
    Main function to run the complete model selection pipeline.
    
    Parameters:
        data_path: path to CSV file
        target_col: name of target column
        task_type: 'CLASSIFICATION' or 'REGRESSION'
        time_col: name of time column (if time series)
        is_timeseries: whether data is time series
        output_dir: directory to save results
    
    Returns:
        dict: comprehensive results
    """
    print("="*80)
    print("ML MODEL SELECTION PIPELINE")
    print("="*80)
    print(f"Task: {task_type}")
    print(f"Time Series: {is_timeseries}")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plots_path = output_path / 'plots'
    plots_path.mkdir(exist_ok=True)
    
    # 1. Load data
    X, y, time_index, is_timeseries = load_and_prepare_data(
        data_path, target_col, time_col, is_timeseries
    )
    
    # 2. Build preprocessing pipeline
    preprocessor, numeric_features, categorical_features = build_preprocessing_pipeline(X)
    
    # 3. Train/Val/Test split
    if is_timeseries:
        # Chronological split
        n = len(X)
        train_end = int(0.6 * n)
        val_end = int(0.8 * n)
        
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        y_test = y.iloc[val_end:]
    else:
        # Random split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 of 0.8 = 0.2
        )
    
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 4. Define scoring
    if task_type.upper() == 'CLASSIFICATION':
        if len(np.unique(y)) == 2:
            scoring = 'roc_auc'
        else:
            scoring = 'accuracy'
    else:
        scoring = 'neg_mean_squared_error'
    
    # 5. Choose optimal CV folds
    optimal_k, k_scores = choose_optimal_cv_folds(
        X_train, y_train, task_type, is_timeseries, preprocessor, scoring
    )
    plt.savefig(plots_path / 'cv_folds_vs_score.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Get candidate models
    models = get_candidate_models(task_type, preprocessor)
    print(f"\nCandidate models: {list(models.keys())}")
    
    # 7. Nested CV
    cv_results = nested_cv_search(
        X_train, y_train, models, task_type, is_timeseries, optimal_k, scoring
    )
    
    # 8. Select best model
    best_model_name = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_cv_score'])
    best_params = cv_results[best_model_name]['best_params']
    best_pipeline_template, _ = models[best_model_name]
    
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"{'='*80}")
    print(f"Best Params: {best_params}")
    print(f"Mean CV Score: {cv_results[best_model_name]['mean_cv_score']:.4f}")
    
    # 9. Overfitting checks
    print(f"\n{'='*80}")
    print("OVERFITTING ANALYSIS")
    print(f"{'='*80}")
    
    # Before tuning (default params)
    default_model = best_pipeline_template
    default_model.fit(X_train, y_train)
    
    train_pred_default = default_model.predict(X_train)
    val_pred_default = default_model.predict(X_val)
    
    if task_type.upper() == 'CLASSIFICATION':
        train_score_default = accuracy_score(y_train, train_pred_default)
        val_score_default = accuracy_score(y_val, val_pred_default)
    else:
        train_score_default = r2_score(y_train, train_pred_default)
        val_score_default = r2_score(y_val, val_pred_default)
    
    is_overfitting_default, gap_default = detect_overfitting(train_score_default, val_score_default)
    
    print(f"\nBEFORE TUNING (Default Hyperparameters):")
    print(f"  Train Score: {train_score_default:.4f}")
    print(f"  Val Score: {val_score_default:.4f}")
    print(f"  Gap: {gap_default:.2%}")
    print(f"  Overfitting: {'YES' if is_overfitting_default else 'NO'}")
    
    # After tuning (best params)
    best_model = best_pipeline_template.set_params(**best_params)
    
    # Train on train+val
    X_train_val = pd.concat([X_train, X_val]) if isinstance(X_train, pd.DataFrame) else np.vstack([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val]) if isinstance(y_train, pd.Series) else np.concatenate([y_train, y_val])
    
    best_model.fit(X_train_val, y_train_val)
    
    train_val_pred = best_model.predict(X_train_val)
    test_pred = best_model.predict(X_test)
    
    if task_type.upper() == 'CLASSIFICATION':
        train_val_score = accuracy_score(y_train_val, train_val_pred)
        test_score = accuracy_score(y_test, test_pred)
    else:
        train_val_score = r2_score(y_train_val, train_val_pred)
        test_score = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
    
    is_overfitting_tuned, gap_tuned = detect_overfitting(train_val_score, test_score)
    
    print(f"\nAFTER TUNING (Best Hyperparameters):")
    print(f"  Train+Val Score: {train_val_score:.4f}")
    print(f"  Test Score: {test_score:.4f}")
    if task_type.upper() == 'REGRESSION':
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Gap: {gap_tuned:.2%}")
    print(f"  Overfitting: {'YES' if is_overfitting_tuned else 'NO'}")
    
    # 10. Learning and validation curves
    print(f"\n{'='*80}")
    print("GENERATING LEARNING AND VALIDATION CURVES")
    print(f"{'='*80}")
    
    if is_timeseries:
        cv_for_curves = TimeSeriesSplit(n_splits=min(5, optimal_k))
    else:
        cv_for_curves = KFold(n_splits=min(5, optimal_k), shuffle=True, random_state=42)
    
    plot_learning_curve(
        best_model, X_train_val, y_train_val, scoring, cv_for_curves, 
        task_type, plots_path / 'learning_curve_best_model.png'
    )
    print("  ✓ Learning curve saved")
    
    # Validation curve for key parameter
    if 'max_depth' in best_params:
        param_name = 'model__max_depth'
        param_range = [None, 5, 10, 20, 30]
    elif 'C' in best_params or 'alpha' in best_params:
        param_name = 'model__C' if 'C' in best_params else 'model__alpha'
        param_range = [0.01, 0.1, 1, 10, 100]
    elif 'learning_rate' in best_params:
        param_name = 'model__learning_rate'
        param_range = [0.01, 0.05, 0.1, 0.2, 0.3]
    else:
        param_name = None
    
    if param_name:
        try:
            plot_validation_curve(
                best_model, X_train_val, y_train_val, param_name, param_range,
                scoring, cv_for_curves, plots_path / 'validation_curve_best_model.png'
            )
            print("  ✓ Validation curve saved")
        except:
            print("  ⚠ Could not generate validation curve")
    
    # 11. Compile results
    results = {
        'task_type': task_type,
        'is_timeseries': is_timeseries,
        'optimal_cv_folds': optimal_k,
        'cv_fold_scores': {str(k): {'mean': float(v[0]), 'std': float(v[1])} 
                          for k, v in k_scores.items()},
        'model_results': {
            name: {
                'mean_cv_score': float(res['mean_cv_score']),
                'std_cv_score': float(res['std_cv_score']),
                'best_params': res['best_params']
            }
            for name, res in cv_results.items()
        },
        'best_model': best_model_name,
        'best_params': best_params,
        'overfitting_analysis': {
            'before_tuning': {
                'train_score': float(train_score_default),
                'val_score': float(val_score_default),
                'gap': float(gap_default),
                'is_overfitting': bool(is_overfitting_default)
            },
            'after_tuning': {
                'train_val_score': float(train_val_score),
                'test_score': float(test_score),
                'gap': float(gap_tuned),
                'is_overfitting': bool(is_overfitting_tuned)
            }
        },
        'test_metrics': {
            'score': float(test_score)
        }
    }
    
    if task_type.upper() == 'REGRESSION':
        results['test_metrics']['rmse'] = float(test_rmse)
        results['test_metrics']['mae'] = float(test_mae)
    
    # Save results
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary report
    print(f"\n{'='*80}")
    print("SUMMARY REPORT")
    print(f"{'='*80}")
    print(f"\nChosen CV Folds: {optimal_k}")
    print(f"  Reason: Diminishing returns threshold (1% improvement)")
    print(f"\nCandidate Models Performance:")
    for name, res in cv_results.items():
        print(f"  {name}: {res['mean_cv_score']:.4f} (+/- {res['std_cv_score']:.4f})")
    print(f"\nSelected Best Model: {best_model_name}")
    print(f"  Best Hyperparameters: {best_params}")
    print(f"\nTrain vs Validation vs Test Metrics:")
    print(f"  Before Tuning:")
    print(f"    Train: {train_score_default:.4f}, Val: {val_score_default:.4f}")
    print(f"  After Tuning:")
    print(f"    Train+Val: {train_val_score:.4f}, Test: {test_score:.4f}")
    if task_type.upper() == 'REGRESSION':
        print(f"    Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}")
    print(f"\nOverfitting Analysis:")
    print(f"  Before Tuning: {'OVERFITTING DETECTED' if is_overfitting_default else 'No significant overfitting'}")
    print(f"  After Tuning: {'OVERFITTING DETECTED' if is_overfitting_tuned else 'No significant overfitting'}")
    print(f"  Tuning {'reduced' if gap_tuned < gap_default else 'did not reduce'} the train-test gap")
    print(f"\nResults saved to: {output_path}")
    
    return results, best_model, X_test, y_test, X_train_val, y_train_val
