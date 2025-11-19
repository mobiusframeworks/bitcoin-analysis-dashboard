#!/usr/bin/env python3
"""
Overfitting Detection and Data Leakage Checker

Checks for:
1. Data leakage (using future information)
2. Look-ahead bias
3. Duplicate/redundant features
4. Overfitting indicators
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def check_data_leakage(X, y, time_col=None, target_col=None):
    """
    Check for data leakage in features.
    
    Returns:
        dict: leakage report
    """
    report = {
        'has_leakage': False,
        'issues': [],
        'warnings': []
    }
    
    # Check if features contain target information
    if target_col and target_col in X.columns:
        report['has_leakage'] = True
        report['issues'].append(f"Target column '{target_col}' found in features!")
    
    # Check for highly correlated features (potential leakage)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = X[numeric_cols].corr()
        # Check for perfect or near-perfect correlations
        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i+1:]:
                corr = abs(corr_matrix.loc[col1, col2])
                if corr > 0.99:
                    report['has_leakage'] = True
                    report['issues'].append(f"Near-perfect correlation between '{col1}' and '{col2}': {corr:.4f}")
                elif corr > 0.95:
                    report['warnings'].append(f"High correlation between '{col1}' and '{col2}': {corr:.4f}")
    
    # Check for duplicate columns
    duplicate_cols = X.columns[X.columns.duplicated()].tolist()
    if duplicate_cols:
        report['has_leakage'] = True
        report['issues'].append(f"Duplicate columns found: {duplicate_cols}")
    
    return report


def check_lookahead_bias(X, y, time_col, target_col):
    """
    Check for look-ahead bias in time series data.
    
    Look-ahead bias occurs when:
    - Using future information to predict past
    - Using same-day features to predict same-day target
    - Not properly lagging features
    """
    report = {
        'has_lookahead': False,
        'issues': [],
        'warnings': []
    }
    
    if time_col is None or time_col not in X.columns:
        report['warnings'].append("No time column available - cannot check for look-ahead bias")
        return report
    
    # Sort by time
    df = pd.concat([X, y], axis=1)
    df = df.sort_values(time_col)
    
    # Check if features are from same time period as target
    # This is a simplified check - in practice, we'd need to know the exact timing
    report['warnings'].append("Look-ahead bias check requires domain knowledge of feature timing")
    
    return report


def detect_overfitting(train_score, val_score, test_score=None, threshold=0.05):
    """
    Detect overfitting based on train/val/test score gaps.
    
    Parameters:
        train_score: Training set score
        val_score: Validation set score
        test_score: Test set score (optional)
        threshold: Relative gap threshold (default 0.05 = 5%)
    
    Returns:
        dict: Overfitting report
    """
    report = {
        'is_overfitting': False,
        'severity': 'none',
        'train_val_gap': None,
        'train_test_gap': None,
        'val_test_gap': None,
        'issues': []
    }
    
    # Calculate gaps
    if val_score != 0:
        train_val_gap = abs(train_score - val_score) / abs(val_score)
        report['train_val_gap'] = train_val_gap
        
        if train_val_gap > threshold:
            report['is_overfitting'] = True
            if train_val_gap > 0.20:
                report['severity'] = 'severe'
            elif train_val_gap > 0.10:
                report['severity'] = 'moderate'
            else:
                report['severity'] = 'mild'
            
            report['issues'].append(
                f"Train-Val gap ({train_val_gap:.2%}) exceeds threshold ({threshold:.2%})"
            )
    
    if test_score is not None and test_score != 0:
        train_test_gap = abs(train_score - test_score) / abs(test_score)
        val_test_gap = abs(val_score - test_score) / abs(test_score)
        
        report['train_test_gap'] = train_test_gap
        report['val_test_gap'] = val_test_gap
        
        if train_test_gap > threshold:
            report['is_overfitting'] = True
            report['issues'].append(
                f"Train-Test gap ({train_test_gap:.2%}) exceeds threshold ({threshold:.2%})"
            )
        
        if val_test_gap > 0.10:  # Val and test should be similar
            report['warnings'] = report.get('warnings', [])
            report['warnings'].append(
                f"Val-Test gap ({val_test_gap:.2%}) is large - possible data distribution shift"
            )
    
    # Check for suspiciously high scores (potential data leakage or trivial task)
    # For financial time series, R² > 0.95 is suspicious (predicting price from previous price is too easy)
    if train_score > 0.95 and val_score > 0.95:
        report['is_overfitting'] = True
        report['severity'] = 'severe'
        report['issues'].append(
            f"Suspiciously high scores (Train: {train_score:.4f}, Val: {val_score:.4f}) - "
            "likely data leakage or trivial prediction task (e.g., predicting price from previous price)"
        )
    elif train_score > 0.90 and val_score > 0.90:
        report['warnings'] = report.get('warnings', [])
        report['warnings'].append(
            f"Very high scores (Train: {train_score:.4f}, Val: {val_score:.4f}) - "
            "may indicate trivial prediction task or data leakage"
        )
    
    return report


def check_feature_quality(X, y):
    """
    Check feature quality and redundancy.
    
    Returns:
        dict: Feature quality report
    """
    report = {
        'duplicate_features': [],
        'constant_features': [],
        'highly_correlated_pairs': [],
        'warnings': []
    }
    
    # Check for constant features
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X[col].nunique() <= 1:
            report['constant_features'].append(col)
    
    # Check for duplicate features (same values)
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            if X[col1].equals(X[col2]):
                report['duplicate_features'].append((col1, col2))
    
    # Check correlations
    if len(numeric_cols) > 1:
        corr_matrix = X[numeric_cols].corr()
        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i+1:]:
                corr = abs(corr_matrix.loc[col1, col2])
                if corr > 0.95:
                    report['highly_correlated_pairs'].append((col1, col2, corr))
    
    return report


def comprehensive_overfitting_check(X, y, train_score, val_score, test_score=None,
                                   time_col=None, target_col=None, threshold=0.05):
    """
    Comprehensive overfitting and data leakage check.
    
    Returns:
        dict: Complete report
    """
    report = {
        'data_leakage': check_data_leakage(X, y, time_col, target_col),
        'lookahead_bias': check_lookahead_bias(X, y, time_col, target_col) if time_col else {},
        'overfitting': detect_overfitting(train_score, val_score, test_score, threshold),
        'feature_quality': check_feature_quality(X, y),
        'overall_status': 'PASS',
        'recommendations': []
    }
    
    # Determine overall status
    if report['data_leakage']['has_leakage']:
        report['overall_status'] = 'FAIL - Data Leakage Detected'
    elif report['overfitting']['is_overfitting']:
        if report['overfitting']['severity'] == 'severe':
            report['overall_status'] = 'FAIL - Severe Overfitting'
        else:
            report['overall_status'] = 'WARNING - Overfitting Detected'
    
    # Generate recommendations
    if report['data_leakage']['has_leakage']:
        report['recommendations'].append("Remove features that contain target information")
        report['recommendations'].append("Ensure features are from past time periods only")
    
    if report['overfitting']['is_overfitting']:
        report['recommendations'].append("Increase regularization")
        report['recommendations'].append("Reduce model complexity")
        report['recommendations'].append("Add more training data")
        report['recommendations'].append("Use feature selection to remove redundant features")
    
    if report['feature_quality']['duplicate_features']:
        report['recommendations'].append("Remove duplicate features")
    
    if report['feature_quality']['highly_correlated_pairs']:
        report['recommendations'].append("Remove or combine highly correlated features")
    
    return report

