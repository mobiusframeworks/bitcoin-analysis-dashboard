#!/usr/bin/env python3
"""
Comprehensive Data Leakage and Look-Ahead Bias Detector

Checks for:
1. Same-day OHLC leakage (using open/high/low to predict close)
2. Look-ahead bias (using future information)
3. Target leakage (target in features)
4. Perfect correlations
5. Time-based leakage
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


def detect_ohlc_leakage(feature_names, target_col='close'):
    """
    Detect if OHLC features from same day are being used to predict close.
    
    This is data leakage because:
    - close is between high and low
    - close is related to open
    - Using same-day OHLC to predict same-day close is trivial
    """
    issues = []
    warnings_list = []
    
    # Check for same-day OHLC features
    ohlc_patterns = {
        'open': ['open', 'prev_open'],
        'high': ['high', 'prev_high'],
        'low': ['low', 'prev_low'],
        'close': ['close', 'prev_close']
    }
    
    target_lower = target_col.lower()
    
    # Check if target is close and we have same-day OHLC
    if 'close' in target_lower:
        for feature in feature_names:
            feature_lower = feature.lower()
            
            # Check for same-day open/high/low (not lagged)
            if any(pattern in feature_lower for pattern in ['open', 'high', 'low']) and \
               'prev' not in feature_lower and 'lag' not in feature_lower:
                issues.append({
                    'type': 'OHLC_LEAKAGE',
                    'feature': feature,
                    'issue': f'Same-day {feature} used to predict close (data leakage)',
                    'severity': 'CRITICAL'
                })
            
            # Check for prev_open/high/low with prev_close (still problematic if same day)
            if 'prev_' in feature_lower and any(x in feature_lower for x in ['open', 'high', 'low']):
                if any('prev_close' in f.lower() for f in feature_names):
                    warnings_list.append({
                        'type': 'OHLC_WARNING',
                        'feature': feature,
                        'issue': f'prev_{feature} and prev_close may be from same day',
                        'severity': 'WARNING'
                    })
    
    return issues, warnings_list


def detect_lookahead_bias(feature_names, time_col=None):
    """
    Detect look-ahead bias - using future information to predict past.
    """
    issues = []
    
    # Check for forward-looking indicators
    forward_patterns = [
        'forward', 'future', 'ahead', 'next', 'shift(-', 'shift_neg',
        'lead', 'shift_forward'
    ]
    
    for feature in feature_names:
        feature_lower = feature.lower()
        
        # Check for forward-looking patterns
        if any(pattern in feature_lower for pattern in forward_patterns):
            issues.append({
                'type': 'LOOKAHEAD_BIAS',
                'feature': feature,
                'issue': f'Feature {feature} appears to use future information',
                'severity': 'CRITICAL'
            })
        
        # Check for negative shifts (Python shift(-n) means future)
        if 'shift' in feature_lower and '-' in feature_lower:
            issues.append({
                'type': 'LOOKAHEAD_BIAS',
                'feature': feature,
                'issue': f'Feature {feature} uses negative shift (future data)',
                'severity': 'CRITICAL'
            })
    
    return issues


def detect_target_leakage(feature_names, target_col):
    """
    Detect if target column is in features.
    """
    issues = []
    
    target_lower = target_col.lower()
    
    for feature in feature_names:
        feature_lower = feature.lower()
        
        # Exact match
        if feature_lower == target_lower:
            issues.append({
                'type': 'TARGET_LEAKAGE',
                'feature': feature,
                'issue': f'Target column {target_col} found in features',
                'severity': 'CRITICAL'
            })
        
        # Partial match (e.g., 'close' in 'prev_close' when target is 'close')
        if target_lower in feature_lower and 'prev' not in feature_lower and 'lag' not in feature_lower:
            if feature_lower != target_lower:  # Already caught above
                issues.append({
                    'type': 'TARGET_LEAKAGE',
                    'feature': feature,
                    'issue': f'Feature {feature} contains target name {target_col}',
                    'severity': 'WARNING'
                })
    
    return issues


def detect_perfect_correlations(X, feature_names, threshold=0.99):
    """
    Detect features with perfect or near-perfect correlations.
    """
    issues = []
    
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X, columns=feature_names)
    
    corr_matrix = X_df.corr().abs()
    
    for i, feat1 in enumerate(corr_matrix.columns):
        for feat2 in corr_matrix.columns[i+1:]:
            corr = corr_matrix.loc[feat1, feat2]
            if corr > threshold:
                issues.append({
                    'type': 'PERFECT_CORRELATION',
                    'feature1': feat1,
                    'feature2': feat2,
                    'correlation': corr,
                    'issue': f'{feat1} and {feat2} have correlation {corr:.4f} (redundant)',
                    'severity': 'HIGH' if corr > 0.99 else 'MEDIUM'
                })
    
    return issues


def detect_time_leakage(feature_names, time_col=None):
    """
    Detect if time-based features leak information.
    """
    issues = []
    
    # Check for time features that might leak
    time_patterns = ['date', 'time', 'timestamp', 'unix', 'year', 'month', 'day']
    
    for feature in feature_names:
        feature_lower = feature.lower()
        
        # Check for raw time features
        if any(pattern in feature_lower for pattern in time_patterns):
            if time_col and feature_lower == time_col.lower():
                # Time column itself is OK if removed from features
                continue
            
            # Check if it's a derived time feature (OK) vs raw time (leakage)
            if not any(x in feature_lower for x in ['days_from', 'months_from', 'since', 'cycle']):
                issues.append({
                    'type': 'TIME_LEAKAGE',
                    'feature': feature,
                    'issue': f'Raw time feature {feature} may leak information',
                    'severity': 'MEDIUM'
                })
    
    return issues


def comprehensive_leakage_check(X, y, feature_names, target_col='close', time_col=None):
    """
    Comprehensive data leakage and look-ahead bias check.
    
    Returns:
        report: dict with all leakage issues
    """
    report = {
        'ohlc_leakage': {'issues': [], 'warnings': []},
        'lookahead_bias': {'issues': []},
        'target_leakage': {'issues': []},
        'perfect_correlations': {'issues': []},
        'time_leakage': {'issues': []},
        'overall_status': 'PASS',
        'critical_issues': [],
        'recommendations': []
    }
    
    # Check OHLC leakage
    ohlc_issues, ohlc_warnings = detect_ohlc_leakage(feature_names, target_col)
    report['ohlc_leakage']['issues'] = ohlc_issues
    report['ohlc_leakage']['warnings'] = ohlc_warnings
    
    # Check look-ahead bias
    lookahead_issues = detect_lookahead_bias(feature_names, time_col)
    report['lookahead_bias']['issues'] = lookahead_issues
    
    # Check target leakage
    target_issues = detect_target_leakage(feature_names, target_col)
    report['target_leakage']['issues'] = target_issues
    
    # Check perfect correlations
    if X is not None:
        corr_issues = detect_perfect_correlations(X, feature_names)
        report['perfect_correlations']['issues'] = corr_issues
    
    # Check time leakage
    time_issues = detect_time_leakage(feature_names, time_col)
    report['time_leakage']['issues'] = time_issues
    
    # Collect all critical issues
    all_issues = ohlc_issues + lookahead_issues + target_issues
    report['critical_issues'] = [issue for issue in all_issues if issue.get('severity') == 'CRITICAL']
    
    # Determine overall status
    if len(report['critical_issues']) > 0:
        report['overall_status'] = 'FAIL - Critical Data Leakage Detected'
    elif len(all_issues) > 0:
        report['overall_status'] = 'WARNING - Potential Issues Detected'
    else:
        report['overall_status'] = 'PASS - No Data Leakage Detected'
    
    # Generate recommendations
    if len(ohlc_issues) > 0:
        report['recommendations'].append(
            "Remove same-day OHLC features. Only use LAGGED OHLC (prev_open, prev_high, prev_low, prev_close)"
        )
    
    if len(lookahead_issues) > 0:
        report['recommendations'].append(
            "Remove features that use future information. Only use past data (shift(1) or higher, not shift(-1))"
        )
    
    if len(target_issues) > 0:
        report['recommendations'].append(
            f"Remove target column '{target_col}' from features"
        )
    
    if len(corr_issues) > 0:
        report['recommendations'].append(
            f"Remove {len(corr_issues)} redundant features with perfect correlations"
        )
    
    return report


def remove_leaky_features(feature_names, leakage_report):
    """
    Remove features identified as having data leakage.
    
    Returns:
        clean_features: list of features without leakage
        removed_features: list of removed features
    """
    to_remove = set()
    
    # Remove features with critical issues
    for issue in leakage_report['critical_issues']:
        to_remove.add(issue['feature'])
    
    # Remove OHLC leakage features
    for issue in leakage_report['ohlc_leakage']['issues']:
        to_remove.add(issue['feature'])
    
    # Remove look-ahead bias features
    for issue in leakage_report['lookahead_bias']['issues']:
        to_remove.add(issue['feature'])
    
    # Remove target leakage features
    for issue in leakage_report['target_leakage']['issues']:
        if issue.get('severity') == 'CRITICAL':
            to_remove.add(issue['feature'])
    
    clean_features = [f for f in feature_names if f not in to_remove]
    removed_features = list(to_remove)
    
    return clean_features, removed_features



