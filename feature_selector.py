#!/usr/bin/env python3
"""
Feature Selection Module

Removes redundant features and prevents overfitting by:
1. Removing highly correlated features
2. Separating FRED vs Technical features
3. Using mutual information for feature selection
4. Removing features that don't add predictive power
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    VarianceThreshold
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def identify_feature_categories(feature_names):
    """
    Categorize features into FRED, Technical, Halving, etc.
    
    Returns:
        dict: {category: [feature_names]}
    """
    categories = {
        'FRED': [],
        'Technical': [],
        'Halving': [],
        'OnChain': [],
        'Other': []
    }
    
    for feature in feature_names:
        feature_lower = feature.lower()
        if any(x in feature_lower for x in ['m2', 'net_liquidity', 'fedfunds', 'cpi', 'gfdebtn', 'walcl', 'dgs10', 'unrate']):
            categories['FRED'].append(feature)
        elif any(x in feature_lower for x in ['sma', 'ema', 'rsi', 'macd', 'volatility', 'return', 'roc', 'slope', 'ratio']):
            categories['Technical'].append(feature)
        elif any(x in feature_lower for x in ['halving', 'days_from', 'months_from']):
            categories['Halving'].append(feature)
        elif any(x in feature_lower for x in ['asopr', 'thermo', 'stablecoin']):
            categories['OnChain'].append(feature)
        else:
            categories['Other'].append(feature)
    
    return categories


def remove_highly_correlated_features(X, y, threshold=0.95, feature_names=None):
    """
    Remove features that are highly correlated with each other.
    
    Keeps the feature that has higher correlation with target.
    
    Returns:
        selected_features: list of feature names to keep
        removed_features: list of feature names removed
    """
    if feature_names is None:
        feature_names = list(X.columns) if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
    
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X, columns=feature_names)
    
    # Calculate correlation matrix
    corr_matrix = X_df.corr().abs()
    
    # Calculate correlation with target
    if isinstance(y, pd.Series):
        target_corr = X_df.corrwith(y).abs()
    else:
        target_corr = pd.Series([abs(np.corrcoef(X_df.iloc[:, i], y)[0, 1]) 
                                 for i in range(X_df.shape[1])], 
                                index=feature_names)
    
    # Find highly correlated pairs
    to_remove = set()
    to_keep = set(feature_names)
    
    for i, col1 in enumerate(corr_matrix.columns):
        if col1 in to_remove:
            continue
        for col2 in corr_matrix.columns[i+1:]:
            if col2 in to_remove:
                continue
            
            corr = corr_matrix.loc[col1, col2]
            if corr > threshold:
                # Keep the one with higher correlation to target
                if target_corr[col1] >= target_corr[col2]:
                    to_remove.add(col2)
                else:
                    to_remove.add(col1)
                    break  # col1 was removed, move to next
    
    selected_features = [f for f in feature_names if f not in to_remove]
    removed_features = list(to_remove)
    
    return selected_features, removed_features


def remove_cross_category_redundancy(X, y, feature_names, threshold=0.90):
    """
    Remove redundant features between FRED and Technical categories.
    
    If a FRED feature and Technical feature are highly correlated,
    keep the one with higher predictive power.
    
    Returns:
        selected_features: list of feature names to keep
        removal_report: dict with details
    """
    categories = identify_feature_categories(feature_names)
    
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X, columns=feature_names)
    
    # Calculate correlation with target
    if isinstance(y, pd.Series):
        target_corr = X_df.corrwith(y).abs()
    else:
        target_corr = pd.Series([abs(np.corrcoef(X_df.iloc[:, i], y)[0, 1]) 
                                 for i in range(X_df.shape[1])], 
                                index=feature_names)
    
    # Check cross-category correlations
    to_remove = set()
    removal_report = {
        'fred_technical_conflicts': [],
        'removed_features': []
    }
    
    # Check FRED vs Technical
    for fred_feat in categories['FRED']:
        if fred_feat not in X_df.columns or fred_feat in to_remove:
            continue
        for tech_feat in categories['Technical']:
            if tech_feat not in X_df.columns or tech_feat in to_remove:
                continue
            
            corr = abs(X_df[fred_feat].corr(X_df[tech_feat]))
            if corr > threshold:
                # Keep the one with higher correlation to target
                if target_corr[fred_feat] >= target_corr[tech_feat]:
                    to_remove.add(tech_feat)
                    removal_report['fred_technical_conflicts'].append({
                        'fred_feature': fred_feat,
                        'technical_feature': tech_feat,
                        'correlation': corr,
                        'removed': tech_feat,
                        'reason': 'FRED feature has higher target correlation'
                    })
                else:
                    to_remove.add(fred_feat)
                    removal_report['fred_technical_conflicts'].append({
                        'fred_feature': fred_feat,
                        'technical_feature': tech_feat,
                        'correlation': corr,
                        'removed': fred_feat,
                        'reason': 'Technical feature has higher target correlation'
                    })
                    break  # fred_feat was removed
    
    selected_features = [f for f in feature_names if f not in to_remove]
    removal_report['removed_features'] = list(to_remove)
    
    return selected_features, removal_report


def select_features_mutual_info(X, y, feature_names, k=20):
    """
    Select top k features using mutual information.
    
    Returns:
        selected_features: list of top k feature names
        scores: dict of {feature: score}
    """
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    # Calculate mutual information
    mi_scores = mutual_info_regression(X_array, y, random_state=42)
    
    # Get top k features
    top_k_indices = np.argsort(mi_scores)[-k:][::-1]
    selected_features = [feature_names[i] for i in top_k_indices]
    
    scores = {feature_names[i]: float(mi_scores[i]) for i in range(len(feature_names))}
    
    return selected_features, scores


def comprehensive_feature_selection(X, y, feature_names=None, 
                                   correlation_threshold=0.95,
                                   cross_category_threshold=0.90,
                                   use_mutual_info=True,
                                   top_k=None):
    """
    Comprehensive feature selection pipeline.
    
    Steps:
    1. Remove constant/low variance features
    2. Remove highly correlated features within categories
    3. Remove redundant features between FRED and Technical
    4. Select top features using mutual information (optional)
    
    Returns:
        selected_features: list of selected feature names
        report: dict with selection details
    """
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X, columns=feature_names)
    
    report = {
        'initial_features': len(feature_names),
        'steps': []
    }
    
    # Step 1: Remove constant features
    variance_selector = VarianceThreshold(threshold=0.01)
    X_variance = variance_selector.fit_transform(X_df)
    kept_indices = variance_selector.get_support()
    current_features = [feature_names[i] for i in range(len(feature_names)) if kept_indices[i]]
    
    report['steps'].append({
        'step': 'Remove constant features',
        'removed': len(feature_names) - len(current_features),
        'remaining': len(current_features)
    })
    
    X_df = X_df[current_features]
    
    # Step 2: Remove highly correlated features
    selected_features, removed = remove_highly_correlated_features(
        X_df, y, threshold=correlation_threshold, feature_names=current_features
    )
    
    report['steps'].append({
        'step': 'Remove highly correlated features',
        'threshold': correlation_threshold,
        'removed': len(removed),
        'remaining': len(selected_features),
        'removed_features': removed[:10]  # First 10
    })
    
    X_df = X_df[selected_features]
    
    # Step 3: Remove cross-category redundancy (FRED vs Technical)
    selected_features, cross_report = remove_cross_category_redundancy(
        X_df, y, selected_features, threshold=cross_category_threshold
    )
    
    report['steps'].append({
        'step': 'Remove cross-category redundancy (FRED vs Technical)',
        'threshold': cross_category_threshold,
        'removed': len(cross_report['removed_features']),
        'remaining': len(selected_features),
        'conflicts': len(cross_report['fred_technical_conflicts']),
        'conflict_details': cross_report['fred_technical_conflicts'][:5]  # First 5
    })
    
    X_df = X_df[selected_features]
    
    # Step 4: Mutual information selection (optional)
    if use_mutual_info and top_k and top_k < len(selected_features):
        selected_features, mi_scores = select_features_mutual_info(
            X_df, y, selected_features, k=top_k
        )
        
        report['steps'].append({
            'step': 'Mutual information selection',
            'top_k': top_k,
            'remaining': len(selected_features),
            'top_features': sorted(selected_features, 
                                  key=lambda x: mi_scores.get(x, 0), 
                                  reverse=True)[:10]
        })
    
    report['final_features'] = len(selected_features)
    report['selected_features'] = selected_features
    report['feature_categories'] = identify_feature_categories(selected_features)
    
    return selected_features, report


def apply_feature_selection(X, y, feature_names=None, **kwargs):
    """
    Apply feature selection and return filtered X.
    
    Returns:
        X_selected: filtered feature matrix
        selected_features: list of selected feature names
        report: selection report
    """
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    selected_features, report = comprehensive_feature_selection(
        X, y, feature_names, **kwargs
    )
    
    if isinstance(X, pd.DataFrame):
        X_selected = X[selected_features]
    else:
        # Convert to DataFrame, select, convert back
        X_df = pd.DataFrame(X, columns=feature_names)
        X_selected = X_df[selected_features].values
    
    return X_selected, selected_features, report



