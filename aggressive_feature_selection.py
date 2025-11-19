#!/usr/bin/env python3
"""
Aggressive Feature Selection to Reduce Overfitting

Strategies:
1. Remove highly correlated features (more aggressive thresholds)
2. Use VIF (Variance Inflation Factor) to detect multicollinearity
3. Remove features that are linear combinations of others
4. Use recursive feature elimination
5. Remove features that don't add information beyond existing features
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')


def calculate_vif(X, feature_names=None):
    """
    Calculate Variance Inflation Factor (VIF) for each feature.
    
    VIF > 10 indicates high multicollinearity (redundancy).
    VIF > 5 is also concerning.
    
    Returns:
        vif_df: DataFrame with VIF scores
    """
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    # Standardize for VIF calculation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_array)
    
    # Calculate VIF
    vif_data = []
    for i in range(X_scaled.shape[1]):
        try:
            vif = variance_inflation_factor(X_scaled, i)
            vif_data.append({'feature': feature_names[i], 'vif': vif})
        except:
            vif_data.append({'feature': feature_names[i], 'vif': np.inf})
    
    vif_df = pd.DataFrame(vif_data).sort_values('vif', ascending=False)
    return vif_df


def remove_redundant_by_correlation(X, y, feature_names=None, threshold=0.85):
    """
    Aggressively remove redundant features based on correlation.
    
    Strategy:
    1. Calculate correlation matrix
    2. For each pair with correlation > threshold:
       - Calculate correlation with target
       - Remove the one with lower target correlation
    3. Iterate until no more redundant pairs
    
    Returns:
        selected_features: list of non-redundant features
        removal_log: list of removed features with reasons
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
    
    # Calculate correlation with target
    if isinstance(y, pd.Series):
        target_corr = X_df.corrwith(y).abs()
    else:
        target_corr = pd.Series([abs(np.corrcoef(X_df.iloc[:, i], y)[0, 1]) 
                                 for i in range(X_df.shape[1])], 
                                index=feature_names)
    
    selected_features = feature_names.copy()
    removal_log = []
    
    # Iteratively remove redundant features
    max_iterations = len(feature_names)
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        removed_this_iteration = False
        
        # Calculate correlation matrix for remaining features
        corr_matrix = X_df[selected_features].corr().abs()
        
        # Find highly correlated pairs
        for i, feat1 in enumerate(selected_features):
            if feat1 not in X_df.columns:
                continue
            for feat2 in selected_features[i+1:]:
                if feat2 not in X_df.columns:
                    continue
                
                corr = corr_matrix.loc[feat1, feat2]
                if corr > threshold:
                    # Remove the one with lower target correlation
                    if target_corr[feat1] >= target_corr[feat2]:
                        selected_features.remove(feat2)
                        removal_log.append({
                            'removed': feat2,
                            'kept': feat1,
                            'correlation': corr,
                            'reason': f'Lower target correlation than {feat1}'
                        })
                    else:
                        selected_features.remove(feat1)
                        removal_log.append({
                            'removed': feat1,
                            'kept': feat2,
                            'correlation': corr,
                            'reason': f'Lower target correlation than {feat2}'
                        })
                    removed_this_iteration = True
                    break
            
            if removed_this_iteration:
                break
        
        if not removed_this_iteration:
            break  # No more redundant features
    
    return selected_features, removal_log


def remove_redundant_by_vif(X, feature_names=None, vif_threshold=5.0):
    """
    Remove features with high VIF (multicollinearity).
    
    Returns:
        selected_features: list of features with VIF < threshold
        vif_scores: VIF scores for all features
    """
    vif_df = calculate_vif(X, feature_names)
    
    # Remove features with VIF > threshold
    high_vif = vif_df[vif_df['vif'] > vif_threshold]
    selected_features = vif_df[vif_df['vif'] <= vif_threshold]['feature'].tolist()
    
    return selected_features, vif_df, high_vif


def remove_linear_combinations(X, feature_names=None, tolerance=1e-6):
    """
    Remove features that are linear combinations of other features.
    
    Uses matrix rank to identify linearly dependent features.
    
    Returns:
        selected_features: linearly independent features
        removed_features: features that are linear combinations
    """
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_array)
    
    # Find linearly independent features using QR decomposition
    Q, R = np.linalg.qr(X_scaled)
    
    # Find features that are linear combinations
    # (columns of R with near-zero diagonal indicate linear dependence)
    diag_R = np.abs(np.diag(R))
    independent_mask = diag_R > tolerance
    
    selected_features = [feature_names[i] for i in range(len(feature_names)) if independent_mask[i]]
    removed_features = [feature_names[i] for i in range(len(feature_names)) if not independent_mask[i]]
    
    return selected_features, removed_features


def recursive_feature_elimination(X, y, feature_names=None, n_features=10, model=None):
    """
    Use Recursive Feature Elimination to select best features.
    
    Returns:
        selected_features: top n_features
        rankings: feature rankings
    """
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    if model is None:
        model = Ridge(alpha=1.0)
    
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    # Use RFE
    rfe = RFE(estimator=model, n_features_to_select=n_features, step=1)
    rfe.fit(X_array, y)
    
    selected_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
    rankings = {feature_names[i]: int(rfe.ranking_[i]) for i in range(len(feature_names))}
    
    return selected_features, rankings


def aggressive_feature_selection(X, y, feature_names=None,
                                correlation_threshold=0.85,
                                vif_threshold=5.0,
                                use_rfe=True,
                                n_features_final=10):
    """
    Comprehensive aggressive feature selection pipeline.
    
    Steps:
    1. Remove features with high VIF (multicollinearity)
    2. Remove highly correlated features
    3. Remove linear combinations
    4. Use RFE for final selection
    
    Returns:
        selected_features: final selected features
        report: detailed selection report
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
    
    current_features = feature_names.copy()
    
    # Step 1: Remove features with high VIF
    print("Step 1: Removing features with high VIF (multicollinearity)...")
    selected_vif, vif_df, high_vif = remove_redundant_by_vif(
        X_df[current_features], current_features, vif_threshold
    )
    
    removed_vif = [f for f in current_features if f not in selected_vif]
    report['steps'].append({
        'step': 'VIF-based removal',
        'threshold': vif_threshold,
        'removed': len(removed_vif),
        'remaining': len(selected_vif),
        'removed_features': removed_vif,
        'high_vif_features': high_vif['feature'].tolist() if len(high_vif) > 0 else []
    })
    
    current_features = selected_vif
    X_df = X_df[current_features]
    
    # Step 2: Remove highly correlated features
    print("Step 2: Removing highly correlated features...")
    selected_corr, removal_log = remove_redundant_by_correlation(
        X_df, y, current_features, threshold=correlation_threshold
    )
    
    removed_corr = [f for f in current_features if f not in selected_corr]
    report['steps'].append({
        'step': 'Correlation-based removal',
        'threshold': correlation_threshold,
        'removed': len(removed_corr),
        'remaining': len(selected_corr),
        'removed_features': removed_corr[:10],  # First 10
        'removal_log': removal_log[:10]  # First 10
    })
    
    current_features = selected_corr
    X_df = X_df[current_features]
    
    # Step 3: Remove linear combinations
    print("Step 3: Removing linear combinations...")
    selected_linear, removed_linear = remove_linear_combinations(
        X_df, current_features
    )
    
    report['steps'].append({
        'step': 'Linear combination removal',
        'removed': len(removed_linear),
        'remaining': len(selected_linear),
        'removed_features': removed_linear
    })
    
    current_features = selected_linear
    X_df = X_df[current_features]
    
    # Step 4: Recursive Feature Elimination (optional)
    if use_rfe and len(current_features) > n_features_final:
        print(f"Step 4: Recursive Feature Elimination (selecting top {n_features_final})...")
        selected_rfe, rankings = recursive_feature_elimination(
            X_df, y, current_features, n_features=n_features_final
        )
        
        removed_rfe = [f for f in current_features if f not in selected_rfe]
        report['steps'].append({
            'step': 'Recursive Feature Elimination',
            'n_features': n_features_final,
            'removed': len(removed_rfe),
            'remaining': len(selected_rfe),
            'removed_features': removed_rfe,
            'rankings': rankings
        })
        
        current_features = selected_rfe
    
    report['final_features'] = len(current_features)
    report['selected_features'] = current_features
    report['vif_scores'] = vif_df.to_dict('records') if 'vif_df' in locals() else []
    
    return current_features, report


def analyze_feature_redundancy(X, y, feature_names=None):
    """
    Analyze feature redundancy and provide recommendations.
    
    Returns:
        analysis: dict with redundancy analysis
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
    
    analysis = {
        'correlation_analysis': {},
        'vif_analysis': {},
        'recommendations': []
    }
    
    # Correlation analysis
    corr_matrix = X_df.corr().abs()
    high_corr_pairs = []
    for i, feat1 in enumerate(feature_names):
        for feat2 in feature_names[i+1:]:
            corr = corr_matrix.loc[feat1, feat2]
            if corr > 0.85:
                high_corr_pairs.append({
                    'feature1': feat1,
                    'feature2': feat2,
                    'correlation': corr
                })
    
    analysis['correlation_analysis'] = {
        'high_correlation_pairs': high_corr_pairs,
        'num_high_corr_pairs': len(high_corr_pairs)
    }
    
    # VIF analysis
    vif_df = calculate_vif(X_df, feature_names)
    high_vif = vif_df[vif_df['vif'] > 5.0]
    
    analysis['vif_analysis'] = {
        'high_vif_features': high_vif.to_dict('records'),
        'num_high_vif': len(high_vif)
    }
    
    # Recommendations
    if len(high_corr_pairs) > 0:
        analysis['recommendations'].append(
            f"Remove {len(high_corr_pairs)} redundant feature pairs (correlation > 0.85)"
        )
    
    if len(high_vif) > 0:
        analysis['recommendations'].append(
            f"Remove {len(high_vif)} features with high VIF (multicollinearity)"
        )
    
    return analysis



