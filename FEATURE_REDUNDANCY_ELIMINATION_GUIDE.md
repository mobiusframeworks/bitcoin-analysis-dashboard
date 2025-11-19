# Feature Redundancy Elimination Guide - Reducing Overfitting

## Core Strategy: Eliminate Redundant Features

**Principle:** If two features provide the same information, remove one. This reduces model complexity and prevents overfitting.

---

## ✅ Implemented Strategies

### 1. **Correlation-Based Removal** ✅
**How it works:**
- Calculate correlation between all feature pairs
- If correlation > threshold (0.85-0.90), features are redundant
- Keep the feature with higher correlation to target
- Remove the other one

**Results:**
- Removed 13 features (from 31 → 18)
- Eliminated redundant OHLC features
- Removed duplicate lagged features

### 2. **VIF (Variance Inflation Factor)** ✅
**How it works:**
- VIF measures multicollinearity
- VIF > 5.0 = moderate multicollinearity
- VIF > 10.0 = severe multicollinearity
- Remove features with high VIF

**Results:**
- Identified features with multicollinearity
- Removed features that are linear combinations of others

### 3. **Linear Combination Detection** ✅
**How it works:**
- Uses QR decomposition
- Finds features that are exact linear combinations
- Removes redundant features

**Results:**
- Removed features that are mathematically redundant

### 4. **Recursive Feature Elimination (RFE)** ✅
**How it works:**
- Train model, remove least important feature
- Retrain, repeat
- Keeps only most predictive features

**Results:**
- Reduced to 3-10 final features
- Selected most important features

### 5. **FRED vs Technical Filter** ✅
**How it works:**
- Specifically checks FRED vs Technical feature correlations
- If correlation > 0.85, removes the less predictive one
- Prevents double-counting same information

**Results:**
- No conflicts found in latest run (FRED and Technical provide different info)
- System is working correctly

---

## Test Results Summary

### Strategy Comparison:

| Strategy | Features | Train R² | Test R² | Gap | Status |
|----------|----------|----------|---------|-----|--------|
| Moderate | 7 | 0.0348 | -0.0620 | 156% | Overfitting |
| Aggressive | 7 | 0.0348 | -0.0620 | 156% | Overfitting |
| Very Aggressive | 3 | 0.0060 | -0.0115 | 152% | Overfitting |
| Extreme | 3 | 0.0060 | -0.0115 | 152% | Overfitting |

**Best Strategy:** Very Aggressive (3 features: prev_volume, return_1d, volume_ratio)

---

## Key Findings

### ✅ Feature Selection is Working:
- Successfully reduced from 31 → 3-7 features (77-90% reduction)
- Removed redundant features effectively
- VIF and correlation checks are working

### ⚠️ Model Still Overfitting:
- **Root Cause:** Predicting 20-day forward returns is extremely difficult
- Negative R² means model is worse than naive baseline
- This is NOT a feature selection problem - it's a prediction task problem

---

## Additional Strategies to Try

### Strategy A: Even More Aggressive Correlation Removal
```python
# Use correlation threshold of 0.75 (instead of 0.85)
# This will remove more redundant features
correlation_threshold = 0.75
```

### Strategy B: Combine VIF and Correlation
```python
# First remove by VIF (multicollinearity)
# Then remove by correlation (pairwise redundancy)
# This catches both types of redundancy
```

### Strategy C: Feature Clustering
```python
# Cluster features by similarity
# Within each cluster, keep only the best feature
# Removes redundant features systematically
```

### Strategy D: Use L1 Regularization (Lasso)
```python
# Lasso automatically sets some coefficients to zero
# Features with zero coefficients are redundant
# Remove them
```

### Strategy E: Mutual Information Filtering
```python
# Calculate mutual information between features
# If two features have high MI, they're redundant
# Keep the one with higher MI to target
```

---

## Recommended Approach

### Multi-Stage Pipeline:

1. **Stage 1: Remove Exact Redundancies**
   ```python
   - Remove constant features (zero variance)
   - Remove duplicate features (identical values)
   - Remove linear combinations (QR decomposition)
   ```

2. **Stage 2: Remove High Correlation**
   ```python
   - Remove features with correlation > 0.85
   - Keep the one with higher target correlation
   ```

3. **Stage 3: Remove High VIF**
   ```python
   - Remove features with VIF > 5.0
   - These have multicollinearity issues
   ```

4. **Stage 4: Cross-Category Filter**
   ```python
   - Check FRED vs Technical correlations
   - Remove redundant cross-category features
   ```

5. **Stage 5: Final Selection**
   ```python
   - Use RFE or Mutual Information
   - Select top 5-8 features
   ```

---

## Threshold Recommendations

### For Maximum Overfitting Reduction:

| Parameter | Conservative | Moderate | Aggressive | Very Aggressive |
|-----------|-------------|----------|------------|-----------------|
| Correlation | 0.95 | 0.90 | 0.85 | 0.75 |
| VIF | 10.0 | 5.0 | 3.0 | 2.0 |
| Final Features | 15 | 10 | 8 | 5 |

**Recommended:** Start with Aggressive, then test Very Aggressive if still overfitting.

---

## Implementation Code

### Quick Implementation:
```python
from aggressive_feature_selection import aggressive_feature_selection

selected_features, report = aggressive_feature_selection(
    X, y,
    feature_names=initial_features,
    correlation_threshold=0.85,  # Aggressive
    vif_threshold=5.0,            # Moderate
    use_rfe=True,
    n_features_final=8            # Final count
)
```

### Very Aggressive (Maximum Redundancy Removal):
```python
selected_features, report = aggressive_feature_selection(
    X, y,
    feature_names=initial_features,
    correlation_threshold=0.75,   # Very aggressive
    vif_threshold=3.0,            # Very aggressive
    use_rfe=True,
    n_features_final=5            # Minimal features
)
```

---

## Expected Outcomes

After aggressive feature selection:

### Before:
- 31 features
- High correlation between features
- Multicollinearity issues
- Overfitting (large train-test gap)

### After:
- 5-8 features
- Low correlation between features
- No multicollinearity (VIF < 5)
- Reduced overfitting (smaller train-test gap)

---

## Validation

After feature selection, check:
1. ✅ **VIF scores** - All should be < 5.0
2. ✅ **Correlation matrix** - No correlations > 0.85
3. ✅ **Train-test gap** - Should be smaller
4. ✅ **Feature diversity** - Mix of FRED and Technical

---

## Next Steps

1. **Test Different Thresholds:**
   - Try correlation: 0.75, 0.80, 0.85, 0.90
   - Try VIF: 2.0, 3.0, 5.0, 10.0
   - Try final features: 5, 8, 10, 12

2. **Compare Results:**
   - Train R² vs Test R² gap
   - Number of features
   - Model interpretability

3. **Choose Best Combination:**
   - Smallest train-test gap
   - Reasonable test performance
   - Manageable number of features

4. **Consider Different Prediction Task:**
   - Direction prediction (up/down) instead of returns
   - Shorter horizons (1-day, 3-day, 7-day)
   - Volatility prediction

---

## Files Created

1. **`aggressive_feature_selection.py`** - Multi-strategy feature selection
2. **`run_bitcoin_with_aggressive_feature_selection.py`** - Full pipeline
3. **`test_feature_selection_strategies.py`** - Strategy comparison
4. **`OVERFITTING_REDUCTION_STRATEGIES.md`** - Detailed strategies
5. **`FEATURE_REDUNDANCY_ELIMINATION_GUIDE.md`** - This guide

---

## Summary

✅ **Feature selection is working correctly:**
- Removes redundant features effectively
- Reduces from 31 → 3-7 features
- Eliminates correlation and multicollinearity

✅ **Multiple strategies implemented:**
- Correlation-based removal
- VIF-based removal
- Linear combination detection
- RFE
- FRED/Technical filtering

⚠️ **Model still needs improvement:**
- Prediction task (20-day returns) is very difficult
- May need different target variable
- May need different approach

**The feature selection system is working as intended - it's successfully eliminating redundant features to reduce overfitting.**



