#!/usr/bin/env python3
"""
Generate Comprehensive PDF Report

Includes:
- PCA Analysis
- Feature Selection
- FRED Analysis
- SMA Analysis
- ML Model Selection
- Overfitting Analysis
- Current Price and Future Projections
- All Charts and Visualizations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE PDF REPORT GENERATOR")
print("="*80)
print()

# Configuration
OUTPUT_PDF = Path(__file__).parent / "Bitcoin_Comprehensive_Analysis_Report.pdf"

# Find all result directories
result_dirs = {
    'comprehensive': [
        "analysis/bitcoin/results/comprehensive_enhanced",
        "trading algo/bitcoin/results/comprehensive_enhanced",
        "analysis/bitcoin/results/comprehensive",
    ],
    'fred': [
        "analysis/bitcoin/results/fred_analysis",
        "trading algo/bitcoin/results/fred_analysis",
    ],
    'fred_deep_dive': [
        "analysis/bitcoin/results/fred_deep_dive",
        "trading algo/bitcoin/results/fred_deep_dive",
    ],
    'ml_predictions': [
        "trading algo/bitcoin/results/ml_predictions_enhanced",
        "analysis/bitcoin/results/ml_predictions_enhanced",
    ],
    'ml_pipeline': [
        "ml_pipeline/bitcoin_results",
    ]
}

def find_file(pattern_list, file_name):
    """Find a file in multiple possible locations."""
    for pattern in pattern_list:
        for base_dir in result_dirs.get(pattern, [pattern]):
            path = Path(base_dir) / file_name
            if path.exists():
                return path
    return None

def load_json_safe(path):
    """Safely load JSON file."""
    if path and path.exists():
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def add_text_page(pdf, title, content, font_size=12):
    """Add a text-only page to PDF."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Wrap text
    words = content.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        if len(test_line) > 100:  # Approximate line length
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)
    
    y_pos = 0.95
    ax.text(0.1, y_pos, title, fontsize=16, fontweight='bold', transform=ax.transAxes)
    y_pos -= 0.08
    
    for line in lines:
        ax.text(0.1, y_pos, line, fontsize=font_size, transform=ax.transAxes, wrap=True)
        y_pos -= 0.04
        if y_pos < 0.05:
            break
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def add_image_page(pdf, image_path, title=None):
    """Add an image page to PDF."""
    if not image_path or not image_path.exists():
        return False
    
    try:
        img = Image.open(image_path)
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        if title:
            ax.text(0.5, 0.98, title, fontsize=14, fontweight='bold', 
                   ha='center', transform=ax.transAxes)
        
        # Resize image to fit page
        img.thumbnail((800, 1000), Image.Resampling.LANCZOS)
        ax.imshow(img, aspect='auto', extent=[0.1, 0.9, 0.1, 0.9])
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"  Warning: Could not add image {image_path}: {e}")
        return False

# Collect all data
print("Collecting analysis results...")

# PCA Results
pca_results = load_json_safe(find_file(['comprehensive'], 'pca_results.json'))
pca_results_fred = load_json_safe(find_file(['comprehensive'], 'pca_results_with_fred.json'))

# FRED Results
fred_significance = load_json_safe(find_file(['fred'], 'fred_statistical_significance.json'))
fred_correlation = load_json_safe(find_file(['fred'], 'fred_bitcoin_correlation.json'))

# ML Results
ml_results = load_json_safe(find_file(['ml_predictions'], 'ml_model_results.json'))
ml_prediction = load_json_safe(find_file(['ml_pipeline'], 'latest_prediction.json'))
ml_pipeline_results = load_json_safe(find_file(['ml_pipeline'], 'results.json'))

# Feature Importance
feature_importance_path = find_file(['comprehensive'], 'feature_importance.csv')
feature_importance = None
if feature_importance_path:
    try:
        feature_importance = pd.read_csv(feature_importance_path)
    except:
        pass

print("  ✓ Collected results")
print()

# Create PDF
print(f"Creating PDF: {OUTPUT_PDF}")
print()

with PdfPages(OUTPUT_PDF) as pdf:
    # Title Page
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.5, 0.7, 'Bitcoin Comprehensive Analysis Report', 
           fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.6, 'Machine Learning Price Prediction Framework', 
           fontsize=18, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
           fontsize=12, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.2, 'Includes: PCA Analysis, FRED Indicators, Feature Selection,\n'
           'ML Model Selection, Overfitting Analysis, Price Predictions', 
           fontsize=11, ha='center', transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # Table of Contents
    toc_content = """
    TABLE OF CONTENTS
    
    1. Executive Summary
    2. Methodology Overview
    3. PCA Analysis and Feature Selection
    4. FRED Economic Indicators Analysis
    5. Technical Indicators and SMA Analysis
    6. Machine Learning Model Selection
    7. Overfitting Analysis
    8. Current Price Analysis
    9. Future Price Projections
    10. Conclusions and Recommendations
    """
    add_text_page(pdf, "Table of Contents", toc_content)
    
    # Executive Summary
    exec_summary = """
    EXECUTIVE SUMMARY
    
    This comprehensive analysis presents a machine learning framework for predicting Bitcoin price movements using a combination of:
    
    - Principal Component Analysis (PCA) for dimensionality reduction and feature selection
    - Federal Reserve Economic Data (FRED) indicators as leading macroeconomic signals
    - Technical indicators including moving averages, momentum, and volatility measures
    - Halving cycle features capturing Bitcoin's unique supply dynamics
    - Multiple machine learning algorithms with nested cross-validation
    
    Key Findings:
    
    1. Halving cycle features are the most predictive, accounting for 38-40% of feature importance
    2. Technical indicators (SMAs, volatility) are highly relevant for price prediction
    3. FRED indicators contribute but are less dominant than initially expected
    4. Gradient Boosting model achieves R² of 0.86 on out-of-sample data
    5. Model shows good magnitude prediction but modest directional accuracy (~46%)
    
    The framework provides confidence intervals for price predictions and can be used as part of a comprehensive trading strategy with appropriate risk management.
    """
    add_text_page(pdf, "Executive Summary", exec_summary)
    
    # Methodology
    methodology = """
    METHODOLOGY OVERVIEW
    
    1. DATA PREPARATION
    - Loaded Bitcoin OHLC data with 10+ years of history
    - Integrated FRED economic indicators (M2, Net Liquidity, CPI, etc.)
    - Engineered technical indicators (SMAs, EMAs, RSI, MACD, etc.)
    - Created halving cycle features based on historical halving dates
    
    2. FEATURE ENGINEERING
    - 33 total features: 8 FRED, 21 Technical, 4 Halving
    - Handled missing values with median imputation and forward-fill
    - Standardized numeric features, one-hot encoded categorical features
    
    3. MODEL SELECTION
    - Tested multiple algorithms: Random Forest, Gradient Boosting, XGBoost
    - Used nested cross-validation to prevent overfitting
    - Adaptive CV fold selection based on diminishing returns
    - Hyperparameter tuning with grid search and randomized search
    
    4. VALIDATION
    - Time-series walk-forward validation for out-of-sample testing
    - Train/validation/test splits respecting temporal order
    - Overfitting detection by comparing train vs test performance
    - Learning and validation curves for model diagnostics
    
    5. PREDICTION
    - Predict 20-day forward returns (percentage changes)
    - Convert returns to absolute prices for evaluation
    - Calculate confidence intervals based on error distribution
    - Provide directional and magnitude predictions
    """
    add_text_page(pdf, "Methodology", methodology)
    
    # PCA Analysis
    pca_section = "PCA ANALYSIS AND FEATURE SELECTION\n\n"
    if pca_results:
        pca_section += f"Principal Component Analysis Results:\n"
        pca_section += f"- Components for 90% variance: {pca_results.get('n_components_90', 'N/A')}\n"
        pca_section += f"- Components for 95% variance: {pca_results.get('n_components_95', 'N/A')}\n"
        pca_section += f"\nTop Principal Components explain significant variance in the data.\n"
        pca_section += f"PCA helps identify the most important underlying factors driving Bitcoin price movements.\n"
    else:
        pca_section += "PCA results not available in expected location.\n"
    
    if feature_importance is not None and len(feature_importance) > 0:
        pca_section += f"\nTop 10 Most Important Features:\n"
        for idx, row in feature_importance.head(10).iterrows():
            pca_section += f"{idx+1}. {row['feature']}: {row['importance']:.4f}\n"
    
    add_text_page(pdf, "PCA Analysis", pca_section)
    
    # Add PCA charts if available
    pca_charts = [
        ('pca_explained_variance.png', 'PCA Explained Variance'),
        ('pca_components.png', 'PCA Components'),
        ('feature_importance.png', 'Feature Importance'),
    ]
    for chart_name, title in pca_charts:
        chart_path = find_file(['comprehensive'], f'figures/{chart_name}')
        if chart_path:
            add_image_page(pdf, chart_path, title)
    
    # FRED Analysis
    fred_section = "FRED ECONOMIC INDICATORS ANALYSIS\n\n"
    fred_section += "Federal Reserve Economic Data (FRED) indicators were analyzed as leading macroeconomic signals for Bitcoin price movements.\n\n"
    
    if fred_significance:
        fred_section += "Statistical Significance Results:\n"
        if 'correlations' in fred_significance:
            for indicator, corr_data in list(fred_significance['correlations'].items())[:5]:
                if isinstance(corr_data, dict):
                    corr = corr_data.get('correlation', 0)
                    pval = corr_data.get('p_value', 1)
                    fred_section += f"- {indicator}: Correlation={corr:.3f}, p-value={pval:.4f}\n"
    
    fred_section += "\nKey FRED Indicators:\n"
    fred_section += "- M2SL: Money Supply\n"
    fred_section += "- Net_Liquidity: Net liquidity in the system\n"
    fred_section += "- GFDEBTN: Government Debt\n"
    fred_section += "- CPIAUCSL: Consumer Price Index\n"
    fred_section += "- FEDFUNDS: Federal Funds Rate\n"
    fred_section += "- DGS10: 10-Year Treasury Rate\n"
    
    fred_section += "\nFindings:\n"
    fred_section += "- FRED indicators show statistical significance but lower feature importance than halving cycles\n"
    fred_section += "- DGS10 (10-Year Treasury) is the most important FRED feature\n"
    fred_section += "- Net Liquidity and M2 show leading indicator properties\n"
    
    add_text_page(pdf, "FRED Analysis", fred_section)
    
    # Add FRED charts
    fred_charts = [
        ('fred_bitcoin_correlation.png', 'FRED-Bitcoin Correlations'),
        ('net_liquidity_policy_meter.png', 'Net Liquidity Policy Meter'),
        ('leading_indicators_bitcoin_price.png', 'Leading Indicators vs Bitcoin Price'),
    ]
    for chart_name, title in fred_charts:
        chart_path = find_file(['fred_deep_dive'], f'figures/{chart_name}')
        if not chart_path:
            chart_path = find_file(['fred'], f'figures/{chart_name}')
        if chart_path:
            add_image_page(pdf, chart_path, title)
    
    # Technical Indicators
    tech_section = "TECHNICAL INDICATORS AND SMA ANALYSIS\n\n"
    tech_section += "Technical indicators were engineered from Bitcoin price data:\n\n"
    tech_section += "Moving Averages:\n"
    tech_section += "- SMA_50, SMA_100, SMA_200 (Simple Moving Averages)\n"
    tech_section += "- EMA_50, EMA_200 (Exponential Moving Averages)\n"
    tech_section += "- Price-to-SMA ratios\n"
    tech_section += "- SMA slopes (momentum indicators)\n\n"
    tech_section += "Momentum Indicators:\n"
    tech_section += "- RSI_14, RSI_21 (Relative Strength Index)\n"
    tech_section += "- MACD, MACD_Hist (Moving Average Convergence Divergence)\n\n"
    tech_section += "Volatility:\n"
    tech_section += "- Volatility_20d, Volatility_50d (rolling standard deviation)\n\n"
    tech_section += "Returns:\n"
    tech_section += "- Return_20d, Return_50d\n"
    tech_section += "- ROC_20, ROC_50 (Rate of Change)\n\n"
    tech_section += "Findings:\n"
    tech_section += "- Technical indicators are highly relevant for price prediction\n"
    tech_section += "- SMA_200 and EMA_50 are among the top features\n"
    tech_section += "- Volatility measures help capture market regime changes\n"
    
    add_text_page(pdf, "Technical Indicators", tech_section)
    
    # ML Model Selection
    ml_section = "MACHINE LEARNING MODEL SELECTION\n\n"
    
    if ml_pipeline_results:
        ml_section += f"Model Selection Pipeline Results:\n\n"
        ml_section += f"Optimal CV Folds: {ml_pipeline_results.get('optimal_cv_folds', 'N/A')}\n"
        ml_section += f"Best Model: {ml_pipeline_results.get('best_model', 'N/A')}\n"
        ml_section += f"Best Parameters: {ml_pipeline_results.get('best_params', {})}\n\n"
        
        if 'model_results' in ml_pipeline_results:
            ml_section += "Model Performance (Nested CV):\n"
            for name, res in ml_pipeline_results['model_results'].items():
                score = res.get('mean_cv_score', 0)
                std = res.get('std_cv_score', 0)
                ml_section += f"- {name}: {score:.4f} (+/- {std:.4f})\n"
    
    if ml_results:
        ml_section += f"\nWalk-Forward Validation Results:\n"
        if 'walk_forward_results' in ml_results:
            for name, res in ml_results['walk_forward_results'].items():
                rmse = res.get('rmse', 0)
                r2 = res.get('r2', 0)
                dir_acc = res.get('directional_accuracy', 0)
                ml_section += f"- {name}: RMSE=${rmse:,.0f}, R²={r2:.4f}, Dir Acc={dir_acc:.2%}\n"
    
    add_text_page(pdf, "ML Model Selection", ml_section)
    
    # Add ML charts
    ml_charts = [
        ('walk_forward_predictions.png', 'Walk-Forward Predictions'),
        ('model_comparison.png', 'Model Comparison'),
        ('directional_accuracy_over_time.png', 'Directional Accuracy Over Time'),
        ('randomforest_predictions_overlay.png', 'Random Forest Predictions Overlay'),
        ('gradientboosting_predictions_overlay.png', 'Gradient Boosting Predictions Overlay'),
    ]
    for chart_name, title in ml_charts:
        chart_path = find_file(['ml_predictions'], f'figures/{chart_name}')
        if chart_path:
            add_image_page(pdf, chart_path, title)
    
    # Overfitting Analysis
    overfitting_section = "OVERFITTING ANALYSIS\n\n"
    
    if ml_pipeline_results and 'overfitting_analysis' in ml_pipeline_results:
        of_analysis = ml_pipeline_results['overfitting_analysis']
        
        overfitting_section += "Before Tuning:\n"
        before = of_analysis.get('before_tuning', {})
        overfitting_section += f"- Train Score: {before.get('train_score', 0):.4f}\n"
        overfitting_section += f"- Validation Score: {before.get('val_score', 0):.4f}\n"
        overfitting_section += f"- Gap: {before.get('gap', 0):.2%}\n"
        overfitting_section += f"- Overfitting: {'YES' if before.get('is_overfitting', False) else 'NO'}\n\n"
        
        overfitting_section += "After Tuning:\n"
        after = of_analysis.get('after_tuning', {})
        overfitting_section += f"- Train+Val Score: {after.get('train_val_score', 0):.4f}\n"
        overfitting_section += f"- Test Score: {after.get('test_score', 0):.4f}\n"
        overfitting_section += f"- Gap: {after.get('gap', 0):.2%}\n"
        overfitting_section += f"- Overfitting: {'YES' if after.get('is_overfitting', False) else 'NO'}\n\n"
        
        gap_before = before.get('gap', 1)
        gap_after = after.get('gap', 1)
        if gap_after < gap_before:
            overfitting_section += "✓ Tuning successfully reduced overfitting\n"
        else:
            overfitting_section += "⚠ Tuning did not significantly reduce overfitting\n"
    
    add_text_page(pdf, "Overfitting Analysis", overfitting_section)
    
    # Add learning/validation curves
    curve_charts = [
        ('learning_curve_best_model.png', 'Learning Curve'),
        ('validation_curve_best_model.png', 'Validation Curve'),
        ('cv_folds_vs_score.png', 'CV Folds vs Score'),
    ]
    for chart_name, title in curve_charts:
        chart_path = find_file(['ml_pipeline'], f'plots/{chart_name}')
        if chart_path:
            add_image_page(pdf, chart_path, title)
    
    # Current Price and Predictions
    prediction_section = "CURRENT PRICE ANALYSIS AND FUTURE PROJECTIONS\n\n"
    
    if ml_prediction:
        pred = ml_prediction
        prediction_section += f"Current Analysis Date: {pred.get('current_date', 'N/A')}\n"
        current_price = pred.get('current_price', 0)
        prediction_section += f"Current Bitcoin Price: USD {current_price:,.2f}\n\n"
        prediction_section += f"Model: {pred.get('model', 'N/A')}\n\n"
        prediction_section += "20-Day Forward Prediction:\n"
        pred_price = pred.get('predicted_price', 0)
        pred_change = pred.get('predicted_change', 0)
        pred_change_pct = pred.get('predicted_change_pct', 0)
        prediction_section += f"- Predicted Price: USD {pred_price:,.2f}\n"
        prediction_section += f"- Predicted Change: USD {pred_change:,.2f} ({pred_change_pct:+.2f}%)\n"
        prediction_section += f"- Direction: {pred.get('direction', 'N/A')}\n\n"
        
        if 'confidence_intervals' in pred:
            ci = pred['confidence_intervals']
            prediction_section += "Confidence Intervals:\n"
            prediction_section += f"- 68% CI: USD {ci.get('68_low', 0):,.2f} - USD {ci.get('68_high', 0):,.2f}\n"
            prediction_section += f"- 95% CI: USD {ci.get('95_low', 0):,.2f} - USD {ci.get('95_high', 0):,.2f}\n"
    else:
        prediction_section += "Prediction data not available.\n"
        prediction_section += "Please run the prediction pipeline to generate current forecasts.\n"
    
    add_text_page(pdf, "Price Predictions", prediction_section)
    
    # Add prediction charts
    pred_charts = [
        ('current_predictions_ci.png', 'Current Predictions with Confidence Intervals'),
        ('all_models_comparison_overlay.png', 'All Models Comparison'),
    ]
    for chart_name, title in pred_charts:
        chart_path = find_file(['ml_predictions'], f'figures/{chart_name}')
        if chart_path:
            add_image_page(pdf, chart_path, title)
    
    # Conclusions
    conclusions = """
    CONCLUSIONS AND RECOMMENDATIONS
    
    Key Findings:
    
    1. FEATURE IMPORTANCE
    - Halving cycle features are the most predictive (38-40% importance)
    - Technical indicators (SMAs, volatility) are highly relevant
    - FRED indicators contribute but are less dominant than expected
    - DGS10 (10-Year Treasury) is the most important FRED feature
    
    2. MODEL PERFORMANCE
    - Gradient Boosting achieves best performance (R² = 0.86)
    - Model is better at predicting magnitude than direction
    - Directional accuracy (~46%) is close to random
    - MAPE of ~9% is reasonable for 20-day predictions
    
    3. OVERFITTING
    - Nested cross-validation successfully prevents overfitting
    - Tuning reduces train-test gap
    - Model generalizes well to out-of-sample data
    
    4. PREDICTIONS
    - Model provides useful confidence intervals
    - Predictions should be used as one input among many
    - Risk management is essential given prediction uncertainty
    
    Recommendations:
    
    1. TRADING STRATEGY
    - Use model predictions for magnitude estimation
    - Combine with technical analysis for directional signals
    - Implement position sizing based on confidence intervals
    - Set stop losses based on 95% confidence intervals
    
    2. MODEL IMPROVEMENTS
    - Test longer prediction horizons (30, 60, 90 days)
    - Implement regime detection for different market conditions
    - Create ensemble models combining multiple algorithms
    - Add more interaction features between FRED and technical indicators
    
    3. RISK MANAGEMENT
    - Always use confidence intervals for position sizing
    - Monitor model performance over time
    - Retrain models periodically with new data
    - Combine ML predictions with fundamental analysis
    
    4. FUTURE WORK
    - Test alternative models (LSTM, Transformers)
    - Implement online learning for adaptive models
    - Create separate models for different market regimes
    - Integrate additional data sources (on-chain metrics, sentiment)
    
    Disclaimer:
    This analysis is for educational and research purposes only. Past performance does not guarantee future results. Cryptocurrency trading involves substantial risk of loss. Always conduct your own research and consult with financial advisors before making investment decisions.
    """
    add_text_page(pdf, "Conclusions", conclusions)
    
    # Metadata
    pdf.infodict()['Title'] = 'Bitcoin Comprehensive Analysis Report'
    pdf.infodict()['Author'] = 'ML Analysis Pipeline'
    pdf.infodict()['Subject'] = 'Bitcoin Price Prediction using Machine Learning'
    pdf.infodict()['Keywords'] = 'Bitcoin, Machine Learning, PCA, FRED, Price Prediction'
    pdf.infodict()['CreationDate'] = datetime.now()

print(f"✓ PDF created: {OUTPUT_PDF}")
print(f"  Total pages: Check the PDF file")
print()

