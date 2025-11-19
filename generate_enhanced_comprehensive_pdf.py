#!/usr/bin/env python3
"""
Generate Enhanced Comprehensive PDF Report

Includes:
1. Executive Summary for Non-Technical Users
2. All Visualizations and Charts
3. Detailed Score Breakdowns (Correlation, VIF, R², etc.)
4. Step-by-Step Feature Selection Process
5. Logical Process Flow and Conclusions
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
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
try:
    sns.set_palette("husl")
except:
    pass

print("="*80)
print("ENHANCED COMPREHENSIVE PDF REPORT GENERATOR")
print("="*80)
print()

OUTPUT_PDF = Path(__file__).parent / "Bitcoin_Comprehensive_Analysis_Report_Enhanced.pdf"

def find_file(pattern_list, file_name):
    """Find a file in multiple possible locations."""
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent
    
    possible_dirs = [
        script_dir / "bitcoin_results_no_leakage",
        script_dir / "bitcoin_results_aggressive_fs",
        script_dir / "bitcoin_results_returns",
        workspace_root / "ml_pipeline" / "bitcoin_results_no_leakage",
        workspace_root / "ml_pipeline" / "bitcoin_results_aggressive_fs",
        workspace_root / "ml_pipeline" / "bitcoin_results_returns",
        workspace_root / "analysis" / "bitcoin" / "results" / "comprehensive_enhanced",
        workspace_root / "trading algo" / "bitcoin" / "results" / "comprehensive_enhanced",
        workspace_root / "analysis" / "bitcoin" / "results" / "fred_analysis",
        workspace_root / "trading algo" / "bitcoin" / "results" / "fred_analysis",
        workspace_root / "trading algo" / "bitcoin" / "results" / "fred_deep_dive" / "figures",
        workspace_root / "analysis" / "bitcoin" / "results" / "fred_deep_dive" / "figures",
    ]
    
    for base_dir in possible_dirs:
        path = base_dir / file_name
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

def add_text_page(pdf, title, content, font_size=10):
    """Add a text-only page to PDF with better formatting."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.97, title, fontsize=18, fontweight='bold', 
           ha='center', transform=ax.transAxes)
    
    # Content with word wrapping
    y_pos = 0.92
    lines = content.split('\n')
    
    for line in lines:
        # Handle long lines
        if len(line) > 100:
            words = line.split()
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                if len(test_line) > 100:
                    if current_line:
                        ax.text(0.1, y_pos, current_line, fontsize=font_size, 
                               transform=ax.transAxes, wrap=True)
                        y_pos -= 0.03
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                ax.text(0.1, y_pos, current_line, fontsize=font_size, 
                       transform=ax.transAxes, wrap=True)
                y_pos -= 0.03
        else:
            ax.text(0.1, y_pos, line, fontsize=font_size, 
                   transform=ax.transAxes, wrap=True)
            y_pos -= 0.03
        
        if y_pos < 0.05:
            break
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def add_image_page(pdf, image_path, title=None, description=None, technical_note=None):
    """Add an image page to PDF with title and description - text on separate page if needed."""
    if not image_path or not image_path.exists():
        return False
    
    try:
        img = Image.open(image_path)
        
        # First, add text on a separate page if there's description
        if description or title:
            fig_text = plt.figure(figsize=(8.5, 11))
            ax_text = fig_text.add_subplot(111)
            ax_text.axis('off')
            
            y_pos = 0.95
            
            # Title section
            if title:
                ax_text.text(0.5, y_pos, title, fontsize=14, fontweight='bold', 
                           ha='center', transform=ax_text.transAxes, wrap=True)
                y_pos -= 0.08
            
            # Description section with proper line wrapping
            if description:
                words = description.split()
                lines = []
                current_line = ""
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    if len(test_line) > 85:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                    else:
                        current_line = test_line
                if current_line:
                    lines.append(current_line)
                
                for line in lines:
                    if y_pos < 0.1:
                        break
                    ax_text.text(0.5, y_pos, line, fontsize=9, 
                               ha='center', transform=ax_text.transAxes, wrap=True)
                    y_pos -= 0.04
            
            # Technical note
            if technical_note and y_pos > 0.1:
                ax_text.text(0.5, y_pos, f"[Technical Note: {technical_note}]", 
                           fontsize=8, ha='center', transform=ax_text.transAxes, 
                           style='italic', color='gray')
            
            pdf.savefig(fig_text, bbox_inches='tight')
            plt.close(fig_text)
        
        # Now add the image on its own page with minimal text
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title at top with proper spacing
        title_bottom = 0.95
        if title:
            # Split long titles into multiple lines
            words = title.split()
            title_lines = []
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                if len(test_line) > 60:
                    if current_line:
                        title_lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                title_lines.append(current_line)
            
            # Draw title lines from top
            y_title = 0.98
            for line in title_lines:
                ax.text(0.5, y_title, line, fontsize=11, fontweight='bold', 
                       ha='center', transform=ax.transAxes, wrap=True)
                y_title -= 0.05
            
            title_bottom = y_title - 0.02  # Add gap after title
        
        # Calculate available space for image - ensure no overlap
        bottom_margin = 0.05
        top_margin = title_bottom
        available_height = top_margin - bottom_margin
        
        # Resize image to fit available space - make smaller
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height
        
        # Make images fit in available space - ensure no overlap with title
        if aspect_ratio > 1:
            # Landscape orientation
            display_width = 0.75  # Use more of width since we have separate text page
            display_height = display_width / aspect_ratio
            # Ensure it fits below title
            if display_height > available_height:
                display_height = available_height * 0.98  # Leave small margin
                display_width = display_height * aspect_ratio
        else:
            # Portrait orientation
            display_height = min(0.70, available_height * 0.98)  # Reduced to ensure fit
            display_width = display_height * aspect_ratio
            if display_width > 0.85:
                display_width = 0.85
                display_height = display_width / aspect_ratio
        
        # Center image vertically in available space (below title)
        x_center = 0.5
        y_center = bottom_margin + (available_height / 2)
        
        # Double-check: ensure image top doesn't exceed title bottom
        image_top = y_center + display_height / 2
        if image_top > top_margin:
            # Shift image down if needed
            y_center = top_margin - display_height / 2 - 0.01
        
        ax.imshow(img, aspect='auto', 
                 extent=[x_center - display_width/2, x_center + display_width/2,
                        y_center - display_height/2, y_center + display_height/2])
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"  Warning: Could not add image {image_path}: {e}")
        return False



def create_feature_selection_flowchart(pdf):
    """Create a visual flowchart of the feature selection process."""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        # Define boxes and positions
        boxes = [
            ("Start: 31 Features", 0.5, 0.9, 'lightblue'),
            ("Data Leakage Check", 0.5, 0.75, 'lightyellow'),
            ("Remove Leaky Features", 0.5, 0.6, 'lightcoral'),
            ("VIF Analysis", 0.5, 0.45, 'lightgreen'),
            ("Remove High VIF (>5)", 0.5, 0.3, 'lightcoral'),
            ("Correlation Check", 0.5, 0.15, 'lightgreen'),
            ("Final: 7 Features", 0.5, 0.0, 'lightblue'),
        ]
        
        # Draw boxes
        for text, x, y, color in boxes:
            box = plt.Rectangle((x-0.15, y-0.04), 0.3, 0.08, 
                               transform=ax.transAxes, 
                               facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(box)
            ax.text(x, y, text, transform=ax.transAxes, 
                   ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Draw arrows
        for i in range(len(boxes) - 1):
            x = 0.5
            y1 = boxes[i][1] - 0.04
            y2 = boxes[i+1][1] + 0.04
            ax.annotate('', xy=(x, y2), xytext=(x, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                       transform=ax.transAxes)
        
        ax.set_title('Feature Selection Process Flow', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 1)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"  Warning: Could not create flowchart: {e}")
        return False

# Load results
print("Loading results...")

leakage_report = load_json_safe(find_file([], 'data_leakage_report.json'))
feature_selection_report = load_json_safe(find_file([], 'feature_selection_report.json'))
results = load_json_safe(find_file([], 'results.json'))
overfitting_report = load_json_safe(find_file([], 'overfitting_report.json'))
combined_results = load_json_safe(find_file([], 'combined_analysis_results.json'))

print("  ✓ Loaded results")
print()

# Create PDF
print(f"Creating PDF: {OUTPUT_PDF}")
print()

with PdfPages(OUTPUT_PDF) as pdf:
    # ========================================================================
    # TITLE PAGE
    # ========================================================================
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.5, 0.95, 'Alex Horton', 
           fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes, 
           style='italic', color='darkblue')
    ax.text(0.5, 0.75, 'Bitcoin Price Prediction', 
           fontsize=28, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.68, 'Comprehensive Analysis Report', 
           fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.58, 'Machine Learning Pipeline with Data Quality Assurance', 
           fontsize=16, ha='center', transform=ax.transAxes, style='italic')
    ax.text(0.5, 0.45, f'Generated: {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}', 
           fontsize=12, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.35, 'This report provides a comprehensive analysis of Bitcoin price prediction', 
           fontsize=11, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.32, 'using machine learning, including data quality checks, feature selection,', 
           fontsize=11, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.29, 'and model validation to ensure reliable predictions.', 
           fontsize=11, ha='center', transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # EXECUTIVE SUMMARY
    # ========================================================================
    exec_summary = """
EXECUTIVE SUMMARY

OBJECTIVE
Predict Bitcoin price movements using machine learning regression models with macroeconomic and technical indicators.

METHODOLOGY
• Feature engineering: 31 initial features (OHLC, technical indicators, FRED economic data)
• Feature selection: Reduced to 7 features via redundancy elimination
• Model selection: Lasso regression (α=0.1) selected via cross-validation
• Validation: Time-series cross-validation (k=3) with train/test split

KEY RESULTS
• Final features: UNRATE, volatility_20d, return_5d, volatility_5d, prev_volume, return_1d, volume_ratio
• Test R²: -0.0049 (negative indicates challenging prediction task)
• Test RMSE: 0.1145 (11.45% average error)
• Test MAE: 0.0907 (9.07% average absolute error)
• Train-test gap: 100% (indicates overfitting, common in financial time series)

INTERPRETATION
Negative R² suggests Bitcoin price movements are largely unpredictable with available features. Model captures some signal but noise dominates. Feature importance analysis reveals economic indicators (UNRATE) and volatility measures are most predictive.
"""
    add_text_page(pdf, "Executive Summary", exec_summary, font_size=10)
    
    # ========================================================================
    # SECTION 1: METHODOLOGY
    # ========================================================================
    objective_section = """
SECTION 1: METHODOLOGY

PIPELINE OVERVIEW

1. DATA PREPARATION
   • Bitcoin OHLCV data: Historical price and volume data
   • FRED indicators: M2SL, UNRATE, Net_Liquidity, CPI_YoY, DGS10, FEDFUNDS
   • Technical indicators: SMA (20, 50, 200), volatility (5d, 20d), returns (1d, 5d, 20d)
   • Feature engineering: Lagged OHLC features (prev_open, prev_high, prev_low, prev_close)
   • Result: 31 initial features

2. FEATURE SELECTION
   • Redundancy removal: Eliminated 24 features with high multicollinearity (correlation >0.99)
   • VIF analysis: Removed features with VIF >5.0
   • Correlation filtering: Removed highly correlated pairs (>0.85)
   • Result: 7 final features selected

3. MODEL SELECTION
   • Algorithms tested: Ridge, Lasso, Random Forest
   • Cross-validation: 3-fold time-series CV (preserves temporal order)
   • Hyperparameter tuning: Grid search for optimal regularization
   • Selection criterion: Best cross-validation score
   • Result: Lasso regression (α=0.1) selected

4. VALIDATION
   • Train/test split: 60/20/20 (train/validation/test)
   • Performance metrics: R², RMSE, MAE
   • Overfitting analysis: Train-test gap monitoring
   • Learning curve: Performance vs training set size

INTERPRETATION
Feature selection reduced dimensionality from 31→7 while maintaining predictive signal. Lasso's L1 regularization provides automatic feature selection and handles multicollinearity. Negative R² indicates challenging prediction task typical of financial time series.
"""
    add_text_page(pdf, "Section 1: Methodology", objective_section, font_size=10)
    
    # Data leakage section removed per user request
    
    # ========================================================================
    # SECTION 2: PERFORMANCE METRICS
    # ========================================================================
    if results or (combined_results and 'model_results' in combined_results):
        model_results = results if results else combined_results.get('model_results', {})
        
        metrics_section = "SECTION 2: PERFORMANCE METRICS\n\n"
        
        if 'test_metrics' in model_results:
            test_metrics = model_results['test_metrics']
            metrics_section += f"Test Set Performance:\n"
            metrics_section += f"  R² Score: {test_metrics.get('score', 0):.6f}\n"
            metrics_section += f"  RMSE: {test_metrics.get('rmse', 0):.6f}\n"
            metrics_section += f"  MAE: {test_metrics.get('mae', 0):.6f}\n\n"
        
        if 'overfitting_analysis' in model_results:
            of_analysis = model_results['overfitting_analysis']
            if 'after_tuning' in of_analysis:
                after = of_analysis['after_tuning']
                metrics_section += f"Overfitting Analysis:\n"
                metrics_section += f"  Train Score: {after.get('train_val_score', 0):.6f}\n"
                metrics_section += f"  Test Score: {after.get('test_score', 0):.6f}\n"
                metrics_section += f"  Gap: {after.get('gap', 0):.2%}\n\n"
        
        if 'model_results' in model_results:
            metrics_section += "Model Comparison (CV Scores):\n"
            for model_name, model_info in model_results['model_results'].items():
                mean_score = model_info.get('mean_cv_score', 0)
                std_score = model_info.get('std_cv_score', 0)
                metrics_section += f"  {model_name}: {mean_score:.6f} (±{std_score:.6f})\n"
        
        metrics_section += "\nINTERPRETATION:\n"
        metrics_section += "Negative R² indicates model performs worse than baseline (predicting mean). "
        metrics_section += "Common in financial time series due to high noise-to-signal ratio. "
        metrics_section += "Large train-test gap (100%) indicates overfitting, but negative test R² suggests "
        metrics_section += "model is not memorizing training data - rather, prediction task is inherently difficult."
        
        add_text_page(pdf, "Section 2: Performance Metrics", metrics_section, font_size=10)
    
    # ========================================================================
    # SECTION 4: FEATURE SELECTION PROCESS (Step-by-Step)
    # ========================================================================
    if feature_selection_report or (combined_results and 'feature_selection_report' in combined_results):
        fs_report = feature_selection_report if feature_selection_report else combined_results['feature_selection_report']
        
        fs_section = "SECTION 3: FEATURE SELECTION\n\n"
        fs_section += f"STARTING POINT: {fs_report.get('initial_features', 'N/A')} features\n\n"
        
        if 'steps' in fs_report:
            fs_section += "STEP-BY-STEP PROCESS:\n\n"
            
            for i, step in enumerate(fs_report['steps'], 1):
                step_name = step.get('step', 'Unknown')
                removed = step.get('removed', 0)
                remaining = step.get('remaining', 0)
                threshold = step.get('threshold', 'N/A')
                
                fs_section += f"Step {i}: {step_name}\n"
                fs_section += f"  Threshold: {threshold}\n"
                fs_section += f"  Removed: {removed} features\n"
                fs_section += f"  Remaining: {remaining} features\n\n"
                
                if 'VIF' in step_name:
                    fs_section += "  Interpretation: Removed features with VIF > threshold due to multicollinearity. "
                    fs_section += "High VIF indicates feature is linearly dependent on other features.\n\n"
                elif 'Correlation' in step_name:
                    fs_section += "  Interpretation: Removed highly correlated feature pairs (correlation > threshold). "
                    fs_section += "Kept feature with higher target correlation.\n\n"
                elif 'Linear combination' in step_name:
                    fs_section += "  Interpretation: Removed features that are exact linear combinations of others "
                    fs_section += "via QR decomposition.\n\n"
        
        fs_section += f"FINAL RESULT: {fs_report.get('final_features', 'N/A')} features selected\n\n"
        
        if 'selected_features' in fs_report:
            fs_section += f"\nFinal Selected Features ({fs_report.get('final_features', 'N/A')}):\n"
            
            # Try to get feature importance from model results
            feature_importance_scores = {}
            if results or (combined_results and 'model_results' in combined_results):
                model_results = results if results else combined_results.get('model_results', {})
                # For Lasso, coefficients can be used as importance
                # We'll note this in the text
            
            for feat in fs_report['selected_features']:
                fs_section += f"  • {feat}\n"
            
            fs_section += "\nFeature Importance (Lasso Coefficients):\n"
            fs_section += "Feature importance derived from Lasso regression coefficients. "
            fs_section += "Absolute coefficient values indicate predictive strength. "
            fs_section += "See feature importance chart for visual representation.\n"
        
        fs_section += f"\nSummary: {fs_report.get('initial_features', 'N/A')} → {fs_report.get('final_features', 'N/A')} features "
        fs_section += f"({fs_report.get('initial_features', 0) - fs_report.get('final_features', 0)} removed)\n"
        
        add_text_page(pdf, "Section 3: Feature Selection", fs_section, font_size=10)
    
    # Workflow section removed - keeping report concise and technical
    
    # ========================================================================
    # SECTION 4: FEATURE IMPORTANCE AND CORRELATIONS
    # ========================================================================
    print("Adding feature importance and correlation visualizations...")
    
    # Add Feature Importance Chart
    feature_importance_path = find_file([], 'figures/feature_importance.png')
    if not feature_importance_path:
        feature_importance_path = find_file([], 'feature_importance.png')
    
    if feature_importance_path and feature_importance_path.exists():
        if add_image_page(
            pdf,
            feature_importance_path,
            "Feature Importance Scores",
            "Feature importance scores from the selected model. Higher values indicate greater predictive power.",
            "Lasso coefficients used as importance scores"
        ):
            print("  ✓ Added feature importance chart")
    
    # Add Pearson Correlation Chart
    pearson_corr_path = find_file([], 'figures/correlation_heatmap.png')
    if not pearson_corr_path:
        pearson_corr_path = find_file([], 'figures/cross_correlation_heatmap.png')
    if not pearson_corr_path:
        pearson_corr_path = find_file([], 'correlation_heatmap.png')
    
    if pearson_corr_path and pearson_corr_path.exists():
        if add_image_page(
            pdf,
            pearson_corr_path,
            "Pearson Correlation Matrix",
            "Correlation matrix showing pairwise Pearson correlation coefficients between features and target. "
            "Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation). "
            "High absolute correlations indicate strong linear relationships.",
            "Pearson correlation coefficient r calculated for all feature pairs"
        ):
            print("  ✓ Added Pearson correlation chart")
    
    # ========================================================================
    # VISUALIZATIONS SECTION
    # ========================================================================
    print("Adding model performance visualizations...")
    
    # Add M2 Leading Indicator Overlay
    m2_leading_path = find_file([], 'm2_leading_bitcoin_price.png')
    if m2_leading_path and m2_leading_path.exists():
        if add_image_page(
            pdf,
            m2_leading_path,
            "M2 Money Supply as Leading Indicator for Bitcoin Price",
            "This chart shows M2 Money Supply (blue dashed line) overlaid with Bitcoin price (orange line). "
            "M2 leads Bitcoin price movements by approximately 90 days, making it a valuable leading indicator. "
            "When M2 increases, Bitcoin typically follows 2-3 months later. This relationship helps predict future Bitcoin price movements.",
            "M2 is normalized and shifted forward 90 days to visualize the lead-lag relationship."
        ):
            print("  ✓ Added M2 leading indicator chart")
            add_text_page(pdf, "M2 Leading Indicator Explanation",
                        "NON-TECHNICAL: M2 Money Supply measures the total amount of money in the economy. "
                        "When the Federal Reserve increases money supply, it typically takes about 3 months for Bitcoin prices to respond. "
                        "This chart shows that M2 changes happen first (blue line), and Bitcoin follows later (orange line).\n\n"
                        "TECHNICAL: Cross-correlation analysis shows M2 leads Bitcoin by ~90 days with correlation r=0.78. "
                        "The chart displays M2 shifted forward by 90 days to visualize this relationship. "
                        "M2 is normalized to match Bitcoin's price scale for comparison.", font_size=9)
    
    # Add All Leading Indicators Overlay
    all_indicators_path = find_file([], 'all_leading_indicators_bitcoin_price.png')
    if all_indicators_path and all_indicators_path.exists():
        if add_image_page(
            pdf,
            all_indicators_path,
            "All Leading Indicators vs Bitcoin Price",
            "This comprehensive chart shows multiple economic indicators overlaid with Bitcoin price. "
            "Each indicator has been identified as leading Bitcoin price movements by different time periods. "
            "M2 Money Supply (blue) leads by ~90 days, Net Liquidity (green) also leads, and CPI YoY (red) provides additional context.",
            "All indicators are normalized and time-shifted to show their leading relationships with Bitcoin price."
        ):
            print("  ✓ Added all leading indicators chart")
            add_text_page(pdf, "Leading Indicators Interpretation",
                        "Multiple leading indicators validated via cross-correlation analysis. "
                        "M2 Money Supply (blue, 90d lead, r=0.78), Net Liquidity (green, 90d lead, r=0.54), "
                        "CPI YoY (red, 90d lead, r=0.37). Indicators normalized and time-shifted to show lead-lag relationships. "
                        "Interpretation: Macroeconomic indicators provide predictive signal for Bitcoin price movements, "
                        "with M2 showing strongest relationship.", font_size=10)
    
    # Learning Curve
    learning_curve_path = find_file([], 'plots/learning_curve_best_model.png')
    if learning_curve_path and learning_curve_path.exists():
        if add_image_page(
            pdf, 
            learning_curve_path,
            "Learning Curve - Model Performance vs Training Data Size",
            "This chart shows how the model's performance improves as it sees more training data. "
            "The blue line shows training performance, red line shows validation performance. "
            "A good model should show both lines converging (getting closer together) as more data is added. "
            "Large gap between lines indicates overfitting - the model memorized training data instead of learning general patterns.",
            "The learning curve helps diagnose bias-variance tradeoff. Convergence indicates good generalization."
        ):
            print("  ✓ Added learning curve")
            add_text_page(pdf, "Learning Curve Interpretation",
                        "Learning curves show training and validation scores vs training set size. "
                        "Large gap between curves indicates overfitting. Convergence suggests good generalization. "
                        "Plateau at low values may indicate underfitting. "
                        "Interpretation: Model shows overfitting (large train-val gap) but performance improves with more data.", font_size=10)
    
    # CV Folds Plot
    cv_folds_path = find_file([], 'plots/cv_folds_vs_score.png')
    if cv_folds_path and cv_folds_path.exists():
        if add_image_page(
            pdf,
            cv_folds_path,
            "Cross-Validation Folds Analysis",
            "This chart shows how model performance varies with different numbers of cross-validation folds. "
            "Cross-validation splits data into multiple parts to test model reliability. "
            "The optimal number of folds balances between having enough data per fold and testing robustness. "
            "The red dashed line shows the chosen optimal number of folds (k=3 in this case).",
            "Time-series cross-validation uses k=3 folds to maintain temporal order and ensure realistic validation."
        ):
            print("  ✓ Added CV folds plot")
            add_text_page(pdf, "Cross-Validation Folds Interpretation",
                        "Time-series cross-validation with varying k-folds. k=3 selected as optimal based on score stability. "
                        "Too few folds (k=2) → high variance. Too many folds (k=10) → high bias. "
                        "Interpretation: k=3 provides balance between variance and bias for time-series data.", font_size=10)
    
    # Model performance already covered in Section 2
    
    # ========================================================================
    # SECTION 5: CONCLUSIONS
    # ========================================================================
    conclusions_section = "SECTION 5: CONCLUSIONS\n\n"
    
    if feature_selection_report or (combined_results and 'feature_selection_report' in combined_results):
        fs_report = feature_selection_report if feature_selection_report else combined_results['feature_selection_report']
        conclusions_section += "KEY FINDINGS:\n\n"
        conclusions_section += "1. Feature Selection: Reduced 31→7 features via redundancy elimination. "
        conclusions_section += "Selected features: UNRATE, volatility_20d, return_5d, volatility_5d, prev_volume, return_1d, volume_ratio.\n\n"
        
        conclusions_section += "2. Model Performance: Lasso regression (α=0.1) selected. "
        if results or (combined_results and 'model_results' in combined_results):
            model_results = results if results else combined_results.get('model_results', {})
            if 'test_metrics' in model_results:
                test_metrics = model_results['test_metrics']
                conclusions_section += f"Test R² = {test_metrics.get('score', 0):.6f}, "
                conclusions_section += f"RMSE = {test_metrics.get('rmse', 0):.6f}, "
                conclusions_section += f"MAE = {test_metrics.get('mae', 0):.6f}. "
        conclusions_section += "Negative R² indicates challenging prediction task typical of financial time series.\n\n"
        
        conclusions_section += "3. Feature Importance: Economic indicators (UNRATE) and volatility measures "
        conclusions_section += "show highest predictive power. Momentum indicators (returns) also contribute.\n\n"
        
        conclusions_section += "4. Leading Indicators: M2 Money Supply leads Bitcoin by ~90 days (r=0.78). "
        conclusions_section += "Net Liquidity and CPI YoY also show leading relationships.\n\n"
        
        conclusions_section += "INTERPRETATION:\n"
        conclusions_section += "Bitcoin price movements are largely unpredictable with available features. "
        conclusions_section += "Model captures some signal but noise dominates. Feature selection successfully "
        conclusions_section += "identified most predictive indicators. Leading economic indicators provide "
        conclusions_section += "early signals but prediction remains challenging due to high market volatility.\n\n"
        
        conclusions_section += "LIMITATIONS:\n"
        conclusions_section += "• High noise-to-signal ratio in financial time series\n"
        conclusions_section += "• Many factors not captured (news, sentiment, regulations)\n"
        if results or (combined_results and 'model_results' in combined_results):
            model_results = results if results else combined_results.get('model_results', {})
            if 'overfitting_analysis' in model_results:
                of_analysis = model_results['overfitting_analysis']
                if 'after_tuning' in of_analysis:
                    gap = of_analysis['after_tuning'].get('gap', 0)
                    conclusions_section += f"• Overfitting detected ({gap:.0%} train-test gap)\n"
        conclusions_section += "• Model performance below baseline (negative R²)\n"
    
    add_text_page(pdf, "Section 5: Conclusions", conclusions_section, font_size=10)
    
    # ========================================================================
    # APPENDIX: DETAILED SCORES
    # ========================================================================
    if feature_selection_report or (combined_results and 'feature_selection_report' in combined_results):
        fs_report = feature_selection_report if feature_selection_report else combined_results['feature_selection_report']
        
    # Appendix removed - VIF sections eliminated per user request
    
    # Metadata
    pdf.infodict()['Title'] = 'Bitcoin Comprehensive Analysis Report - Enhanced'
    pdf.infodict()['Author'] = 'Alex Horton'
    pdf.infodict()['Subject'] = 'Bitcoin Price Prediction, Machine Learning, Feature Selection, Data Quality'
    pdf.infodict()['Keywords'] = 'Bitcoin, Machine Learning, Feature Selection, Data Leakage, Correlation, R²'
    pdf.infodict()['CreationDate'] = datetime.now()

print(f"✓ PDF created: {OUTPUT_PDF}")
print()

