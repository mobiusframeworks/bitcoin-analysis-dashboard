#!/usr/bin/env python3
"""
Generate Comprehensive PDF Report with Data Leakage Checks

Updates the comprehensive analysis PDF to include:
- Data leakage detection
- Look-ahead bias checks
- OHLC leakage analysis
- Feature selection results
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
print("COMPREHENSIVE PDF REPORT WITH DATA LEAKAGE CHECKS")
print("="*80)
print()

OUTPUT_PDF = Path(__file__).parent / "Bitcoin_Comprehensive_Analysis_Report_With_Leakage_Checks.pdf"

def find_file(pattern_list, file_name):
    """Find a file in multiple possible locations."""
    possible_dirs = [
        "ml_pipeline/bitcoin_results_no_leakage",
        "ml_pipeline/bitcoin_results_aggressive_fs",
        "ml_pipeline/bitcoin_results_returns",
        "analysis/bitcoin/results/comprehensive_enhanced",
        "trading algo/bitcoin/results/comprehensive_enhanced",
        "analysis/bitcoin/results/fred_analysis",
        "trading algo/bitcoin/results/fred_analysis",
    ]
    
    for base_dir in possible_dirs:
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

def add_text_page(pdf, title, content, font_size=11):
    """Add a text-only page to PDF."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    words = content.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        if len(test_line) > 100:
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
        
        img.thumbnail((800, 1000), Image.Resampling.LANCZOS)
        ax.imshow(img, aspect='auto', extent=[0.1, 0.9, 0.1, 0.9])
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        return True
    except:
        return False

# Load results
print("Loading results...")

leakage_report = load_json_safe(find_file([], 'data_leakage_report.json'))
feature_selection_report = load_json_safe(find_file([], 'feature_selection_report.json'))
aggressive_report = load_json_safe(find_file([], 'aggressive_feature_selection_report.json'))
results = load_json_safe(find_file([], 'results.json'))
overfitting_report = load_json_safe(find_file([], 'overfitting_report.json'))
combined_results = load_json_safe(find_file([], 'combined_analysis_results.json'))
pca_results = load_json_safe(find_file([], 'pca_results.json'))
fred_significance = load_json_safe(find_file([], 'fred_statistical_significance.json'))

print("  ✓ Loaded results")
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
    ax.text(0.5, 0.6, 'With Data Leakage and Look-Ahead Bias Checks', 
           fontsize=18, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
           fontsize=12, ha='center', transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # Data Leakage Section (NEW - Comprehensive)
    if leakage_report or (combined_results and 'leakage_report' in combined_results):
        leak_report = leakage_report if leakage_report else combined_results['leakage_report']
        
        leakage_section = "COMPREHENSIVE DATA LEAKAGE AND LOOK-AHEAD BIAS ANALYSIS\n\n"
        leakage_section += f"Overall Status: {leak_report.get('overall_status', 'N/A')}\n\n"
        
        leakage_section += "This section checks for all forms of data leakage that could cause overfitting:\n\n"
        
        # OHLC Leakage
        leakage_section += "1. OHLC LEAKAGE DETECTION\n"
        leakage_section += "   Problem: Using same-day open/high/low to predict same-day close.\n"
        leakage_section += "   Why it's leakage: Close price is always between high and low.\n"
        leakage_section += "   This makes prediction trivial and causes overfitting.\n\n"
        
        ohlc_issues = leak_report.get('ohlc_leakage', {}).get('issues', [])
        if ohlc_issues:
            leakage_section += "   ❌ CRITICAL ISSUES FOUND:\n"
            for issue in ohlc_issues:
                leakage_section += f"   - {issue.get('feature', 'N/A')}: {issue.get('issue', 'N/A')}\n"
            leakage_section += "\n   SOLUTION: Only use LAGGED OHLC features:\n"
            leakage_section += "   - prev_open (previous day's open)\n"
            leakage_section += "   - prev_high (previous day's high)\n"
            leakage_section += "   - prev_low (previous day's low)\n"
            leakage_section += "   - prev_close (previous day's close)\n"
        else:
            leakage_section += "   ✅ PASS - No OHLC leakage detected\n"
            leakage_section += "   All OHLC features are properly lagged.\n"
        
        leakage_section += "\n"
        
        # Look-Ahead Bias
        leakage_section += "2. LOOK-AHEAD BIAS DETECTION\n"
        leakage_section += "   Problem: Using future information to predict past.\n"
        leakage_section += "   Examples: Forward-looking indicators, negative shifts.\n\n"
        
        lookahead_issues = leak_report.get('lookahead_bias', {}).get('issues', [])
        if lookahead_issues:
            leakage_section += "   ❌ CRITICAL ISSUES FOUND:\n"
            for issue in lookahead_issues:
                leakage_section += f"   - {issue.get('feature', 'N/A')}: {issue.get('issue', 'N/A')}\n"
        else:
            leakage_section += "   ✅ PASS - No look-ahead bias detected\n"
            leakage_section += "   All features use only past data.\n"
        
        leakage_section += "\n"
        
        # Target Leakage
        leakage_section += "3. TARGET LEAKAGE DETECTION\n"
        leakage_section += "   Problem: Target column accidentally included in features.\n\n"
        
        target_issues = leak_report.get('target_leakage', {}).get('issues', [])
        if target_issues:
            leakage_section += "   ❌ ISSUES FOUND:\n"
            for issue in target_issues:
                leakage_section += f"   - {issue.get('feature', 'N/A')}: {issue.get('issue', 'N/A')}\n"
        else:
            leakage_section += "   ✅ PASS - No target leakage detected\n"
        
        leakage_section += "\n"
        
        # Perfect Correlations
        leakage_section += "4. PERFECT CORRELATION DETECTION\n"
        leakage_section += "   Problem: Features with correlation > 0.99 are redundant.\n\n"
        
        corr_issues = leak_report.get('perfect_correlations', {}).get('issues', [])
        if corr_issues:
            leakage_section += f"   ⚠ Found {len(corr_issues)} perfect correlations:\n"
            for issue in corr_issues[:5]:
                leakage_section += f"   - {issue.get('feature1', 'N/A')} vs {issue.get('feature2', 'N/A')}: "
                leakage_section += f"{issue.get('correlation', 0):.4f}\n"
        else:
            leakage_section += "   ✅ PASS - No perfect correlations\n"
        
        leakage_section += "\n"
        
        # Critical Issues
        critical_issues = leak_report.get('critical_issues', [])
        if critical_issues:
            leakage_section += "CRITICAL ISSUES SUMMARY:\n"
            for issue in critical_issues:
                leakage_section += f"  - {issue.get('feature', 'N/A')}: {issue.get('issue', 'N/A')}\n"
            leakage_section += "\n"
        
        # Recommendations
        if leak_report.get('recommendations'):
            leakage_section += "RECOMMENDATIONS TO FIX DATA LEAKAGE:\n"
            for rec in leak_report.get('recommendations', []):
                leakage_section += f"  - {rec}\n"
        
        add_text_page(pdf, "Data Leakage Analysis", leakage_section)
    
    # Feature Selection with Leakage Removal
    report = aggressive_report if aggressive_report else feature_selection_report
    if report:
        fs_section = "FEATURE SELECTION WITH LEAKAGE REMOVAL\n\n"
        
        if combined_results and 'removed_leaky_features' in combined_results:
            removed_leaky = combined_results['removed_leaky_features']
            fs_section += f"Leaky Features Removed: {len(removed_leaky)}\n"
            if removed_leaky:
                fs_section += f"  {', '.join(removed_leaky)}\n"
            fs_section += "\n"
        
        fs_section += f"Initial Features: {report.get('initial_features', 'N/A')}\n"
        fs_section += f"After Leakage Removal: {report.get('initial_features', 0) - len(removed_leaky) if 'removed_leaky_features' in locals() else report.get('initial_features', 'N/A')}\n"
        fs_section += f"Final Features: {report.get('final_features', 'N/A')}\n"
        fs_section += f"Total Reduction: {report.get('initial_features', 0) - report.get('final_features', 0)} features\n\n"
        
        if 'steps' in report:
            fs_section += "Selection Steps:\n"
            for step in report['steps']:
                fs_section += f"  {step.get('step', 'N/A')}: Removed {step.get('removed', 0)} features\n"
        
        if 'feature_categories' in report:
            fs_section += "\nFinal Feature Categories:\n"
            for cat, features in report['feature_categories'].items():
                if features:
                    fs_section += f"  {cat}: {len(features)} - {', '.join(features)}\n"
        
        add_text_page(pdf, "Feature Selection", fs_section)
    
    # FRED vs SMA Analysis
    fred_sma_section = "FRED vs SMA REDUNDANCY ANALYSIS\n\n"
    fred_sma_section += "Analysis of whether FRED economic indicators and SMA/Technical indicators are redundant.\n\n"
    
    if report:
        conflicts_found = False
        if 'steps' in report:
            for step in report['steps']:
                if 'conflicts' in step and step.get('conflicts', 0) > 0:
                    conflicts_found = True
                    fred_sma_section += f"Conflicts Found: {step['conflicts']}\n\n"
                    if 'conflict_details' in step:
                        for conflict in step['conflict_details']:
                            fred_sma_section += f"  - {conflict.get('fred_feature', 'N/A')} (FRED) vs "
                            fred_sma_section += f"{conflict.get('technical_feature', 'N/A')} (Technical)\n"
                            fred_sma_section += f"    Correlation: {conflict.get('correlation', 0):.3f}\n"
                            fred_sma_section += f"    Action: Removed {conflict.get('removed', 'N/A')}\n\n"
        
        if not conflicts_found:
            fred_sma_section += "Result: No significant FRED/Technical conflicts found.\n\n"
            fred_sma_section += "Conclusion:\n"
            fred_sma_section += "- FRED and SMA/Technical features provide DIFFERENT information\n"
            fred_sma_section += "- They are complementary, not redundant\n"
            fred_sma_section += "- Both should be included in the model\n"
        
        if 'feature_categories' in report:
            categories = report['feature_categories']
            fred_sma_section += f"\nFinal Distribution:\n"
            fred_sma_section += f"  FRED: {len(categories.get('FRED', []))} features\n"
            fred_sma_section += f"  Technical: {len(categories.get('Technical', []))} features\n"
    
    add_text_page(pdf, "FRED vs SMA Analysis", fred_sma_section)
    
    # Model Performance
    if results:
        perf_section = "MODEL PERFORMANCE\n\n"
        perf_section += f"Best Model: {results.get('best_model', 'N/A')}\n\n"
        
        if 'overfitting_analysis' in results:
            of_analysis = results['overfitting_analysis']
            perf_section += "Performance Metrics:\n"
            after = of_analysis.get('after_tuning', {})
            perf_section += f"  Train+Val R²: {after.get('train_val_score', 0):.4f}\n"
            perf_section += f"  Test R²: {after.get('test_score', 0):.4f}\n"
            perf_section += f"  Gap: {after.get('gap', 0):.2%}\n"
            
            if 'test_metrics' in results:
                test_metrics = results['test_metrics']
                perf_section += f"  Test RMSE: {test_metrics.get('rmse', 0):.4f}\n"
                perf_section += f"  Test MAE: {test_metrics.get('mae', 0):.4f}\n"
        
        add_text_page(pdf, "Model Performance", perf_section)
    
    # Overfitting Analysis
    if overfitting_report:
        of_section = "OVERFITTING ANALYSIS\n\n"
        of_section += f"Status: {overfitting_report.get('overall_status', 'N/A')}\n\n"
        
        if 'overfitting' in overfitting_report:
            of = overfitting_report['overfitting']
            if of.get('is_overfitting', False):
                of_section += f"Overfitting Detected: {of.get('severity', 'unknown')}\n\n"
                for issue in of.get('issues', []):
                    of_section += f"  - {issue}\n"
            else:
                of_section += "No Significant Overfitting\n"
            
            if of.get('train_val_gap'):
                of_section += f"\nTrain-Val Gap: {of['train_val_gap']:.2%}\n"
            if of.get('train_test_gap'):
                of_section += f"Train-Test Gap: {of['train_test_gap']:.2%}\n"
        
        add_text_page(pdf, "Overfitting Analysis", of_section)
    
    # Conclusions
    conclusions = """
    CONCLUSIONS AND RECOMMENDATIONS
    
    DATA LEAKAGE PREVENTION:
    1. Always use LAGGED features (prev_open, prev_high, prev_low, prev_close)
    2. Never use same-day OHLC to predict same-day close
    3. Ensure all features are from past time periods only
    4. Check for look-ahead bias (forward-looking indicators)
    5. Verify target column is not in features
    
    FEATURE SELECTION:
    1. Remove redundant features aggressively
    2. Use VIF to detect multicollinearity
    3. Remove highly correlated features (>0.85)
    4. Check FRED vs Technical redundancy
    5. Keep only most predictive features (5-10)
    
    OVERFITTING REDUCTION:
    1. Lower R² scores are better than overfitted high R²
    2. Focus on test performance, not train performance
    3. Monitor train-test gap continuously
    4. Use strong regularization
    5. Accept that perfect predictions are unrealistic
    
    MODEL VALIDATION:
    1. Only use models that pass all leakage checks
    2. Verify no data leakage before deployment
    3. Test on out-of-sample data
    4. Monitor performance over time
    5. Retrain periodically with new data
    """
    add_text_page(pdf, "Conclusions", conclusions)
    
    # Metadata
    pdf.infodict()['Title'] = 'Bitcoin Comprehensive Analysis with Data Leakage Checks'
    pdf.infodict()['Author'] = 'ML Analysis Pipeline'
    pdf.infodict()['Subject'] = 'Bitcoin Analysis, Data Leakage, Feature Selection'
    pdf.infodict()['Keywords'] = 'Bitcoin, Data Leakage, OHLC, Look-Ahead Bias, Feature Selection'
    pdf.infodict()['CreationDate'] = datetime.now()

print(f"✓ PDF created: {OUTPUT_PDF}")
print()



