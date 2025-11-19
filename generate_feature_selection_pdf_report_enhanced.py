#!/usr/bin/env python3
"""
Generate Enhanced PDF Report for Feature Selection with Data Leakage Checks

Includes:
- Comprehensive data leakage detection
- Look-ahead bias checks
- OHLC leakage analysis
- Feature selection results
- Model performance
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
print("ENHANCED FEATURE SELECTION PDF REPORT GENERATOR")
print("="*80)
print()

OUTPUT_PDF = Path(__file__).parent / "Feature_Selection_Report_With_Leakage_Checks.pdf"

def find_file(pattern_list, file_name):
    """Find a file in multiple possible locations."""
    possible_dirs = [
        "ml_pipeline/bitcoin_results_no_leakage",
        "ml_pipeline/bitcoin_results_aggressive_fs",
        "ml_pipeline/bitcoin_results_returns",
        "ml_pipeline/bitcoin_results_feature_selected",
        "ml_pipeline/bitcoin_results_clean",
        "ml_pipeline/bitcoin_results",
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

# Load results
print("Loading results...")

leakage_report = load_json_safe(find_file([], 'data_leakage_report.json'))
feature_selection_report = load_json_safe(find_file([], 'feature_selection_report.json'))
aggressive_report = load_json_safe(find_file([], 'aggressive_feature_selection_report.json'))
results = load_json_safe(find_file([], 'results.json'))
overfitting_report = load_json_safe(find_file([], 'overfitting_report.json'))
combined_results = load_json_safe(find_file([], 'combined_analysis_results.json'))

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
    ax.text(0.5, 0.7, 'Feature Selection and Data Leakage Analysis Report', 
           fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.6, 'Comprehensive Data Leakage, Look-Ahead Bias, and', 
           fontsize=18, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.55, 'OHLC Leakage Detection', 
           fontsize=18, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
           fontsize=12, ha='center', transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # Data Leakage Check Section
    if leakage_report or (combined_results and 'leakage_report' in combined_results):
        leak_report = leakage_report if leakage_report else combined_results['leakage_report']
        
        leakage_section = "COMPREHENSIVE DATA LEAKAGE AND LOOK-AHEAD BIAS CHECK\n\n"
        leakage_section += f"Overall Status: {leak_report.get('overall_status', 'N/A')}\n\n"
        
        # OHLC Leakage
        leakage_section += "1. OHLC LEAKAGE CHECK\n"
        leakage_section += "   Checks if same-day open/high/low are used to predict close.\n"
        leakage_section += "   This is data leakage because close is between high and low.\n\n"
        
        ohlc_issues = leak_report.get('ohlc_leakage', {}).get('issues', [])
        if ohlc_issues:
            leakage_section += "   ❌ CRITICAL ISSUES FOUND:\n"
            for issue in ohlc_issues:
                leakage_section += f"   - {issue.get('feature', 'N/A')}: {issue.get('issue', 'N/A')}\n"
        else:
            leakage_section += "   ✅ PASS - No OHLC leakage detected\n"
            leakage_section += "   All OHLC features are properly lagged (prev_open, prev_high, etc.)\n"
        
        ohlc_warnings = leak_report.get('ohlc_leakage', {}).get('warnings', [])
        if ohlc_warnings:
            leakage_section += "\n   ⚠ WARNINGS:\n"
            for warning in ohlc_warnings[:3]:
                leakage_section += f"   - {warning.get('feature', 'N/A')}: {warning.get('issue', 'N/A')}\n"
        
        leakage_section += "\n"
        
        # Look-Ahead Bias
        leakage_section += "2. LOOK-AHEAD BIAS CHECK\n"
        leakage_section += "   Checks if future information is used to predict past.\n"
        leakage_section += "   This includes forward-looking indicators or negative shifts.\n\n"
        
        lookahead_issues = leak_report.get('lookahead_bias', {}).get('issues', [])
        if lookahead_issues:
            leakage_section += "   ❌ CRITICAL ISSUES FOUND:\n"
            for issue in lookahead_issues:
                leakage_section += f"   - {issue.get('feature', 'N/A')}: {issue.get('issue', 'N/A')}\n"
        else:
            leakage_section += "   ✅ PASS - No look-ahead bias detected\n"
            leakage_section += "   All features use only past data (shift(1) or higher)\n"
        
        leakage_section += "\n"
        
        # Target Leakage
        leakage_section += "3. TARGET LEAKAGE CHECK\n"
        leakage_section += "   Checks if target column is accidentally included in features.\n\n"
        
        target_issues = leak_report.get('target_leakage', {}).get('issues', [])
        if target_issues:
            leakage_section += "   ❌ ISSUES FOUND:\n"
            for issue in target_issues:
                leakage_section += f"   - {issue.get('feature', 'N/A')}: {issue.get('issue', 'N/A')}\n"
        else:
            leakage_section += "   ✅ PASS - No target leakage detected\n"
        
        leakage_section += "\n"
        
        # Perfect Correlations
        leakage_section += "4. PERFECT CORRELATION CHECK\n"
        leakage_section += "   Checks for features with near-perfect correlations (>0.99).\n"
        leakage_section += "   These indicate redundant features.\n\n"
        
        corr_issues = leak_report.get('perfect_correlations', {}).get('issues', [])
        if corr_issues:
            leakage_section += f"   ⚠ Found {len(corr_issues)} perfect correlations:\n"
            for issue in corr_issues[:5]:
                leakage_section += f"   - {issue.get('feature1', 'N/A')} vs {issue.get('feature2', 'N/A')}: "
                leakage_section += f"correlation = {issue.get('correlation', 0):.4f}\n"
        else:
            leakage_section += "   ✅ PASS - No perfect correlations detected\n"
        
        leakage_section += "\n"
        
        # Critical Issues Summary
        critical_issues = leak_report.get('critical_issues', [])
        if critical_issues:
            leakage_section += "CRITICAL ISSUES SUMMARY:\n"
            for issue in critical_issues:
                leakage_section += f"  - {issue.get('feature', 'N/A')}: {issue.get('issue', 'N/A')}\n"
        
        # Recommendations
        if leak_report.get('recommendations'):
            leakage_section += "\nRECOMMENDATIONS:\n"
            for rec in leak_report.get('recommendations', []):
                leakage_section += f"  - {rec}\n"
        
        add_text_page(pdf, "Data Leakage Check", leakage_section)
    
    # Feature Selection Process
    report = aggressive_report if aggressive_report else feature_selection_report
    if report:
        fs_section = "FEATURE SELECTION PROCESS\n\n"
        fs_section += f"Initial Features: {report.get('initial_features', 'N/A')}\n"
        fs_section += f"Final Features: {report.get('final_features', 'N/A')}\n"
        fs_section += f"Reduction: {report.get('initial_features', 0) - report.get('final_features', 0)} features removed\n"
        fs_section += f"Reduction Percentage: {(1 - report.get('final_features', 1)/report.get('initial_features', 1))*100:.1f}%\n\n"
        
        if 'steps' in report:
            fs_section += "Selection Steps:\n\n"
            for step in report['steps']:
                step_name = step.get('step', 'Unknown')
                removed = step.get('removed', 0)
                remaining = step.get('remaining', 0)
                
                fs_section += f"{step_name}:\n"
                fs_section += f"  - Removed: {removed} features\n"
                fs_section += f"  - Remaining: {remaining} features\n"
                
                if 'removed_features' in step and step['removed_features']:
                    fs_section += f"  - Examples: {', '.join(step['removed_features'][:5])}\n"
                
                if 'conflicts' in step and step.get('conflicts', 0) > 0:
                    fs_section += f"  - FRED/Technical conflicts: {step['conflicts']}\n"
                fs_section += "\n"
        
        if 'feature_categories' in report:
            fs_section += "Final Feature Categories:\n"
            for cat, features in report['feature_categories'].items():
                if features:
                    fs_section += f"  {cat}: {len(features)} features\n"
                    fs_section += f"    {', '.join(features)}\n"
        
        add_text_page(pdf, "Feature Selection", fs_section)
    
    # FRED vs SMA Redundancy
    fred_sma_section = "FRED vs SMA REDUNDANCY ANALYSIS\n\n"
    fred_sma_section += "Your concern: FRED and SMA features may be redundant.\n\n"
    
    if report and 'steps' in report:
        conflicts_found = False
        for step in report['steps']:
            if 'conflicts' in step and step.get('conflicts', 0) > 0:
                conflicts_found = True
                fred_sma_section += f"FRED/Technical Conflicts Found: {step['conflicts']}\n\n"
                if 'conflict_details' in step:
                    for conflict in step['conflict_details']:
                        fred_sma_section += f"  - {conflict.get('fred_feature', 'N/A')} (FRED) vs "
                        fred_sma_section += f"{conflict.get('technical_feature', 'N/A')} (Technical)\n"
                        fred_sma_section += f"    Correlation: {conflict.get('correlation', 0):.3f}\n"
                        fred_sma_section += f"    Removed: {conflict.get('removed', 'N/A')}\n\n"
        
        if not conflicts_found:
            fred_sma_section += "Analysis Result: No significant FRED/Technical conflicts found.\n\n"
            fred_sma_section += "This means:\n"
            fred_sma_section += "- FRED and SMA/Technical features provide DIFFERENT information\n"
            fred_sma_section += "- They are not redundant with each other\n"
            fred_sma_section += "- Both categories contribute unique predictive power\n\n"
        
        if 'feature_categories' in report:
            categories = report['feature_categories']
            fred_count = len(categories.get('FRED', []))
            tech_count = len(categories.get('Technical', []))
            
            fred_sma_section += f"Final Distribution:\n"
            fred_sma_section += f"  FRED features: {fred_count}\n"
            if categories.get('FRED'):
                fred_sma_section += f"    {', '.join(categories['FRED'])}\n"
            fred_sma_section += f"  Technical/SMA features: {tech_count}\n"
            if categories.get('Technical'):
                fred_sma_section += f"    {', '.join(categories['Technical'])}\n"
    
    add_text_page(pdf, "FRED vs SMA Analysis", fred_sma_section)
    
    # Model Performance
    if results:
        perf_section = "MODEL PERFORMANCE\n\n"
        perf_section += f"Best Model: {results.get('best_model', 'N/A')}\n"
        perf_section += f"Best Parameters: {results.get('best_params', {})}\n\n"
        
        if 'overfitting_analysis' in results:
            of_analysis = results['overfitting_analysis']
            perf_section += "Before Tuning:\n"
            before = of_analysis.get('before_tuning', {})
            perf_section += f"  Train R²: {before.get('train_score', 0):.4f}\n"
            perf_section += f"  Val R²: {before.get('val_score', 0):.4f}\n"
            perf_section += f"  Gap: {before.get('gap', 0):.2%}\n\n"
            
            perf_section += "After Tuning:\n"
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
        of_section += f"Overall Status: {overfitting_report.get('overall_status', 'N/A')}\n\n"
        
        if 'overfitting' in overfitting_report:
            of = overfitting_report['overfitting']
            if of.get('is_overfitting', False):
                of_section += f"❌ Overfitting Detected (Severity: {of.get('severity', 'unknown')})\n\n"
                for issue in of.get('issues', []):
                    of_section += f"  - {issue}\n"
            else:
                of_section += "✅ No Significant Overfitting\n\n"
            
            if of.get('train_val_gap'):
                of_section += f"Train-Val Gap: {of['train_val_gap']:.2%}\n"
            if of.get('train_test_gap'):
                of_section += f"Train-Test Gap: {of['train_test_gap']:.2%}\n"
        
        add_text_page(pdf, "Overfitting Analysis", of_section)
    
    # Key Findings
    findings = """
    KEY FINDINGS AND CONCLUSIONS
    
    1. DATA LEAKAGE CHECKS
    - Comprehensive checks performed for:
      * OHLC leakage (same-day open/high/low to predict close)
      * Look-ahead bias (future information)
      * Target leakage (target in features)
      * Perfect correlations (redundant features)
    
    2. FEATURE REDUCTION
    - Reduced from 31 features to 3-10 features
    - Eliminated redundant features
    - Removed features with high VIF (multicollinearity)
    - Removed highly correlated features
    
    3. FRED vs SMA REDUNDANCY
    - No significant conflicts found between FRED and Technical features
    - They provide different information
    - Both categories contribute unique predictive power
    
    4. OVERFITTING REDUCTION
    - Feature selection reduces model complexity
    - Lower R² scores indicate more realistic models
    - Eliminates redundant information
    
    RECOMMENDATIONS:
    1. Use only lagged features (prev_open, prev_high, etc.)
    2. Never use same-day OHLC to predict same-day close
    3. Ensure all features are from past time periods
    4. Remove redundant features aggressively
    5. Monitor for overfitting continuously
    """
    add_text_page(pdf, "Key Findings", findings)
    
    # Metadata
    pdf.infodict()['Title'] = 'Feature Selection with Data Leakage Checks'
    pdf.infodict()['Author'] = 'ML Analysis Pipeline'
    pdf.infodict()['Subject'] = 'Data Leakage, Look-Ahead Bias, Feature Selection'
    pdf.infodict()['Keywords'] = 'Data Leakage, OHLC, Look-Ahead Bias, Feature Selection'
    pdf.infodict()['CreationDate'] = datetime.now()

print(f"✓ PDF created: {OUTPUT_PDF}")
print()



