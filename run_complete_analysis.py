#!/usr/bin/env python3
"""
Complete Analysis Pipeline Runner

Runs the full ML pipeline and generates comprehensive PDF report.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from run_bitcoin_prediction import *
from generate_comprehensive_pdf_report import *

if __name__ == "__main__":
    print("="*80)
    print("COMPLETE BITCOIN ANALYSIS PIPELINE")
    print("="*80)
    print()
    
    # Step 1: Run ML model selection and prediction
    print("STEP 1: Running ML Model Selection and Prediction...")
    print("-" * 80)
    
    # This will be executed when the module is imported
    # The actual execution happens in run_bitcoin_prediction.py
    
    # Step 2: Generate PDF report
    print("\nSTEP 2: Generating Comprehensive PDF Report...")
    print("-" * 80)
    
    # This will be executed when the module is imported
    # The actual execution happens in generate_comprehensive_pdf_report.py
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Review the ML results in: ml_pipeline/bitcoin_results/")
    print("2. Check the PDF report: Bitcoin_Comprehensive_Analysis_Report.pdf")
    print("3. Review predictions in: ml_pipeline/bitcoin_results/latest_prediction.json")
    print()



