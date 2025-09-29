#!/usr/bin/env python3
"""Check the progress of Santander experiments"""

import os
import time
from pathlib import Path

output_dir = Path('outputs/santander-customer-transaction-prediction')

print("Monitoring Santander experiment progress...")
print("=" * 60)

# Check for augmented data files
augmented_dir = Path('data/augmented/santander')
if augmented_dir.exists():
    files = list(augmented_dir.glob('*.csv'))
    print(f"\nAugmented data files created: {len(files)}")
    for f in sorted(files)[-5:]:  # Show last 5
        print(f"  - {f.name}")

# Check for output files
if output_dir.exists():
    # Check comprehensive comparison
    comp_dir = output_dir / 'comprehensive_comparison'
    if comp_dir.exists():
        csv_files = list(comp_dir.glob('*.csv'))
        if csv_files:
            print(f"\nComparison CSV files: {len(csv_files)}")
            for f in csv_files:
                print(f"  - {f.name}")
    
    # Check detailed results
    detail_dir = output_dir / 'detailed_results'
    if detail_dir.exists():
        csv_files = list(detail_dir.glob('*.csv'))
        if csv_files:
            print(f"\nDetailed result files: {len(csv_files)}")
    
    # Check for plots
    plot_files = list(output_dir.glob('*.png'))
    if plot_files:
        print(f"\nPlot files: {len(plot_files)}")
        for f in plot_files:
            print(f"  - {f.name}")
    
    # Check for report
    report_files = list(output_dir.glob('*.txt'))
    if report_files:
        print(f"\nReport files: {len(report_files)}")
        for f in report_files:
            print(f"  - {f.name}")

print("\n" + "=" * 60)
print("The experiment is running in the background. It will test:")
print("- 15 augmentation methods (none + 7 base methods + 7 RUBRIC variants)")
print("- Multiple classifiers for each method")
print("This may take 30-60 minutes depending on your system.")
