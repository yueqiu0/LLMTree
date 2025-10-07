#!/usr/bin/env python3
"""
Script to analyze evaluation results from JSON files and generate AUC and accuracy tables.
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
import re

def extract_dataset_name(filename):
    """Extract dataset name from filename like 'baseline_dt_airline_satisfaction.json'"""
    # Remove 'baseline_dt_' prefix and '.json' suffix
    name = filename.replace('baseline_dt_', '').replace('.json', '')
    return name

def load_json_data(filepath):
    """Load and parse JSON data from file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_metrics(data):
    """Extract tree_auc and tree_accuracy metrics from JSON data"""
    results = {}
    
    if 'results' not in data:
        return results
    
    for train_size, experiments in data['results'].items():
        train_size = int(train_size)
        auc_values = []
        accuracy_values = []
        
        for exp in experiments:
            # Extract tree_auc (handle both dict and float formats)
            if 'tree_auc' in exp:
                auc_value = exp['tree_auc']
                if isinstance(auc_value, dict):
                    # Use macro value if it's a dictionary
                    auc_values.append(auc_value.get('macro', 0.0))
                else:
                    # Use the value directly if it's a number
                    auc_values.append(auc_value)
            
            # Extract tree_accuracy
            if 'tree_accuracy' in exp:
                accuracy_values.append(exp['tree_accuracy'])
        
        if auc_values and accuracy_values:
            results[train_size] = {
                'auc': auc_values,
                'accuracy': accuracy_values
            }
    
    return results

def calculate_statistics(values):
    """Calculate mean and standard deviation"""
    if not values:
        return 0.0, 0.0
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return mean_val, std_val

def format_value(mean, std, decimals=3):
    """Format value as 'mean±std' with specified decimal places"""
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"

def create_tables(all_results):
    """Create AUC and accuracy tables for all datasets"""
    # Collect all unique train sizes across all datasets
    all_train_sizes = set()
    for dataset_name, results in all_results.items():
        all_train_sizes.update(results.keys())
    all_train_sizes = sorted(all_train_sizes)
    
    # Create AUC table
    auc_table = []
    for dataset_name, results in all_results.items():
        row = [dataset_name]
        for train_size in all_train_sizes:
            if train_size in results and 'auc' in results[train_size]:
                mean, std = calculate_statistics(results[train_size]['auc'])
                row.append(format_value(mean, std))
            else:
                row.append('N/A')
        auc_table.append(row)
    
    # Create accuracy table
    accuracy_table = []
    for dataset_name, results in all_results.items():
        row = [dataset_name]
        for train_size in all_train_sizes:
            if train_size in results and 'accuracy' in results[train_size]:
                mean, std = calculate_statistics(results[train_size]['accuracy'])
                row.append(format_value(mean, std))
            else:
                row.append('N/A')
        accuracy_table.append(row)
    
    return auc_table, accuracy_table, all_train_sizes

def save_tables(auc_table, accuracy_table, train_sizes, output_dir):
    """Save tables to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create headers
    headers = ['Dataset'] + [f'{size} samples' for size in train_sizes]
    
    # Save AUC table
    auc_df = pd.DataFrame(auc_table, columns=headers)
    auc_file = os.path.join(output_dir, 'auc_table.csv')
    auc_df.to_csv(auc_file, index=False)
    print(f"AUC table saved to: {auc_file}")
    
    # Save accuracy table
    accuracy_df = pd.DataFrame(accuracy_table, columns=headers)
    accuracy_file = os.path.join(output_dir, 'accuracy_table.csv')
    accuracy_df.to_csv(accuracy_file, index=False)
    print(f"Accuracy table saved to: {accuracy_file}")
    
    # Also save as LaTeX tables
    auc_latex_file = os.path.join(output_dir, 'auc_table.tex')
    with open(auc_latex_file, 'w') as f:
        f.write(auc_df.to_latex(index=False, escape=False))
    print(f"AUC LaTeX table saved to: {auc_latex_file}")
    
    accuracy_latex_file = os.path.join(output_dir, 'accuracy_table.tex')
    with open(accuracy_latex_file, 'w') as f:
        f.write(accuracy_df.to_latex(index=False, escape=False))
    print(f"Accuracy LaTeX table saved to: {accuracy_latex_file}")

def create_csv_tables(all_results):
    """Create CSV tables in the specific format requested by user"""
    # Define the specific format for each dataset
    dataset_formats = {
        'abalone': [2, 4, 8, 16, 32],
        'airline_satisfaction': [2, 4, 8, 16, 32],
        'balance_scale': [3, 6, 12, 24, 48],
        'blood': [2, 4, 8, 16, 32],
        'cmc': [3, 6, 12, 24, 48],
        'diabetes': [2, 4, 8, 16, 32],
        'glioma_grading': [2, 4, 8, 16, 32],
        'heart_statlog': [2, 4, 8, 16, 32],
        'iris': [3, 6, 12, 24, 48],
        'spambase': [2, 4, 8, 16, 32],
        'wine': [3, 6, 12, 24, 48]
    }
    
    # Create AUC CSV data
    auc_csv_data = []
    accuracy_csv_data = []
    
    for dataset_name, train_sizes in dataset_formats.items():
        if dataset_name in all_results:
            for train_size in train_sizes:
                if train_size in all_results[dataset_name]:
                    # Get AUC data
                    if 'auc' in all_results[dataset_name][train_size]:
                        mean, std = calculate_statistics(all_results[dataset_name][train_size]['auc'])
                        auc_value = format_value(mean, std)
                    else:
                        auc_value = "N/A"
                    
                    # Get accuracy data
                    if 'accuracy' in all_results[dataset_name][train_size]:
                        mean, std = calculate_statistics(all_results[dataset_name][train_size]['accuracy'])
                        accuracy_value = format_value(mean, std)
                    else:
                        accuracy_value = "N/A"
                    
                    auc_csv_data.append([dataset_name, train_size, auc_value])
                    accuracy_csv_data.append([dataset_name, train_size, accuracy_value])
                else:
                    auc_csv_data.append([dataset_name, train_size, "N/A"])
                    accuracy_csv_data.append([dataset_name, train_size, "N/A"])
    
    return auc_csv_data, accuracy_csv_data

def main():
    """Main function to process all JSON files and generate tables"""
    # Define paths
    input_dir = "output/evaluate"
    output_dir = "output/sta"
    
    # Find all baseline_dt_*.json files
    json_files = []
    for filename in os.listdir(input_dir):
        if filename.startswith('baseline_dt_') and filename.endswith('.json'):
            json_files.append(filename)
    
    print(f"Found {len(json_files)} JSON files to process:")
    for f in sorted(json_files):
        print(f"  - {f}")
    
    # Process each JSON file
    all_results = {}
    
    for filename in sorted(json_files):
        dataset_name = extract_dataset_name(filename)
        filepath = os.path.join(input_dir, filename)
        
        print(f"\nProcessing {filename}...")
        data = load_json_data(filepath)
        
        if data is None:
            print(f"  Skipped {filename} due to loading error")
            continue
        
        results = extract_metrics(data)
        if results:
            all_results[dataset_name] = results
            print(f"  Extracted data for {len(results)} train sizes")
        else:
            print(f"  No valid results found in {filename}")
    
    if not all_results:
        print("No valid results found in any files!")
        return
    
    print(f"\nSuccessfully processed {len(all_results)} datasets")
    
    # Create tables
    print("\nCreating tables...")
    auc_table, accuracy_table, train_sizes = create_tables(all_results)
    
    # Save tables
    print("\nSaving tables...")
    save_tables(auc_table, accuracy_table, train_sizes, output_dir)
    
    # Create CSV format tables
    print("\nCreating CSV format tables...")
    auc_csv_data, accuracy_csv_data = create_csv_tables(all_results)
    
    # Save CSV format tables
    auc_csv_file = os.path.join(output_dir, 'auc_results.csv')
    with open(auc_csv_file, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['Data', 'Shot', 'CART'])  # Header
        writer.writerows(auc_csv_data)
    print(f"AUC CSV table saved to: {auc_csv_file}")
    
    accuracy_csv_file = os.path.join(output_dir, 'accuracy_results.csv')
    with open(accuracy_csv_file, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['Data', 'Shot', 'CART'])  # Header
        writer.writerows(accuracy_csv_data)
    print(f"Accuracy CSV table saved to: {accuracy_csv_file}")
    
    # Print CSV tables to console
    print(f"\n" + "="*80)
    print("AUC TABLE (CSV Format):")
    print("="*80)
    print("Data,Shot,CART")
    for row in auc_csv_data:
        print(f"{row[0]},{row[1]},{row[2]}")
    
    print(f"\n" + "="*80)
    print("ACCURACY TABLE (CSV Format):")
    print("="*80)
    print("Data,Shot,CART")
    for row in accuracy_csv_data:
        print(f"{row[0]},{row[1]},{row[2]}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  - Processed {len(all_results)} datasets")
    print(f"  - Train sizes: {train_sizes}")
    print(f"  - Tables saved to: {output_dir}/")
    
    # Print a sample of the tables
    print(f"\nSample AUC table (first 3 rows):")
    for i, row in enumerate(auc_table[:3]):
        print(f"  {row}")
    
    print(f"\nSample accuracy table (first 3 rows):")
    for i, row in enumerate(accuracy_table[:3]):
        print(f"  {row}")

if __name__ == "__main__":
    main()
