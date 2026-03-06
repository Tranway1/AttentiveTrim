#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Analysis Script for AITQA Dataset
Combines AttentiveTrim method results and RAG baseline results in one big table.
ENHANCED VERSION:
- Supports both 'aitqa' and 'aitqa_full_table' baseline directories
- Automatically detects and reads dataset file for row usage calculation
"""

import json
import os
import pandas as pd
import re
import argparse
from pathlib import Path
from collections import defaultdict

# Model name mapping for AttentiveTrim methods
MODEL_MAPPING = {
    'llama3.2_1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'qwen3_8b': 'Qwen/Qwen3-8B',
    'qwen3_14b': 'Qwen/Qwen3-14B'
}

def parse_method_filename(filename):
    """
    Parse AttentiveTrim method filename like 'farest_top4_doc_0_eval.json'
    Returns: (method_type, budget, doc_id) e.g., ('farest', 4, 0)
    """
    pattern = r'(\w+)_top(\d+)_doc_(\d+)_eval\.json'
    match = re.match(pattern, filename)
    if match:
        method_type = match.group(1)
        budget = int(match.group(2))
        doc_id = int(match.group(3))
        return method_type, budget, doc_id
    return None, None, None

def parse_baseline_eval_directory(dirname):
    """
    Parse RAG baseline directory name like 'aitqa_UAE-Large-V1_top5' or 'aitqa_full_table'
    Returns: (embedding_model, top_k, is_full_table) 
    e.g., ('UAE-Large-V1', 5, False) or ('full_table', None, True)
    """
    # Check for full table pattern first
    if dirname == 'aitqa_full_table':
        return 'full_table', None, True
    
    # Check for regular RAG pattern
    pattern = r'aitqa_(.+)_top(\d+)'
    match = re.match(pattern, dirname)
    if match:
        embedding_model = match.group(1)
        top_k = int(match.group(2))
        return embedding_model, top_k, False
    
    return None, None, False

def parse_doc_filename(filename):
    """
    Parse filename like 'baseline_doc_0_eval.json'
    Returns: doc_id e.g., 0
    """
    pattern = r'baseline_doc_(\d+)_eval\.json'
    match = re.match(pattern, filename)
    if match:
        doc_id = int(match.group(1))
        return doc_id
    return None

def load_eval_data(filepath):
    """Load full evaluation data from JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def count_table_rows(table_content):
    """
    Count the number of rows in a table.
    The table is stored as a JSON string with structure:
    {
      "column_header": [...],
      "row_header": [...],
      "data": [[row1], [row2], ...],
      "id": "..."
    }
    """
    if isinstance(table_content, str):
        try:
            # Try to parse as JSON first
            table_obj = json.loads(table_content)
            if isinstance(table_obj, dict) and 'data' in table_obj:
                # Count rows in the data field
                data = table_obj['data']
                if isinstance(data, list):
                    return len(data)
        except json.JSONDecodeError:
            # If not JSON, fall back to newline counting
            lines = [line.strip() for line in table_content.split('\n') if line.strip()]
            # Subtract 1 for header row
            num_rows = max(0, len(lines) - 1)
            return num_rows
    elif isinstance(table_content, dict):
        # Already parsed as dict
        if 'data' in table_content:
            data = table_content['data']
            if isinstance(data, list):
                return len(data)
    elif isinstance(table_content, list):
        # If it's a list of rows, count them (excluding header)
        num_rows = max(0, len(table_content) - 1)
        return num_rows
    
    # Unknown format
    return None

def auto_find_dataset_file(method_models):
    """
    Automatically search for aitqa_processed.json in common locations.
    Searches in method model directories.
    """
    print("\n🔍 Auto-detecting dataset file...")
    
    # Common paths to check
    search_paths = []
    
    # Check in each method model directory
    for model in method_models:
        search_paths.extend([
            Path(model) / 'data' / 'datasets' / 'aitqa_processed.json',
            Path(model) / 'datasets' / 'aitqa_processed.json',
            Path(model) / 'aitqa_processed.json',
        ])
    
    # Also check some common global locations
    search_paths.extend([
        Path('data') / 'datasets' / 'aitqa_processed.json',
        Path('datasets') / 'aitqa_processed.json',
        Path('aitqa_processed.json'),
        Path('../data') / 'datasets' / 'aitqa_processed.json',
    ])
    
    # Try each path
    for path in search_paths:
        if path.exists():
            print(f"✅ Found dataset file: {path}")
            return path
    
    print("⚠️  Dataset file not found in common locations")
    print("   Searched in:")
    for path in search_paths[:5]:  # Show first 5 search paths
        print(f"     - {path}")
    print("   ...")
    
    return None

def load_dataset_row_counts(dataset_file):
    """
    Load dataset and extract row counts for each document.
    Returns: dict mapping doc_id to number of rows
    """
    if dataset_file is None:
        return {}
    
    print(f"\n📖 Loading dataset to extract row counts: {dataset_file}")
    
    if not dataset_file.exists():
        print(f"⚠️  WARNING: Dataset file not found: {dataset_file}")
        print(f"   Row usage statistics will not be calculated.")
        return {}
    
    try:
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        
        documents = dataset.get('documents', [])
        if not documents:
            print(f"⚠️  WARNING: No documents found in dataset")
            return {}
        
        print(f"   Processing {len(documents)} documents...")
        row_counts = {}
        failed_docs = []
        
        for doc in documents:
            doc_id = doc['document_id']
            content = doc.get('content', '')
            
            if not content:
                failed_docs.append(doc_id)
                continue
            
            num_rows = count_table_rows(content)
            if num_rows is not None and num_rows > 0:
                row_counts[doc_id] = num_rows
            else:
                failed_docs.append(doc_id)
        
        print(f"✅ Extracted row counts for {len(row_counts)}/{len(documents)} documents")
        
        if failed_docs and len(failed_docs) <= 5:
            print(f"   ⚠️  Failed to count rows for docs: {failed_docs}")
        elif failed_docs:
            print(f"   ⚠️  Failed to count rows for {len(failed_docs)} documents")
        
        # Show some examples
        if row_counts:
            print(f"   Sample row counts:")
            for doc_id in sorted(row_counts.keys())[:5]:
                print(f"     doc_{doc_id}: {row_counts[doc_id]} rows")
            if len(row_counts) > 5:
                print(f"     ... and {len(row_counts) - 5} more")
        else:
            print(f"   ⚠️  No row counts extracted - check content format")
        
        return row_counts
        
    except Exception as e:
        print(f"⚠️  WARNING: Error loading dataset: {e}")
        print(f"   Row usage statistics will not be calculated.")
        return {}

def print_separator(char='=', length=80):
    """Print a separator line"""
    print(char * length)

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive analysis of AITQA results - Methods + Baselines (Enhanced)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze with auto-detection (recommended)
  python analyze_aitqa_comprehensive.py
  
  # Specify dataset file manually
  python analyze_aitqa_comprehensive.py --dataset-file path/to/aitqa_processed.json
  
  # Specific method models and types
  python analyze_aitqa_comprehensive.py --method-models llama3.2_1b qwen3_8b
  
  # Include only farest method type
  python analyze_aitqa_comprehensive.py --method-types farest
  
  # Custom directories
  python analyze_aitqa_comprehensive.py --baseline-eval-dir baselines/evaluation --output-dir results
        """
    )
    
    parser.add_argument('--dataset-file', type=str, default=None,
                       help='Path to aitqa_processed.json (optional, will auto-detect if not provided)')
    parser.add_argument('--method-models', nargs='+', 
                       choices=list(MODEL_MAPPING.keys()),
                       default=list(MODEL_MAPPING.keys()),
                       help='Which AttentiveTrim method models to include (default: all)')
    parser.add_argument('--method-types', nargs='+',
                       choices=['farest', 'baseline', 'raw'],
                       default=['farest', 'baseline', 'raw'],
                       help='Which method types to include (default: all)')
    parser.add_argument('--baseline-eval-dir', type=str, default='baselines/evaluation',
                       help='Base directory containing RAG baseline evaluation results (default: baselines/evaluation)')
    parser.add_argument('--output-dir', type=str, default='analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE AITQA ANALYSIS - METHODS + BASELINES (ENHANCED)")
    print("="*80)
    print(f"Method models: {', '.join(args.method_models)}")
    print(f"Method types: {', '.join(args.method_types)}")
    print(f"Baseline eval dir: {args.baseline_eval_dir}")
    print(f"Output directory: {output_dir}")
    
    # ========================================================================
    # AUTO-DETECT OR LOAD DATASET FILE
    # ========================================================================
    if args.dataset_file:
        dataset_file = Path(args.dataset_file)
        print(f"Dataset file (manual): {dataset_file}")
    else:
        dataset_file = auto_find_dataset_file(args.method_models)
        if dataset_file:
            print(f"Dataset file (auto-detected): {dataset_file}")
    
    table_row_counts = load_dataset_row_counts(dataset_file)
    
    # ========================================================================
    # LOAD ATTENTIVETRIM METHOD RESULTS
    # ========================================================================
    print()
    print_separator('-')
    print("📊 LOADING ATTENTIVETRIM METHOD RESULTS")
    print_separator('-')
    
    # method_data[model_name][method_type][budget][doc_id] = full_eval_data
    method_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    method_budgets = set()
    all_docs = set()
    
    for model_name in args.method_models:
        method_dir = Path(model_name) / 'evaluation' / 'aitqa'
        
        print(f"\n📂 Checking {model_name}...")
        print(f"   Looking in: {method_dir.absolute()}")
        
        if not method_dir.exists():
            print(f"   ❌ Directory not found!")
            print(f"   Please ensure {method_dir} exists")
            continue
        
        # List all files to debug
        all_files = list(method_dir.glob('*.json'))
        print(f"   Found {len(all_files)} JSON files in directory")
        
        # Track what we're loading per model
        model_stats = defaultdict(int)
        
        for filename in sorted(os.listdir(method_dir)):
            if filename.endswith('_eval.json'):
                method_type, budget, doc_id = parse_method_filename(filename)
                if method_type and budget is not None and doc_id is not None:
                    # Check if this method_type is requested
                    if method_type not in args.method_types:
                        continue
                    
                    filepath = method_dir / filename
                    data = load_eval_data(filepath)
                    if data is not None:
                        method_data[model_name][method_type][budget][doc_id] = data
                        method_budgets.add(budget)
                        all_docs.add(doc_id)
                        model_stats[f"{method_type}_top{budget}"] += 1
                        
                        if model_stats[f"{method_type}_top{budget}"] <= 2:  # Show first 2 per config
                            acc = data.get('accuracy', 0)
                            matches = data.get('total_matches', 0)
                            total = data.get('total_questions', 0)
                            print(f"   ✓ {method_type}_top{budget} doc={doc_id}, acc={acc:.3f} ({matches}/{total})")
        
        # Print summary for this model
        if model_stats:
            print(f"   ✅ Loaded files for {model_name}:")
            for config, count in sorted(model_stats.items()):
                print(f"      - {config}: {count} documents")
        else:
            print(f"   ⚠️  No evaluation files found for requested types: {args.method_types}")
            if all_files:
                print(f"   Sample files found: {[f.name for f in all_files[:3]]}")
    
    if not method_data:
        print("\n⚠️  WARNING: No AttentiveTrim method results loaded!")
    
    method_budgets_sorted = sorted(method_budgets)
    print(f"\n📈 All method budgets found: {[f'top{b}' for b in method_budgets_sorted]}")
    print(f"📈 Total method models loaded: {len(method_data)}")
    print(f"📈 Method types loaded: {args.method_types}")
    
    # ========================================================================
    # LOAD RAG BASELINE RESULTS
    # ========================================================================
    print()
    print_separator('-')
    print("📊 LOADING RAG BASELINE RESULTS")
    print_separator('-')
    
    baseline_eval_dir = Path(args.baseline_eval_dir)
    
    print(f"Looking in: {baseline_eval_dir.absolute()}")
    
    # baseline_data[embedding_model][top_k][doc_id] = full_eval_data
    # Note: for full_table, top_k will be None
    baseline_data = defaultdict(lambda: defaultdict(dict))
    baseline_models = set()
    baseline_top_k = set()
    
    if not baseline_eval_dir.exists():
        print(f"❌ Baseline evaluation directory not found: {baseline_eval_dir.absolute()}")
        print(f"   Please ensure the directory exists")
    else:
        # List all directories to debug
        all_dirs = [d for d in baseline_eval_dir.iterdir() if d.is_dir()]
        print(f"\nFound {len(all_dirs)} directories in {baseline_eval_dir}")
        if all_dirs:
            print(f"Sample directories: {[d.name for d in all_dirs[:5]]}")
        
        # Scan for evaluation result directories
        config_dirs = []
        for item in baseline_eval_dir.iterdir():
            if item.is_dir():
                embedding_model, top_k, is_full_table = parse_baseline_eval_directory(item.name)
                if embedding_model:
                    config_dirs.append((item, embedding_model, top_k, is_full_table))
                else:
                    print(f"   ⚠️  Skipping directory (doesn't match pattern): {item.name}")
        
        if not config_dirs:
            print(f"\n❌ No baseline configuration directories found!")
            print(f"   Expected patterns:")
            print(f"     - aitqa_<embedding_model>_top<k> (e.g., aitqa_UAE-Large-V1_top5)")
            print(f"     - aitqa_full_table")
        else:
            print(f"\n📂 Found {len(config_dirs)} baseline configuration directories")
            print(f"   Configurations:")
            for _, em, tk, is_full in config_dirs:
                if is_full:
                    print(f"     - {em} (full table)")
                else:
                    print(f"     - {em} (top-{tk})")
        
        for config_dir, embedding_model, top_k, is_full_table in sorted(config_dirs):
            if is_full_table:
                print(f"\n📂 Loading {embedding_model} (full table)...")
            else:
                print(f"\n📂 Loading {embedding_model} (top-{top_k})...")
            print(f"   From: {config_dir}")
            
            baseline_models.add(embedding_model)
            if top_k is not None:
                baseline_top_k.add(top_k)
            
            # List files to debug
            all_files = list(config_dir.glob('*.json'))
            print(f"   Found {len(all_files)} JSON files")
            
            file_count = 0
            for filename in sorted(os.listdir(config_dir)):
                if filename.endswith('_eval.json') and filename.startswith('baseline_doc'):
                    doc_id = parse_doc_filename(filename)
                    if doc_id is not None:
                        filepath = config_dir / filename
                        data = load_eval_data(filepath)
                        if data is not None:
                            # Store with top_k as key (None for full_table)
                            baseline_data[embedding_model][top_k][doc_id] = data
                            all_docs.add(doc_id)
                            file_count += 1
                            
                            if file_count <= 3:
                                acc = data.get('accuracy', 0)
                                matches = data.get('total_matches', 0)
                                total = data.get('total_questions', 0)
                                print(f"   ✓ doc={doc_id}, acc={acc:.3f} ({matches}/{total})")
            
            if file_count > 3:
                print(f"   ... and {file_count - 3} more files")
            
            if file_count > 0:
                print(f"   ✅ Loaded {file_count} files")
            else:
                print(f"   ❌ No evaluation files found!")
                if all_files:
                    print(f"   Sample files: {[f.name for f in all_files[:3]]}")
    
    baseline_models_sorted = sorted(baseline_models)
    baseline_top_k_sorted = sorted([k for k in baseline_top_k if k is not None])
    has_full_table = None in [top_k for configs in baseline_data.values() for top_k in configs.keys()]
    
    if baseline_models:
        print(f"\n📈 Baseline models found: {baseline_models_sorted}")
        if baseline_top_k_sorted:
            print(f"📈 Baseline top-k values: {baseline_top_k_sorted}")
        if has_full_table:
            print(f"📈 Full table baseline: Yes")
        print(f"📈 Total baseline configurations: {sum(len(configs) for configs in baseline_data.values())}")
    else:
        print(f"\n⚠️  WARNING: No RAG baseline results loaded!")
    
    print_separator('-')
    
    # ========================================================================
    # VALIDATE DATA BEFORE PROCEEDING
    # ========================================================================
    
    if not method_data and not baseline_data:
        print()
        print("="*80)
        print("❌ ERROR: No data loaded!")
        print("="*80)
        print("\nPlease check:")
        print("1. AttentiveTrim method directories exist:")
        for model_name in args.method_models:
            method_dir = Path(model_name) / 'evaluation' / 'aitqa'
            print(f"   - {method_dir.absolute()}")
        print()
        print("2. RAG baseline evaluation directory exists:")
        print(f"   - {baseline_eval_dir.absolute()}")
        print()
        return
    
    # ========================================================================
    # CREATE COMPREHENSIVE COMPARISON TABLE
    # ========================================================================
    docs_sorted = sorted(all_docs)
    
    if not docs_sorted:
        print()
        print("="*80)
        print("❌ ERROR: No documents found in any results!")
        print("="*80)
        return
    
    print()
    print_separator()
    print("📋 CREATING COMPREHENSIVE COMPARISON TABLE")
    print_separator()
    print(f"Total documents: {len(docs_sorted)}")
    print(f"AttentiveTrim models: {len(args.method_models)}")
    print(f"RAG baseline models: {len(baseline_models)}")
    print()
    
    # Build column list: method models first, then baselines
    columns = []
    column_metadata = {}  # Track what each column represents
    
    # Add AttentiveTrim method columns
    for model_name in sorted(method_data.keys()):
        for method_type in sorted(method_data[model_name].keys()):
            for budget in method_budgets_sorted:
                if budget in method_data[model_name][method_type]:
                    col_name = f'{model_name}_{method_type}_top{budget}'
                    columns.append(col_name)
                    column_metadata[col_name] = {
                        'type': 'method', 
                        'model': model_name, 
                        'method_type': method_type,
                        'budget': budget
                    }
    
    # Add RAG baseline columns
    for embedding_model in baseline_models_sorted:
        for top_k in sorted([k for k in baseline_data[embedding_model].keys()]):
            if top_k is None:
                # Full table baseline
                col_name = f'aitqa_full_table'
                columns.append(col_name)
                column_metadata[col_name] = {
                    'type': 'baseline', 
                    'model': embedding_model, 
                    'top_k': None, 
                    'is_full_table': True
                }
            else:
                col_name = f'RAG_{embedding_model}_top{top_k}'
                columns.append(col_name)
                column_metadata[col_name] = {
                    'type': 'baseline', 
                    'model': embedding_model, 
                    'top_k': top_k,
                    'is_full_table': False
                }
    
    print(f"Total columns: {len(columns)}")
    print(f"  - AttentiveTrim: {sum(1 for c in columns if column_metadata[c]['type'] == 'method')}")
    print(f"  - RAG Baseline: {sum(1 for c in columns if column_metadata[c]['type'] == 'baseline')}")
    
    if columns:
        print("\nColumn breakdown:")
        print("  AttentiveTrim columns:")
        for col in columns:
            if column_metadata[col]['type'] == 'method':
                print(f"    - {col}")
        print("  RAG Baseline columns:")
        for col in columns:
            if column_metadata[col]['type'] == 'baseline':
                print(f"    - {col}")
    print()
    
    # Create DataFrame
    df = pd.DataFrame(index=[f'doc_{d}' for d in docs_sorted], columns=columns)
    
    # Fill in data
    print("Filling table with accuracy data...")
    
    # Fill AttentiveTrim method data
    method_entries = 0
    for doc_id in docs_sorted:
        row_name = f'doc_{doc_id}'
        
        for model_name in method_data.keys():
            for method_type in method_data[model_name].keys():
                for budget in method_budgets_sorted:
                    if budget in method_data[model_name][method_type]:
                        col_name = f'{model_name}_{method_type}_top{budget}'
                        if doc_id in method_data[model_name][method_type][budget]:
                            df.loc[row_name, col_name] = method_data[model_name][method_type][budget][doc_id].get('accuracy', None)
                            method_entries += 1
    
    print(f"  ✓ Filled {method_entries} AttentiveTrim method entries")
    
    # Fill RAG baseline data
    baseline_entries = 0
    for doc_id in docs_sorted:
        row_name = f'doc_{doc_id}'
        
        for embedding_model in baseline_data.keys():
            for top_k in baseline_data[embedding_model].keys():
                if top_k is None:
                    col_name = f'aitqa_full_table'
                else:
                    col_name = f'RAG_{embedding_model}_top{top_k}'
                
                if doc_id in baseline_data[embedding_model][top_k]:
                    df.loc[row_name, col_name] = baseline_data[embedding_model][top_k][doc_id].get('accuracy', None)
                    baseline_entries += 1
    
    print(f"  ✓ Filled {baseline_entries} RAG baseline entries")
    print(f"  Total entries: {method_entries + baseline_entries}")
    
    # ========================================================================
    # CALCULATE ROW USAGE STATISTICS
    # ========================================================================
    print()
    print_separator('-')
    print("📊 CALCULATING ROW USAGE STATISTICS")
    print_separator('-')
    
    # Calculate row usage with or without dataset
    row_usage_data = defaultdict(dict)
    avg_row_usage = {}
    
    if not table_row_counts:
        print("⚠️  No row counts from dataset - showing budget (rows used) instead of percentage")
        print(f"   table_row_counts is empty (size: {len(table_row_counts)})")
        
        # Fallback: just show the budget as "rows used"
        for col in columns:
            meta = column_metadata[col]
            
            if meta['type'] == 'method':
                budget = meta['budget']
                # Store budget as the "row usage" (not a percentage, just number of rows)
                avg_row_usage[col] = budget  # Just the number, not a percentage
                
            else:  # baseline
                is_full_table = meta.get('is_full_table', False)
                if is_full_table:
                    avg_row_usage[col] = float('inf')  # Represents "all rows"
                else:
                    top_k = meta.get('top_k')
                    avg_row_usage[col] = top_k if top_k else None
        
        configs_with_usage = sum(1 for v in avg_row_usage.values() if v is not None and v != float('inf'))
        print(f"✓ Showing rows used for {configs_with_usage} configurations (not percentages)")
        
    else:
        print(f"✓ Using row counts for {len(table_row_counts)} documents")
        if table_row_counts:
            sample_doc_id = list(table_row_counts.keys())[0]
            sample_rows = table_row_counts[sample_doc_id]
            print(f"  Sample: doc_{sample_doc_id} has {sample_rows} rows")
        
        # For each configuration, calculate row usage percentage per document
        for col in columns:
            meta = column_metadata[col]
            
            if meta['type'] == 'method':
                budget = meta['budget']
                
                for doc_id in docs_sorted:
                    if doc_id in table_row_counts:
                        total_rows = table_row_counts[doc_id]
                        if total_rows > 0:
                            # Cap at total_rows (can't use more rows than exist)
                            rows_used = min(budget, total_rows)
                            usage_pct = (rows_used / total_rows) * 100
                            row_usage_data[col][doc_id] = usage_pct
                        else:
                            row_usage_data[col][doc_id] = None
                    else:
                        row_usage_data[col][doc_id] = None
            
            else:  # baseline
                top_k = meta['top_k']
                is_full_table = meta.get('is_full_table', False)
                
                for doc_id in docs_sorted:
                    if is_full_table:
                        # Full table means 100% usage
                        row_usage_data[col][doc_id] = 100.0
                    elif top_k and doc_id in table_row_counts:
                        total_rows = table_row_counts[doc_id]
                        if total_rows > 0:
                            # Cap at total_rows (can't use more rows than exist)
                            rows_used = min(top_k, total_rows)
                            usage_pct = (rows_used / total_rows) * 100
                            row_usage_data[col][doc_id] = usage_pct
                        else:
                            row_usage_data[col][doc_id] = None
                    else:
                        row_usage_data[col][doc_id] = None
        
        # Calculate average row usage per configuration
        for col in columns:
            usages = [u for u in row_usage_data[col].values() if u is not None]
            if usages:
                avg_row_usage[col] = sum(usages) / len(usages)
            else:
                avg_row_usage[col] = None
        
        # Count how many configs have row usage data
        configs_with_usage = sum(1 for v in avg_row_usage.values() if v is not None)
        print(f"✓ Calculated row usage percentages for {configs_with_usage}/{len(columns)} configurations")
    
    # ========================================================================
    # CALCULATE WEIGHTED AVERAGES (by question count)
    # ========================================================================
    print()
    print("Calculating weighted averages (by question count)...")
    print("  Formula: accuracy = total_matches / total_questions across ALL documents")
    avg_row = {}
    column_stats = {}  # Track stats for each column
    
    for col in columns:
        meta = column_metadata[col]
        total_questions = 0
        total_matches = 0
        num_docs = 0
        
        if meta['type'] == 'method':
            model_name = meta['model']
            method_type = meta['method_type']
            budget = meta['budget']
            for doc_id in docs_sorted:
                if doc_id in method_data[model_name][method_type][budget]:
                    data = method_data[model_name][method_type][budget][doc_id]
                    total_questions += data.get('total_questions', 0)
                    total_matches += data.get('total_matches', 0)
                    num_docs += 1
        else:  # baseline
            embedding_model = meta['model']
            top_k = meta['top_k']
            for doc_id in docs_sorted:
                if doc_id in baseline_data[embedding_model][top_k]:
                    data = baseline_data[embedding_model][top_k][doc_id]
                    total_questions += data.get('total_questions', 0)
                    total_matches += data.get('total_matches', 0)
                    num_docs += 1
        
        if total_questions > 0:
            # Weighted average: total_matches / total_questions
            avg_row[col] = total_matches / total_questions
            column_stats[col] = {
                'total_questions': total_questions,
                'total_matches': total_matches,
                'num_documents': num_docs,
                'weighted_accuracy': total_matches / total_questions,
                'avg_row_usage': avg_row_usage.get(col)
            }
        else:
            avg_row[col] = None
            column_stats[col] = {
                'total_questions': 0,
                'total_matches': 0,
                'num_documents': 0,
                'weighted_accuracy': None,
                'avg_row_usage': avg_row_usage.get(col)
            }
    
    print(f"  ✓ Calculated weighted averages for {len(columns)} columns")
    
    # Add average row (note: this is weighted by question count)
    df.loc['average'] = avg_row
    
    # Save to CSV
    csv_path = output_dir / 'comprehensive_accuracy_comparison.csv'
    df.to_csv(csv_path)
    
    # Also save a version with calculation notes
    notes_path = output_dir / 'README_accuracy_calculation.txt'
    with open(notes_path, 'w') as f:
        f.write("AITQA Comprehensive Accuracy Analysis (Enhanced)\n")
        f.write("=" * 80 + "\n\n")
        f.write("ACCURACY CALCULATION METHOD:\n\n")
        f.write("The 'average' row in comprehensive_accuracy_comparison.csv shows\n")
        f.write("WEIGHTED ACCURACY calculated as:\n\n")
        f.write("  Weighted Accuracy = (Total Matches) / (Total Questions)\n\n")
        f.write("Where:\n")
        f.write("  - Total Matches = Sum of all matches (score ≥ 7) across all documents\n")
        f.write("  - Total Questions = Sum of all questions across all documents\n\n")
        f.write("This weights each document by its number of questions, giving equal\n")
        f.write("importance to each question rather than each document.\n\n")
        f.write("Per-document accuracies are shown in rows doc_0, doc_1, etc.\n\n")
        f.write("Column naming:\n")
        f.write("  - AttentiveTrim methods: {model_name}_{method_type}_top{budget}\n")
        f.write("  - RAG baselines: RAG_{embedding_model}_top{k}\n")
        f.write("  - RAG full table: aitqa_full_table\n\n")
        f.write("ROW USAGE CALCULATION:\n\n")
        f.write("Average row usage shows what percentage of table rows were used:\n\n")
        f.write("  Row Usage % = (min(Rows Used, Total Rows) / Total Rows) × 100\n\n")
        f.write("For each document, this is averaged across all documents.\n")
        f.write("  - Full table baseline: 100% (uses all rows)\n")
        f.write("  - top-k methods: (min(k, total_rows) / total_rows) × 100\n")
        f.write("  - Capped at 100% when k > total_rows (can't use more than exist)\n\n")
        f.write("Example:\n")
        f.write("  - Table with 3 rows, top-4 selected: min(4,3)/3 = 100% (not 133%)\n")
        f.write("  - Table with 8 rows, top-4 selected: min(4,8)/8 = 50%\n\n")
        if dataset_file:
            f.write(f"Dataset file used: {dataset_file}\n")
        else:
            f.write("Dataset file: Auto-detection failed - row usage not calculated\n")
    
    print(f"✅ Table saved: {csv_path}")
    print(f"✅ Calculation notes saved: {notes_path}")
    print()
    
    # Display sample of the table
    print_separator()
    print("📊 COMPREHENSIVE TABLE (First 5 Documents)")
    print_separator()
    print(df.head(5).to_string())
    print()
    print("...")
    print()
    print("Average row (weighted by question count):")
    print(df.loc[['average']].to_string())
    print_separator()
    print("Note: The 'average' row shows weighted accuracy = total_matches / total_questions")
    print("      This accounts for documents having different numbers of questions.")
    print_separator()
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    print()
    print_separator()
    print("📈 SUMMARY STATISTICS (Weighted by Question Count)")
    print_separator()
    
    print("\nAttentiveTrim Method Models:")
    for col in columns:
        if column_metadata[col]['type'] == 'method':
            stats = column_stats[col]
            if stats['weighted_accuracy'] is not None:
                acc = stats['weighted_accuracy']
                row_usage = stats['avg_row_usage']
                if row_usage and not table_row_counts:
                    # Just showing budget, not percentage
                    row_usage_str = f", uses {int(row_usage)} rows"
                elif row_usage:
                    # Showing percentage
                    row_usage_str = f", {row_usage:.1f}% rows"
                else:
                    row_usage_str = ""
                print(f"  {col:40s}: {acc:.4f} ({acc*100:.2f}%) "
                      f"[{stats['total_matches']}/{stats['total_questions']} questions, "
                      f"{stats['num_documents']} docs{row_usage_str}]")
    
    print("\nRAG Baseline Models:")
    for col in columns:
        if column_metadata[col]['type'] == 'baseline':
            stats = column_stats[col]
            if stats['weighted_accuracy'] is not None:
                acc = stats['weighted_accuracy']
                row_usage = stats['avg_row_usage']
                if row_usage == float('inf'):
                    row_usage_str = ", uses all rows"
                elif row_usage and not table_row_counts:
                    row_usage_str = f", uses {int(row_usage)} rows"
                elif row_usage:
                    row_usage_str = f", {row_usage:.1f}% rows"
                else:
                    row_usage_str = ""
                print(f"  {col:40s}: {acc:.4f} ({acc*100:.2f}%) "
                      f"[{stats['total_matches']}/{stats['total_questions']} questions, "
                      f"{stats['num_documents']} docs{row_usage_str}]")
    
    print_separator()
    
    # ========================================================================
    # ROW USAGE ANALYSIS
    # ========================================================================
    if configs_with_usage > 0:
        print()
        print_separator()
        print("📏 ROW USAGE ANALYSIS")
        print_separator()
        print()
        print("Average Row Usage (% of table rows used):")
        print()
        
        # Sort by row usage
        sorted_by_usage = sorted(
            [(col, avg_row_usage[col]) for col in columns if avg_row_usage[col] is not None],
            key=lambda x: x[1]
        )
        
        print("AttentiveTrim Methods:")
        for col, usage in sorted_by_usage:
            if column_metadata[col]['type'] == 'method':
                meta = column_metadata[col]
                acc = column_stats[col]['weighted_accuracy']
                acc_str = f"acc={acc:.4f}" if acc else "acc=N/A"
                print(f"  {col:40s}: {usage:6.1f}% rows  ({acc_str})")
        
        print("\nRAG Baselines:")
        for col, usage in sorted_by_usage:
            if column_metadata[col]['type'] == 'baseline':
                meta = column_metadata[col]
                acc = column_stats[col]['weighted_accuracy']
                acc_str = f"acc={acc:.4f}" if acc else "acc=N/A"
                print(f"  {col:40s}: {usage:6.1f}% rows  ({acc_str})")
        
        print_separator()
        
        # Efficiency analysis (accuracy per % row usage)
        print()
        print("📊 EFFICIENCY ANALYSIS (Accuracy per % Row Usage)")
        print_separator()
        print()
        
        efficiency_data = []
        for col in columns:
            acc = column_stats[col]['weighted_accuracy']
            usage = avg_row_usage[col]
            if acc is not None and usage is not None and usage > 0:
                efficiency = acc / usage
                efficiency_data.append((col, efficiency, acc, usage))
        
        efficiency_data.sort(key=lambda x: x[1], reverse=True)
        
        print("Top configurations by efficiency (higher is better):")
        for i, (col, eff, acc, usage) in enumerate(efficiency_data[:10], 1):
            col_type = "Method" if column_metadata[col]['type'] == 'method' else "RAG"
            print(f"  {i:2d}. [{col_type:6s}] {col:40s}: "
                  f"eff={eff:.6f} (acc={acc:.4f}, {usage:.1f}% rows)")
        
        print_separator()
    
    # ========================================================================
    # TOP PERFORMERS
    # ========================================================================
    print()
    print_separator()
    print("🏆 TOP PERFORMERS (Ranked by Weighted Accuracy)")
    print_separator()
    
    # Sort by average accuracy
    sorted_cols = sorted(
        [(col, avg_row[col]) for col in columns if avg_row[col] is not None],
        key=lambda x: x[1],
        reverse=True
    )
    
    print("\nTop 10 Configurations by Weighted Accuracy:")
    for i, (col, acc) in enumerate(sorted_cols[:10], 1):
        col_type = "Method" if column_metadata[col]['type'] == 'method' else "RAG"
        stats = column_stats[col]
        row_usage = stats['avg_row_usage']
        if row_usage == float('inf'):
            row_usage_str = ", uses all rows"
        elif row_usage and not table_row_counts:
            row_usage_str = f", uses {int(row_usage)} rows"
        elif row_usage:
            row_usage_str = f", {row_usage:.1f}% rows"
        else:
            row_usage_str = ""
        print(f"  {i:2d}. [{col_type:6s}] {col:40s}: {acc:.4f} ({acc*100:.2f}%) "
              f"[{stats['total_matches']}/{stats['total_questions']} questions{row_usage_str}]")
    
    if len(sorted_cols) > 10:
        print(f"\n... ({len(sorted_cols) - 10} more configurations)")
        print(f"\nLowest Configuration:")
        col, acc = sorted_cols[-1]
        col_type = "Method" if column_metadata[col]['type'] == 'method' else "RAG"
        stats = column_stats[col]
        row_usage = stats['avg_row_usage']
        if row_usage == float('inf'):
            row_usage_str = ", uses all rows"
        elif row_usage and not table_row_counts:
            row_usage_str = f", uses {int(row_usage)} rows"
        elif row_usage:
            row_usage_str = f", {row_usage:.1f}% rows"
        else:
            row_usage_str = ""
        print(f"  [{col_type:6s}] {col:40s}: {acc:.4f} ({acc*100:.2f}%) "
              f"[{stats['total_matches']}/{stats['total_questions']} questions{row_usage_str}]")
    
    print_separator()
    
    # ========================================================================
    # SAVE METADATA
    # ========================================================================
    metadata = {
        "total_documents": len(docs_sorted),
        "total_configurations": len(columns),
        "method_models": sorted(method_data.keys()),
        "method_budgets": method_budgets_sorted,
        "baseline_models": baseline_models_sorted,
        "baseline_top_k": baseline_top_k_sorted,
        "has_full_table_baseline": has_full_table,
        "column_info": column_metadata,
        "column_statistics": column_stats,
        "average_row_usage": avg_row_usage,
        "dataset_file": str(dataset_file) if dataset_file else "Auto-detection failed",
        "note": "Accuracy is weighted average: total_matches / total_questions across all documents",
        "row_usage_note": "Row usage shows average % of table rows used per configuration",
        "top_performers": [
            {
                "rank": i, 
                "config": col, 
                "accuracy": acc, 
                "type": column_metadata[col]['type'],
                "total_questions": column_stats[col]['total_questions'],
                "total_matches": column_stats[col]['total_matches'],
                "num_documents": column_stats[col]['num_documents'],
                "avg_row_usage": column_stats[col]['avg_row_usage']
            }
            for i, (col, acc) in enumerate(sorted_cols[:10], 1)
        ]
    }
    
    metadata_path = output_dir / 'comprehensive_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # ========================================================================
    # SAVE COMPREHENSIVE REPORT
    # ========================================================================
    report_path = output_dir / 'comprehensive_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE AITQA ANALYSIS - METHODS + BASELINES (ENHANCED)\n")
        f.write("="*80 + "\n\n")
        
        # Configuration
        f.write("CONFIGURATION:\n")
        f.write(f"  Method models: {', '.join(args.method_models)}\n")
        f.write(f"  Method types: {', '.join(args.method_types)}\n")
        f.write(f"  Baseline eval dir: {args.baseline_eval_dir}\n")
        f.write(f"  Output directory: {output_dir}\n")
        if dataset_file:
            f.write(f"  Dataset file: {dataset_file}\n")
        else:
            f.write(f"  Dataset file: Auto-detection failed\n")
        f.write("\n")
        
        # Summary
        f.write("="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Total documents: {len(docs_sorted)}\n")
        f.write(f"Total configurations: {len(columns)}\n")
        f.write(f"  - AttentiveTrim: {sum(1 for c in columns if column_metadata[c]['type'] == 'method')}\n")
        f.write(f"  - RAG Baseline: {sum(1 for c in columns if column_metadata[c]['type'] == 'baseline')}\n")
        if configs_with_usage > 0:
            f.write(f"Row usage calculated for: {configs_with_usage} configurations\n")
        f.write("\n")
        
        # Comprehensive table sample
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE TABLE (First 5 Documents)\n")
        f.write("="*80 + "\n")
        f.write(df.head(5).to_string() + "\n\n")
        f.write("...\n\n")
        f.write("Average row (weighted by question count):\n")
        f.write(df.loc[['average']].to_string() + "\n")
        f.write("="*80 + "\n")
        f.write("Note: The 'average' row shows weighted accuracy = total_matches / total_questions\n")
        f.write("      This accounts for documents having different numbers of questions.\n")
        f.write("="*80 + "\n\n")
        
        # Summary statistics
        f.write("="*80 + "\n")
        f.write("SUMMARY STATISTICS (Weighted by Question Count)\n")
        f.write("="*80 + "\n\n")
        
        f.write("AttentiveTrim Method Models:\n")
        for col in columns:
            if column_metadata[col]['type'] == 'method':
                stats = column_stats[col]
                if stats['weighted_accuracy'] is not None:
                    acc = stats['weighted_accuracy']
                    row_usage = stats['avg_row_usage']
                    row_usage_str = f", {row_usage:.1f}% rows" if row_usage else ""
                    f.write(f"  {col:40s}: {acc:.4f} ({acc*100:.2f}%) "
                          f"[{stats['total_matches']}/{stats['total_questions']} questions, "
                          f"{stats['num_documents']} docs{row_usage_str}]\n")
        
        f.write("\nRAG Baseline Models:\n")
        for col in columns:
            if column_metadata[col]['type'] == 'baseline':
                stats = column_stats[col]
                if stats['weighted_accuracy'] is not None:
                    acc = stats['weighted_accuracy']
                    row_usage = stats['avg_row_usage']
                    row_usage_str = f", {row_usage:.1f}% rows" if row_usage else ""
                    f.write(f"  {col:40s}: {acc:.4f} ({acc*100:.2f}%) "
                          f"[{stats['total_matches']}/{stats['total_questions']} questions, "
                          f"{stats['num_documents']} docs{row_usage_str}]\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Row usage analysis
        if configs_with_usage > 0:
            f.write("="*80 + "\n")
            f.write("ROW USAGE ANALYSIS\n")
            f.write("="*80 + "\n\n")
            f.write("Average Row Usage (% of table rows used):\n\n")
            
            sorted_by_usage = sorted(
                [(col, avg_row_usage[col]) for col in columns if avg_row_usage[col] is not None],
                key=lambda x: x[1]
            )
            
            f.write("AttentiveTrim Methods:\n")
            for col, usage in sorted_by_usage:
                if column_metadata[col]['type'] == 'method':
                    acc = column_stats[col]['weighted_accuracy']
                    acc_str = f"acc={acc:.4f}" if acc else "acc=N/A"
                    f.write(f"  {col:40s}: {usage:6.1f}% rows  ({acc_str})\n")
            
            f.write("\nRAG Baselines:\n")
            for col, usage in sorted_by_usage:
                if column_metadata[col]['type'] == 'baseline':
                    acc = column_stats[col]['weighted_accuracy']
                    acc_str = f"acc={acc:.4f}" if acc else "acc=N/A"
                    f.write(f"  {col:40s}: {usage:6.1f}% rows  ({acc_str})\n")
            
            f.write("\n" + "="*80 + "\n\n")
            
            # Efficiency analysis
            f.write("="*80 + "\n")
            f.write("EFFICIENCY ANALYSIS (Accuracy per % Row Usage)\n")
            f.write("="*80 + "\n\n")
            
            efficiency_data = []
            for col in columns:
                acc = column_stats[col]['weighted_accuracy']
                usage = avg_row_usage[col]
                if acc is not None and usage is not None and usage > 0:
                    efficiency = acc / usage
                    efficiency_data.append((col, efficiency, acc, usage))
            
            efficiency_data.sort(key=lambda x: x[1], reverse=True)
            
            f.write("Top configurations by efficiency (higher is better):\n")
            for i, (col, eff, acc, usage) in enumerate(efficiency_data[:10], 1):
                col_type = "Method" if column_metadata[col]['type'] == 'method' else "RAG"
                f.write(f"  {i:2d}. [{col_type:6s}] {col:40s}: "
                      f"eff={eff:.6f} (acc={acc:.4f}, {usage:.1f}% rows)\n")
            
            f.write("\n" + "="*80 + "\n\n")
        
        # Top performers
        f.write("="*80 + "\n")
        f.write("TOP PERFORMERS (Ranked by Weighted Accuracy)\n")
        f.write("="*80 + "\n\n")
        
        f.write("Top 10 Configurations by Weighted Accuracy:\n")
        for i, (col, acc) in enumerate(sorted_cols[:10], 1):
            col_type = "Method" if column_metadata[col]['type'] == 'method' else "RAG"
            stats = column_stats[col]
            row_usage = stats['avg_row_usage']
            row_usage_str = f", {row_usage:.1f}% rows" if row_usage else ""
            f.write(f"  {i:2d}. [{col_type:6s}] {col:40s}: {acc:.4f} ({acc*100:.2f}%) "
                  f"[{stats['total_matches']}/{stats['total_questions']} questions{row_usage_str}]\n")
        
        if len(sorted_cols) > 10:
            f.write(f"\n... ({len(sorted_cols) - 10} more configurations)\n\n")
            f.write("Lowest Configuration:\n")
            col, acc = sorted_cols[-1]
            col_type = "Method" if column_metadata[col]['type'] == 'method' else "RAG"
            stats = column_stats[col]
            row_usage = stats['avg_row_usage']
            row_usage_str = f", {row_usage:.1f}% rows" if row_usage else ""
            f.write(f"  [{col_type:6s}] {col:40s}: {acc:.4f} ({acc*100:.2f}%) "
                  f"[{stats['total_matches']}/{stats['total_questions']} questions{row_usage_str}]\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Output files
        f.write("="*80 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("="*80 + "\n")
        f.write(f"  Accuracy table: {csv_path.name}\n")
        f.write(f"  Calculation notes: {notes_path.name}\n")
        f.write(f"  Metadata: {metadata_path.name}\n")
        f.write(f"  This report: {report_path.name}\n")
        f.write("="*80 + "\n")
    
    print(f"📁 Report: {report_path}")
    
    print()
    print("="*80)
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE (ENHANCED)")
    print("="*80)
    print(f"📁 Accuracy table: {csv_path}")
    print(f"📁 Calculation notes: {notes_path}")
    print(f"📁 Metadata: {metadata_path}")
    print(f"📁 Comprehensive report: {report_path}")
    print(f"📊 Total configurations: {len(columns)}")
    print(f"   - AttentiveTrim: {sum(1 for c in columns if column_metadata[c]['type'] == 'method')}")
    print(f"   - RAG Baseline: {sum(1 for c in columns if column_metadata[c]['type'] == 'baseline')}")
    print(f"📄 Documents: {len(docs_sorted)}")
    if configs_with_usage > 0:
        print(f"📏 Row usage calculated for: {configs_with_usage} configurations")
    print()
    print("NOTE: The 'average' row shows weighted accuracy (total_matches / total_questions)")
    print("      This accounts for documents having different numbers of questions.")
    if configs_with_usage > 0:
        print("      Row usage shows average % of table rows used per configuration.")
    print("      Full analysis saved to: " + str(report_path))
    print("="*80)

if __name__ == '__main__':
    main()