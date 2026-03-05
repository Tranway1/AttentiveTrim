import json
import os
import pandas as pd
import re
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Model name mapping (must match data_reader.py and unit_window.py)
MODEL_MAPPING = {
    'llama3.2_1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'qwen3_8b': 'Qwen/Qwen3-8B',
    'qwen3_14b': 'Qwen/Qwen3-14B'
}

# RAG baseline directory (not model-specific)
RAG_BASELINE_DIR = "baselines/evaluation/quality"

AVAILABLE_EMBEDDING_MODELS = [
    'UAE-Large-V1',
    'bflhc/Octen-Embedding-4B',
    'Qwen3-Embedding-8B'
]

def load_ground_truth(filepath):
    """Load ground truth data"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def compute_avg_token_usage(file_data: Dict) -> Optional[float]:
    """
    Compute average token usage percentage for a single evaluation file.
    
    Formula: avg_token_pct = (sum(tokens_extracted / total_tokens for each doc) / num_docs) * 100
    
    Returns:
        Average token usage percentage, or None if cannot compute
    """
    documents = file_data.get('files', [])
    if not isinstance(documents, list) or not documents:
        return None
    
    token_ratios: List[float] = []
    
    for doc in documents:
        tokens_used_raw = doc.get('tokens_extracted', doc.get('tokens_used'))
        total_tokens_raw = doc.get('total_tokens')
        
        if tokens_used_raw is not None and total_tokens_raw not in (None, 0):
            try:
                tokens_used = float(tokens_used_raw)
                total_tokens = float(total_tokens_raw)
                if total_tokens > 0:
                    token_ratios.append(tokens_used / total_tokens)
            except Exception:
                pass
    
    if token_ratios:
        avg_ratio = sum(token_ratios) / len(token_ratios)
        return avg_ratio * 100  # Convert to percentage
    else:
        return None

def compute_avg_token_usage_rag(file_data: Dict) -> Optional[float]:
    """
    Compute average token usage percentage for RAG baseline.
    
    Returns:
        Average token usage percentage, or None if cannot compute
    """
    questions = file_data.get('questions', [])
    if not isinstance(questions, list) or not questions:
        return None
    
    token_ratios: List[float] = []
    
    for q in questions:
        tokens_used_raw = q.get('tokens_extracted', q.get('tokens_used'))
        total_tokens_raw = q.get('total_tokens')
        
        if tokens_used_raw is not None and total_tokens_raw not in (None, 0):
            try:
                tokens_used = float(tokens_used_raw)
                total_tokens = float(total_tokens_raw)
                if total_tokens > 0:
                    token_ratios.append(tokens_used / total_tokens)
            except Exception:
                pass
    
    if token_ratios:
        avg_ratio = sum(token_ratios) / len(token_ratios)
        return avg_ratio * 100  # Convert to percentage
    else:
        return None

def parse_our_method_filename(filename):
    """
    Parse filename like: farest_budget_0.005_about_how_big_is_the_cleopatra_ship_options_a_quite_large_enough_for_at_least_a_gpt4.json
    Returns: budget value
    """
    match = re.match(r'farest_budget_([\d.]+)_.*\.json', filename)
    if match:
        return float(match.group(1))
    return None

def parse_rag_filename(filename):
    """
    Parse filename like: rag-bflhc-Octen-Embedding-4B-top1_doc_1_gpt4.json
    Returns: (model, top_k, doc_id)
    """
    # Pattern: rag-{model}-top{k}_doc_{num}_gpt4.json
    match = re.match(r'rag-(.+)-top(\d+)_doc_(\d+)_gpt4\.json', filename)
    if match:
        model = match.group(1)
        top_k = int(match.group(2))
        doc_id = int(match.group(3))
        return model, top_k, doc_id
    return None, None, None

def calculate_accuracy_our_method(result_data):
    """Calculate accuracy from our method results"""
    if 'files' not in result_data:
        return {}
    
    doc_accuracies = {}
    for file_entry in result_data['files']:
        doc_id = file_entry['file']
        match = file_entry.get('match', False)
        
        if doc_id not in doc_accuracies:
            doc_accuracies[doc_id] = {'correct': 0, 'total': 0}
        
        doc_accuracies[doc_id]['total'] += 1
        if match:
            doc_accuracies[doc_id]['correct'] += 1
    
    # Convert to accuracy
    for doc_id in doc_accuracies:
        total = doc_accuracies[doc_id]['total']
        correct = doc_accuracies[doc_id]['correct']
        doc_accuracies[doc_id] = correct / total if total > 0 else 0.0
    
    return doc_accuracies

def calculate_accuracy_rag(result_data):
    """Calculate accuracy from RAG baseline results"""
    if 'questions' not in result_data:
        return 0.0
    
    total = len(result_data['questions'])
    correct = sum(1 for q in result_data['questions'] if q.get('match', False))
    
    return correct / total if total > 0 else 0.0

def analyze_our_method(directory, model_name):
    """Analyze all results from our method for a specific model"""
    results = defaultdict(lambda: defaultdict(dict))  # budget -> doc_id -> accuracy
    token_usage = defaultdict(list)  # budget -> [token_usage_percentages]
    question_counts = defaultdict(lambda: defaultdict(int))  # budget -> doc_id -> num_questions
    
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return results, token_usage, question_counts
    
    for filename in os.listdir(directory):
        if not filename.endswith('.json'):
            continue
        
        budget = parse_our_method_filename(filename)
        if budget is None:
            continue
        
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        doc_accuracies = calculate_accuracy_our_method(data)
        
        # Compute token usage for this file
        token_pct = compute_avg_token_usage(data)
        if token_pct is not None:
            token_usage[budget].append(token_pct)
        
        for doc_id, accuracy in doc_accuracies.items():
            # Extract numeric part from doc_id (e.g., "doc_133" -> 133)
            if isinstance(doc_id, str) and doc_id.startswith('doc_'):
                doc_num = int(doc_id.split('_')[1])
            else:
                doc_num = doc_id
            
            if doc_num not in results[budget]:
                results[budget][doc_num] = []
            results[budget][doc_num].append(accuracy)
            question_counts[budget][doc_num] += 1
    
    # Average multiple questions for same doc/budget
    averaged_results = defaultdict(dict)
    for budget in results:
        for doc_num in results[budget]:
            averaged_results[budget][doc_num] = sum(results[budget][doc_num]) / len(results[budget][doc_num])
    
    return averaged_results, token_usage, question_counts

def analyze_rag_baselines(directory):
    """Analyze all RAG baseline results"""
    results = defaultdict(lambda: defaultdict(dict))  # model -> top_k -> doc_id -> accuracy
    token_usage = defaultdict(lambda: defaultdict(list))  # model -> top_k -> [token_usage_percentages]
    question_counts = defaultdict(lambda: defaultdict(dict))  # model -> top_k -> doc_id -> num_questions
    
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return results, token_usage, question_counts
    
    for filename in os.listdir(directory):
        if not filename.endswith('.json'):
            continue
        
        model, top_k, doc_id = parse_rag_filename(filename)
        if model is None:
            continue
        
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        accuracy = calculate_accuracy_rag(data)
        results[model][top_k][doc_id] = accuracy
        
        # Track number of questions for this document
        num_questions = len(data.get('questions', []))
        question_counts[model][top_k][doc_id] = num_questions
        
        # Compute token usage
        token_pct = compute_avg_token_usage_rag(data)
        if token_pct is not None:
            token_usage[model][top_k].append(token_pct)
    
    return results, token_usage, question_counts

def create_comparison_table(all_model_results, all_model_question_counts, rag_results, rag_question_counts):
    """
    Create a comparison table with documents as rows and all model-budget combinations + RAG as columns.
    
    Args:
        all_model_results: dict mapping model_name -> {budget -> {doc_id -> accuracy}}
        all_model_question_counts: dict mapping model_name -> {budget -> {doc_id -> num_questions}}
        rag_results: dict mapping rag_model -> {top_k -> {doc_id -> accuracy}}
        rag_question_counts: dict mapping rag_model -> {top_k -> {doc_id -> num_questions}}
    """
    
    # Get all unique document IDs
    all_doc_ids = set()
    for model_name in all_model_results:
        for budget in all_model_results[model_name]:
            all_doc_ids.update(all_model_results[model_name][budget].keys())
    for rag_model in rag_results:
        for top_k in rag_results[rag_model]:
            all_doc_ids.update(rag_results[rag_model][top_k].keys())
    
    all_doc_ids = sorted(all_doc_ids)
    
    # Create column names
    columns = ['document_id']
    
    # Add AttentiveTrim columns for all models
    model_budget_configs = []
    for model_name in sorted(all_model_results.keys()):
        budgets = sorted(all_model_results[model_name].keys())
        for budget in budgets:
            columns.append(f'{model_name}_budget_{budget}')
            model_budget_configs.append((model_name, budget))
    
    # Add RAG columns
    rag_configs = []
    for rag_model in sorted(rag_results.keys()):
        for top_k in sorted(rag_results[rag_model].keys()):
            rag_model_short = rag_model.split('/')[-1] if '/' in rag_model else rag_model
            columns.append(f'RAG_{rag_model_short}_top{top_k}')
            rag_configs.append((rag_model, top_k))
    
    # Build data rows
    data_rows = []
    for doc_id in all_doc_ids:
        row = [f'doc_{doc_id}']
        
        # Add AttentiveTrim results for all models
        for model_name, budget in model_budget_configs:
            accuracy = all_model_results[model_name][budget].get(doc_id, None)
            row.append(f'{accuracy:.4f}' if accuracy is not None else 'N/A')
        
        # Add RAG results
        for rag_model, top_k in rag_configs:
            accuracy = rag_results[rag_model][top_k].get(doc_id, None)
            row.append(f'{accuracy:.4f}' if accuracy is not None else 'N/A')
        
        data_rows.append(row)
    
    # Calculate question-weighted averages
    avg_row = ['AVERAGE (Question-Weighted)']
    
    # Average for AttentiveTrim models
    for model_name, budget in model_budget_configs:
        total_correct = sum(
            all_model_results[model_name][budget][doc_id] * all_model_question_counts[model_name][budget][doc_id]
            for doc_id in all_doc_ids 
            if doc_id in all_model_results[model_name][budget]
        )
        total_questions = sum(
            all_model_question_counts[model_name][budget][doc_id]
            for doc_id in all_doc_ids 
            if doc_id in all_model_question_counts[model_name][budget]
        )
        avg = total_correct / total_questions if total_questions > 0 else 0.0
        avg_row.append(f'{avg:.4f}')
    
    # Average for RAG
    for rag_model, top_k in rag_configs:
        total_correct = sum(
            rag_results[rag_model][top_k][doc_id] * rag_question_counts[rag_model][top_k][doc_id]
            for doc_id in all_doc_ids 
            if doc_id in rag_results[rag_model][top_k]
        )
        total_questions = sum(
            rag_question_counts[rag_model][top_k][doc_id]
            for doc_id in all_doc_ids 
            if doc_id in rag_question_counts[rag_model][top_k]
        )
        avg = total_correct / total_questions if total_questions > 0 else 0.0
        avg_row.append(f'{avg:.4f}')
    
    data_rows.append(avg_row)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=columns)
    return df

def create_summary_report(all_model_results, all_model_token_usage, all_model_question_counts, 
                         rag_results, rag_token_usage, rag_question_counts):
    """Create a summary report with average accuracy and token usage for each model-budget/k combination"""
    
    summary_rows = []
    
    # Add AttentiveTrim results for all models
    for model_name in sorted(all_model_results.keys()):
        for budget in sorted(all_model_results[model_name].keys()):
            # Calculate weighted average accuracy (by number of questions)
            total_correct = 0
            total_questions = 0
            for doc_id in all_model_results[model_name][budget]:
                accuracy = all_model_results[model_name][budget][doc_id]
                num_questions = all_model_question_counts[model_name][budget][doc_id]
                total_correct += accuracy * num_questions
                total_questions += num_questions
            
            avg_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
            
            # Calculate average token usage
            token_usages = all_model_token_usage[model_name].get(budget, [])
            avg_token_usage = sum(token_usages) / len(token_usages) if token_usages else None
            
            summary_rows.append({
                'Method': f'AttentiveTrim ({model_name})',
                'Configuration': f'budget_{budget}',
                'Budget/Top-K': budget,
                'Average Accuracy': f'{avg_accuracy:.4f}',
                'Average Token Usage (%)': f'{avg_token_usage:.2f}' if avg_token_usage is not None else 'N/A',
                'Num Documents': len(all_model_results[model_name][budget]),
                'Num Questions': total_questions
            })
    
    # Add RAG baseline results
    for rag_model in sorted(rag_results.keys()):
        rag_model_short = rag_model.split('/')[-1] if '/' in rag_model else rag_model
        for top_k in sorted(rag_results[rag_model].keys()):
            # Calculate weighted average accuracy (by number of questions)
            total_correct = 0
            total_questions = 0
            for doc_id in rag_results[rag_model][top_k]:
                accuracy = rag_results[rag_model][top_k][doc_id]
                num_questions = rag_question_counts[rag_model][top_k][doc_id]
                total_correct += accuracy * num_questions
                total_questions += num_questions
            
            avg_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
            
            # Calculate average token usage
            token_usages = rag_token_usage.get(rag_model, {}).get(top_k, [])
            avg_token_usage = sum(token_usages) / len(token_usages) if token_usages else None
            
            summary_rows.append({
                'Method': f'RAG-{rag_model_short}',
                'Configuration': f'top{top_k}',
                'Budget/Top-K': top_k,
                'Average Accuracy': f'{avg_accuracy:.4f}',
                'Average Token Usage (%)': f'{avg_token_usage:.2f}' if avg_token_usage is not None else 'N/A',
                'Num Documents': len(rag_results[rag_model][top_k]),
                'Num Questions': total_questions
            })
    
    df = pd.DataFrame(summary_rows)
    return df

def main():
    parser = argparse.ArgumentParser(
        description='Analyze Quality dataset results across all models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all models (default)
  python analyze_quality.py
  
  # Analyze specific models only
  python analyze_quality.py --models llama3.2_1b qwen3_8b
  
  # Custom output directory
  python analyze_quality.py --output-dir my_analysis
        """
    )
    
    parser.add_argument('--models', nargs='+', 
                       choices=list(MODEL_MAPPING.keys()),
                       default=list(MODEL_MAPPING.keys()),
                       help='Which models to analyze (default: all models)')
    parser.add_argument('--output-dir', type=str, default='analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--dataset-dir', type=str, default='data/datasets',
                       help='Directory containing processed dataset')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    output_csv = os.path.join(output_dir, "quality_analysis_comparison.csv")
    summary_csv = os.path.join(output_dir, "quality_summary_report.csv")
    
    print("="*120)
    print("QUALITY DATASET ANALYSIS - MULTI-MODEL COMPARISON")
    print("="*120)
    print(f"Models to analyze: {', '.join(args.models)}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check if RAG baseline directory exists
    if not os.path.exists(RAG_BASELINE_DIR):
        print(f"ERROR: RAG baseline directory {RAG_BASELINE_DIR} not found!")
        return
    
    # Load ground truth
    ground_truth_file = os.path.join(args.dataset_dir, "quality_processed.json")
    if os.path.exists(ground_truth_file):
        ground_truth = load_ground_truth(ground_truth_file)
        print(f"✅ Loaded ground truth with {len(ground_truth['documents'])} documents")
    else:
        print(f"⚠️  Warning: Ground truth file {ground_truth_file} not found")
        ground_truth = None
    
    # Analyze AttentiveTrim results for all models
    all_model_results = {}
    all_model_token_usage = {}
    all_model_question_counts = {}
    
    for model_name in args.models:
        model_eval_dir = os.path.join(model_name, "evaluation", "quality")
        
        print(f"\n📊 Analyzing {model_name}...")
        print(f"   Directory: {model_eval_dir}")
        
        if not os.path.exists(model_eval_dir):
            print(f"   ⚠️  Directory not found, skipping...")
            continue
        
        results, token_usage, question_counts = analyze_our_method(model_eval_dir, model_name)
        
        if results:
            all_model_results[model_name] = results
            all_model_token_usage[model_name] = token_usage
            all_model_question_counts[model_name] = question_counts
            
            print(f"   ✅ Found {len(results)} budgets:")
            for budget in sorted(results.keys()):
                total_correct = sum(results[budget][doc_id] * question_counts[budget][doc_id] 
                                   for doc_id in results[budget])
                total_questions = sum(question_counts[budget].values())
                avg_acc = total_correct / total_questions if total_questions > 0 else 0
                
                token_usages = token_usage.get(budget, [])
                avg_token = sum(token_usages) / len(token_usages) if token_usages else 0
                print(f"      Budget {budget}: {len(results[budget])} docs, {total_questions} questions, "
                      f"Avg Acc: {avg_acc:.4f}, Avg Token: {avg_token:.2f}%")
        else:
            print(f"   ⚠️  No results found")
    
    if not all_model_results:
        print("\n❌ ERROR: No AttentiveTrim results found for any model!")
        return
    
    # Analyze RAG baselines
    print(f"\n📊 Analyzing RAG baselines...")
    print(f"   Directory: {RAG_BASELINE_DIR}")
    rag_results, rag_token_usage, rag_question_counts = analyze_rag_baselines(RAG_BASELINE_DIR)
    
    if rag_results:
        print(f"   ✅ Found {len(rag_results)} RAG models:")
        for rag_model in sorted(rag_results.keys()):
            print(f"      Model {rag_model}:")
            for top_k in sorted(rag_results[rag_model].keys()):
                total_correct = sum(rag_results[rag_model][top_k][doc_id] * rag_question_counts[rag_model][top_k][doc_id]
                                   for doc_id in rag_results[rag_model][top_k])
                total_questions = sum(rag_question_counts[rag_model][top_k].values())
                avg_acc = total_correct / total_questions if total_questions > 0 else 0
                
                token_usages = rag_token_usage.get(rag_model, {}).get(top_k, [])
                avg_token = sum(token_usages) / len(token_usages) if token_usages else 0
                print(f"         top_{top_k}: {len(rag_results[rag_model][top_k])} docs, {total_questions} questions, "
                      f"Avg Acc: {avg_acc:.4f}, Avg Token: {avg_token:.2f}%")
    else:
        print(f"   ⚠️  No RAG baseline results found")
    
    # Create comparison table
    print("\n📊 Creating detailed comparison table...")
    df = create_comparison_table(all_model_results, all_model_question_counts, 
                                 rag_results, rag_question_counts)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"   ✅ Saved to {output_csv}")
    
    # Create summary report
    print("\n📊 Creating summary report...")
    summary_df = create_summary_report(all_model_results, all_model_token_usage, all_model_question_counts,
                                      rag_results, rag_token_usage, rag_question_counts)
    summary_df.to_csv(summary_csv, index=False)
    print(f"   ✅ Saved to {summary_csv}")
    
    # Display summary
    print("\n" + "="*120)
    print("SUMMARY REPORT (Question-Weighted Accuracy)")
    print("="*120)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*120)
    print("DETAILED COMPARISON (first 10 rows)")
    print("="*120)
    print(df.head(10).to_string(index=False))
    
    print("\n" + "="*120)
    print("✅ Analysis complete!")
    print("="*120)
    print(f"📁 Full detailed results: {output_csv}")
    print(f"📁 Summary report: {summary_csv}")
    print(f"📁 All files saved to '{output_dir}/' directory")
    print("\n💡 Note: Average Accuracy in summary is weighted by number of questions per document")
    print("="*120)

if __name__ == "__main__":
    main()