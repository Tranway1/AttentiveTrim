"""
Compute average accuracy over all k values for each RAG embedding model and question.
Processes both 'notice' and 'paper' datasets.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Evaluation method for computing accuracy
ROUGE_MATCH_THRESHOLD = 0.5


def parse_bool(value) -> bool:
    """Convert various truthy values into bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def compute_accuracy(file_data: Dict, evaluation_method: str) -> float:
    """
    Compute accuracy for a single evaluation file.
    """
    documents = file_data.get('files', [])
    if not isinstance(documents, list) or not documents:
        return 0.0

    total_docs = len(documents)
    correct_docs = 0

    for doc in documents:
        match_value = doc.get('match')
        is_correct = False
        
        if evaluation_method in ['rouge', 'embedding']:
            # For embedding/rouge, check similarity/match score
            try:
                score = float(match_value) if match_value is not None else doc.get('similarity', 0)
                is_correct = score >= ROUGE_MATCH_THRESHOLD
            except (TypeError, ValueError):
                is_correct = False
        else:
            # For LLM methods (gpt, gemini, gpt4), check boolean match
            is_correct = parse_bool(match_value)

        if is_correct:
            correct_docs += 1

    return correct_docs / total_docs if total_docs > 0 else 0.0


def load_and_analyze_rag_results(dataset: str, evaluation_method: str, base_dir: Path = Path("baselines/evaluation")):
    """
    Load RAG results and compute average accuracy for each model-question pair.
    
    Args:
        dataset: Dataset name (e.g., 'paper', 'notice')
        evaluation_method: Evaluation method to filter by ('embedding', 'gpt', 'rouge', etc.)
        base_dir: Base directory containing evaluation results
    """
    eval_dir = base_dir / dataset
    
    if not eval_dir.exists():
        print(f"❌ Directory not found: {eval_dir}")
        return {}

    # Pattern to parse RAG filenames
    eval_pattern = re.compile(
        r"^rag-(?P<model>.+?)-top(?P<topk>[0-9]+)_(?P<question>.+?)_(?P<method>[a-z0-9]+)\.json$",
        re.IGNORECASE,
    )

    # Store results: {model: {question: {k: accuracy}}}
    results: Dict[str, Dict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    
    files_processed = 0
    files_skipped = 0
    
    for eval_path in sorted(eval_dir.glob("rag-*.json")):
        match = eval_pattern.match(eval_path.name)
        if not match:
            continue
            
        model_name = match.group("model")
        top_k = int(match.group("topk"))
        question_slug = match.group("question")
        eval_file_method = match.group("method")
        
        # Normalize evaluation method names (same logic as original code)
        normalized_method = eval_file_method
        if eval_file_method in ['gpt4', 'gemini']:
            normalized_method = 'gpt'
        
        # Filter by evaluation method (CRITICAL: same logic as generate_figures_enhanced.py)
        if normalized_method != evaluation_method and evaluation_method != 'gpt':
            files_skipped += 1
            continue
        
        if evaluation_method == 'gpt' and eval_file_method not in ['gpt', 'gpt4', 'gemini']:
            files_skipped += 1
            continue
        
        try:
            with open(eval_path, "r") as f:
                eval_data = json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading {eval_path.name}: {e}")
            continue

        # Get the actual question text
        question_text = eval_data.get("question", question_slug.replace("_", " "))
        
        # Compute accuracy
        accuracy = compute_accuracy(eval_data, normalized_method)
        
        # Store result
        results[model_name][question_text][top_k] = accuracy
        files_processed += 1

    print(f"  📁 Files processed: {files_processed}, skipped: {files_skipped}")
    return results


def print_results(dataset: str, results: Dict):
    """
    Print results in a clear, organized format.
    """
    if not results:
        print(f"\n❌ No results found for {dataset}")
        return
    
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset.upper()}")
    print(f"{'='*80}")
    
    for model in sorted(results.keys()):
        print(f"\n📊 Model: {model}")
        print(f"{'-'*80}")
        
        for question in sorted(results[model].keys()):
            k_accuracies = results[model][question]
            
            if not k_accuracies:
                continue
            
            # Compute average across all k values
            avg_accuracy = sum(k_accuracies.values()) / len(k_accuracies)
            
            # Sort k values for display
            sorted_k = sorted(k_accuracies.items())
            
            print(f"\n  Question: {question}")
            print(f"  {'─'*76}")
            
            # Show individual k values
            for k, acc in sorted_k:
                print(f"    k={k:2d}: {acc:.4f} ({acc*100:.2f}%)")
            
            # Show average
            print(f"  {'─'*76}")
            print(f"    AVERAGE: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")


def save_summary_table(dataset: str, results: Dict, output_dir: Path, eval_method: str = 'gpt'):
    """
    Save a summary table to CSV for easy analysis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataset}_rag_avg_accuracy_{eval_method}.csv"
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("Model,Question,")
        
        # Find all unique k values across all results
        all_k_values = set()
        for model_data in results.values():
            for question_data in model_data.values():
                all_k_values.update(question_data.keys())
        
        sorted_k = sorted(all_k_values)
        for k in sorted_k:
            f.write(f"k={k},")
        f.write("Average\n")
        
        # Write data rows
        for model in sorted(results.keys()):
            for question in sorted(results[model].keys()):
                k_accuracies = results[model][question]
                
                # Escape question if it contains commas
                question_escaped = f'"{question}"' if ',' in question else question
                
                f.write(f"{model},{question_escaped},")
                
                # Write accuracy for each k value
                for k in sorted_k:
                    acc = k_accuracies.get(k, '')
                    if acc != '':
                        f.write(f"{acc:.4f},")
                    else:
                        f.write(",")
                
                # Write average
                avg_accuracy = sum(k_accuracies.values()) / len(k_accuracies)
                f.write(f"{avg_accuracy:.4f}\n")
    
    print(f"💾 Individual report saved: {output_file}")


def save_combined_summary_table(all_results: Dict[str, Dict], output_dir: Path, eval_method: str = 'gpt'):
    """
    Save a combined summary table with all datasets together.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"combined_rag_avg_accuracy_{eval_method}.csv"
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("Dataset,Model,Question,")
        
        # Find all unique k values across ALL datasets
        all_k_values = set()
        for dataset_results in all_results.values():
            for model_data in dataset_results.values():
                for question_data in model_data.values():
                    all_k_values.update(question_data.keys())
        
        sorted_k = sorted(all_k_values)
        for k in sorted_k:
            f.write(f"k={k},")
        f.write("Average\n")
        
        # Write data rows for each dataset
        for dataset in sorted(all_results.keys()):
            results = all_results[dataset]
            
            for model in sorted(results.keys()):
                for question in sorted(results[model].keys()):
                    k_accuracies = results[model][question]
                    
                    # Escape question if it contains commas
                    question_escaped = f'"{question}"' if ',' in question else question
                    
                    f.write(f"{dataset},{model},{question_escaped},")
                    
                    # Write accuracy for each k value
                    for k in sorted_k:
                        acc = k_accuracies.get(k, '')
                        if acc != '':
                            f.write(f"{acc:.4f},")
                        else:
                            f.write(",")
                    
                    # Write average
                    avg_accuracy = sum(k_accuracies.values()) / len(k_accuracies)
                    f.write(f"{avg_accuracy:.4f}\n")
    
    print(f"\n💾 Combined report saved: {output_file}")
    print(f"   📊 Includes {len(all_results)} datasets with all models and questions")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute average RAG accuracy across all k values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process notice and paper datasets with GPT evaluation method (default)
  python compute_avg_rag_accuracy.py
  
  # Explicitly specify GPT evaluation
  python compute_avg_rag_accuracy.py --eval-method gpt
  
  # Use embedding evaluation method
  python compute_avg_rag_accuracy.py --eval-method embedding
  
  # Specify custom base directory
  python compute_avg_rag_accuracy.py --base-dir /path/to/baselines/evaluation
  
  # Process only specific datasets
  python compute_avg_rag_accuracy.py --datasets notice paper
  
  # Custom output directory
  python compute_avg_rag_accuracy.py --output-dir results/rag_analysis
        """
    )
    
    parser.add_argument('--base-dir', type=str, default='baselines/evaluation',
                       help='Base directory containing RAG evaluation results (default: baselines/evaluation)')
    parser.add_argument('--datasets', nargs='+', type=str, default=['notice', 'paper'],
                       help='Datasets to process (default: notice paper)')
    parser.add_argument('--output-dir', type=str, default='analysis/rag_avg_accuracy',
                       help='Output directory for CSV files (default: analysis/rag_avg_accuracy)')
    parser.add_argument('--eval-method', type=str, default='gpt',
                       choices=['embedding', 'gpt', 'rouge', 'gpt4', 'gemini'],
                       help='Evaluation method to use (default: gpt)')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    
    print("\n" + "="*80)
    print("RAG ACCURACY ANALYSIS")
    print("Computing average accuracy over all k values")
    print("="*80)
    print(f"Base directory: {base_dir}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Evaluation method: {args.eval_method}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Store all results for combined report
    all_results = {}
    
    for dataset in args.datasets:
        print(f"\n🔍 Processing dataset: {dataset}")
        results = load_and_analyze_rag_results(dataset, args.eval_method, base_dir)
        
        if results:
            print_results(dataset, results)
            save_summary_table(dataset, results, output_dir, args.eval_method)
            all_results[dataset] = results
        else:
            print(f"\n❌ No RAG results found for {dataset} with {args.eval_method} evaluation")
    
    # Generate combined report if we have results from multiple datasets
    if len(all_results) > 0:
        print(f"\n{'='*80}")
        print("📊 GENERATING COMBINED REPORT")
        print(f"{'='*80}")
        save_combined_summary_table(all_results, output_dir, args.eval_method)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    
    # Summary of generated files
    if all_results:
        print("\n📁 Generated files:")
        for dataset in all_results.keys():
            print(f"   • {dataset}_rag_avg_accuracy_{args.eval_method}.csv")
        if len(all_results) > 0:
            print(f"   • combined_rag_avg_accuracy_{args.eval_method}.csv")
        print(f"\n📂 All files saved in: {output_dir}/")
    else:
        print("\n⚠️  No files generated (no results found)")


if __name__ == "__main__":
    main()