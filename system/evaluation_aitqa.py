#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation Script for AITQA Dataset - Attention-Based Method with Top-K Support
Evaluates individual result files (one per question) against ground truth.
Outputs results grouped by document and supports different top-k values.
PRESERVES table metadata: row headers, column headers, selected row data
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
import os
import re
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

MODEL_MAPPING = {
    'llama3.2_1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'qwen3_8b': 'Qwen/Qwen3-8B',
    'qwen3_14b': 'Qwen/Qwen3-14B'
}

# ============================================================================
# LLM API CLIENT
# ============================================================================

class GPT4Client:
    """OpenAI GPT-4 API client."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.api_key = api_key
        self.model_name = model
        self.client = OpenAI(api_key=api_key)
        print(f"✅ GPT-4 client initialized with model: {model}")
    
    def evaluate(self, prompt: str) -> Tuple[float, str]:
        """Evaluate using GPT-4."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_completion_tokens=100
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract score
            if response_text.isdigit():
                score = float(response_text)
            else:
                # Try to extract first number from response
                score_match = re.search(r'(\d+)', response_text)
                if score_match:
                    score = float(score_match.group(1))
                else:
                    print(f"⚠️  Warning: Could not parse score from response")
                    print(f"Response: {response_text}")
                    score = 0
            
            score = max(0, min(10, score))  # Clamp to 0-10
            return score, response_text
        except Exception as e:
            print(f"❌ Error calling GPT-4 API: {e}")
            return 0, f"Error: {str(e)}"

def create_llm_client(model_name: str = "gpt-4o-mini") -> GPT4Client:
    """Create GPT-4 client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return GPT4Client(api_key, model_name)

# ============================================================================
# EVALUATION PROMPT
# ============================================================================

def get_aitqa_evaluation_prompt(question: str, ground_truth: str, prediction: str) -> str:
    """
    Returns evaluation prompt for AITQA dataset.
    
    AITQA is a table QA dataset where answers are typically numbers, dates, or short phrases.
    """
    return f"""You are evaluating a table question-answering system.

Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {prediction}

Evaluate if the prediction correctly answers the question based on the ground truth.

Guidelines:
- For numerical answers: Match if the numbers are the same (ignore formatting like commas, currency symbols)
  Example: "$5,813" matches "5813" and "$5,813 million"
- For dates/years: Match if the year/date is the same
  Example: "2016" matches "year 2016"
- For text answers: Match if the key information is the same (allow minor wording differences)
  Example: "4,137 million gallons" matches "4137 gallons"
- Ignore minor formatting differences (commas, dollar signs, units if clearly implied)

Scoring (0-10):
10: Perfect match or semantically equivalent
7-9: Mostly correct with minor differences in formatting or wording
4-6: Partially correct but missing key information
0-3: Incorrect or irrelevant

Output ONLY a number 0-10:"""

# ============================================================================
# DATASET LOADING
# ============================================================================

def load_processed_dataset(dataset_name: str, dataset_dir: str = 'data/datasets') -> Dict:
    """
    Load processed dataset file containing ground truth.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'aitqa')
        dataset_dir: Directory containing processed dataset files
    
    Returns:
        Dictionary containing the full dataset
    """
    processed_file = Path(dataset_dir) / f'{dataset_name}_processed.json'
    
    if not processed_file.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_file}")
    
    print(f"📖 Loading dataset from: {processed_file}")
    
    with open(processed_file) as f:
        dataset = json.load(f)
    
    num_docs = len(dataset.get('documents', []))
    print(f"✅ Loaded dataset with {num_docs} documents")
    
    return dataset

# ============================================================================
# RESULT LOADING (INDIVIDUAL FILES) WITH TOP-K SUPPORT
# ============================================================================

def discover_available_k_values(dataset_name: str, attention_type: str,
                                results_dir: str = 'pred_att') -> Set[int]:
    """
    Discover all available top-k values for the given dataset and attention type.
    
    Returns:
        Set of available k values
    """
    results_path = Path(results_dir) / dataset_name
    
    if not results_path.exists():
        return set()
    
    # Pattern to match result files with top-k
    pattern = f"results-{attention_type}-top*-{dataset_name}_*.json"
    result_files = list(results_path.glob(pattern))
    
    k_values = set()
    for file_path in result_files:
        # Extract k value from filename: results-{attention_type}-top{k}-...
        match = re.search(r'-top(\d+)-', file_path.name)
        if match:
            k = int(match.group(1))
            k_values.add(k)
    
    return k_values

def load_individual_results(dataset_name: str, attention_type: str, top_k: int,
                            results_dir: str = 'pred_att') -> Dict[int, List[Dict]]:
    """
    Loads individual result files for a specific top-k value and groups them by document_id.
    Each input file contains one question's result.
    
    Args:
        dataset_name: Name of the dataset
        attention_type: Type of attention ('raw', 'farest', 'baseline')
        top_k: The top-k value to load results for
        results_dir: Directory containing results
    
    Returns:
        Dict mapping document_id to list of results for that document
    """
    results_path = Path(results_dir) / dataset_name
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")
    
    # Pattern to match individual result files with specific top-k
    pattern = f"results-{attention_type}-top{top_k}-{dataset_name}_*.json"
    result_files = list(results_path.glob(pattern))
    
    if not result_files:
        raise FileNotFoundError(f"No result files found matching: {pattern}")
    
    print(f"📂 Found {len(result_files)} individual result files for top-{top_k}")
    
    # Group by document_id
    grouped = defaultdict(list)
    
    for file_path in sorted(result_files):
        with open(file_path) as f:
            result = json.load(f)
            doc_id = result.get('document_id')
            grouped[doc_id].append(result)
    
    print(f"📋 Grouped into {len(grouped)} documents")
    
    return dict(grouped)

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_document(doc_id: int, results: List[Dict], dataset: Dict,
                     llm_client: GPT4Client, top_k: int) -> Dict:
    """
    Evaluate all questions for a single document.
    
    Args:
        doc_id: Document ID
        results: List of result dictionaries for this document
        dataset: Full dataset with ground truth
        llm_client: GPT-4 client for evaluation
        top_k: The top-k value being evaluated
    
    Returns:
        Dictionary with evaluation results for this document
    """
    print(f"\n{'='*60}")
    print(f"📄 Evaluating Document {doc_id} (top-{top_k})")
    print(f"{'='*60}")
    print(f"Questions: {len(results)}")
    
    # Find ground truth for this document
    doc_data = None
    for doc in dataset.get('documents', []):
        if doc['document_id'] == doc_id:
            doc_data = doc
            break
    
    if not doc_data:
        print(f"⚠️  Warning: No ground truth found for document {doc_id}")
        return None
    
    # Create ground truth lookup
    gt_lookup = {}
    for q in doc_data.get('questions', []):
        question_id = q['question_id']
        gt_lookup[question_id] = {
            'question': q['question'],
            'answer': q.get('answer', '')
        }
    
    # Evaluate each result
    evaluated_results = []
    total_score = 0
    total_matches = 0
    
    for result in results:
        question_id = result.get('question_id')
        question = result.get('question')
        prediction = result.get('result', '')
        
        # Get ground truth
        gt_data = gt_lookup.get(question_id)
        if not gt_data:
            print(f"⚠️  No ground truth for question_id: {question_id}")
            continue
        
        ground_truth = gt_data['answer']
        
        # Evaluate with LLM
        evaluation_prompt = get_aitqa_evaluation_prompt(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction
        )
        
        score, rationale = llm_client.evaluate(evaluation_prompt)
        
        # Count as match if score >= 7
        match = score >= 7
        if match:
            total_matches += 1
        total_score += score
        
        print(f"Q: {question[:60]}...")
        print(f"   GT: {ground_truth[:60]}...")
        print(f"   Pred: {prediction[:60]}...")
        print(f"   Score: {score}, Match: {match}")
        
        # Build evaluation result with all metadata
        eval_entry = {
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "score": score,
            "match": match,
            "rationale": rationale,
            "top_k": result.get('top_k', top_k),
            "tokens_extracted": result.get('tokens_extracted'),
            "total_tokens": result.get('total_tokens'),
            "num_rows_selected": result.get('num_rows_selected'),
            "total_rows": result.get('total_rows'),
            "selected_row_idx": result.get('selected_row_indices'),
            "selected_row_data": result.get('selected_row_data'),
            "timing": result.get('timing', {})
        }
        
        # Preserve table-specific metadata if present
        if 'selected_row_headers' in result:
            eval_entry['selected_row_headers'] = result['selected_row_headers']
        if 'column_headers' in result:
            eval_entry['column_headers'] = result['column_headers']
        if 'attention_type' in result:
            eval_entry['attention_type'] = result['attention_type']
        if 'use_kv_cache' in result:
            eval_entry['use_kv_cache'] = result['use_kv_cache']
        
        evaluated_results.append(eval_entry)
    
    # Create document evaluation summary
    doc_evaluation = {
        "document_id": doc_id,
        "top_k": top_k,
        "total_questions": len(evaluated_results),
        "total_matches": total_matches,
        "average_score": total_score / len(evaluated_results) if evaluated_results else 0,
        "accuracy": total_matches / len(evaluated_results) if evaluated_results else 0,
        "results": evaluated_results
    }
    
    print(f"\n📊 Document {doc_id} (top-{top_k}) Summary:")
    print(f"   Matches (≥7): {total_matches}/{len(evaluated_results)}")
    print(f"   Accuracy: {doc_evaluation['accuracy']:.2%}")
    print(f"   Avg Score: {doc_evaluation['average_score']:.2f}")
    
    return doc_evaluation

# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def evaluate_for_k(dataset: Dict, dataset_name: str, attention_type: str, top_k: int,
                   results_dir: str, output_dir: Path, llm_client: GPT4Client):
    """
    Evaluate results for a specific top-k value.
    
    Args:
        dataset: Loaded dataset with ground truth
        dataset_name: Name of the dataset
        attention_type: Type of attention
        top_k: The top-k value to evaluate
        results_dir: Directory containing results
        output_dir: Directory to save evaluation results
        llm_client: GPT-4 client for evaluation
    """
    print("\n" + "="*60)
    print(f"EVALUATING: {attention_type.upper()} with TOP-{top_k}")
    print("="*60)
    
    # Load individual result files grouped by document
    try:
        grouped_results = load_individual_results(
            dataset_name, attention_type, top_k, results_dir
        )
    except Exception as e:
        print(f"❌ Error loading results for top-{top_k}: {e}")
        return None
    
    # Evaluate each document
    all_evaluations = []
    total_questions = 0
    total_matches = 0
    total_score = 0
    
    for doc_id in sorted(grouped_results.keys()):
        results = grouped_results[doc_id]
        
        try:
            doc_eval = evaluate_document(doc_id, results, dataset, llm_client, top_k)
            if doc_eval:
                all_evaluations.append(doc_eval)
                total_questions += doc_eval['total_questions']
                total_matches += doc_eval['total_matches']
                total_score += doc_eval['average_score'] * doc_eval['total_questions']
                
                # Save individual document evaluation
                doc_output_file = output_dir / f"{attention_type}_top{top_k}_doc_{doc_id}_eval.json"
                with open(doc_output_file, 'w') as f:
                    json.dump(doc_eval, f, indent=2)
                print(f"✅ Saved: {doc_output_file.name}")
        
        except Exception as e:
            print(f"❌ Error evaluating document {doc_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create summary for this k value
    summary = {
        "dataset": dataset_name,
        "attention_type": attention_type,
        "top_k": top_k,
        "total_documents": len(all_evaluations),
        "total_questions": total_questions,
        "total_matches": total_matches,
        "overall_accuracy": total_matches / total_questions if total_questions else 0,
        "average_score": total_score / total_questions if total_questions else 0,
        "documents": all_evaluations
    }
    
    # Save summary
    summary_file = output_dir / f"{attention_type}_top{top_k}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print(f"📊 SUMMARY: {attention_type.upper()} with TOP-{top_k}")
    print("="*60)
    print(f"Total Documents: {len(all_evaluations)}")
    print(f"Total Questions: {total_questions}")
    print(f"Total Matches (≥7): {total_matches}")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.2%}")
    print(f"Average Score: {summary['average_score']:.2f}")
    print(f"✅ Summary saved: {summary_file}")
    print("="*60)
    
    return summary

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate AITQA attention-based results with top-k support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Llama3.2-1B results for top-3 rows
  python evaluation_aitqa_topk.py --dataset aitqa --attention-type raw --top-k 3 --model llama3.2_1b
  
  # Evaluate all available k values for Qwen3-8B with farest attention
  python evaluation_aitqa_topk.py --dataset aitqa --attention-type farest --top-k all --model qwen3_8b
  
  # Evaluate specific k values for Qwen3-14B: 1, 3, and 5
  python evaluation_aitqa_topk.py --dataset aitqa --attention-type baseline --top-k 1,3,5 --model qwen3_14b
        """
    )
    
    parser.add_argument('--dataset', type=str, default='aitqa',
                       help='Dataset name (default: aitqa)')
    parser.add_argument('--attention-type', type=str, required=True,
                       choices=['raw', 'farest', 'baseline'],
                       help='Type of attention used')
    parser.add_argument('--top-k', type=str, required=True,
                       help='Top-k value(s) to evaluate. Options: single value (e.g., "3"), '
                            'comma-separated values (e.g., "1,3,5"), or "all" for all available k values')
    
    # ✅ NEW: Model selection (must match preprocessing and inference)
    parser.add_argument('--model', type=str,
                       default="llama3.2_1b",
                       choices=["llama3.2_1b", "qwen3_8b", "qwen3_14b"],
                       help='Model used for preprocessing and inference (MUST match!)')
    
    parser.add_argument('--results-dir', type=str, default='pred_att',
                       help='Directory containing results')
    parser.add_argument('--dataset-dir', type=str, default='data/datasets',
                       help='Directory containing processed dataset')
    parser.add_argument('--output-dir', type=str, default='evaluation',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # ✅ Resolve model name from mapping
    model_name = MODEL_MAPPING.get(args.model, args.model)
    
    print("\n" + "="*60)
    print("AITQA EVALUATION PIPELINE WITH TOP-K SUPPORT")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model} ({model_name})")
    print(f"Attention Type: {args.attention_type}")
    
    # ✅ WARNING: Check model consistency
    print(f"\n⚠️  IMPORTANT: Ensure preprocessing and inference used the same model!")
    print(f"   Current model: {args.model} ({model_name})")
    print()
    
    # ✅ Setup directories with model prefix
    base_data_dir = os.path.join(args.model, args.dataset_dir)
    dataset_dir = os.path.join(base_data_dir, '')
    
    base_results_dir = os.path.join(args.model, args.results_dir)
    base_output_dir = os.path.join(args.model, args.output_dir)
    
    print(f"📂 Directory structure:")
    print(f"   Dataset:     {dataset_dir}")
    print(f"   Results:     {base_results_dir}/")
    print(f"   Output:      {base_output_dir}/")
    print()
    
    # Initialize LLM client
    try:
        llm_client = create_llm_client()
    except Exception as e:
        print(f"❌ Error initializing LLM client: {e}")
        return
    
    # Load processed dataset (contains ground truth)
    try:
        dataset = load_processed_dataset(args.dataset, dataset_dir)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"   Expected file: {os.path.join(dataset_dir, args.dataset)}_processed.json")
        print(f"   Make sure you ran data_reader.py with --model {args.model}")
        return
    
    # Parse top-k argument
    k_values = []
    if args.top_k.lower() == 'all':
        # Discover all available k values
        available_k = discover_available_k_values(args.dataset, args.attention_type, base_results_dir)
        if not available_k:
            print(f"❌ No results found for {args.attention_type} attention")
            print(f"   Expected directory: {base_results_dir}/{args.dataset}/")
            print(f"   Make sure you ran unit_window_aitqa.py with --model {args.model}")
            return
        k_values = sorted(available_k)
        print(f"🔍 Discovered k values: {k_values}")
    else:
        # Parse comma-separated list or single value
        try:
            k_values = [int(k.strip()) for k in args.top_k.split(',')]
        except ValueError:
            print(f"❌ Error: Invalid top-k format: {args.top_k}")
            print("   Use a single value (e.g., '3'), comma-separated (e.g., '1,3,5'), or 'all'")
            return
    
    print(f"📊 Will evaluate {len(k_values)} k value(s): {k_values}")
    
    # Create output directory with model prefix
    output_dir = Path(base_output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate for each k value
    all_summaries = []
    for k in k_values:
        summary = evaluate_for_k(
            dataset=dataset,
            dataset_name=args.dataset,
            attention_type=args.attention_type,
            top_k=k,
            results_dir=base_results_dir,
            output_dir=output_dir,
            llm_client=llm_client
        )
        if summary:
            all_summaries.append(summary)
    
    # Create combined summary if multiple k values
    if len(all_summaries) > 1:
        combined_summary = {
            "dataset": args.dataset,
            "attention_type": args.attention_type,
            "k_values": k_values,
            "summaries": all_summaries,
            "comparison": []
        }
        
        # Add comparison data
        for summary in all_summaries:
            combined_summary["comparison"].append({
                "top_k": summary["top_k"],
                "accuracy": summary["overall_accuracy"],
                "average_score": summary["average_score"],
                "total_questions": summary["total_questions"]
            })
        
        # Save combined summary
        combined_file = output_dir / f"{args.attention_type}_all_k_comparison.json"
        with open(combined_file, 'w') as f:
            json.dump(combined_summary, f, indent=2)
        
        # Print comparison table
        print("\n" + "="*60)
        print(f"📊 COMPARISON ACROSS K VALUES: {args.attention_type.upper()}")
        print("="*60)
        print(f"{'Top-K':<10} {'Accuracy':<15} {'Avg Score':<15} {'Questions':<15}")
        print("-" * 60)
        for comp in combined_summary["comparison"]:
            print(f"{comp['top_k']:<10} {comp['accuracy']:<15.2%} {comp['average_score']:<15.2f} {comp['total_questions']:<15}")
        print("="*60)
        print(f"✅ Comparison saved: {combined_file}")
    
    print("\n" + "="*60)
    print(f"✅ COMPLETE: All evaluations finished")
    print("="*60)
    print(f"📁 Results directory: {output_dir}")

if __name__ == "__main__":
    main()