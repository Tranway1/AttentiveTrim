#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation Script for AITQA Dataset - RAG Baseline Method
Evaluates RAG baseline results against ground truth.
Imports evaluation functions from evaluation_aitqa.py for consistency.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys
import os

# Add parent directory to path to import from the main evaluation script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import evaluation components from the attention evaluation script in parent directory
from evaluation_aitqa import (
    create_llm_client,
    get_aitqa_evaluation_prompt,
    load_processed_dataset,
    GPT4Client
)

# ============================================================================
# CONFIGURATION
# ============================================================================

AVAILABLE_EMBEDDING_MODELS = [
    'UAE-Large-V1',
    'bflhc-Octen-Embedding-4B',
    'Qwen3-Embedding-8B'
]

AVAILABLE_TOP_K = [1, 3, 5, 10]  # Common top-k values

# ============================================================================
# BASELINE RESULT LOADING
# ============================================================================

def load_baseline_results(dataset_name: str, 
                          results_dir: str = 'pred_att',
                          embedding_model: str = None,
                          top_k: int = None) -> Dict[int, Dict]:
    """
    Loads RAG baseline result files grouped by document.
    Each file contains all questions for one document.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'aitqa')
        results_dir: Directory containing baseline results
        embedding_model: Name of embedding model
        top_k: Top-K value used during generation
    
    Returns:
        Dict mapping document_id to the full result dict for that document
    """
    # UPDATED: New directory structure - pred_att/{dataset}/top{k}/{model}/
    if embedding_model and top_k:
        results_path = Path(results_dir) / dataset_name / f'top{top_k}' / embedding_model
    elif embedding_model:
        # Fallback for old structure
        results_path = Path(results_dir) / f'{dataset_name}_baseline_{embedding_model}'
    else:
        results_path = Path(results_dir) / f'{dataset_name}_baseline'
    
    if not results_path.exists():
        raise FileNotFoundError(f"Baseline results directory not found: {results_path}")
    
    # Pattern to match baseline result files: results-aitqa-{doc_id}.json
    pattern = f"results-{dataset_name}-*.json"
    result_files = list(results_path.glob(pattern))
    
    if not result_files:
        raise FileNotFoundError(
            f"No baseline result files found in {results_path} matching: {pattern}"
        )
    
    print(f"📂 Found {len(result_files)} baseline result files in {results_path}")
    
    # Load and group by document_id
    grouped = {}
    
    for file_path in sorted(result_files):
        with open(file_path) as f:
            result_doc = json.load(f)
            doc_id = result_doc.get('document_id')
            grouped[doc_id] = result_doc
    
    print(f"📋 Loaded {len(grouped)} documents")
    
    return grouped

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_document(doc_id: int, doc_results: Dict, dataset: Dict,
                     llm_client: GPT4Client) -> Dict:
    """
    Evaluate all questions for a single document.
    
    Args:
        doc_id: Document ID
        doc_results: Full results dict for this document (contains 'results' list)
        dataset: Full dataset with ground truth
        llm_client: GPT-4 client for evaluation
    
    Returns:
        Dictionary with evaluation results for this document
    """
    print(f"\n{'='*60}")
    print(f"📄 Evaluating Document {doc_id}")
    print(f"{'='*60}")
    
    results = doc_results.get('results', [])
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
        
        # Evaluate with LLM using the same prompt as attention method
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
        
        # Build evaluation result preserving baseline metadata
        eval_entry = {
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "score": score,
            "match": match,
            "evaluation_rationale": rationale,
            "baseline_rationale": result.get('rationale', ''),
            "timing": result.get('timing', {}),
            "total_tokens": result.get('total_tokens'),
            "rag_metadata": result.get('rag_metadata', {})
        }
        
        evaluated_results.append(eval_entry)
    
    # Create document evaluation summary
    doc_evaluation = {
        "document_id": doc_id,
        "total_questions": len(evaluated_results),
        "total_matches": total_matches,
        "average_score": total_score / len(evaluated_results) if evaluated_results else 0,
        "accuracy": total_matches / len(evaluated_results) if evaluated_results else 0,
        "results": evaluated_results
    }
    
    print(f"\n📊 Document {doc_id} Summary:")
    print(f"   Matches (≥7): {total_matches}/{len(evaluated_results)}")
    print(f"   Accuracy: {doc_evaluation['accuracy']:.2%}")
    print(f"   Avg Score: {doc_evaluation['average_score']:.2f}")
    
    return doc_evaluation

# ============================================================================
# EVALUATION PIPELINE FOR SINGLE CONFIGURATION
# ============================================================================

def evaluate_single_configuration(dataset_name: str, dataset: Dict, llm_client: GPT4Client,
                                 results_dir: str, output_base_dir: str,
                                 embedding_model: str, top_k: int) -> Dict:
    """
    Evaluate a single configuration (one embedding model + one top_k).
    
    Returns:
        Summary dictionary with results
    """
    config_name = f"{embedding_model}_top{top_k}"
    
    print("\n" + "="*80)
    print(f"🔄 EVALUATING CONFIGURATION: {config_name}")
    print("="*80)
    print(f"Embedding Model: {embedding_model}")
    print(f"Top-K: {top_k}")
    print()
    
    # Load baseline result files grouped by document
    try:
        grouped_results = load_baseline_results(dataset_name, results_dir, embedding_model, top_k)
    except Exception as e:
        print(f"❌ Error loading baseline results for {config_name}: {e}")
        return None
    
    # Create output directory
    output_dir = Path(output_base_dir) / f"{dataset_name}_{embedding_model}_top{top_k}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each document
    all_evaluations = []
    total_questions = 0
    total_matches = 0
    total_score = 0
    
    for doc_id in sorted(grouped_results.keys()):
        doc_results = grouped_results[doc_id]
        
        try:
            doc_eval = evaluate_document(doc_id, doc_results, dataset, llm_client)
            if doc_eval:
                all_evaluations.append(doc_eval)
                total_questions += doc_eval['total_questions']
                total_matches += doc_eval['total_matches']
                total_score += doc_eval['average_score'] * doc_eval['total_questions']
                
                # Save individual document evaluation
                doc_output_file = output_dir / f"baseline_doc_{doc_id}_eval.json"
                with open(doc_output_file, 'w') as f:
                    json.dump(doc_eval, f, indent=2)
                print(f"✅ Saved: {doc_output_file.name}")
        
        except Exception as e:
            print(f"❌ Error evaluating document {doc_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create overall summary
    summary = {
        "dataset": dataset_name,
        "method": "rag_baseline",
        "embedding_model": embedding_model,
        "top_k": top_k,
        "total_documents": len(all_evaluations),
        "total_questions": total_questions,
        "total_matches": total_matches,
        "overall_accuracy": total_matches / total_questions if total_questions else 0,
        "average_score": total_score / total_questions if total_questions else 0,
        "documents": all_evaluations
    }
    
    # Save summary
    summary_file = output_dir / "baseline_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print(f"📊 SUMMARY - {config_name}")
    print("="*60)
    print(f"Total Documents: {len(all_evaluations)}")
    print(f"Total Questions: {total_questions}")
    print(f"Total Matches (≥7): {total_matches}")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.2%}")
    print(f"Average Score: {summary['average_score']:.2f}")
    print()
    print(f"✅ Summary saved: {summary_file}")
    print(f"📁 Results directory: {output_dir}")
    print("="*60)
    
    return summary

# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate AITQA RAG baseline results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate specific configuration
  %(prog)s --embedding-model UAE-Large-V1 --top-k 5
  
  # Evaluate all top-k for one model
  %(prog)s --embedding-model UAE-Large-V1 --top-k all
  
  # Evaluate all models for one top-k
  %(prog)s --embedding-model all --top-k 5
  
  # Evaluate ALL combinations (all models × all top-k values)
  %(prog)s --embedding-model all --top-k all
        """
    )
    
    parser.add_argument('--dataset', type=str, default='aitqa',
                       help='Dataset name (default: aitqa)')
    parser.add_argument('--embedding-model', type=str, required=True,
                       help=f'Embedding model name or "all". Options: {", ".join(AVAILABLE_EMBEDDING_MODELS + ["all"])}')
    parser.add_argument('--top-k', type=str, required=True,
                       help=f'Top-K value or "all". Common values: {", ".join(map(str, AVAILABLE_TOP_K))}')
    parser.add_argument('--results-dir', type=str, default='pred_att',
                       help='Directory containing baseline results (default: pred_att)')
    parser.add_argument('--dataset-dir', type=str, default='../llama3.2_1b/data/datasets',
                       help='Directory containing processed dataset (default: ../llama3.2_1b/data/datasets)')
    parser.add_argument('--output-dir', type=str, default='evaluation/aitqa',
                       help='Directory to save evaluation results (default: evaluation/aitqa)')
    
    args = parser.parse_args()
    
    # Determine which embedding models to evaluate
    if args.embedding_model.lower() == 'all':
        models_to_eval = AVAILABLE_EMBEDDING_MODELS
        print(f"\n🚀 Evaluating ALL embedding models: {models_to_eval}")
    else:
        if args.embedding_model not in AVAILABLE_EMBEDDING_MODELS:
            print(f"❌ Error: Unknown embedding model '{args.embedding_model}'")
            print(f"   Available models: {', '.join(AVAILABLE_EMBEDDING_MODELS)}")
            print(f"   Or use 'all' to evaluate all models")
            return 1
        models_to_eval = [args.embedding_model]
    
    # Determine which top-k values to evaluate
    if args.top_k.lower() == 'all':
        top_k_to_eval = AVAILABLE_TOP_K
        print(f"🚀 Evaluating ALL top-k values: {top_k_to_eval}")
    else:
        try:
            top_k_value = int(args.top_k)
            top_k_to_eval = [top_k_value]
        except ValueError:
            print(f"❌ Error: Invalid top-k value '{args.top_k}'")
            print(f"   Must be an integer or 'all'")
            return 1
    
    print("\n" + "="*60)
    print("AITQA RAG BASELINE EVALUATION PIPELINE")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Method: RAG Baseline")
    print(f"Embedding Models: {models_to_eval}")
    print(f"Top-K Values: {top_k_to_eval}")
    print(f"Total Configurations: {len(models_to_eval) * len(top_k_to_eval)}")
    print(f"Results Dir: {args.results_dir}")
    print()
    
    # Initialize LLM client (same as attention evaluation)
    try:
        llm_client = create_llm_client()
    except Exception as e:
        print(f"❌ Error initializing LLM client: {e}")
        return 1
    
    # Load processed dataset (contains ground truth)
    try:
        dataset = load_processed_dataset(args.dataset, args.dataset_dir)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return 1
    
    # Evaluate all combinations
    all_summaries = []
    
    for model_idx, embedding_model in enumerate(models_to_eval, 1):
        for k_idx, top_k in enumerate(top_k_to_eval, 1):
            config_num = (model_idx - 1) * len(top_k_to_eval) + k_idx
            total_configs = len(models_to_eval) * len(top_k_to_eval)
            
            print(f"\n{'='*80}")
            print(f"⏳ CONFIGURATION {config_num}/{total_configs}")
            print(f"{'='*80}")
            
            try:
                summary = evaluate_single_configuration(
                    args.dataset, dataset, llm_client,
                    args.results_dir, args.output_dir,
                    embedding_model, top_k
                )
                
                if summary:
                    all_summaries.append(summary)
                    
            except Exception as e:
                print(f"❌ Error evaluating {embedding_model} with top-{top_k}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Create aggregate summary if multiple configurations
    if len(all_summaries) > 1:
        aggregate_summary = {
            "dataset": args.dataset,
            "method": "rag_baseline",
            "total_configurations": len(all_summaries),
            "configurations": []
        }
        
        print("\n" + "="*80)
        print("📊 AGGREGATE SUMMARY - ALL CONFIGURATIONS")
        print("="*80)
        
        for summary in all_summaries:
            config_summary = {
                "embedding_model": summary["embedding_model"],
                "top_k": summary["top_k"],
                "accuracy": summary["overall_accuracy"],
                "average_score": summary["average_score"],
                "total_questions": summary["total_questions"]
            }
            aggregate_summary["configurations"].append(config_summary)
            
            print(f"\n{summary['embedding_model']} (top-{summary['top_k']}):")
            print(f"  Accuracy: {summary['overall_accuracy']:.2%}")
            print(f"  Avg Score: {summary['average_score']:.2f}")
            print(f"  Questions: {summary['total_questions']}")
        
        # Save aggregate summary
        aggregate_file = Path(args.output_dir) / f"{args.dataset}_aggregate_summary.json"
        with open(aggregate_file, 'w') as f:
            json.dump(aggregate_summary, f, indent=2)
        
        print(f"\n✅ Aggregate summary saved: {aggregate_file}")
        print("="*80)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)
    print(f"Successfully evaluated: {len(all_summaries)}/{len(models_to_eval) * len(top_k_to_eval)} configurations")
    print(f"Results directory: {args.output_dir}")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())