#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation Script for RAG Baselines
Loads RAG prediction results and evaluates them using the same evaluation functions
as the main AttentiveTrim evaluation.

UPDATED: 
- Supports paper, notice datasets
- Uses correct embedding models (UAE-Large-V1, bflhc/Octen-Embedding-4B, Qwen3-Embedding-8B)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys
import warnings
import re
from collections import defaultdict

warnings.filterwarnings('ignore')

# Predefined models and top-k values (UPDATED TO MATCH ACTUAL MODELS)
AVAILABLE_EMBEDDING_MODELS = [
    'UAE-Large-V1',
    'bflhc-Octen-Embedding-4B',  # Note: / becomes - in filenames
    'Qwen3-Embedding-8B'
]
TOP_K_VALUES = [1, 3, 5, 7, 10]

# Add parent directory to path to import from evaluation.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import evaluation functions and utilities from main evaluation.py
from evaluation import (
    evaluate_results_llm,
    evaluate_results_embedding,
    create_llm_client,
    load_processed_dataset,
    build_groundtruth_for_question,
    sanitize_question_for_filename,
)

try:
    from sentence_transformers import SentenceTransformer
    print("✅ Embedding libraries initialized")
except ImportError as e:
    print(f"⚠️  Warning: Could not import embedding dependencies: {e}")

# ============================================================================
# RAG-SPECIFIC RESULT LOADING
# ============================================================================

def parse_rag_filename(filename: str) -> Dict:
    """
    Parse RAG result filename for paper/notice datasets.
    
    Format: results-rag-{model}-top{k}-{dataset}_doc{doc_id}_q{question_id}.json
    Example: results-rag-Qwen3-Embedding-8B-top1-paper_doc0_q0_0.json
    """
    pattern = re.compile(
        r"^results-rag-(?P<model>.+?)-top(?P<topk>\d+)-(?P<dataset>[^_]+)_doc(?P<doc_id>\d+)_q(?P<question_id>[\d_]+)\.json$"
    )
    
    match = pattern.match(filename)
    if not match:
        return None
    
    return {
        'model': match.group('model'),
        'top_k': int(match.group('topk')),
        'dataset': match.group('dataset'),
        'doc_id': int(match.group('doc_id')),
        'question_id': match.group('question_id'),
        'format': 'standard'
    }

def load_and_group_rag_results(
    dataset_name: str,
    model_filter: str = None,
    top_k_filter: int = None,
    results_dir: str = 'baselines/pred_att'
) -> Dict[str, Dict[str, Dict[int, List[Dict]]]]:
    """
    Load RAG prediction files and group them by question, model, and top-k.
    
    UPDATED: Handles both standard (paper/notice) dataset formats.
    
    Args:
        dataset_name: Name of the dataset (paper, notice)
        model_filter: Optional embedding model name to filter
        top_k_filter: Optional top-k value to filter
        results_dir: Directory containing RAG prediction files
    
    Returns:
        Dictionary mapping: question -> model -> top_k -> list of results
        Example: {"What is the artifact?": {"Qwen3-Embedding-8B": {1: [...], 3: [...]}}}
    """
    results_path = Path(__file__).resolve().parent / results_dir / dataset_name
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")
    
    print(f"📂 Loading RAG results from: {results_path}")
    print(f"   Dataset type: {dataset_name}")
    
    # Group by question -> model -> top_k
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for file_path in sorted(results_path.glob("results-rag-*.json")):
        # Parse filename based on dataset type
        parsed = parse_rag_filename(file_path.name)
        if not parsed:
            print(f"⚠️  Could not parse filename: {file_path.name}")
            continue
        
        model = parsed['model']
        top_k = parsed['top_k']
        
        # Apply filters if specified
        if model_filter and model != model_filter:
            continue
        if top_k_filter is not None and top_k != top_k_filter:
            continue
        
        # Load result file
        try:
            with open(file_path) as f:
                result = json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading {file_path.name}: {e}")
            continue
        
        # Get question from result file
        question = result.get('question', '')
        
        if not question:
            print(f"⚠️  No question found in {file_path.name}")
            continue
        
        # Add metadata from filename and file to result
        result['_metadata'] = {
            'filename': file_path.name,
            'parsed': parsed,
        }
        
        grouped[question][model][top_k].append(result)
    
    # Convert to regular dict
    result_dict = {}
    for question, models in grouped.items():
        result_dict[question] = {}
        for model, top_ks in models.items():
            result_dict[question][model] = dict(top_ks)
    
    print(f"📋 Grouped into {len(result_dict)} unique questions")
    for question, models in list(result_dict.items())[:3]:
        print(f"   '{question[:60]}...':")
        for model, top_ks in models.items():
            print(f"      {model}: top-k values {sorted(top_ks.keys())}")
    if len(result_dict) > 3:
        print(f"   ... and {len(result_dict) - 3} more questions")
    
    return result_dict

def aggregate_rag_results_to_dict(results_list: List[Dict], question: str,
                                           processed_dataset: Dict) -> Dict:
    """
    Aggregate RAG results for paper/notice datasets (standard format).
    
    Args:
        results_list: List of RAG prediction results for this question
        question: Question text
        processed_dataset: The processed dataset with metadata
    
    Returns:
        Dictionary compatible with evaluate_results_llm/embedding functions
    """
    print(f"   📦 Aggregating {len(results_list)} standard RAG results")
    
    # Create a mapping from document_id to original_path
    doc_id_to_path = {}
    for doc in processed_dataset.get('documents', []):
        doc_id = doc['document_id']
        original_path = doc.get('metadata', {}).get('original_path', f'doc_{doc_id}')
        doc_id_to_path[doc_id] = original_path
    
    # Create aggregated results structure
    aggregated = {
        'question': question,
        'files': []
    }
    
    for result in results_list:
        doc_id = result.get('document_id')
        predicted = result.get('result', '')
        
        # Get token usage from RAG result
        token_usage = result.get('token_usage', {})
        context_tokens = token_usage.get('context_tokens')
        document_total_tokens = token_usage.get('document_total_tokens')
        
        # Use original file path to match with ground truth
        original_path = doc_id_to_path.get(doc_id, f'doc_{doc_id}')
        
        aggregated['files'].append({
            'file': original_path,
            'result': predicted,
            'tokens_extracted': context_tokens,  # For figure generation
            'total_tokens': document_total_tokens,
            # Store additional RAG metadata
            'rag_metadata': {
                'embedding_model': result.get('embedding_model'),
                'top_k_chunks': result.get('top_k_chunks'),
                'chunks_retrieved': token_usage.get('chunks_retrieved'),
                'chunk_token_counts': token_usage.get('chunk_token_counts'),
            }
        })
    
    return aggregated


# ============================================================================
# MAIN EVALUATION PIPELINE FOR RAG
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG baseline results')
    
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['paper', 'notice'],
                       help='Dataset name (paper, notice)')
    parser.add_argument('--model', type=str, default=None,
                       help='Filter by embedding model name (e.g., Qwen3-Embedding-8B, bflhc-Octen-Embedding-4B)')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Filter by top-k value')
    parser.add_argument('--all-models', action='store_true',
                       help='Evaluate all predefined models and top-k combinations')
    parser.add_argument('--method', type=str, default='both',
                       choices=['llm', 'embedding', 'both'], help="Evaluation method")
    parser.add_argument('--llm-model', type=str, default='gemini',
                       choices=['gemini', 'gpt4'], help="LLM model for evaluation")
    parser.add_argument('--results-dir', type=str, default='pred_att',
                       help='Directory containing RAG prediction files (relative to baselines/)')
    parser.add_argument('--dataset-dir', type=str, default='../llama3.2_1b/data/datasets',
                       help='Directory containing processed dataset files')
    parser.add_argument('--output-dir', type=str, default='evaluation',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Determine which model/top-k combinations to evaluate
    if args.all_models:
        # Evaluate all predefined combinations
        combinations = []
        for model in AVAILABLE_EMBEDDING_MODELS:
            for top_k in TOP_K_VALUES:
                combinations.append((model, top_k))
        print(f"\n🔄 --all-models flag set: Will evaluate {len(combinations)} combinations")
        print(f"   Models: {AVAILABLE_EMBEDDING_MODELS}")
        print(f"   Top-K values: {TOP_K_VALUES}")
    else:
        # Evaluate single combination or all available
        if args.model or args.top_k is not None:
            combinations = [(args.model, args.top_k)]
            print(f"\n🔍 Evaluating specific filter:")
            if args.model:
                print(f"   Model: {args.model}")
            if args.top_k is not None:
                print(f"   Top-K: {args.top_k}")
        else:
            # No filters - evaluate whatever is found
            combinations = [(None, None)]
            print(f"\n🔍 No filters specified - will evaluate all found results")
    
    print("\n" + "="*60)
    print("RAG BASELINE EVALUATION PIPELINE")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Evaluation Method: {args.method}")
    if args.method in ['llm', 'both']:
        print(f"LLM Model: {args.llm_model}")
    print()
    
    # Initialize LLM client if needed
    llm_client = None
    if args.method in ['llm', 'both']:
        try:
            llm_client = create_llm_client(args.llm_model)
        except Exception as e:
            print(f"❌ Error initializing LLM client: {e}")
            return
    
    # Load embedding model if needed
    embedding_model = None
    if args.method in ['embedding', 'both']:
        try:
            print("📥 Loading Qwen3-Embedding-0.6B model...")
            embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            return
    
    # Load processed dataset once (contains ground truth)
    dataset_dir = Path(__file__).resolve().parent / args.dataset_dir
    try:
        dataset = load_processed_dataset(args.dataset, str(dataset_dir))
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Create output directory
    output_root = Path(__file__).resolve().parent / args.output_dir / args.dataset
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    total_evaluations = 0
    successful_llm = 0
    successful_embedding = 0
    failed_evaluations = []
    
    # Process each model/top-k combination
    for combo_idx, (model_filter, top_k_filter) in enumerate(combinations, 1):
        if len(combinations) > 1:
            print(f"\n{'#'*60}")
            print(f"COMBINATION {combo_idx}/{len(combinations)}")
            if model_filter:
                print(f"Model: {model_filter}")
            if top_k_filter is not None:
                print(f"Top-K: {top_k_filter}")
            print(f"{'#'*60}")
        
        # Load and group RAG results for this combination
        try:
            grouped_results = load_and_group_rag_results(
                args.dataset,
                model_filter=model_filter,
                top_k_filter=top_k_filter,
                results_dir=args.results_dir
            )
        except Exception as e:
            print(f"❌ Error loading RAG results: {e}")
            if args.all_models:
                print(f"   Skipping this combination and continuing...")
                continue
            else:
                return
        
        if not grouped_results:
            print(f"⚠️  No RAG results found for this combination")
            if args.all_models:
                print(f"   Skipping and continuing...")
                continue
            else:
                print("❌ Exiting")
                return
        
        # Process each question
        for question, models in grouped_results.items():
            safe_question = sanitize_question_for_filename(question)
            
            print(f"\n{'='*60}")
            print(f"QUESTION: {question[:80]}...")
            print(f"{'='*60}")
            
            # Build ground truth once per question
            try:
                groundtruth_data = build_groundtruth_for_question(dataset, question)
            except Exception as e:
                print(f"❌ Error building ground truth: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Process each model
            for model_name, top_k_dict in models.items():
                print(f"\n📊 Model: {model_name}")
                
                # Process each top-k value for this model
                for top_k, rag_results in sorted(top_k_dict.items()):
                    total_evaluations += 1
                    
                    print(f"\n   ├─ Top-K: {top_k} ({len(rag_results)} result files)")
                    
                    # Aggregate results
                    try:
                        aggregated_results = aggregate_rag_results_to_dict(
                            rag_results, question, dataset
                        )
                        print(f"   ├─ Aggregated into {len(aggregated_results['files'])} document results")
                    except Exception as e:
                        print(f"   ❌ Error aggregating results: {e}")
                        import traceback
                        traceback.print_exc()
                        failed_evaluations.append((question, model_name, top_k, "aggregation", str(e)))
                        continue
                    
                    # Run evaluations
                    if args.method in ['llm', 'both']:
                        llm_output_file = output_root / f"rag-{model_name}-top{top_k}_{safe_question}_{args.llm_model}.json"
                        
                        # Skip if file already exists
                        if llm_output_file.exists():
                            print(f"   ⏭️  LLM evaluation already exists, skipping")
                        else:
                            try:
                                evaluate_results_llm(
                                    aggregated_results,
                                    groundtruth_data,
                                    llm_output_file,
                                    args.dataset,
                                    llm_client
                                )
                                successful_llm += 1
                                print(f"   ✅ LLM evaluation completed")
                            except Exception as e:
                                print(f"   ❌ Error in LLM evaluation: {e}")
                                failed_evaluations.append((question, model_name, top_k, "llm_eval", str(e)))
                    
                    if args.method in ['embedding', 'both']:
                        embedding_output_file = output_root / f"rag-{model_name}-top{top_k}_{safe_question}_embedding.json"
                        
                        # Skip if file already exists
                        if embedding_output_file.exists():
                            print(f"   ⏭️  Embedding evaluation already exists, skipping")
                        else:
                            try:
                                evaluate_results_embedding(
                                    aggregated_results,
                                    groundtruth_data,
                                    embedding_output_file,
                                    embedding_model
                                )
                                successful_embedding += 1
                                print(f"   ✅ Embedding evaluation completed")
                            except Exception as e:
                                print(f"   ❌ Error in embedding evaluation: {e}")
                                failed_evaluations.append((question, model_name, top_k, "embedding_eval", str(e)))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Total model/top-k combinations evaluated: {total_evaluations}")
    if args.method in ['llm', 'both']:
        print(f"Successful LLM evaluations: {successful_llm}/{total_evaluations}")
    if args.method in ['embedding', 'both']:
        print(f"Successful embedding evaluations: {successful_embedding}/{total_evaluations}")
    
    if failed_evaluations:
        print(f"\n⚠️  Failed evaluations: {len(failed_evaluations)}")
        for question, model, top_k, stage, error in failed_evaluations[:5]:
            print(f"   - {question[:40]}... | {model} top-{top_k} (failed at {stage})")
        if len(failed_evaluations) > 5:
            print(f"   ... and {len(failed_evaluations) - 5} more")
    
    print("\n" + "="*60)
    print("✅ RAG EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved in: {args.output_dir}/{args.dataset}/")

if __name__ == "__main__":
    main()