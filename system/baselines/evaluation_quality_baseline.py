#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation Script for Quality Dataset RAG Baselines
Groups evaluation results by DOCUMENT (not by question).

Key differences from evaluation_rag.py:
- Quality dataset specific
- Output: One JSON file per document containing all questions for that document
- Uses evaluation prompts and methods from evaluation.py
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

# Predefined models and top-k values
AVAILABLE_EMBEDDING_MODELS = [
    'UAE-Large-V1',
    'bflhc-Octen-Embedding-4B',
    'Qwen3-Embedding-8B'
]
TOP_K_VALUES = [1, 3, 5, 7, 10]

# Add parent directory to path to import from evaluation.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import evaluation utilities from main evaluation.py
from evaluation import (
    get_evaluation_prompt,
    create_llm_client,
    load_processed_dataset,
)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    print("✅ Embedding libraries initialized")
except ImportError as e:
    print(f"⚠️  Warning: Could not import embedding dependencies: {e}")

# ============================================================================
# QUALITY DATASET RAG RESULT LOADING
# ============================================================================

def parse_rag_filename_quality(filename: str) -> Dict:
    """
    Parse RAG result filename for quality dataset.
    
    Format: results-rag-{model}-top{k}-quality_{question_sanitized}.json
    Example: results-rag-UAE-Large-V1-top5-quality_what_is_the_main_theme.json
    """
    pattern = re.compile(
        r"^results-rag-(?P<model>.+?)-top(?P<topk>\d+)-quality_(?P<question_safe>.+)\.json$"
    )
    
    match = pattern.match(filename)
    if not match:
        return None
    
    return {
        'model': match.group('model'),
        'top_k': int(match.group('topk')),
        'question_safe': match.group('question_safe'),
    }

def load_quality_rag_results(
    model_filter: str = None,
    top_k_filter: int = None,
    results_dir: str = 'baselines/pred_att'
) -> Dict[str, Dict[int, List[Dict]]]:
    """
    Load quality RAG prediction files and group by model and top-k.
    
    Quality dataset structure: each file contains results for ALL documents for ONE question.
    Format: {
        "question": "...",
        "files": [
            {"file": "doc_id", "result": "...", "token_usage": {...}},
            ...
        ]
    }
    
    Args:
        model_filter: Optional embedding model name to filter
        top_k_filter: Optional top-k value to filter
        results_dir: Directory containing RAG prediction files
    
    Returns:
        Dictionary mapping: model -> top_k -> list of result files
    """
    results_path = Path(__file__).resolve().parent / results_dir / 'quality'
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")
    
    print(f"📂 Loading Quality RAG results from: {results_path}")
    
    # Group by model -> top_k
    grouped = defaultdict(lambda: defaultdict(list))
    
    for file_path in sorted(results_path.glob("results-rag-*.json")):
        # Parse filename
        parsed = parse_rag_filename_quality(file_path.name)
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
        
        # Add metadata
        result['_metadata'] = {
            'filename': file_path.name,
            'parsed': parsed,
        }
        
        grouped[model][top_k].append(result)
    
    # Convert to regular dict
    result_dict = {}
    for model, top_ks in grouped.items():
        result_dict[model] = dict(top_ks)
    
    print(f"📋 Loaded results for {len(result_dict)} models")
    for model, top_ks in result_dict.items():
        for top_k, results in top_ks.items():
            print(f"   {model} top-{top_k}: {len(results)} question files")
    
    return result_dict

# ============================================================================
# DOCUMENT-CENTRIC ORGANIZATION
# ============================================================================

def organize_by_document(rag_results: List[Dict], dataset: Dict) -> Dict[str, List[Dict]]:
    """
    Reorganize RAG results from question-centric to document-centric.
    
    Input: List of RAG result files, each containing:
        {
            "question": "...",
            "files": [
                {"file": "2", "result": "...", "token_usage": {...}},
                {"file": "5", "result": "...", "token_usage": {...}},
                ...
            ]
        }
    
    Output: Dictionary mapping document_id -> list of question results
        {
            "doc_2": [
                {"question": "...", "result": "...", "token_usage": {...}},
                ...
            ],
            ...
        }
    
    Args:
        rag_results: List of RAG result dictionaries
        dataset: Processed dataset for mapping doc_id to formatted identifiers
    
    Returns:
        Dictionary mapping document_id to list of question results
    """
    print("\n📦 Organizing results by document...")
    
    # Create mapping from raw doc_id to formatted identifier
    doc_id_to_path = {}
    for doc in dataset.get('documents', []):
        doc_id = str(doc['document_id'])
        original_path = doc.get('metadata', {}).get('original_path', f'doc_{doc_id}')
        doc_id_to_path[doc_id] = original_path
    
    # Group by document
    doc_to_questions = defaultdict(list)
    
    for rag_result in rag_results:
        question = rag_result.get('question', '')
        
        # Each file in the result represents one document's answer to this question
        for file_result in rag_result.get('files', []):
            raw_doc_id = str(file_result.get('file', ''))
            doc_id = doc_id_to_path.get(raw_doc_id, f'doc_{raw_doc_id}')
            
            # Store question result for this document
            doc_to_questions[doc_id].append({
                'question': question,
                'result': file_result.get('result', ''),
                'token_usage': file_result.get('token_usage', {}),
            })
    
    print(f"✅ Organized into {len(doc_to_questions)} documents")
    for doc_id, questions in list(doc_to_questions.items())[:3]:
        print(f"   {doc_id}: {len(questions)} questions")
    if len(doc_to_questions) > 3:
        print(f"   ... and {len(doc_to_questions) - 3} more documents")
    
    return dict(doc_to_questions)

# ============================================================================
# GROUND TRUTH EXTRACTION
# ============================================================================

def get_document_ground_truth(dataset: Dict, doc_id: str) -> List[Dict]:
    """
    Get all ground truth questions and answers for a specific document.
    
    Args:
        dataset: Processed dataset
        doc_id: Document identifier (e.g., "doc_2")
    
    Returns:
        List of question/answer pairs:
        [
            {"question": "...", "answer": "A"},
            {"question": "...", "answer": "C"},
            ...
        ]
    """
    # Find the document
    for doc in dataset.get('documents', []):
        raw_doc_id = str(doc['document_id'])
        original_path = doc.get('metadata', {}).get('original_path', f'doc_{raw_doc_id}')
        
        if original_path == doc_id:
            # Return all questions for this document
            return [
                {
                    'question': q['question'],
                    'answer': q.get('answer', '')
                }
                for q in doc.get('questions', [])
            ]
    
    return []

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_document_llm(
    doc_id: str,
    question_results: List[Dict],
    ground_truth_list: List[Dict],
    llm_client,
    model_name: str,
    top_k: int
) -> Dict:
    """
    Evaluate all questions for a single document using LLM.
    
    Args:
        doc_id: Document identifier
        question_results: List of question results for this document
        ground_truth_list: List of ground truth question/answer pairs
        llm_client: LLM client for evaluation
        model_name: Embedding model name
        top_k: Top-k value
    
    Returns:
        Evaluation results for this document
    """
    print(f"\n📄 Evaluating document: {doc_id}")
    print(f"   Questions to evaluate: {len(question_results)}")
    
    # Create mapping from question to ground truth
    question_to_gt = {gt['question']: gt['answer'] for gt in ground_truth_list}
    
    evaluated_questions = []
    total_score = 0
    correct_count = 0
    
    for q_result in question_results:
        question = q_result['question']
        prediction = q_result['result']
        token_usage = q_result['token_usage']
        
        # Get ground truth
        ground_truth = question_to_gt.get(question)
        if not ground_truth:
            print(f"   ⚠️  No ground truth found for question: {question[:50]}...")
            continue
        
        # Get evaluation prompt
        evaluation_prompt = get_evaluation_prompt(
            dataset_name='quality',
            question=question,
            ground_truth=ground_truth,
            prediction=prediction
        )
        
        # Run LLM evaluation
        score, rationale = llm_client.evaluate(evaluation_prompt)
        
        # For quality dataset: score is 10 (correct) or 0 (incorrect)
        match = score >= 7
        if match:
            correct_count += 1
        total_score += score
        
        print(f"   {'✓' if match else '✗'} Q: {question[:60]}... | GT: {ground_truth} | Score: {score}")
        
        evaluated_questions.append({
            'question': question,
            'groundtruth': ground_truth,
            'result': prediction,
            'score': score,
            'match': match,
            'rationale': rationale,
            'tokens_extracted': token_usage.get('context_tokens'),
            'total_tokens': token_usage.get('document_total_tokens'),
        })
    
    # Calculate summary statistics
    total_questions = len(evaluated_questions)
    accuracy = correct_count / total_questions if total_questions > 0 else 0
    avg_score = total_score / total_questions if total_questions > 0 else 0
    
    return {
        'document_id': doc_id,
        'model': model_name,
        'top_k': top_k,
        'questions': evaluated_questions,
        'summary': {
            'total_questions': total_questions,
            'correct_answers': correct_count,
            'accuracy': accuracy,
            'average_score': avg_score,
        }
    }

def evaluate_document_embedding(
    doc_id: str,
    question_results: List[Dict],
    ground_truth_list: List[Dict],
    embedding_model,
    model_name: str,
    top_k: int
) -> Dict:
    """
    Evaluate all questions for a single document using embedding similarity.
    
    Args:
        doc_id: Document identifier
        question_results: List of question results for this document
        ground_truth_list: List of ground truth question/answer pairs
        embedding_model: Sentence transformer model
        model_name: Embedding model name
        top_k: Top-k value
    
    Returns:
        Evaluation results for this document
    """
    print(f"\n📄 Evaluating document: {doc_id}")
    print(f"   Questions to evaluate: {len(question_results)}")
    
    # Create mapping from question to ground truth
    question_to_gt = {gt['question']: gt['answer'] for gt in ground_truth_list}
    
    evaluated_questions = []
    total_similarity = 0
    
    for q_result in question_results:
        question = q_result['question']
        prediction = q_result['result']
        token_usage = q_result['token_usage']
        
        # Get ground truth
        ground_truth = question_to_gt.get(question)
        if not ground_truth:
            print(f"   ⚠️  No ground truth found for question: {question[:50]}...")
            continue
        
        # Compute embeddings
        embeddings = embedding_model.encode([ground_truth, prediction])
        
        # Calculate cosine similarity
        cos_sim = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]
        
        # Clamp to [0, 1]
        similarity = max(0.0, min(1.0, float(cos_sim)))
        total_similarity += similarity
        
        print(f"   Q: {question[:60]}... | Similarity: {similarity:.4f}")
        
        evaluated_questions.append({
            'question': question,
            'groundtruth': ground_truth,
            'result': prediction,
            'similarity': similarity,
            'tokens_extracted': token_usage.get('context_tokens'),
            'total_tokens': token_usage.get('document_total_tokens'),
        })
    
    # Calculate summary statistics
    total_questions = len(evaluated_questions)
    avg_similarity = total_similarity / total_questions if total_questions > 0 else 0
    
    return {
        'document_id': doc_id,
        'model': model_name,
        'top_k': top_k,
        'questions': evaluated_questions,
        'summary': {
            'total_questions': total_questions,
            'average_similarity': avg_similarity,
        }
    }

# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate Quality dataset RAG results (grouped by document)')
    
    parser.add_argument('--model', type=str, default=None,
                       help='Filter by embedding model name')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Filter by top-k value')
    parser.add_argument('--all-models', action='store_true',
                       help='Evaluate all predefined models and top-k combinations')
    parser.add_argument('--method', type=str, default='llm',
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
        combinations = []
        for model in AVAILABLE_EMBEDDING_MODELS:
            for top_k in TOP_K_VALUES:
                combinations.append((model, top_k))
        print(f"\n🔄 --all-models flag set: Will evaluate {len(combinations)} combinations")
    else:
        if args.model or args.top_k is not None:
            combinations = [(args.model, args.top_k)]
        else:
            combinations = [(None, None)]
    
    print("\n" + "="*60)
    print("QUALITY DATASET RAG EVALUATION (GROUPED BY DOCUMENT)")
    print("="*60)
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
    
    # Load quality dataset
    dataset_dir = Path(__file__).resolve().parent / args.dataset_dir
    try:
        dataset = load_processed_dataset('quality', str(dataset_dir))
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Create output directory
    output_root = Path(__file__).resolve().parent / args.output_dir / 'quality'
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    total_documents = 0
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
        
        # Load RAG results
        try:
            rag_results = load_quality_rag_results(
                model_filter=model_filter,
                top_k_filter=top_k_filter,
                results_dir=args.results_dir
            )
        except Exception as e:
            print(f"❌ Error loading RAG results: {e}")
            if args.all_models:
                continue
            else:
                return
        
        if not rag_results:
            print(f"⚠️  No RAG results found")
            if args.all_models:
                continue
            else:
                return
        
        # Process each model and top-k
        for model_name, top_k_dict in rag_results.items():
            for top_k, result_files in sorted(top_k_dict.items()):
                print(f"\n{'='*60}")
                print(f"Processing: {model_name} with top-{top_k}")
                print(f"{'='*60}")
                
                # Organize results by document
                try:
                    doc_to_questions = organize_by_document(result_files, dataset)
                except Exception as e:
                    print(f"❌ Error organizing by document: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Evaluate each document
                for doc_id, question_results in doc_to_questions.items():
                    total_documents += 1
                    
                    # Get ground truth for this document
                    ground_truth_list = get_document_ground_truth(dataset, doc_id)
                    if not ground_truth_list:
                        print(f"⚠️  No ground truth found for {doc_id}")
                        continue
                    
                    # Run evaluations
                    if args.method in ['llm', 'both']:
                        llm_output_file = output_root / f"rag-{model_name}-top{top_k}_{doc_id}_{args.llm_model}.json"
                        
                        if llm_output_file.exists():
                            print(f"   ⏭️  LLM evaluation already exists for {doc_id}, skipping")
                        else:
                            try:
                                eval_result = evaluate_document_llm(
                                    doc_id, question_results, ground_truth_list,
                                    llm_client, model_name, top_k
                                )
                                
                                # Save result
                                with open(llm_output_file, 'w') as f:
                                    json.dump(eval_result, f, indent=4)
                                
                                successful_llm += 1
                                print(f"   ✅ LLM evaluation saved: {llm_output_file.name}")
                                print(f"      Accuracy: {eval_result['summary']['accuracy']:.2%} "
                                      f"({eval_result['summary']['correct_answers']}/{eval_result['summary']['total_questions']})")
                            except Exception as e:
                                print(f"   ❌ Error in LLM evaluation for {doc_id}: {e}")
                                failed_evaluations.append((doc_id, model_name, top_k, "llm_eval", str(e)))
                    
                    if args.method in ['embedding', 'both']:
                        embedding_output_file = output_root / f"rag-{model_name}-top{top_k}_{doc_id}_embedding.json"
                        
                        if embedding_output_file.exists():
                            print(f"   ⏭️  Embedding evaluation already exists for {doc_id}, skipping")
                        else:
                            try:
                                eval_result = evaluate_document_embedding(
                                    doc_id, question_results, ground_truth_list,
                                    embedding_model, model_name, top_k
                                )
                                
                                # Save result
                                with open(embedding_output_file, 'w') as f:
                                    json.dump(eval_result, f, indent=4)
                                
                                successful_embedding += 1
                                print(f"   ✅ Embedding evaluation saved: {embedding_output_file.name}")
                                print(f"      Avg Similarity: {eval_result['summary']['average_similarity']:.4f}")
                            except Exception as e:
                                print(f"   ❌ Error in embedding evaluation for {doc_id}: {e}")
                                failed_evaluations.append((doc_id, model_name, top_k, "embedding_eval", str(e)))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    print(f"Total documents evaluated: {total_documents}")
    if args.method in ['llm', 'both']:
        print(f"Successful LLM evaluations: {successful_llm}/{total_documents}")
    if args.method in ['embedding', 'both']:
        print(f"Successful embedding evaluations: {successful_embedding}/{total_documents}")
    
    if failed_evaluations:
        print(f"\n⚠️  Failed evaluations: {len(failed_evaluations)}")
        for doc_id, model, top_k, stage, error in failed_evaluations[:5]:
            print(f"   - {doc_id} | {model} top-{top_k} (failed at {stage})")
        if len(failed_evaluations) > 5:
            print(f"   ... and {len(failed_evaluations) - 5} more")
    
    print("\n" + "="*60)
    print("✅ QUALITY EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved in: {args.output_dir}/quality/")
    print("📁 Output format: One JSON file per document with all questions evaluated")

if __name__ == "__main__":
    main()