#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit Window for Attention-Based Context Selection
-------------------------------------------------
Uses pre-computed attention scores to select context windows and generate answers.

This script assumes preprocessing and differential computation have been completed
by data_reader.py.

IMPORTANT: Must use the same model/tokenizer as preprocessing!

Usage:
    # Generate results using raw attention (with Llama tokenizer - default)
    python unit_window.py --dataset paper --use-raw --model llama3.2_1b
    
    # Generate results using Qwen3-8B tokenizer (must match preprocessing)
    python unit_window.py --dataset paper --use-farest --model qwen3_8b
    
    # Generate results using Qwen3-14B tokenizer (must match preprocessing)
    python unit_window.py --dataset paper --use-farest --model qwen3_14b
    
    # Specify custom budgets
    python unit_window.py --dataset paper --use-raw --budgets 0.01 0.05 0.10 0.50 --model llama3.2_1b
"""

import sys
import os
import warnings
import logging
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import time
import google.generativeai as genai

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['ACCELERATE_LOG_LEVEL'] = 'error'

class NoFlushStreamHandler(logging.StreamHandler):
    def flush(self):
        try:
            super().flush()
        except (PermissionError, OSError):
            pass

logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s',
    handlers=[NoFlushStreamHandler(sys.stdout)]
)

warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

from transformers import AutoTokenizer

tokenizer = None

# Model name mapping (must match data_reader.py) - UPDATED WITH FULL NAMES
MODEL_MAPPING = {
    'llama3.2_1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'qwen3_8b': 'Qwen/Qwen3-8B',
    'qwen3_14b': 'Qwen/Qwen3-14B'
}

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_tokenizer(model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
    """Initialize tokenizer for decoding."""
    global tokenizer
    
    if tokenizer is not None:
        print("✅ Tokenizer already initialized.")
        return
    
    print(f"📥 Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir="/tmp/huggingface-cache", trust_remote_code=True
    )
    print("✅ Tokenizer initialized successfully.")

def initialize_gemini():
    """Initialize Gemini API."""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ Gemini API configured")
    else:
        print("⚠️ Warning: GEMINI_API_KEY not found in environment variables")
        print("   Set it with: export GEMINI_API_KEY='your-api-key'")

# ============================================================================
# WINDOW-BASED CONTEXT SELECTION (FIXED 2% WINDOW WITH MERGING)
# ============================================================================

def apply_budget_to_attention(attention_scores: np.ndarray, tokens: List[str], 
                              budget: float) -> Tuple[List[List[str]], List[str]]:
    """
    Selects windows using a fixed 2% window size with adaptive merging strategy.
    
    Strategy:
    - Window size is fixed at 2% of document length
    - If budget <= 2%: Return single best window
    - If budget > 2%: Select windows in descending score order, merging overlaps
    
    Args:
        attention_scores: Attention scores for each token
        tokens: List of tokens
        budget: Percentage of tokens to select (total budget)
    
    Returns:
        Tuple of (list_of_token_lists, list_of_decoded_texts)
    """
    global tokenizer
    if tokenizer is None: 
        initialize_tokenizer()

    if attention_scores.size == 0 or not tokens:
        return [[]], [""]

    if budget >= 0.02:
        # Fixed window size at 2% of document length if budget is >= 0.02
        window_size = max(1, int(len(attention_scores) * 0.02))
        total_budget_tokens = max(10, int(len(attention_scores) * budget))
    else:
        window_size = max(1, int(len(attention_scores) * budget))
        total_budget_tokens = max(1, int(len(attention_scores) * budget))

    
    print(f"    - Fixed window size: {window_size} tokens (2% of {len(attention_scores)})")
    print(f"    - Total budget: {total_budget_tokens} tokens ({budget*100:.1f}%)")
    
    # Special case: if budget <= 2%, return single best window
    if budget <= 0.02:
        if window_size >= len(attention_scores):
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            # ✅ FIX: Filter out None values before decoding
            token_ids = [tid for tid in token_ids if tid is not None]
            if not token_ids:
                return [tokens], [""]
            return [tokens], [tokenizer.decode(token_ids)]
        
        # Find single best window
        best_score = -float('inf')
        best_start = 0
        
        for i in range(len(attention_scores) - window_size + 1):
            window_sum = np.sum(attention_scores[i:i + window_size])
            if window_sum > best_score:
                best_score = window_sum
                best_start = i
        
        selected_tokens = tokens[best_start:best_start + window_size]
        token_ids = tokenizer.convert_tokens_to_ids(selected_tokens)
        # ✅ FIX: Filter out None values before decoding
        token_ids = [tid for tid in token_ids if tid is not None]
        if not token_ids:
            selected_text = ""
        else:
            selected_text = tokenizer.decode(token_ids)
        
        print(f"    - Budget ≤ 2%: Selected single best window at position {best_start}-{best_start + window_size}")
        return [selected_tokens], [selected_text]
    
    # For budget > 2%: Use sliding window with merging strategy
    if window_size >= len(attention_scores):
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        # ✅ FIX: Filter out None values before decoding
        token_ids = [tid for tid in token_ids if tid is not None]
        if not token_ids:
            return [tokens], [""]
        return [tokens], [tokenizer.decode(token_ids)]
    
    # Calculate scores for all possible windows
    window_scores = []
    for i in range(len(attention_scores) - window_size + 1):
        window_sum = np.sum(attention_scores[i:i + window_size])
        window_avg = window_sum / window_size
        window_scores.append({
            'start_idx': i,
            'end_idx': i + window_size,
            'avg_score': window_avg
        })
    
    # Sort by average score (highest first)
    window_scores.sort(key=lambda x: x['avg_score'], reverse=True)
    
    # Greedy selection with merging
    selected_ranges = []  # List of (start, end) tuples
    total_tokens_selected = 0
    
    for candidate in window_scores:
        if total_tokens_selected >= total_budget_tokens:
            break
        
        cand_start = candidate['start_idx']
        cand_end = candidate['end_idx']
        
        # Check for overlap with any selected range
        overlapping_indices = []
        for idx, (sel_start, sel_end) in enumerate(selected_ranges):
            if cand_start < sel_end and cand_end > sel_start:  # Overlap detected
                overlapping_indices.append(idx)
        
        if overlapping_indices:
            # Merge with all overlapping ranges
            ranges_to_merge = [selected_ranges[idx] for idx in overlapping_indices]
            ranges_to_merge.append((cand_start, cand_end))
            
            # Find the merged span
            merged_start = min(r[0] for r in ranges_to_merge)
            merged_end = max(r[1] for r in ranges_to_merge)
            
            # Calculate tokens from old overlapping ranges
            old_tokens = sum(selected_ranges[idx][1] - selected_ranges[idx][0] 
                           for idx in overlapping_indices)
            
            # Remove old overlapping ranges (reverse order to preserve indices)
            for idx in sorted(overlapping_indices, reverse=True):
                del selected_ranges[idx]
            
            # Add merged range
            selected_ranges.append((merged_start, merged_end))
            
            # Update total tokens
            new_tokens = merged_end - merged_start
            total_tokens_selected = total_tokens_selected - old_tokens + new_tokens
            
            print(f"    - Merged: {cand_start}-{cand_end} → {merged_start}-{merged_end} "
                  f"(total: {total_tokens_selected} tokens)")
        else:
            # No overlap - add as new range
            new_tokens = cand_end - cand_start
            
            # Check if adding this would exceed budget
            if total_tokens_selected + new_tokens > total_budget_tokens:
                # Truncate the window to fit remaining budget
                available_tokens = total_budget_tokens - total_tokens_selected
                if available_tokens > 0:
                    truncated_end = cand_start + available_tokens
                    selected_ranges.append((cand_start, truncated_end))
                    total_tokens_selected += available_tokens
                    print(f"    - Added truncated window: {cand_start}-{truncated_end} "
                          f"(budget reached: {total_tokens_selected}/{total_budget_tokens})")
                break
            else:
                selected_ranges.append((cand_start, cand_end))
                total_tokens_selected += new_tokens
                print(f"    - Added new window: {cand_start}-{cand_end} "
                      f"(total: {total_tokens_selected} tokens)")
    
    # Sort ranges by start position to maintain document order
    selected_ranges.sort(key=lambda x: x[0])
    
    # Extract tokens and decode each range
    selected_token_lists = []
    selected_texts = []
    
    for start, end in selected_ranges:
        range_tokens = tokens[start:end]
        token_ids = tokenizer.convert_tokens_to_ids(range_tokens)
        
        # ✅ FIX: Filter out None values before decoding
        token_ids = [tid for tid in token_ids if tid is not None]
        
        if not token_ids:
            range_text = ""
        else:
            range_text = tokenizer.decode(token_ids)
        
        selected_token_lists.append(range_tokens)
        selected_texts.append(range_text)
    
    print(f"    - Final selection: {len(selected_ranges)} range(s), "
          f"{total_tokens_selected}/{total_budget_tokens} tokens ({total_tokens_selected/len(attention_scores)*100:.1f}%)")
    
    return selected_token_lists, selected_texts

# ============================================================================
# ANSWER GENERATION (NO KV CACHE)
# ============================================================================

def generate_answer_with_rationale(context: str, question: str, 
                                   doc_id: str = None) -> Tuple[str, str]:
    """
    Generate an answer from the provided (reduced) context using Gemini API.
    
    Args:
        context: The selected context text (can be a list or string)
        question: The question to answer
        doc_id: Document ID (unused, kept for backward compatibility)
    
    Returns:
        Tuple of (answer, rationale)
    """
    try:
        # Convert context to string if it's a list
        if isinstance(context, list):
            context_text = "\n".join(context)
        else:
            context_text = context
        
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        prompt = f"""Based on the following context, answer the question concisely.

Context: {context_text}

Question: {question}

Answer:"""
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,  # Deterministic output
                max_output_tokens=150,
            )
        )
        
        # Extract answer
        answer = response.text.strip()
        
        # Create rationale
        rationale = f"Generated answer using Gemini API without KV cache from a reduced context of {len(context_text.split())} words."

        return answer if answer else "None", rationale
        
    except Exception as e:
        print(f"❌ Error during Gemini API call: {e}")
        return "None", f"Error occurred: {str(e)}"

# ============================================================================
# PERFORMANCE REPORT - REMOVED
# ============================================================================
# Performance report generation has been removed

# ============================================================================
# DOCUMENT SUMMARY REPORT (DEPRECATED - keeping for backward compatibility)
# ============================================================================

def save_document_summary(dataset_name: str, doc_id: int, attention_type: str,
                         budget: float, question_timings: List[Dict],
                         output_dir: Path):
    """
    Save a per-document summary report with timing for all queries.
    
    DEPRECATED: Use save_performance_report instead.
    This function is kept for backward compatibility but is no longer called.
    """
    
    # Calculate summary statistics
    total_time = sum(q['timing']['total'] for q in question_timings)
    total_load_time = sum(q['timing']['load_attention'] for q in question_timings)
    total_budget_time = sum(q['timing']['apply_budget'] for q in question_timings)
    total_generate_time = sum(q['timing']['generate_answer'] for q in question_timings)
    
    avg_time_per_question = total_time / len(question_timings) if question_timings else 0
    
    summary = {
        "dataset": dataset_name,
        "document_id": doc_id,
        "attention_type": attention_type,
        "budget": budget,
        "total_questions": len(question_timings),
        "questions": question_timings,
        "summary": {
            "total_time": round(total_time, 3),
            "total_load_attention": round(total_load_time, 3),
            "total_apply_budget": round(total_budget_time, 3),
            "total_generate_answer": round(total_generate_time, 3),
            "average_time_per_question": round(avg_time_per_question, 3)
        }
    }
    
    # Save summary file
    summary_file = output_dir / f"summary_doc{doc_id}_{attention_type}_{budget:.3f}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"   Total questions: {len(question_timings)}")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Avg per question: {avg_time_per_question:.3f}s")
    print(f"   Generation time: {total_generate_time:.3f}s")
    print(f"   Saved to: {summary_file.name}")

# ============================================================================
# RESULT GENERATION PIPELINES
# ============================================================================

def generate_results_for_attention_type(dataset_name: str, dataset: Dict, 
                                       attention_type: str, budgets: List[float],
                                       token_dir: str, attention_dir: str, 
                                       output_dir: str):
    """
    Generate evaluation results for a specific attention type.
    
    Args:
        dataset_name: Name of the dataset
        dataset: Dataset dictionary
        attention_type: One of 'raw', 'farest', 'baseline'
        budgets: List of budget percentages
        token_dir: Directory containing token files
        attention_dir: Directory containing attention files
        output_dir: Directory to save results
    """
    print(f"\n{'='*60}")
    print(f"GENERATING RESULTS: {attention_type.upper()} attention")
    print(f"{'='*60}")
    
    # Initialize
    initialize_gemini()
    
   
    dataset_output_dir = Path(output_dir) / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine attention file suffix
    if attention_type == 'raw':
        attention_suffix = ''
    elif attention_type == 'farest':
        attention_suffix = '_farest'
    elif attention_type == 'baseline':
        attention_suffix = '_baseline'
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
    
    total_processed = 0
    total_skipped = 0
    
    # Process each document
    for doc in dataset['documents']:
        doc_id = doc['document_id']
        
        # Load tokens once per document
        token_file = Path(token_dir) / f"{dataset_name}_{doc_id}.json"
        if not token_file.exists():
            print(f"⚠️ Missing token file for document {doc_id}")
            total_skipped += len(doc['questions']) * len(budgets)
            continue
        
        with open(token_file, 'r') as f:
            tokens = json.load(f)
        
        # Process each budget
        for budget in budgets:
            print(f"\n{'='*60}")
            print(f"📄 Document {doc_id} - Budget {budget:.3f}")
            print(f"{'='*60}")

            # Process each question
            for question_data in doc['questions']:
                question_id = question_data['question_id']
                question = question_data['question']
                
                print(f"\n🔄 Processing: Q: {question[:40]}..., Budget: {budget:.3f}")
                
                # Create results filename with truncated question text
                safe_question = question.replace("?", "").replace("/", "_").replace(" ", "_")
                # Truncate if too long to avoid filesystem limits (typical limit ~255 chars for filename)
                # Reserve space for prefix, keeping question part to max 80 chars
                max_question_length = 80
                if len(safe_question) > max_question_length:
                    safe_question = safe_question[:max_question_length]
                results_file = dataset_output_dir / f"results-{attention_type}-{budget:.3f}-{dataset_name}_{doc_id}_{safe_question}.json"
                
                if results_file.exists():
                    print(f"    ✅ Results file already exists. Skipping.")
                    continue
                
                try:
                    file_start_time = time.time()
                    
                    # Load pre-computed attention scores
                    attention_file = Path(attention_dir) / f"{dataset_name}_{question_id}{attention_suffix}.npy"
                    
                    if not attention_file.exists():
                        print(f"    ⚠️ Missing attention file: {attention_file.name}")
                        total_skipped += 1
                        continue
                    
                    load_start = time.time()
                    attention_scores = np.load(attention_file)
                    load_time = time.time() - load_start
                    
                    # Apply budget to select context
                    budget_start = time.time()
                    selected_token_lists, selected_texts = apply_budget_to_attention(
                        attention_scores, tokens, budget
                    )
                    total_tokens_extracted = sum(len(token_list) for token_list in selected_token_lists)
                    budget_time = time.time() - budget_start
                    
                    # Generate answer
                    generate_start = time.time()
                    answer, rationale = generate_answer_with_rationale(
                        selected_texts, 
                        question,
                        doc_id=doc_id
                    )
                    print(f"    - Generated answer: {answer}")
                    generate_time = time.time() - generate_start
                    
                    file_total_time = time.time() - file_start_time
                    
                    # Create timing dictionary
                    timing_dict = {
                        "total": file_total_time,
                        "load_attention": load_time,
                        "apply_budget": budget_time,
                        "generate_answer": generate_time
                    }
                    
                    # Save results
                    result = {
                        "dataset": dataset_name,
                        "document_id": doc_id,
                        "question_id": question_id,
                        "question": question,
                        "budget": budget,
                        "attention_type": attention_type,
                        "use_kv_cache": False,
                        "result": answer,
                        "rationale": rationale,
                        "timing": timing_dict,
                        "tokens_extracted": total_tokens_extracted,
                        "total_tokens": len(tokens)
                    }
                    
                    with open(results_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    print(f"    ✅ Saved results to: {results_file.name}")
                    total_processed += 1
                    
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    total_skipped += 1
    
    print(f"\n📊 {attention_type.upper()} SUMMARY:")
    print(f"   ✅ Processed: {total_processed}")
    print(f"   ⚠️  Skipped: {total_skipped}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unit window for attention-based context selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate results using raw attention with Llama tokenizer (default)
  python unit_window.py --dataset paper --use-raw --model llama3.2_1b
  
  # Generate results using farest differential with Qwen3-8B tokenizer
  python unit_window.py --dataset notice --use-farest --model qwen3_8b
  
  # Generate results using farest differential with Qwen3-14B tokenizer
  python unit_window.py --dataset notice --use-farest --model qwen3_14b
  
  # Generate results using all attention types
  python unit_window.py --dataset paper --use-raw --use-farest --use-baseline --model llama3.2_1b
  
  # Specify custom budgets
  python unit_window.py --dataset paper --use-raw --budgets 0.01 0.05 0.10 0.50 --model llama3.2_1b
        """
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing processed data')
    parser.add_argument('--output-dir', type=str, default='pred_att',
                       help='Directory to save evaluation results')
    
    # ✅ UPDATED: Model selection with full names (must match preprocessing)
    parser.add_argument('--model', type=str,
                       default="llama3.2_1b",
                       choices=["llama3.2_1b", "qwen3_8b", "qwen3_14b"],
                       help='Model to use for tokenization (MUST match preprocessing!)')
    
    # Attention type selection
    parser.add_argument('--use-raw', action='store_true',
                       help='Use raw attention scores')
    parser.add_argument('--use-farest', action='store_true',
                       help='Use farest question differential attention')
    parser.add_argument('--use-baseline', action='store_true',
                       help='Use baseline differential attention')
    
    # Budget configuration
    parser.add_argument('--budgets', type=float, nargs='+',
                       default=[0.005, 0.010, 0.050, 0.100, 0.150, 0.400],
                       help='List of budget percentages (e.g., 0.01 0.05, 0.15 0.10)')
    
    args = parser.parse_args()
    
    # ✅ Resolve model name from mapping
    model_name = MODEL_MAPPING.get(args.model, args.model)
    
    print("="*60)
    print("UNIT WINDOW FOR ATTENTION-BASED CONTEXT SELECTION")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model} ({model_name})")
    print(f"Budgets: {args.budgets}")
    print(f"Attention types:")
    print(f"  - Raw: {args.use_raw}")
    print(f"  - Farest: {args.use_farest}")
    print(f"  - Baseline: {args.use_baseline}")
    
    # ✅ WARNING: Check model consistency
    print(f"\n⚠️  IMPORTANT: Ensure preprocessing was done with the same model!")
    print(f"   Current model: {args.model} ({model_name})")
    print(f"   If you preprocessed with a different model, re-run data_reader.py")
    
    # Check that at least one attention type is selected
    if not (args.use_raw or args.use_farest or args.use_baseline):
        print("\n❌ ERROR: No attention type selected!")
        print("   Use at least one of: --use-raw, --use-farest, --use-baseline")
        return
    
    # ✅ Setup directories with model prefix
    base_data_dir = os.path.join(args.model, args.data_dir)
    token_dir = os.path.join(base_data_dir, 'tokens')
    attention_dir = os.path.join(base_data_dir, 'attention_summary')
    dataset_dir = os.path.join(base_data_dir, 'datasets')
    
    # ✅ Output directory with model prefix
    base_output_dir = os.path.join(args.model, args.output_dir)
    
    print(f"\n📂 Directory structure:")
    print(f"   Input data:  {base_data_dir}/")
    print(f"   Tokens:      {token_dir}")
    print(f"   Attention:   {attention_dir}")
    print(f"   Dataset:     {dataset_dir}")
    print(f"   Output:      {base_output_dir}/")
    
    # Check directories exist
    if not os.path.exists(token_dir):
        print(f"\n❌ ERROR: Token directory not found: {token_dir}")
        print(f"   Expected directory: {args.model}/data/tokens/")
        print("   Run data_reader.py first with the same --model argument")
        return
    
    if not os.path.exists(attention_dir):
        print(f"\n❌ ERROR: Attention directory not found: {attention_dir}")
        print(f"   Expected directory: {args.model}/data/attention_summary/")
        print("   Run data_reader.py first with the same --model argument")
        return
    
    # Load processed dataset
    dataset_file = os.path.join(dataset_dir, f'{args.dataset}_processed.json')
    if not os.path.exists(dataset_file):
        print(f"\n❌ ERROR: Processed dataset not found: {dataset_file}")
        print(f"   Expected file: {args.model}/data/datasets/{args.dataset}_processed.json")
        print("   Run data_reader.py first with the same --model and --dataset arguments")
        return
    
    print(f"\n📖 Loading processed dataset from: {dataset_file}")
    with open(dataset_file) as f:
        dataset = json.load(f)
    
    print(f"✅ Loaded dataset: {dataset['dataset_name']}")
    print(f"   Documents: {len(dataset['documents'])}")
    
    # ✅ Initialize tokenizer with the correct model
    print(f"\n📥 Initializing tokenizer for {model_name}...")
    initialize_tokenizer(model_name)
    
    # Generate results for each selected attention type
    if args.use_raw:
        generate_results_for_attention_type(
            args.dataset, dataset, 'raw', args.budgets,
            token_dir, attention_dir, base_output_dir
        )
    
    if args.use_farest:
        generate_results_for_attention_type(
            args.dataset, dataset, 'farest', args.budgets,
            token_dir, attention_dir, base_output_dir
        )
    
    if args.use_baseline:
        generate_results_for_attention_type(
            args.dataset, dataset, 'baseline', args.budgets,
            token_dir, attention_dir, base_output_dir
        )
    
    print("\n" + "="*60)
    print(f"✅ COMPLETE: All results generated for '{args.dataset}'")
    print("="*60)
    print(f"📁 Results saved in: {os.path.join(base_output_dir, args.dataset)}")

if __name__ == "__main__":
    main()