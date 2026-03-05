#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot CDF comparison between Context Reduction and RAG baselines.
Supports multiple models, attention types, budgets, and RAG configurations.
For each combination, creates a separate output folder with all question plots.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from collections import defaultdict
import re
import os

MODEL_MAPPING = {
    'llama3.2_1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'qwen3_8b': 'Qwen/Qwen3-8B',
    'qwen3_14b': 'Qwen/Qwen3-14B'
}

# RAG model name mapping for file matching
RAG_MODEL_PATTERNS = {
    'Qwen3-Embedding-8B': 'Qwen3-Embedding-8B',
    'Qwen3-Embedding-14B': 'Qwen3-Embedding-14B',
    'qwen3_8b': 'Qwen3-Embedding-8B',
    'qwen3_14b': 'Qwen3-Embedding-14B',
}

def sanitize_question_for_filename(question: str, max_length: int = 80) -> str:
    """Sanitize question text for use in filename."""
    safe = question.lower()
    safe = re.sub(r'[^\w\s-]', '', safe)
    safe = re.sub(r'[-\s]+', '_', safe)
    safe = safe.strip('_')
    if len(safe) > max_length:
        safe = safe[:max_length].rstrip('_')
    if not safe:
        safe = "unknown_question"
    return safe


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


def extract_similarity_scores(file_data: Dict) -> List[float]:
    """
    Extract similarity scores from evaluation file.
    
    Returns:
        List of similarity scores (0-1 range), clamped to [0, 1]
    """
    documents = file_data.get('files', [])
    similarity_scores = []
    
    for doc in documents:
        similarity = doc.get('similarity')
        if similarity is not None:
            try:
                score = float(similarity)
                # Clamp to [0, 1]
                score = max(0.0, min(1.0, score))
                similarity_scores.append(score)
            except (TypeError, ValueError):
                pass
    
    return similarity_scores


def plot_cdf(data: List[float], label: str, color=None, linestyle='-', marker=''):
    """
    Plot complementary CDF: P(X > x).
    
    Args:
        data: List of values
        label: Label for the line
        color: Line color
        linestyle: Line style
        marker: Marker style
    """
    if not data:
        return
    
    # Sort the data in ascending order
    sorted_data = np.sort(data)
    
    # Calculate the complementary CDF values: P(X > x)
    cdf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Append the maximum x value to extend line to 0
    sorted_data = np.append(sorted_data, sorted_data[-1])
    cdf = np.append(cdf, 0)
    
    # Plot the CDF
    plt.plot(sorted_data, cdf, label=label, color=color, linestyle=linestyle, 
             marker=marker, markersize=4, linewidth=2)


def load_single_cr_file(
    dataset: str,
    model: str,
    attention_type: str,
    budget: float,
    question: str,
    evaluation_dir: str = "evaluation"
) -> Optional[Dict]:
    """
    Load a single context reduction evaluation file.
    
    Returns:
        File data dict if found, None otherwise
    """
    eval_path = Path(model) / evaluation_dir / dataset
    
    # Pattern: {attention_type}_budget_{budget:.3f}_{question}_embedding.json
    safe_question = sanitize_question_for_filename(question)
    filename = f"{attention_type}_budget_{budget:.3f}_{safe_question}_embedding.json"
    file_path = eval_path / filename
    
    # Also try with different question sanitization
    if not file_path.exists():
        # Try pattern matching
        pattern = f"{attention_type}_budget_{budget:.3f}_*_embedding.json"
        candidates = list(eval_path.glob(pattern))
        
        for candidate in candidates:
            try:
                with open(candidate) as f:
                    file_data = json.load(f)
                if file_data.get('question', '').strip().lower() == question.strip().lower():
                    return file_data
            except Exception:
                continue
        return None
    
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        return None


def load_single_rag_file(
    dataset: str,
    rag_model: str,
    top_k: int,
    question: str,
    evaluation_dir: str = "baselines/evaluation"
) -> Optional[Dict]:
    """
    Load a single RAG evaluation file.
    
    Returns:
        File data dict if found, None otherwise
    """
    eval_path = Path(evaluation_dir) / dataset
    
    # Get the model pattern to match in filenames
    model_pattern = RAG_MODEL_PATTERNS.get(rag_model, rag_model)
    
    # Pattern: rag-{model}-top{k}_{question}_embedding.json
    safe_question = sanitize_question_for_filename(question)
    filename = f"rag-{model_pattern}-top{top_k}_{safe_question}_embedding.json"
    file_path = eval_path / filename
    
    # Also try pattern matching
    if not file_path.exists():
        pattern = f"rag-{model_pattern}-top{top_k}_*_embedding.json"
        candidates = list(eval_path.glob(pattern))
        
        for candidate in candidates:
            try:
                with open(candidate) as f:
                    file_data = json.load(f)
                if file_data.get('question', '').strip().lower() == question.strip().lower():
                    return file_data
            except Exception:
                continue
        return None
    
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        return None


def discover_available_models(evaluation_dir: str = "evaluation") -> List[str]:
    """
    Discover all available models by looking for model directories.
    
    Returns:
        List of available model names
    """
    models = []
    for model_name in MODEL_MAPPING.keys():
        model_path = Path(model_name) / evaluation_dir
        if model_path.exists():
            models.append(model_name)
    return sorted(models)


def discover_available_attention_types(
    dataset: str,
    model: str,
    evaluation_dir: str = "evaluation"
) -> List[str]:
    """
    Discover all available attention types for a specific model and dataset.
    
    Returns:
        List of attention type names found
    """
    eval_path = Path(model) / evaluation_dir / dataset
    
    if not eval_path.exists():
        return []
    
    # Look for files matching pattern: {attention_type}_budget_*_embedding.json
    all_files = list(eval_path.glob("*_budget_*_embedding.json"))
    
    attention_types = set()
    for file_path in all_files:
        # Extract attention type from filename
        parts = file_path.name.split('_budget_')
        if len(parts) >= 2:
            attention_type = parts[0]
            attention_types.add(attention_type)
    
    return sorted(list(attention_types))


def discover_available_budgets(
    dataset: str,
    model: str,
    attention_type: str,
    evaluation_dir: str = "evaluation"
) -> List[float]:
    """
    Discover all available budgets for a specific model, attention type, and dataset.
    
    Returns:
        Sorted list of budget values found
    """
    eval_path = Path(model) / evaluation_dir / dataset
    
    if not eval_path.exists():
        return []
    
    # Pattern: {attention_type}_budget_{budget:.3f}_*_embedding.json
    pattern = f"{attention_type}_budget_*_embedding.json"
    budgets = set()
    
    for file_path in eval_path.glob(pattern):
        # Extract budget from filename
        match = re.search(r'budget_(\d+\.\d+)', file_path.name)
        if match:
            budgets.add(float(match.group(1)))
    
    return sorted(list(budgets))


def discover_available_rag_models(
    dataset: str,
    rag_evaluation_dir: str = "baselines/evaluation"
) -> List[str]:
    """
    Discover all available RAG models for a specific dataset.
    
    Returns:
        List of RAG model names found
    """
    rag_path = Path(rag_evaluation_dir) / dataset
    
    if not rag_path.exists():
        return []
    
    # Pattern: rag-{model}-top{k}_*_embedding.json
    rag_models = set()
    
    for file_path in rag_path.glob("rag-*_embedding.json"):
        # Extract model name from filename
        match = re.match(r'rag-(.+?)-top\d+', file_path.name)
        if match:
            rag_models.add(match.group(1))
    
    return sorted(list(rag_models))


def discover_available_top_k(
    dataset: str,
    rag_model: str,
    rag_evaluation_dir: str = "baselines/evaluation"
) -> List[int]:
    """
    Discover all available top-k values for a specific RAG model and dataset.
    
    Returns:
        Sorted list of top-k values found
    """
    rag_path = Path(rag_evaluation_dir) / dataset
    
    if not rag_path.exists():
        return []
    
    # Get the model pattern to match in filenames
    model_pattern = RAG_MODEL_PATTERNS.get(rag_model, rag_model)
    
    # Pattern: rag-{model}-top{k}_*_embedding.json
    top_ks = set()
    
    for file_path in rag_path.glob(f"rag-{model_pattern}-top*_embedding.json"):
        # Extract top-k from filename
        match = re.search(r'-top(\d+)_', file_path.name)
        if match:
            top_ks.add(int(match.group(1)))
    
    return sorted(list(top_ks))


def discover_questions(
    dataset: str,
    cr_models: List[str],
    cr_types: List[str],
    rag_models: List[str],
    evaluation_dir: str = "evaluation",
    rag_evaluation_dir: str = "baselines/evaluation"
) -> List[str]:
    """
    Discover all unique questions from CR and RAG evaluation files.
    
    Returns:
        Sorted list of unique questions
    """
    questions = set()
    
    # Discover from CR files
    for model in cr_models:
        for attention_type in cr_types:
            eval_path = Path(model) / evaluation_dir / dataset
            if eval_path.exists():
                pattern = f"{attention_type}_budget_*_embedding.json"
                for file_path in eval_path.glob(pattern):
                    try:
                        with open(file_path) as f:
                            file_data = json.load(f)
                        question = file_data.get('question', '')
                        if question:
                            questions.add(question)
                    except Exception:
                        continue
    
    # Discover from RAG files
    rag_path = Path(rag_evaluation_dir) / dataset
    if rag_path.exists():
        for file_path in rag_path.glob("rag-*_embedding.json"):
            try:
                with open(file_path) as f:
                    file_data = json.load(f)
                question = file_data.get('question', '')
                if question:
                    questions.add(question)
            except Exception:
                continue
    
    return sorted(list(questions))


def generate_combination_name(
    cr_configs: List[Tuple[str, str]],
    budgets: List[float],
    rag_configs: List[str],
    top_ks: List[int]
) -> str:
    """
    Generate a descriptive folder name for this combination.
    
    Args:
        cr_configs: List of (model, attention_type) tuples
        budgets: List of budgets
        rag_configs: List of RAG model names
        top_ks: List of top-k values
    
    Returns:
        Folder name string
    """
    parts = []
    
    # CR part
    if cr_configs:
        cr_parts = []
        for model, atype in cr_configs:
            cr_parts.append(f"{model}_{atype}")
        cr_str = "+".join(cr_parts)
        budget_str = "_".join([f"{b:.3f}".replace(".", "p") for b in sorted(budgets)])
        parts.append(f"CR_{cr_str}_b{budget_str}")
    
    # RAG part
    if rag_configs:
        rag_parts = []
        for rag_model in rag_configs:
            # Shorten model name
            short_name = rag_model.replace('Qwen3-Embedding-', 'Q3E-').replace('qwen3_', 'q3_')
            rag_parts.append(short_name)
        rag_str = "+".join(rag_parts)
        topk_str = "_".join([str(k) for k in sorted(top_ks)])
        parts.append(f"RAG_{rag_str}_k{topk_str}")
    
    return "_vs_".join(parts) if len(parts) > 1 else parts[0] if parts else "default"


def plot_question_cdf_multi(
    question: str,
    cr_configs: List[Tuple[str, str]],  # List of (model, attention_type)
    budgets: List[float],
    rag_configs: List[str],  # List of RAG model names
    top_ks: List[int],
    output_dir: str,
    dataset: str,
    evaluation_dir: str,
    rag_evaluation_dir: str
):
    """
    Plot CDF comparison for a single question with multiple configurations.
    
    Args:
        question: Question text
        cr_configs: List of (model, attention_type) tuples for CR
        budgets: List of budgets to plot
        rag_configs: List of RAG model names
        top_ks: List of top-k values to plot
        output_dir: Output directory for figures
        dataset: Dataset name
        evaluation_dir: CR evaluation directory
        rag_evaluation_dir: RAG evaluation directory
    """
    plt.figure(figsize=(14, 8))
    
    # Color schemes - expanded for more combinations
    cr_color_palette = [
        '#1f77b4', '#aec7e8',  # Blues
        '#2ca02c', '#98df8a',  # Greens
        '#d62728', '#ff9896',  # Reds
        '#9467bd', '#c5b0d5',  # Purples
        '#8c564b', '#c49c94',  # Browns
        '#e377c2', '#f7b6d2',  # Pinks
    ]
    
    rag_color_palette = [
        '#ff7f0e', '#ffbb78',  # Oranges
        '#bcbd22', '#dbdb8d',  # Yellows
        '#17becf', '#9edae5',  # Cyans
    ]
    
    plot_count = 0
    
    # Plot CR configurations
    print(f"   📈 Plotting {len(cr_configs)} CR configs × {len(budgets)} budgets")
    for config_idx, (model, attention_type) in enumerate(cr_configs):
        for budget_idx, budget in enumerate(sorted(budgets)):
            # Try to load file
            file_data = load_single_cr_file(
                dataset, model, attention_type, budget, question, evaluation_dir
            )
            
            if file_data is None:
                print(f"      ⚠️  Skipping {model}/{attention_type}/budget={budget:.3f} (not found)")
                continue
            
            similarities = extract_similarity_scores(file_data)
            avg_token = compute_avg_token_usage(file_data)
            
            if similarities:
                color_idx = (config_idx * len(budgets) + budget_idx) % len(cr_color_palette)
                color = cr_color_palette[color_idx]
                token_str = f"{avg_token:.1f}%" if avg_token is not None else "N/A"
                
                # Short model name for legend
                short_model = model.replace('llama3.2_', 'L3.2-').replace('qwen3_', 'Q3-')
                label = f"CR {short_model}/{attention_type[:4]} b={budget:.3f} ({token_str})"
                
                plot_cdf(similarities, label, color=color, linestyle='-', marker='')
                plot_count += 1
    
    # Plot RAG configurations
    print(f"   📊 Plotting {len(rag_configs)} RAG models × {len(top_ks)} top-k values")
    for rag_idx, rag_model in enumerate(rag_configs):
        for topk_idx, top_k in enumerate(sorted(top_ks)):
            # Try to load file
            file_data = load_single_rag_file(
                dataset, rag_model, top_k, question, rag_evaluation_dir
            )
            
            if file_data is None:
                print(f"      ⚠️  Skipping RAG {rag_model}/top-{top_k} (not found)")
                continue
            
            similarities = extract_similarity_scores(file_data)
            avg_token = compute_avg_token_usage(file_data)
            
            if similarities:
                color_idx = (rag_idx * len(top_ks) + topk_idx) % len(rag_color_palette)
                color = rag_color_palette[color_idx]
                token_str = f"{avg_token:.1f}%" if avg_token is not None else "N/A"
                
                # Short model name for legend
                short_rag = rag_model.replace('Qwen3-Embedding-', 'Q3E-').replace('qwen3_', 'Q3-')
                label = f"RAG {short_rag} k={top_k} ({token_str})"
                
                plot_cdf(similarities, label, color=color, linestyle='--', marker='')
                plot_count += 1
    
    if plot_count == 0:
        print(f"      ⚠️  No data found for question, skipping plot")
        plt.close()
        return
    
    # Customize plot
    plt.xlabel('Similarity Score', fontsize=12)
    plt.ylabel('P(Similarity > x)', fontsize=12)
    
    # Truncate question for title if too long
    title_question = question if len(question) <= 60 else question[:57] + "..."
    plt.title(f'CDF: {title_question}', fontsize=13)
    
    plt.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2 if plot_count > 8 else 1)
    plt.grid(True, alpha=0.3)
    plt.xlim(1.01, -0.01)  # Reverse x-axis
    plt.ylim(-0.05, 1.05)
    
    # Save figure
    safe_question = sanitize_question_for_filename(question)
    filename = f"cdf_{dataset}_{safe_question}.pdf"
    
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"      ✅ Saved: {output_dir / filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot CDF comparison with multiple model and configuration support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover and use ALL available configurations
  python generate_cdf_enhanced.py --dataset paper \\
    --models all \\
    --attention-types all \\
    --budgets all \\
    --rag-models all \\
    --rag-top-k all
  
  # Use all models with specific attention type and budgets
  python generate_cdf_enhanced.py --dataset paper \\
    --models all \\
    --attention-types raw \\
    --budgets 0.005 0.01 0.02
  
  # Use specific models with all available attention types
  python generate_cdf_enhanced.py --dataset paper \\
    --models llama3.2_1b qwen3_8b \\
    --attention-types all \\
    --budgets all \\
    --rag-models all \\
    --rag-top-k all
  
  # Compare multiple models with same attention type
  python generate_cdf_enhanced.py --dataset paper \\
    --models llama3.2_1b qwen3_8b \\
    --attention-types raw \\
    --budgets 0.005 0.01 0.02 \\
    --rag-models Qwen3-Embedding-8B \\
    --rag-top-k 1 3 5
  
  # Compare multiple attention types with same model
  python generate_cdf_enhanced.py --dataset paper \\
    --models llama3.2_1b \\
    --attention-types raw farest baseline \\
    --budgets 0.005 0.01 \\
    --rag-models Qwen3-Embedding-8B Qwen3-Embedding-14B \\
    --rag-top-k 1 3
  
  # All RAG baselines with all top-k values
  python generate_cdf_enhanced.py --dataset paper \\
    --rag-models all \\
    --rag-top-k all
  
  # Specific model with all budgets vs all RAG
  python generate_cdf_enhanced.py --dataset paper \\
    --models qwen3_8b \\
    --attention-types farest \\
    --budgets all \\
    --rag-models all \\
    --rag-top-k all
        """
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., paper, notice, quality)')
    
    # Context Reduction configurations
    parser.add_argument('--models', nargs='+', type=str, default=None,
                       help='Models for context reduction (can specify multiple). Use "all" to auto-discover. '
                            'Choices: llama3.2_1b, qwen3_8b, qwen3_14b, or "all"')
    parser.add_argument('--attention-types', nargs='+', type=str, default=None,
                       help='Attention types (e.g., raw farest baseline). Use "all" to auto-discover all available types.')
    parser.add_argument('--budgets', nargs='+', type=str, default=None,
                       help='Budgets to plot (applied to all CR configs). Use "all" to auto-discover. '
                            'Example: --budgets 0.005 0.01 0.02 or --budgets all')
    
    # RAG baseline configurations
    parser.add_argument('--rag-models', nargs='+', type=str, default=None,
                       help='RAG models (e.g., Qwen3-Embedding-8B Qwen3-Embedding-14B). Use "all" to auto-discover.')
    parser.add_argument('--rag-top-k', nargs='+', type=str, default=None,
                       help='Top-k values for RAG (applied to all RAG models). Use "all" to auto-discover. '
                            'Example: --rag-top-k 1 3 5 or --rag-top-k all')
    
    # Directory configurations
    parser.add_argument('--evaluation-dir', type=str, default='evaluation',
                       help='Directory containing CR evaluation results (relative to model dir)')
    parser.add_argument('--rag-evaluation-dir', type=str, default='baselines/evaluation',
                       help='Directory containing RAG evaluation results')
    parser.add_argument('--output-dir', type=str, default='figures/cdf_comparison',
                       help='Base directory to save output figures')
    
    # Filtering
    parser.add_argument('--questions', nargs='+', type=str, default=None,
                       help='Specific questions to plot (substring match)')
    
    args = parser.parse_args()
    
    # ===== AUTO-DISCOVERY LOGIC =====
    
    # Discover models if "all" is specified
    if args.models and "all" in args.models:
        print("🔍 Auto-discovering available models...")
        discovered_models = discover_available_models(args.evaluation_dir)
        if discovered_models:
            args.models = discovered_models
            print(f"   ✅ Found models: {', '.join(args.models)}")
        else:
            print(f"   ⚠️  No models found in {args.evaluation_dir}/")
            args.models = []
    elif args.models:
        # Validate specified models
        valid_models = ["llama3.2_1b", "qwen3_8b", "qwen3_14b"]
        invalid = [m for m in args.models if m not in valid_models]
        if invalid:
            parser.error(f"Invalid models: {invalid}. Choices: {valid_models} or 'all'")
    
    # Discover attention types if "all" is specified
    if args.attention_types and "all" in args.attention_types:
        print("🔍 Auto-discovering available attention types...")
        if not args.models:
            parser.error("Cannot use 'all' for --attention-types without specifying --models")
        
        all_attention_types = set()
        for model in args.models:
            types_for_model = discover_available_attention_types(args.dataset, model, args.evaluation_dir)
            all_attention_types.update(types_for_model)
            print(f"   {model}: {', '.join(types_for_model) if types_for_model else 'none'}")
        
        if all_attention_types:
            args.attention_types = sorted(list(all_attention_types))
            print(f"   ✅ Using attention types: {', '.join(args.attention_types)}")
        else:
            print(f"   ⚠️  No attention types found")
            args.attention_types = []
    
    # Discover budgets if "all" is specified
    if args.budgets and "all" in [str(b) for b in args.budgets]:
        print("🔍 Auto-discovering available budgets...")
        if not args.models or not args.attention_types:
            parser.error("Cannot use 'all' for --budgets without specifying --models and --attention-types")
        
        all_budgets = set()
        for model in args.models:
            for attention_type in args.attention_types:
                budgets_for_config = discover_available_budgets(
                    args.dataset, model, attention_type, args.evaluation_dir
                )
                all_budgets.update(budgets_for_config)
                if budgets_for_config:
                    print(f"   {model}/{attention_type}: {', '.join(f'{b:.3f}' for b in budgets_for_config)}")
        
        if all_budgets:
            args.budgets = sorted(list(all_budgets))
            print(f"   ✅ Using budgets: {', '.join(f'{b:.3f}' for b in args.budgets)}")
        else:
            print(f"   ⚠️  No budgets found")
            args.budgets = []
    else:
        # Convert string budgets to float
        if args.budgets:
            try:
                args.budgets = [float(b) for b in args.budgets]
            except ValueError as e:
                parser.error(f"Invalid budget value: {e}")
    
    # Discover RAG models if "all" is specified
    if args.rag_models and "all" in args.rag_models:
        print("🔍 Auto-discovering available RAG models...")
        discovered_rag = discover_available_rag_models(args.dataset, args.rag_evaluation_dir)
        if discovered_rag:
            args.rag_models = discovered_rag
            print(f"   ✅ Found RAG models: {', '.join(args.rag_models)}")
        else:
            print(f"   ⚠️  No RAG models found in {args.rag_evaluation_dir}/{args.dataset}/")
            args.rag_models = []
    
    # Discover RAG top-k if "all" is specified
    if args.rag_top_k and "all" in [str(k) for k in args.rag_top_k]:
        print("🔍 Auto-discovering available RAG top-k values...")
        if not args.rag_models:
            parser.error("Cannot use 'all' for --rag-top-k without specifying --rag-models")
        
        all_top_ks = set()
        for rag_model in args.rag_models:
            topks_for_model = discover_available_top_k(args.dataset, rag_model, args.rag_evaluation_dir)
            all_top_ks.update(topks_for_model)
            if topks_for_model:
                print(f"   {rag_model}: {', '.join(str(k) for k in topks_for_model)}")
        
        if all_top_ks:
            args.rag_top_k = sorted(list(all_top_ks))
            print(f"   ✅ Using top-k values: {', '.join(str(k) for k in args.rag_top_k)}")
        else:
            print(f"   ⚠️  No top-k values found")
            args.rag_top_k = []
    else:
        # Convert string top-k to int
        if args.rag_top_k:
            try:
                args.rag_top_k = [int(k) for k in args.rag_top_k]
            except ValueError as e:
                parser.error(f"Invalid top-k value: {e}")
    
    # ===== END AUTO-DISCOVERY LOGIC =====
    
    # Determine what methods are being used
    has_cr = args.models is not None and args.attention_types is not None
    has_rag = args.rag_models is not None
    
    # Validate that at least one method is specified
    if not has_cr and not has_rag:
        parser.error("Must specify at least one method:\n"
                    "  - Context Reduction: --models + --attention-types + --budgets\n"
                    "  - RAG Baselines: --rag-models + --rag-top-k\n"
                    "  - Both methods can be combined for comparison")
    
    # Validate and setup CR configurations
    if has_cr:
        # Check that models and attention types are not empty after auto-discovery
        if not args.models or len(args.models) == 0:
            parser.error("--models resulted in no models (check that model directories exist)")
        if not args.attention_types or len(args.attention_types) == 0:
            parser.error("--attention-types resulted in no attention types")
        if args.budgets is None or len(args.budgets) == 0:
            parser.error("--budgets is required when using --models and --attention-types")
        
        cr_configs = [(m, t) for m in args.models for t in args.attention_types]
    else:
        # CR not being used - set empty defaults
        cr_configs = []
        if args.budgets is None:
            args.budgets = []
    
    # Validate and setup RAG configurations
    if has_rag:
        # Check that RAG models are not empty after auto-discovery
        if not args.rag_models or len(args.rag_models) == 0:
            parser.error("--rag-models resulted in no models (check baselines directory)")
        if args.rag_top_k is None or len(args.rag_top_k) == 0:
            parser.error("--rag-top-k is required when using --rag-models")
        
        rag_configs = args.rag_models
    else:
        # RAG not being used - set empty defaults
        rag_configs = []
        if args.rag_top_k is None:
            args.rag_top_k = []
    
    print("\n" + "="*70)
    print("CDF COMPARISON PLOT GENERATION (ENHANCED)")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print()
    
    # Display CR configurations
    if cr_configs:
        print("✅ Context Reduction Configurations:")
        for model, atype in cr_configs:
            print(f"  • {model} / {atype}")
        print(f"  Budgets: {sorted(args.budgets)}")
    else:
        print("⊗  Context Reduction: SKIPPED (not specified)")
    
    print()
    
    # Display RAG configurations
    if rag_configs:
        print("✅ RAG Baseline Configurations:")
        for rag_model in rag_configs:
            print(f"  • {rag_model}")
        print(f"  Top-K values: {sorted(args.rag_top_k)}")
    else:
        print("⊗  RAG Baselines: SKIPPED (not specified)")
    
    print()
    
    # Generate combination folder name
    combo_name = generate_combination_name(
        cr_configs, args.budgets, rag_configs, args.rag_top_k
    )
    
    output_path = Path(args.output_dir) / args.dataset / combo_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output Directory: {output_path}")
    print()
    
    # Discover all questions
    print("🔍 Discovering questions...")
    cr_models = args.models if args.models else []
    cr_types = args.attention_types if args.attention_types else []
    
    questions = discover_questions(
        args.dataset,
        cr_models,
        cr_types,
        rag_configs,
        args.evaluation_dir,
        args.rag_evaluation_dir
    )
    
    print(f"📊 Found {len(questions)} unique questions")
    if questions:
        for q in questions[:5]:
            print(f"   • {q}")
        if len(questions) > 5:
            print(f"   ... and {len(questions) - 5} more")
    print()
    
    # Filter questions if specified
    if args.questions:
        filtered_questions = []
        for q in questions:
            if any(filter_q.lower() in q.lower() for filter_q in args.questions):
                filtered_questions.append(q)
        questions = filtered_questions
        print(f"📌 Filtered to {len(questions)} questions based on filters")
        print()
    
    if not questions:
        print("❌ No questions found!")
        return
    
    # Generate plots
    print(f"🎨 Generating {len(questions)} CDF plots...")
    print()
    
    for idx, question in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] {question[:60]}...")
        
        plot_question_cdf_multi(
            question,
            cr_configs,
            args.budgets,
            rag_configs,
            args.rag_top_k,
            output_path,
            args.dataset,
            args.evaluation_dir,
            args.rag_evaluation_dir
        )
    
    print()
    print("="*70)
    print("✅ CDF PLOTS COMPLETE")
    print("="*70)
    print(f"Generated {len(questions)} figures in:")
    print(f"  {output_path}/")
    print()
    

if __name__ == "__main__":
    main()