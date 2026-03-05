"""
Generate accuracy vs. token-usage figures for evaluation outputs.
Enhanced version with support for multiple models and auto-discovery.

CORRECT FORMULA for average token usage:
    avg_token_pct = (sum(tokens_used / total_tokens for each doc) / num_docs) * 100

For RAG: tokens_used = context_tokens from evaluation files (now includes tokens_extracted)
For others: tokens_used = tokens_extracted from evaluation files

MODIFICATIONS:
- Removed point labels
- Generate separate shared legend
- Reduced figure size for grid layout
- Increased font sizes for better readability
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

# Model name mapping (must match data_reader.py and unit_window.py)
MODEL_MAPPING = {
    'llama3.2_1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'qwen3_8b': 'Qwen/Qwen3-8B',
    'qwen3_14b': 'Qwen/Qwen3-14B'
}

# RAG model name mapping for file matching
# Maps various naming conventions to canonical model names
RAG_MODEL_PATTERNS = {
    # Qwen models
    'Qwen3-Embedding-8B': 'Qwen3-Embedding-8B',
    'Qwen3-Embedding-14B': 'Qwen3-Embedding-14B',
    'qwen3_8b': 'Qwen3-Embedding-8B',
    'qwen3_14b': 'Qwen3-Embedding-14B',
    
    # UAE models
    'UAE-Large-V1': 'UAE-Large-V1',
    
    # Octen models (slash may be converted to underscore in filenames)
    'bflhc/Octen-Embedding-4B': 'Octen-Embedding-4B',
    'bflhc_Octen-Embedding-4B': 'Octen-Embedding-4B',
    'Octen-Embedding-4B': 'Octen-Embedding-4B',
}

# Base directory (current script location)
BASE_DIR = Path(__file__).resolve().parent

# Matplotlib defaults - OPTIMIZED FOR SMALLER FIGURES WITH LARGER FONTS
plt.rcParams['figure.figsize'] = (8, 6)  # Reduced from (10, 6)
plt.rcParams['font.size'] = 20  # Increased from 16
plt.rcParams['axes.labelsize'] = 24  # Increased from 18
plt.rcParams['axes.titlesize'] = 26  # Increased from 20
plt.rcParams['legend.fontsize'] = 20  # Increased from 16
plt.rcParams['xtick.labelsize'] = 20  # Increased from 16
plt.rcParams['ytick.labelsize'] = 20  # Increased from 16
plt.rcParams['lines.linewidth'] = 2.5  # Reduced from 3
plt.rcParams['lines.markersize'] = 8  # Reduced from 10

# --- CONFIGURATION: Style mappings for your methods ---
PATTERN_LABELS = {
    'raw': 'Raw Attention',
    'baseline': 'Differential (Baseline)',
    'farest': 'Differential (Farthest)',
    'farthest': 'SAGE Farthest',
    'diff-far': 'Attentive Trim (Diff-Far)',
}

PATTERN_COLORS = {
    'raw': '#1f77b4',
    'baseline': '#ff7f0e',
    'farest': '#2ca02c',
    'diff-far': '#d62728',
}

PATTERN_MARKERS = {
    'raw': 'o',
    'baseline': 's',
    'farest': '^',
    'diff-far': 'D',
}

COLOR_CYCLE = [
    '#9467bd', '#8c564b', '#e377c2',
    '#7f7f7f', '#bcbd22', '#17becf',
    '#1a55FF', '#FF7F50', '#2E8B57',
]
DYNAMIC_COLOR_MAP: Dict[str, str] = {}
DYNAMIC_MARKERS = ['x', 'P', '*', 'v', '<', '>']
DYNAMIC_MARKER_MAP: Dict[str, str] = {}

EVAL_METHOD_CHOICES = {'gpt', 'gemini', 'gpt4', 'embedding', 'rouge'}
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


def sanitize_question(question: str) -> str:
    """Convert question to safe filename format."""
    return question.replace("?", "").replace("/", "_").replace(" ", "_")


def format_pattern_label(pattern: str, model: Optional[str] = None) -> str:
    """Format pattern name for legend, optionally including model."""
    base_label = PATTERN_LABELS.get(pattern, pattern.replace('_', ' ').title())
    
    if pattern.startswith('rag-'):
        rag_core = pattern[len('rag-'):]
        base_label = f"RAG {rag_core.replace('-', ' ').title()}"
    
    # Add model prefix if provided and multiple models are being compared
    if model:
        short_model = model.replace('llama3.2_', 'L3.2-').replace('qwen3_', 'Q3-')
        return f"{short_model}/{base_label}"
    
    return base_label


def get_pattern_color(pattern: str) -> Optional[str]:
    if pattern in PATTERN_COLORS:
        return PATTERN_COLORS[pattern]
    if pattern not in DYNAMIC_COLOR_MAP:
        color = COLOR_CYCLE[len(DYNAMIC_COLOR_MAP) % len(COLOR_CYCLE)]
        DYNAMIC_COLOR_MAP[pattern] = color
    return DYNAMIC_COLOR_MAP[pattern]


def get_pattern_marker(pattern: str) -> str:
    if pattern in PATTERN_MARKERS:
        return PATTERN_MARKERS[pattern]
    if pattern not in DYNAMIC_MARKER_MAP:
        idx = len(DYNAMIC_MARKER_MAP) % len(DYNAMIC_MARKERS)
        DYNAMIC_MARKER_MAP[pattern] = DYNAMIC_MARKERS[idx]
    return DYNAMIC_MARKER_MAP[pattern]


def discover_available_models(evaluation_dir: str = "evaluation") -> List[str]:
    """Discover all available models by looking for model directories."""
    models = []
    for model_name in MODEL_MAPPING.keys():
        model_path = Path(model_name) / evaluation_dir
        if model_path.exists():
            models.append(model_name)
    return sorted(models)


def discover_available_patterns(
    dataset: str,
    model: str,
    evaluation_dir: str = "evaluation"
) -> List[str]:
    """Discover all available patterns for a specific model and dataset."""
    eval_path = Path(model) / evaluation_dir / dataset
    
    if not eval_path.exists():
        return []
    
    patterns = set()
    for file_path in eval_path.glob("*.json"):
        # Skip RAG files
        if file_path.name.startswith('rag-'):
            continue
        
        # Try to extract pattern from filename
        # Format: results-{type}-{budget}-{dataset}_{query}-location_eval_{method}.json
        # or: {type}_budget_{budget}_{query}_{method}.json
        match = re.match(r"^results-(?P<type>.+?)-(?P<budget>[0-9.]+)-", file_path.name)
        if match:
            patterns.add(match.group('type'))
        else:
            match = re.match(r"(?P<type>.+)_budget_(?P<budget>[0-9.]+)", file_path.name)
            if match:
                patterns.add(match.group('type'))
    
    return sorted(list(patterns))


def discover_available_rag_models(
    dataset: str,
    rag_evaluation_dir: str = "baselines/evaluation"
) -> List[str]:
    """Discover all available RAG models for a specific dataset."""
    rag_path = Path(rag_evaluation_dir) / dataset
    
    if not rag_path.exists():
        print(f"   ⚠️  RAG path does not exist: {rag_path}")
        return []
    
    rag_models = set()
    files_found = list(rag_path.glob("rag-*.json"))
    
    if not files_found:
        print(f"   ⚠️  No rag-*.json files found in {rag_path}")
        return []
    
    print(f"   Found {len(files_found)} RAG files in {rag_path}")
    
    for file_path in files_found:
        # Pattern: rag-{model}-top{k}_...
        # The model name might contain underscores, hyphens, or other characters
        # Extract everything between "rag-" and "-top"
        match = re.match(r'rag-(.+?)-top\d+', file_path.name)
        if match:
            model_name = match.group(1)
            rag_models.add(model_name)
            print(f"      • Extracted model: '{model_name}' from {file_path.name}")
        else:
            print(f"      ⚠️  Could not parse: {file_path.name}")
    
    return sorted(list(rag_models))


def generate_combination_name(
    cr_configs: List[Tuple[str, str]],
    rag_configs: List[str]
) -> str:
    """Generate a descriptive folder name for this combination."""
    parts = []
    
    # CR part
    if cr_configs:
        cr_parts = []
        for model, pattern in cr_configs:
            cr_parts.append(f"{model}_{pattern}")
        cr_str = "+".join(cr_parts[:3])  # Limit to avoid too long names
        if len(cr_parts) > 3:
            cr_str += f"+{len(cr_parts)-3}more"
        parts.append(f"CR_{cr_str}")
    
    # RAG part
    if rag_configs:
        rag_parts = []
        for rag_model in rag_configs:
            short_name = rag_model.replace('Qwen3-Embedding-', 'Q3E-').replace('qwen3_', 'q3_')
            rag_parts.append(short_name)
        rag_str = "+".join(rag_parts[:2])
        if len(rag_parts) > 2:
            rag_str += f"+{len(rag_parts)-2}more"
        parts.append(f"RAG_{rag_str}")
    
    return "_vs_".join(parts) if len(parts) > 1 else parts[0] if parts else "default"


def parse_evaluation_filename(path: Path) -> Optional[Tuple[str, float, str, Optional[str]]]:
    """Extract (type, budget, query_slug, eval_method) from filename."""
    # New format: results-{type}-{budget}-{dataset}_{query}-location_eval_{method}.json
    pattern = re.compile(
        r"^results-(?P<type>.+?)-(?P<budget>[0-9.]+)-(?P<dataset>[^_]+)_(?P<query>.+)-location_eval_(?P<eval_method>[a-z0-9]+)\.json$"
    )

    match = pattern.match(path.name)
    
    if not match:
        # Legacy format: {type}_budget_{budget}_{query}_{method}.json
        legacy_pattern = re.compile(r'(?P<type>.+)_budget_(?P<budget>[0-9.]+)_(?P<query>.+)_(?P<eval_method>[a-z0-9]+)\.json')
        match_legacy = legacy_pattern.match(path.name)
        if match_legacy:
            return (
                match_legacy.group('type'),
                float(match_legacy.group('budget')),
                match_legacy.group('query'),
                match_legacy.group('eval_method')
            )
        return None

    method_type = match.group('type')
    budget_str = match.group('budget')
    query_slug = match.group('query')
    eval_method = match.group('eval_method')

    try:
        budget_val = float(budget_str)
    except ValueError:
        return None

    return method_type, budget_val, query_slug, eval_method


def compute_accuracy_and_avg_tokens(
    file_data: Dict,
    evaluation_method: str,
) -> Tuple[float, Optional[float]]:
    """
    Compute accuracy and average token usage percentage for a single evaluation file.
    
    Formula: avg_token_pct = (sum(tokens_extracted / total_tokens for each doc) / num_docs) * 100
    """
    documents = file_data.get('files', [])
    if not isinstance(documents, list) or not documents:
        return 0.0, None

    total_docs = len(documents)
    correct_docs = 0
    token_ratios: List[float] = []

    for doc in documents:
        # Compute accuracy
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

        # Compute token ratio for this document
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

    accuracy = correct_docs / total_docs if total_docs > 0 else 0.0
    
    if token_ratios:
        avg_ratio = sum(token_ratios) / len(token_ratios)
    else:
        avg_ratio = None

    return accuracy, avg_ratio


def load_rag_accuracy_results(
    dataset: str,
    evaluation_method: str,
    rag_model_filter: Optional[List[str]] = None,
    top_k_filter: Optional[List[int]] = None,
) -> Dict[str, Dict[str, List[Dict]]]:
    """Load RAG results from baselines/evaluation directory."""
    eval_dir = BASE_DIR / "baselines" / "evaluation" / dataset
    
    if not eval_dir.exists():
        print(f"  ⚠️  RAG evaluation directory not found: {eval_dir}")
        return {}

    rag_results: Dict[str, Dict[str, List[Dict]]] = {}
    
    eval_pattern = re.compile(
        r"^rag-(?P<model>.+?)-top(?P<topk>[0-9]+)_(?P<question>.+?)_(?P<method>[a-z0-9]+)\.json$",
        re.IGNORECASE,
    )

    files_processed = 0
    files_filtered = 0
    
    for eval_path in sorted(eval_dir.glob("rag-*.json")):
        match = eval_pattern.match(eval_path.name)
        if not match:
            print(f"  ⚠️  Filename doesn't match pattern: {eval_path.name}")
            continue
            
        model_name = match.group("model")
        top_k = int(match.group("topk"))
        question_slug = match.group("question")
        eval_file_method = match.group("method")
        
        # Filter by top-k if specified
        if top_k_filter is not None and top_k not in top_k_filter:
            files_filtered += 1
            continue
        
        # Filter by RAG model if specified
        if rag_model_filter:
            # Normalize model names for comparison
            # Check multiple variations to handle different naming conventions
            model_matches = False
            
            # Get canonical pattern
            model_pattern = RAG_MODEL_PATTERNS.get(model_name, model_name)
            
            for filter_model in rag_model_filter:
                filter_pattern = RAG_MODEL_PATTERNS.get(filter_model, filter_model)
                
                # Check exact matches
                if (model_name == filter_model or 
                    model_name == filter_pattern or
                    model_pattern == filter_model or
                    model_pattern == filter_pattern):
                    model_matches = True
                    break
                
                # Check if filter is contained in model name (partial match)
                if (filter_model.replace('/', '_') in model_name or
                    model_name in filter_model.replace('/', '_')):
                    model_matches = True
                    break
            
            if not model_matches:
                files_filtered += 1
                continue
        
        # Normalize evaluation method names
        normalized_method = eval_file_method
        if eval_file_method in ['gpt4', 'gemini']:
            normalized_method = 'gpt'
        
        if normalized_method != evaluation_method and evaluation_method != 'gpt':
            files_filtered += 1
            continue
        
        if evaluation_method == 'gpt' and eval_file_method not in ['gpt', 'gpt4', 'gemini']:
            files_filtered += 1
            continue
        
        pattern_name = f"rag-{model_name}"

        try:
            with open(eval_path, "r") as f:
                eval_data = json.load(f)
        except Exception as e:
            print(f"  ⚠️  Error loading {eval_path.name}: {e}")
            continue

        documents = eval_data.get("files", [])
        if not documents:
            continue

        question_text = eval_data.get("question")
        if not question_text:
            continue

        accuracy, avg_ratio = compute_accuracy_and_avg_tokens(eval_data, normalized_method)
        
        if avg_ratio is None:
            continue
        
        avg_token_pct = avg_ratio * 100

        rag_results.setdefault(question_text, {}).setdefault(pattern_name, []).append(
            {
                "token_pct": avg_token_pct,
                "accuracy": accuracy,
                "budget": float(top_k),
                "source_file": eval_path.name,
            }
        )
        
        files_processed += 1

    if rag_results:
        print(f"  ✅ Loaded {len(rag_results)} questions from {files_processed} RAG evaluation files")
        if files_filtered > 0:
            print(f"     (Filtered out {files_filtered} files due to model/method mismatch)")
    else:
        print(f"  ⚠️  No RAG results loaded. Processed {files_processed} files, filtered {files_filtered} files")
    
    return rag_results


def load_results_for_config(
    dataset: str,
    model: str,
    pattern: str,
    evaluation_method: str,
    evaluation_dir: str = "evaluation",
    budget_filter: Optional[List[float]] = None,
) -> Dict[str, List[Dict]]:
    """Load results for a specific model and pattern combination."""
    dataset_dir = Path(model) / evaluation_dir / dataset
    if not dataset_dir.exists():
        return {}

    results: Dict[str, List[Dict]] = {}

    for path in sorted(dataset_dir.glob("*.json")):
        if path.name.startswith('rag-'):
            continue
            
        parsed = parse_evaluation_filename(path)
        if not parsed:
            continue

        method_type, budget_val, query_slug, eval_method = parsed
        
        # Filter by pattern
        if method_type != pattern:
            continue
        
        # Filter by budget if specified
        if budget_filter is not None and budget_val not in budget_filter:
            continue
        
        # Filter by evaluation method
        normalized_method = eval_method
        if eval_method in ['gpt4', 'gemini']:
            normalized_method = 'gpt'
        
        if normalized_method and normalized_method != evaluation_method:
            if evaluation_method == 'gpt' and eval_method not in ['gpt', 'gpt4', 'gemini']:
                continue
            elif evaluation_method != 'gpt':
                continue

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            continue

        question_text = data.get("question") or query_slug.replace("_", " ")
        
        accuracy, avg_ratio = compute_accuracy_and_avg_tokens(data, normalized_method or evaluation_method)
        avg_token_pct = (avg_ratio * 100) if avg_ratio is not None else (budget_val * 100)

        results.setdefault(question_text, []).append({
            "token_pct": avg_token_pct,
            "accuracy": accuracy,
            "budget": budget_val,
            "source_file": path.name,
        })

    return results


def generate_shared_legend(
    all_configs: List[Tuple[str, str, str, str]],  # (config_key, label, color, marker)
    output_path: Path
):
    """
    Generate a separate figure containing only the legend.
    
    Args:
        all_configs: List of tuples (config_key, label, color, marker)
        output_path: Path to save the legend figure
    """
    # Create dummy figure for legend
    fig_legend = plt.figure(figsize=(12, 3))
    
    # Create legend handles
    legend_handles = []
    for config_key, label, color, marker in all_configs:
        handle = Line2D([0], [0], 
                       color=color, 
                       marker=marker, 
                       linestyle='-',
                       linewidth=2.5,
                       markersize=8,
                       label=label)
        legend_handles.append(handle)
    
    # Create legend with more columns for flatter layout
    num_items = len(legend_handles)
    if num_items <= 4:
        ncol = num_items
    elif num_items <= 8:
        ncol = 4
    elif num_items <= 12:
        ncol = 6
    else:
        ncol = min(8, num_items)
    
    legend = fig_legend.legend(
        handles=legend_handles,
        loc='center',
        ncol=ncol,
        frameon=True,
        fontsize=18,
        handlelength=2.0,
        handletextpad=0.8,
        columnspacing=1.5
    )
    
    # Remove axes
    fig_legend.gca().set_axis_off()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig_legend)
    print(f"    ✅ Saved shared legend: {output_path.name}")


def generate_figures_multi(
    dataset: str,
    cr_configs: List[Tuple[str, str]],
    rag_configs: List[str],
    output_dir: Path,
    evaluation_method: str,
    evaluation_dir: str = "evaluation",
    budget_filter: Optional[List[float]] = None,
    top_k_filter: Optional[List[int]] = None,
):
    """Generate line plots with multiple model/pattern combinations."""
    print(f"\n📊 Generating figures for dataset '{dataset}' ({evaluation_method})...")

    # Collect all data
    all_question_data: Dict[str, Dict[str, List[Dict]]] = {}
    
    # Load CR data
    print(f"\n📂 Loading Context Reduction results...")
    if budget_filter:
        print(f"   Budget filter: {budget_filter}")
    for model, pattern in cr_configs:
        print(f"   Loading {model}/{pattern}...")
        if pattern == "farest":
            config_key = f"_farthest"
        else: 
            config_key = f"{model}_{pattern}"
        
        results = load_results_for_config(
            dataset, model, pattern, evaluation_method, evaluation_dir, budget_filter
        )
        
        for question, entries in results.items():
            all_question_data.setdefault(question, {})[config_key] = entries
    
    # Load RAG data
    if rag_configs:
        print(f"\n📂 Loading RAG results...")
        if top_k_filter:
            print(f"   Top-K filter: {top_k_filter}")
        rag_results = load_rag_accuracy_results(dataset, evaluation_method, rag_configs, top_k_filter)
        
        for question, pattern_map in rag_results.items():
            for pattern, entries in pattern_map.items():
                all_question_data.setdefault(question, {})[pattern] = entries
    
    if not all_question_data:
        print("❌ No evaluation data found to plot.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    skipped = 0

    # Determine if we need to show model names in labels
    use_model_in_label = len(set(model for model, _ in cr_configs)) > 1

    # Collect all unique configurations for shared legend
    all_configs_for_legend = []
    seen_configs = set()

    # First pass: collect all unique configurations
    for question, pattern_data in all_question_data.items():
        for config_key in sorted(pattern_data.keys()):
            if config_key in seen_configs:
                continue
            seen_configs.add(config_key)
            
            # Extract label, color, marker
            if config_key.startswith('rag-'):
                label = format_pattern_label(config_key)
                color = get_pattern_color(config_key)
                marker = get_pattern_marker(config_key)
            else:
                # config_key format: {model}_{pattern}
                parts = config_key.split('_', 1)
                if len(parts) == 2:
                    model, pattern = parts
                    label = format_pattern_label(pattern, model if use_model_in_label else None)
                    color = get_pattern_color(pattern)
                    marker = get_pattern_marker(pattern)
                else:
                    label = config_key
                    color = get_pattern_color(config_key)
                    marker = get_pattern_marker(config_key)
            
            all_configs_for_legend.append((config_key, label, color, marker))

    # Generate shared legend
    legend_path = output_dir / "shared_legend.pdf"
    generate_shared_legend(all_configs_for_legend, legend_path)

    # Second pass: generate individual plots
    for question, pattern_data in all_question_data.items():
        print(f"\n  📈 Processing: {question[:60]}...")
        safe_question = sanitize_question(question)

        fig, ax = plt.subplots(figsize=(8, 6))  # Reduced from (10, 6)
        has_data = False

        # Plot all configurations
        for config_key in sorted(pattern_data.keys()):
            entries = sorted(pattern_data[config_key], key=lambda e: e["token_pct"])
            xs = [entry["token_pct"] for entry in entries]
            ys = [entry["accuracy"] for entry in entries]

            # Extract model and pattern for labeling
            if config_key.startswith('rag-'):
                label = format_pattern_label(config_key)
                color = get_pattern_color(config_key)
                marker = get_pattern_marker(config_key)
            else:
                # config_key format: {model}_{pattern}
                parts = config_key.split('_', 1)
                if len(parts) == 2:
                    model, pattern = parts
                    label = format_pattern_label(pattern, model if use_model_in_label else None)
                    color = get_pattern_color(pattern)
                    marker = get_pattern_marker(pattern)
                else:
                    label = config_key
                    color = get_pattern_color(config_key)
                    marker = get_pattern_marker(config_key)

            ax.plot(
                xs,
                ys,
                marker=marker,
                label=label,
                color=color,
                linewidth=2.5,
                markersize=8,
                linestyle="-",
            )

            has_data = True

        if not has_data:
            print("    ⚠️  No method data found for this question. Skipping.")
            skipped += 1
            plt.close(fig)
            continue

        # Set labels with larger font sizes
        ax.set_xlabel("Average tokens used (%)", fontsize=30, fontweight='bold')
        ax.set_ylabel("Accuracy", fontsize=35, fontweight='bold')
        # ax.set_title(question, fontsize=26, fontweight='bold', pad=20, wrap=True)
        
        # Customize tick label fonts (axis numbers)
        ax.tick_params(axis='both', which='major', labelsize=25, width=1.5, length=5)
        
        ax.grid(True, alpha=0.3, linestyle="--")
        
        # Remove individual legend (use shared legend instead)
        # ax.legend(loc="best", framealpha=0.9)
        
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()

        output_path = output_dir / f"{dataset}_{safe_question}.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"    ✅ Saved figure: {output_path.name}")
        created += 1

    print(f"\n  📊 Summary: {created} figures created, {skipped} skipped")
    print(f"  📊 Shared legend saved: {legend_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate accuracy vs. token-usage figures with multi-model support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover and plot ALL configurations
  python generate_figures_enhanced_modified.py --dataset paper \\
    --models all --patterns all \\
    --rag-models all --eval-method gpt
  
  # Compare multiple models with same pattern
  python generate_figures_enhanced_modified.py --dataset paper \\
    --models llama3.2_1b qwen3_8b \\
    --patterns farest \\
    --rag-models all --eval-method gpt
  
  # Compare multiple patterns with same model
  python generate_figures_enhanced_modified.py --dataset paper \\
    --models qwen3_8b \\
    --patterns raw farest baseline \\
    --rag-models Qwen3-Embedding-8B --eval-method embedding
  
  # Filter by specific budgets for CR
  python generate_figures_enhanced_modified.py --dataset paper \\
    --models qwen3_8b --patterns farest \\
    --budgets 0.1 0.2 0.5 --eval-method gpt
  
  # Filter by specific top-k for RAG
  python generate_figures_enhanced_modified.py --dataset paper \\
    --rag-models Qwen3-Embedding-8B \\
    --top-k 5 10 --eval-method embedding
  
  # Combine CR and RAG with specific budgets and top-k
  python generate_figures_enhanced_modified.py --dataset paper \\
    --models qwen3_8b --patterns farest \\
    --budgets 0.2 0.5 \\
    --rag-models UAE-Large-V1 \\
    --top-k 5 10 --eval-method gpt
  
  # Context Reduction only (no RAG)
  python generate_figures_enhanced_modified.py --dataset paper \\
    --models all --patterns all --eval-method gpt
  
  # RAG baselines only (no CR)
  python generate_figures_enhanced_modified.py --dataset paper \\
    --rag-models all --eval-method gpt
        """
    )

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    
    # Context Reduction configurations
    parser.add_argument('--models', nargs='+', type=str, default=None,
                       help='Models for context reduction (can specify multiple). Use "all" to auto-discover.')
    parser.add_argument('--patterns', nargs='+', type=str, default=None,
                       help='Attention types/patterns (e.g., raw farest baseline). Use "all" to auto-discover.')
    parser.add_argument('--budgets', nargs='+', type=str, default=None,
                       help='Budget values for CR (e.g., 0.1 0.2 0.5). Use "all" for all budgets. If not specified, all budgets are included.')
    
    # RAG baseline configurations
    parser.add_argument('--rag-models', nargs='+', type=str, default=None,
                       help='RAG models (e.g., Qwen3-Embedding-8B). Use "all" to auto-discover.')
    parser.add_argument('--top-k', nargs='+', type=str, default=None,
                       help='Top-K values for RAG (e.g., 5 10 20). Use "all" for all top-k values. If not specified, all top-k values are included.')
    
    # Other configurations
    parser.add_argument("--output-dir", type=str, default="figures/line_chart", 
                       help="Base output directory (default: figures/line_chart)")
    parser.add_argument("--eval-method", type=str, default="gpt", 
                       choices=sorted(EVAL_METHOD_CHOICES),
                       help="Evaluation method to use")
    parser.add_argument('--evaluation-dir', type=str, default='evaluation',
                       help='Directory containing CR evaluation results (relative to model dir)')

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
            print(f"   ⚠️  No models found")
            args.models = []
    elif args.models:
        # Validate specified models
        valid_models = ["llama3.2_1b", "qwen3_8b", "qwen3_14b"]
        invalid = [m for m in args.models if m not in valid_models]
        if invalid:
            parser.error(f"Invalid models: {invalid}. Choices: {valid_models} or 'all'")
    
    # Discover patterns if "all" is specified
    if args.patterns and "all" in args.patterns:
        print("🔍 Auto-discovering available patterns...")
        if not args.models:
            parser.error("Cannot use 'all' for --patterns without specifying --models")
        
        all_patterns = set()
        for model in args.models:
            patterns_for_model = discover_available_patterns(args.dataset, model, args.evaluation_dir)
            all_patterns.update(patterns_for_model)
            print(f"   {model}: {', '.join(patterns_for_model) if patterns_for_model else 'none'}")
        
        if all_patterns:
            args.patterns = sorted(list(all_patterns))
            print(f"   ✅ Using patterns: {', '.join(args.patterns)}")
        else:
            print(f"   ⚠️  No patterns found")
            args.patterns = []
    
    # Discover RAG models if "all" is specified
    if args.rag_models and "all" in args.rag_models:
        print("🔍 Auto-discovering available RAG models...")
        discovered_rag = discover_available_rag_models(args.dataset)
        if discovered_rag:
            args.rag_models = discovered_rag
            print(f"   ✅ Found RAG models: {', '.join(args.rag_models)}")
        else:
            print(f"   ⚠️  No RAG models found")
            args.rag_models = []
    
    # ===== END AUTO-DISCOVERY LOGIC =====
    
    # ===== PARSE BUDGET AND TOP-K FILTERS =====
    
    # Parse budgets for CR
    budget_filter = None
    if args.budgets:
        if "all" in args.budgets:
            budget_filter = None  # None means include all budgets
            print("📊 Budget filter: ALL (no filtering)")
        else:
            try:
                budget_filter = [float(b) for b in args.budgets]
                print(f"📊 Budget filter: {budget_filter}")
            except ValueError as e:
                parser.error(f"Invalid budget values: {args.budgets}. Must be numbers or 'all'")
    
    # Parse top-k for RAG
    top_k_filter = None
    if args.top_k:
        if "all" in args.top_k:
            top_k_filter = None  # None means include all top-k values
            print("📊 Top-K filter: ALL (no filtering)")
        else:
            try:
                top_k_filter = [int(k) for k in args.top_k]
                print(f"📊 Top-K filter: {top_k_filter}")
            except ValueError as e:
                parser.error(f"Invalid top-k values: {args.top_k}. Must be integers or 'all'")
    
    # ===== END BUDGET AND TOP-K PARSING =====
    
    # Determine what methods are being used
    has_cr = args.models is not None and args.patterns is not None
    has_rag = args.rag_models is not None
    
    # Validate that at least one method is specified
    if not has_cr and not has_rag:
        parser.error("Must specify at least one method:\n"
                    "  - Context Reduction: --models + --patterns\n"
                    "  - RAG Baselines: --rag-models\n"
                    "  - Both methods can be combined for comparison")
    
    # Validate and setup CR configurations
    if has_cr:
        if not args.models or len(args.models) == 0:
            parser.error("--models resulted in no models")
        if not args.patterns or len(args.patterns) == 0:
            parser.error("--patterns resulted in no patterns")
        
        cr_configs = [(m, p) for m in args.models for p in args.patterns]
    else:
        cr_configs = []
    
    # Validate and setup RAG configurations
    if has_rag:
        if not args.rag_models or len(args.rag_models) == 0:
            parser.error("--rag-models resulted in no models")
        
        rag_configs = args.rag_models
    else:
        rag_configs = []
    
    print("\n" + "="*70)
    print("FIGURE GENERATOR (ENHANCED - MODIFIED FOR OVERLEAF)")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Evaluation Method: {args.eval_method}")
    if budget_filter is not None:
        print(f"Budget Filter: {budget_filter}")
    else:
        print(f"Budget Filter: ALL (no filtering)")
    if top_k_filter is not None:
        print(f"Top-K Filter: {top_k_filter}")
    else:
        print(f"Top-K Filter: ALL (no filtering)")
    print()
    
    # Display CR configurations
    if cr_configs:
        print("✅ Context Reduction Configurations:")
        for model, pattern in cr_configs:
            print(f"  • {model} / {pattern}")
    else:
        print("⊗  Context Reduction: SKIPPED (not specified)")
    
    print()
    
    # Display RAG configurations
    if rag_configs:
        print("✅ RAG Baseline Configurations:")
        for rag_model in rag_configs:
            print(f"  • {rag_model}")
    else:
        print("⊗  RAG Baselines: SKIPPED (not specified)")
    
    print()
    
    # Generate combination folder name
    combo_name = generate_combination_name(cr_configs, rag_configs)
    output_path = Path(args.output_dir) / args.dataset / combo_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output Directory: {output_path}")
    print("="*70)

    generate_figures_multi(
        dataset=args.dataset,
        cr_configs=cr_configs,
        rag_configs=rag_configs,
        output_dir=output_path,
        evaluation_method=args.eval_method,
        evaluation_dir=args.evaluation_dir,
        budget_filter=budget_filter,
        top_k_filter=top_k_filter,
    )

    print("\n" + "="*70)
    print("✅ DONE")
    print("="*70)
    print(f"📁 Figures saved in: {output_path}/")
    print(f"📁 Use shared_legend.pdf for your Overleaf document")

if __name__ == "__main__":
    main()