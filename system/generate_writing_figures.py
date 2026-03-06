#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNIFIED CDF PLOTTING SCRIPT - VERSION 2.0
1. Notice Dataset: farest vs RAG baselines.
2. Paper Dataset: farest vs RAG baselines.
3. Paper Dataset (Attention Type): Comparison of Raw, Baseline, and Farthest (Solid).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

# ============================================================
# 1. HARD-CODED CONFIGURATIONS
# ============================================================

# RAG Comparison (farest vs RAG)
PAPER_RAG_CONFIG = {
    'dataset': 'paper', 'model': 'qwen3_8b', 'attention_type': 'farest',
    'budgets': [0.005, 0.01, 0.1, 0.15],
    'rag_models': ['Qwen3-Embedding-8B', 'UAE-Large-V1', 'bflhc-Octen-Embedding-4B'],
    'rag_top_k': [1, 7, 10],
    'budget_colors': {0.005: '#08519C', 0.01: '#9370B8', 0.1: '#66C2A5', 0.15: '#C33225'},
    'rag_k_colors': {1: '#9370B8', 7: '#66C2A5', 10: '#C33225'},
    'best_rag_color': '#D95F02',
    'question_to_rag': {
        'Qwen3-Embedding-8B': [
            "What is the artifact?", "What is the authors of the paper?", 
            "What is the domain of this paper?", "What is the main contribution of the paper?", 
            "What is the number of authors?", "What is the publication year and venue of the paper?", 
            "What is the publication year of the paper?", "What is the type of population being studied or designed?", 
            "What is the type of study of this paper?"
        ],
        'UAE-Large-V1': ["What is the theory?", "What is the title of the paper?"],
        'bflhc-Octen-Embedding-4B': ["What is the type of contribution?"]
    }
}

# NEW: Attention Type Comparison Configuration
ATTENTION_COMP_CONFIG = {
    'dataset': 'paper', 'model': 'qwen3_8b',
    'attention_types': ['raw', 'baseline', 'farest'],
    'budgets': [0.005, 0.1, 0.15, 0.4],
    'colors': {0.005: '#08519C', 0.1: '#9370B8', 0.15: '#66C2A5', 0.4: '#C33225'},
    'line_styles': {
        'farest': '-',      # Solid as requested
        'baseline': '--',   # Dashed
        'raw': '-.'         # Dash-dot
    }
}

NOTICE_CONFIG = {
    'dataset': 'notice', 'model': 'qwen3_8b', 'attention_type': 'farest',
    'budgets': [0.05, 0.22, 0.38, 0.53, 0.74],
    'rag_models': ['Qwen3-Embedding-8B', 'UAE-Large-V1', 'bflhc-Octen-Embedding-4B'],
    'rag_top_k': [1, 3, 5, 7, 10],
    'budget_colors': {0.05: '#C6DBEF', 0.22: '#9370B8', 0.38: '#66C2A5', 0.53: '#E6B854', 0.74: '#C33225'},
    'best_rag_color': '#D95F02',
    'rag_k_colors': {1: '#C6DBEF', 3: '#9370B8', 5: '#66C2A5', 7: '#E6B854', 10: '#C33225'},
    'question_to_rag': {
        'Qwen3-Embedding-8B': ["What is the civil penalty amount suggested for the probable violation?"],
        'UAE-Large-V1': ["What are the state abbreviation and ZIP code of the company?", "What is the date of the notice?", "What is the name of the company?"],
        'bflhc-Octen-Embedding-4B': ["What are the violation items?", "What is the full title of the sender in the notice?", "What is the type of violation item?", "What statutory authority is used to propose the Compliance Order?"]
    }
}

EVALUATION_DIR = 'evaluation'
RAG_EVALUATION_DIR = 'baselines/evaluation'
OUTPUT_BASE_DIR = Path('figures/cdf_comparison')

# ============================================================
# 2. CORE UTILITY FUNCTIONS
# ============================================================

def sanitize_question(question: str) -> str:
    safe = question.lower()
    safe = re.sub(r'[^\w\s-]', '', safe)
    safe = re.sub(r'[-\s]+', '_', safe)
    return safe.strip('_')[:80]

def extract_scores(file_data: Dict) -> List[float]:
    docs = file_data.get('files', [])
    return [max(0.0, min(1.0, float(d['similarity']))) for d in docs if d.get('similarity') is not None]

def compute_tokens(file_data: Dict) -> float:
    docs = file_data.get('files', [])
    ratios = [float(d.get('tokens_extracted', d.get('tokens_used', 0))) / float(d.get('total_tokens', 1)) for d in docs]
    return (sum(ratios) / len(ratios) * 100) if ratios else 0.0

def plot_cdf_line(ax, data: List[float], label: str, color, style='-', width=2.5):  # Reduced from 3.5
    if not data: return
    sorted_data = np.sort(data)
    cdf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    sorted_data = np.append(sorted_data, sorted_data[-1])
    cdf = np.append(cdf, 0)
    ax.plot(sorted_data, cdf, label=label, color=color, linestyle=style, linewidth=width)

# ============================================================
# 3. FILE LOADING LOGIC
# ============================================================

def load_json(path: Path, att_type: str, budget: float, question: str) -> Optional[Dict]:
    safe_q = sanitize_question(question)
    f_name = f"{att_type}_budget_{budget:.3f}_{safe_q}_embedding.json"
    if (path / f_name).exists():
        with open(path / f_name) as f: return json.load(f)
    for cand in path.glob(f"{att_type}_budget_{budget:.3f}_*_embedding.json"):
        with open(cand) as f: 
            data = json.load(f)
        if data.get('question', '').strip().lower() == question.strip().lower(): return data
    return None

def load_rag_json(dataset: str, model: str, k: int, question: str) -> Optional[Dict]:
    path = Path(RAG_EVALUATION_DIR) / dataset
    safe_q = sanitize_question(question)
    f_name = f"rag-{model}-top{k}_{safe_q}_embedding.json"
    if (path / f_name).exists():
        with open(path / f_name) as f: return json.load(f)
    return None

# ============================================================
# 4. PLOTTING FUNCTIONS
# ============================================================

def generate_rag_comparison(question: str, config: Dict, output_dir: Path):
    # Smaller figure size for grid layout
    fig, ax = plt.subplots(figsize=(8, 6))  # Reduced from (16, 9)
    handles, labels = [], []
    for b in sorted(config['budgets']):
        data = load_json(Path(config['model']) / EVALUATION_DIR / config['dataset'], config['attention_type'], b, question)
        if data:
            s, t = extract_scores(data), compute_tokens(data)
            lbl = f"SAGE {t:.1f}%"
            clr = config['budget_colors'].get(b, '#888888')
            line, = ax.plot([], [], color=clr, linestyle='-', linewidth=2.5, label=lbl)
            plot_cdf_line(ax, s, lbl, clr, '-')
            handles.append(line); labels.append(lbl)

    for k in sorted(config['rag_top_k']):
        best_scores, best_tokens, best_mean = None, 0.0, -1.0
        for model in config['rag_models']:
            data = load_rag_json(config['dataset'], model, k, question)
            if data:
                s = extract_scores(data)
                mean_s = float(np.mean(s)) if s else -1.0
                if mean_s > best_mean:
                    best_mean = mean_s
                    best_scores = s
                    best_tokens = compute_tokens(data)
        if best_scores:
            lbl = f"Best RAG {best_tokens:.1f}%"
            clr = config['rag_k_colors'].get(k, '#888888')
            line, = ax.plot([], [], color=clr, linestyle='--', linewidth=2.5, label=lbl)
            plot_cdf_line(ax, best_scores, lbl, clr, '--')
            handles.append(line); labels.append(lbl)

    ax.set_xlabel('Similarity Score', fontsize=35, fontweight='bold')  # Reduced from 40
    ax.set_ylabel('P(Similarity > x)', fontsize=35, fontweight='bold')  # Reduced from 40
    # ax.set_title(question, fontsize=24, fontweight='bold')
    
    # Customize tick label fonts (axis numbers) - adjusted for smaller figure
    ax.tick_params(axis='both', which='major', labelsize=25, width=1.5, length=5)  # Reduced from 32
    
    # Added bottom padding to prevent label overlap
    ax.set_xlim(1.01, -0.01); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f"cdf_{config['dataset']}_{sanitize_question(question)}.pdf")
    plt.close()
    return handles, labels

def generate_attention_comparison(question: str, config: Dict, output_dir: Path):
    # Smaller figure size for grid layout
    fig, ax = plt.subplots(figsize=(8, 6))  # Reduced from (16, 9)
    handles, labels = [], []

    for att in config['attention_types']:
        style = config['line_styles'][att]
        # Change 'farest' to 'Farthest' for the legend
        display_att = "Farthest" if att == 'farest' else ("Fixed Question" if att == 'baseline' else att.capitalize())
        
        for b in config['budgets']:
            data = load_json(Path(config['model']) / EVALUATION_DIR / config['dataset'], att, b, question)
            if data:
                s = extract_scores(data)
                clr = config['colors'][b]
                lbl = f"{display_att} (b={b})"
                line, = ax.plot([], [], color=clr, linestyle=style, linewidth=2.5, label=lbl)  # Reduced from 3.5
                plot_cdf_line(ax, s, lbl, clr, style)
                handles.append(line); labels.append(lbl)

    ax.set_xlabel('Similarity Score', fontsize=35, fontweight='bold')  # Reduced from 22
    ax.set_ylabel('P(Similarity > x)', fontsize=35, fontweight='bold')  # Reduced from 22
    # ax.set_title(f"Attention Comparison: {question}", fontsize=20, fontweight='bold')  # Reduced from 24
    
    # Customize tick label fonts (axis numbers) - adjusted for smaller figure
    ax.tick_params(axis='both', which='major', labelsize=25, width=1.5, length=5)  # Reduced from 18
    
    # Added bottom padding to prevent label overlap
    ax.set_xlim(1.01, -0.01); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f"attention_comp_{sanitize_question(question)}.pdf")
    plt.close()
    return handles, labels

def save_legend(handles, labels, path, name, ncol=None):
    """
    Save a separate legend figure with maximally flat layout.
    
    Args:
        handles: List of legend handles
        labels: List of legend labels
        path: Output directory path
        name: Output filename
        ncol: Number of columns (auto-determined if None)
    """
    if not handles: 
        return
    
    # Remove duplicates while preserving order
    unique = dict(zip(labels, handles))
    unique_handles = list(unique.values())
    unique_labels = list(unique.keys())
    num_items = len(unique_labels)
    
    # Auto-determine optimal number of columns for flat layout
    if ncol is None:
        if num_items <= 4:
            ncol = num_items  # All in one row
        elif num_items <= 8:
            ncol = 4  # 2 rows max
        elif num_items <= 12:
            ncol = 6  # 2 rows max
        elif num_items <= 16:
            ncol = 8  # 2 rows max
        else:
            ncol = min(10, num_items)  # Cap at 10 for readability
    
    # Calculate appropriate figure width based on columns and content
    # Estimate width needed: ~3-4 inches per column
    fig_width = max(24, ncol * 3.5)
    # Height scales with number of rows
    num_rows = (num_items + ncol - 1) // ncol  # Ceiling division
    fig_height = max(2, num_rows * 0.8)
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.legend(
        unique_handles, 
        unique_labels, 
        loc='center', 
        ncol=ncol, 
        fontsize=16, 
        frameon=True, 
        edgecolor='black',
        columnspacing=1.0,  # Spacing between columns
        handlelength=2.0,   # Length of legend handles
        handletextpad=0.5   # Space between handle and text
    )
    plt.axis('off')
    plt.savefig(path / name, bbox_inches='tight', dpi=300)
    plt.close()

# ============================================================
# 5. MAIN
# ============================================================

def main():
    # 1. NOTICE (RAG vs Budget)
    out_n = OUTPUT_BASE_DIR / "notice"
    out_n.mkdir(parents=True, exist_ok=True)
    qs_n = [q for qs in NOTICE_CONFIG['question_to_rag'].values() for q in qs]
    h_n, l_n = None, None
    print("Processing Notice Dataset...")
    for q in qs_n:
        h, l = generate_rag_comparison(q, NOTICE_CONFIG, out_n)
        if h and not h_n: h_n, l_n = h, l
    save_legend(h_n, l_n, out_n, "legend_notice.pdf")  # Auto-determine ncol

    # 2. PAPER (RAG vs Budget)
    out_p = OUTPUT_BASE_DIR / "paper"
    out_p.mkdir(parents=True, exist_ok=True)
    qs_p = [q for qs in PAPER_RAG_CONFIG['question_to_rag'].values() for q in qs]
    h_p, l_p = None, None
    print("Processing Paper RAG Comparison...")
    for q in qs_p:
        h, l = generate_rag_comparison(q, PAPER_RAG_CONFIG, out_p)
        if h and not h_p: h_p, l_p = h, l
    save_legend(h_p, l_p, out_p, "legend_paper.pdf")  # Auto-determine ncol

    # 3. PAPER (Attention Type Comparison)
    out_a = OUTPUT_BASE_DIR / "paper_attention"
    out_a.mkdir(parents=True, exist_ok=True)
    h_a, l_a = None, None
    print("Processing Paper Attention Type Comparison...")
    for q in qs_p:
        h, l = generate_attention_comparison(q, ATTENTION_COMP_CONFIG, out_a)
        if h and not h_a: h_a, l_a = h, l
    save_legend(h_a, l_a, out_a, "legend_attention_paper.pdf")  # Auto-determine ncol

    print("\n✅ Task completed. Farthest is now solid and correctly labeled.")

if __name__ == "__main__":
    main()