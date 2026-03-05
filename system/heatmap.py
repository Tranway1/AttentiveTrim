#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive Attention Heatmap - HTML Version
--------------------------------------------

Creates interactive HTML visualizations for attention scores where tokens
are clickable to reveal their exact attention scores.

The generated HTML files are standalone and can be viewed in:
- Any web browser (Chrome, Firefox, Safari, etc.)
- VSCode (right-click HTML file → "Open with Live Server" or "Open Preview")

Example usage:
    # Generate all questions in the dataset
    python attention_heatmap_html.py --dataset notice --pattern farest
    
    # Generate specific question ID
    python attention_heatmap_html.py --dataset paper --pattern baseline --question-id 0_0
    
    # Generate all questions for a specific document
    python attention_heatmap_html.py --dataset notice --pattern farest --document-id 5
    
    # Generate multiple specific questions
    python attention_heatmap_html.py --dataset paper --questions 0_0 0_2 1_1
    
    # Use linear scaling instead of log scaling
    python attention_heatmap_html.py --dataset notice --pattern raw --no-log-scale
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

BASE_DIR = Path(__file__).resolve().parent

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

# Model name mapping (must match data_reader.py and unit_window.py)
MODEL_MAPPING = {
    'llama3.2_1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'qwen3_8b': 'Qwen/Qwen3-8B',
    'qwen3_14b': 'Qwen/Qwen3-14B'
}

# Default relative paths (will be prefixed with model directory)
DEFAULT_DATA_DIR = "data"
DEFAULT_ATTENTION_DIR = "data/attention_summary"
DEFAULT_TOKEN_DIR = "data/tokens"
DEFAULT_DATASET_DIR = "data/datasets"
DEFAULT_OUTPUT_DIR = "figures_attention_html"

# Common PDF/encoding mojibake replacements
MOJIBAKE_REPLACEMENTS = {
    "âĢĻ": "'",
    "âĢľ": '"',
    "âĢĵ": "-",
    "âĢĿ": '"',
    "âĢĺ": '"',
    "âĢĲ": '"',
    "âĢ¢": "-",
    "âĢ¦": "...",
    "âĢĶ": "-",
    "âĦ¢": "-",
    "Â§": "§",
    "Â§Â§": "§§",
    "Â½": "1/2",
    "Â¼": "1/4",
    "Â¾": "3/4",
    "âģ": "+/-",
    "Ħ": "H",
}


def load_dataset(dataset: str, dataset_dir: Path = DEFAULT_DATASET_DIR) -> Dict:
    """Load the processed dataset JSON."""
    path = dataset_dir / f"{dataset}_processed.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def build_question_lookup(dataset_data: Dict) -> Dict[str, Dict]:
    """Build a lookup: question_id -> {question, farest_question_id, document_id}."""
    lookup: Dict[str, Dict] = {}
    for doc in dataset_data.get("documents", []):
        doc_id = doc.get("document_id")
        for q in doc.get("questions", []):
            question_id = str(q["question_id"])
            lookup[question_id] = {
                "question": q.get("question", ""),
                "farest_question_id": q.get("farest_question_id"),
                "document_id": doc_id,
            }
    return lookup


def load_tokens(dataset: str, doc_id: int, token_dir: Path = DEFAULT_TOKEN_DIR) -> List[str]:
    """Load token strings for a document."""
    token_path = token_dir / f"{dataset}_{doc_id}.json"
    if not token_path.exists():
        raise FileNotFoundError(f"Token file not found: {token_path}")
    with open(token_path, "r") as f:
        return json.load(f)


def load_attention_file(dataset: str, question_id: str, pattern: str, attention_dir: Path) -> np.ndarray:
    """Load the attention array for a question and pattern."""
    suffix = "" if pattern == "raw" else f"_{pattern}"
    path = attention_dir / f"{dataset}_{question_id}{suffix}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Attention file not found: {path}")
    return np.load(path)


def create_error_html(
    error_message: str,
    dataset: str,
    output_path: Path,
):
    """Create an error HTML page when IDs are out of bounds."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error - Out of Bounds</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .error-container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .error-icon {{
            font-size: 72px;
            color: #e74c3c;
            margin-bottom: 20px;
        }}
        h1 {{
            color: #e74c3c;
            margin-bottom: 20px;
        }}
        .error-message {{
            font-size: 18px;
            color: #333;
            margin-bottom: 30px;
            line-height: 1.6;
        }}
        .dataset-info {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="error-container">
        <div class="error-icon">⚠️</div>
        <h1>Question or Document Out of Bounds</h1>
        <div class="error-message">
            {escape_html(error_message)}
        </div>
        <div class="dataset-info">
            Dataset: <strong>{escape_html(dataset.upper())}</strong>
        </div>
    </div>
</body>
</html>
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"⚠️  Created error page: {output_path}")


def sanitize_filename(text: str) -> str:
    """Convert an arbitrary string into a safe filename fragment."""
    safe_chars = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe_chars.append(ch)
        elif ch.isspace():
            safe_chars.append("_")
        else:
            safe_chars.append("_")
    sanitized = "".join(safe_chars).strip("_")
    return sanitized or "figure"


def normalize_token_for_display(token: str) -> str:
    """Convert tokenizer output into display-friendly text."""
    if token is None:
        return ""
    
    replacements = {
        "Ġ": "",
        "▁": "",
        "Ċ": "\n",
        "\\n": "\n",
    }
    
    text = token
    for prefix, sub in (("Ġ", " "), ("▁", " ")):
        if text.startswith(prefix):
            text = sub + text[len(prefix):]
            break
    
    for marker, sub in replacements.items():
        text = text.replace(marker, sub)
    
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    
    text = text.replace("Â", "")
    return text


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))


def score_to_color(score: float, vmin: float, vmax: float, use_log: bool = True) -> str:
    """
    Convert attention score to RGB color string using coolwarm colormap with logarithmic scaling.
    
    Args:
        score: The attention score to convert
        vmin: Minimum score in the dataset
        vmax: Maximum score in the dataset
        use_log: If True, apply log-scaling to both position normalization and color saturation
                 for better mid-range differentiation (recommended by advisor)
    """
    # Apply log-scaling for better color differentiation
    if use_log:
        # Handle the sign separately for symmetric log scaling
        sign = 1 if score >= 0 else -1
        abs_score = abs(score)
        abs_vmax = max(abs(vmin), abs(vmax))
        
        # Apply log1p (log(1+x)) to handle values near zero
        if abs_vmax > 0:
            log_score = sign * np.log1p(abs_score) / np.log1p(abs_vmax)
        else:
            log_score = 0.0
        
        # Normalize to [0, 1]
        normalized = (log_score + 1.0) / 2.0
    else:
        # Linear normalization
        if vmax == vmin:
            normalized = 0.5
        else:
            normalized = (score - vmin) / (vmax - vmin)
    
    # Apply logarithmic scaling to make mid-range values more saturated
    # This compresses the dynamic range and makes all non-zero scores more visible
    
    # Coolwarm colormap with log-scaled saturation
    if normalized < 0.5:
        # Blue side: pure blue at 0, fading to white at 0.5
        t = normalized * 2  # 0 to 1
        # Apply log scaling: log(1 + x) to handle x=0
        if t > 0:
            t = np.log1p(t * 10) / np.log1p(10)  # Scale to [0, 1] range
        r = int(t * 255)
        g = int(t * 255)
        b = 255
    else:
        # Red side: white at 0.5, fading to pure red at 1.0
        t = (normalized - 0.5) * 2  # 0 to 1
        # Apply log scaling
        if t > 0:
            t = np.log1p(t * 10) / np.log1p(10)  # Scale to [0, 1] range
        r = 255
        g = int((1 - t) * 255)
        b = int((1 - t) * 255)
    
    return f"rgb({r},{g},{b})"


def create_html_visualization(
    tokens: List[str],
    scores: np.ndarray,
    question_text: str,
    farest_text: Optional[str],
    dataset: str,
    doc_id: int,
    question_id: str,
    pattern: str,
    output_path: Path,
    use_log_scale: bool = True,
):
    """Create an interactive HTML visualization with clickable tokens."""
    
    if len(tokens) == 0 or len(scores) == 0:
        print(f"⚠️  Skipping empty data for question {question_id}")
        return
    
    # Align lengths
    aligned_len = min(len(tokens), len(scores))
    tokens = tokens[:aligned_len]
    scores = scores[:aligned_len]
    
    # Calculate score range for color mapping
    vmin = float(np.min(scores))
    vmax = float(np.max(scores))
    vmax_abs = max(abs(vmin), abs(vmax), 1e-6)
    vmin, vmax = -vmax_abs, vmax_abs
    
    # Build HTML
    html_parts = []
    
    # HTML header with styles and JavaScript
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Attention Visualization</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0 0 10px 0;
            font-size: 24px;
            color: #333;
        }
        .metadata {
            color: #666;
            font-size: 14px;
            margin: 5px 0;
        }
        .question-box {
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 6px;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
        }
        .question-label {
            font-weight: bold;
            color: #856404;
            margin-bottom: 5px;
        }
        .farest-box {
            background-color: #d1ecf1;
            padding: 15px;
            border-radius: 6px;
            margin: 10px 0;
            border-left: 4px solid #17a2b8;
        }
        .farest-label {
            font-weight: bold;
            color: #0c5460;
            margin-bottom: 5px;
        }
        .legend {
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .legend-title {
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 16px;
        }
        .color-bar {
            height: 30px;
            background: linear-gradient(to right, 
                rgb(0,0,255) 0%,
                rgb(100,100,255) 15%,
                rgb(180,180,255) 35%,
                rgb(255,255,255) 50%,
                rgb(255,180,180) 65%,
                rgb(255,100,100) 85%,
                rgb(255,0,0) 100%);
            border-radius: 4px;
            margin: 10px 0;
        }
        .color-labels {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #666;
        }
        .instruction {
            font-size: 14px;
            color: #666;
            font-style: italic;
            margin-top: 10px;
        }
        .token-container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            line-height: 2;
            font-size: 14px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .token {
            display: inline-block;
            padding: 2px 4px;
            margin: 1px;
            border-radius: 3px;
            cursor: pointer;
            transition: transform 0.1s, box-shadow 0.1s;
            border: 1px solid transparent;
        }
        .token:hover {
            transform: scale(1.05);
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            z-index: 10;
            position: relative;
            border: 1px solid rgba(0,0,0,0.3);
        }
        .token:active {
            transform: scale(0.98);
            box-shadow: 0 1px 4px rgba(0,0,0,0.3);
        }
        .tooltip {
            position: absolute;
            background-color: rgba(0, 0, 0, 0.85);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 14px;
            z-index: 1000;
            display: none;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            font-family: 'Courier New', monospace;
            white-space: nowrap;
            pointer-events: none;
        }
        .tooltip.show {
            display: block;
        }
        .stats {
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stats-title {
            font-weight: bold;
            margin-bottom: 10px;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
        .stat-row:last-child {
            border-bottom: none;
        }
    </style>
</head>
<body>
""")
    
    pattern_name = pattern
    if pattern_name == "farest":
        pattern_name = "farthest"

    # Header section
    legend_title = "Attention Score Legend (Log-Scaled Colors)" if use_log_scale else "Attention Score Legend"
    scaling_note = "📊 Colors use logarithmic scaling for better visibility of mid-range scores" if use_log_scale else "📊 Colors use linear scaling"
    
    html_parts.append(f"""
    <div class="header">
        <h1>Attention Score Visualization</h1>
        <div class="metadata">Dataset: <strong>{escape_html(dataset.upper())}</strong> | Document ID: <strong>{doc_id}</strong> | Pattern: <strong>{pattern_name}</strong> | Question ID: <strong>{escape_html(question_id)}</strong></div>
        
        <div class="question-box">
            <div class="question-label">Qa+ (Current Question):</div>
            <div>{escape_html(question_text)}</div>
        </div>
        
        <div class="farest-box">
            <div class="farest-label">Qb- (Farest Question):</div>
            <div>{escape_html(farest_text or 'N/A')}</div>
        </div>
    </div>
    
    <div class="legend">
        <div class="legend-title">{legend_title}</div>
        <div class="color-bar"></div>
        <div class="color-labels">
            <span>← Negative (Qb-): {vmin:.4f}</span>
            <span>Neutral: 0.0000</span>
            <span>Positive (Qa+): {vmax:.4f} →</span>
        </div>
        <div class="instruction">🖱️ <strong>Click any token</strong> to see its attention score</div>
        <div class="instruction" style="margin-top: 5px;">{scaling_note}</div>
    </div>
    
    <div class="token-container" id="tokenContainer">
""")
    
    # Add tokens with colors
    for idx, (token, score) in enumerate(zip(tokens, scores)):
        display_token = normalize_token_for_display(token)
        if display_token == "":
            display_token = " "
        
        escaped_token = escape_html(display_token)
        color = score_to_color(score, vmin, vmax, use_log=use_log_scale)
        
        # Log-scaled opacity to make colors more visible
        abs_score = abs(score)
        abs_vmax = max(abs(vmin), abs(vmax))
        
        if abs_vmax > 0:
            if use_log_scale:
                # Log-scale the opacity
                log_alpha = np.log1p(abs_score) / np.log1p(abs_vmax)
                # Apply additional log scaling for better mid-range visibility
                log_alpha = np.log1p(log_alpha * 10) / np.log1p(10)
                alpha = max(0.5, min(1.0, log_alpha * 1.5))
            else:
                # Linear opacity with log boost
                linear_alpha = abs_score / abs_vmax
                linear_alpha = np.log1p(linear_alpha * 10) / np.log1p(10)
                alpha = max(0.5, min(1.0, linear_alpha * 1.5))
        else:
            alpha = 0.5
        
        html_parts.append(
            f'<span class="token" style="background-color: {color}; opacity: {alpha:.2f};" '
            f'data-score="{score:.6f}" data-index="{idx}">{escaped_token}</span>'
        )
    
    html_parts.append("""
    </div>
    
    <div class="stats">
        <div class="stats-title">Statistics</div>
""")
    
    # Add statistics
    html_parts.append(f"""
        <div class="stat-row">
            <span>Total Tokens:</span>
            <span>{len(tokens)}</span>
        </div>
        <div class="stat-row">
            <span>Min Score:</span>
            <span>{vmin:.6f}</span>
        </div>
        <div class="stat-row">
            <span>Max Score:</span>
            <span>{vmax:.6f}</span>
        </div>
        <div class="stat-row">
            <span>Mean Score:</span>
            <span>{np.mean(scores):.6f}</span>
        </div>
        <div class="stat-row">
            <span>Std Dev:</span>
            <span>{np.std(scores):.6f}</span>
        </div>
    </div>
    
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        const tokens = document.querySelectorAll('.token');
        const tooltip = document.getElementById('tooltip');
        
        tokens.forEach(token => {{
            token.addEventListener('click', function(e) {{
                const score = this.getAttribute('data-score');
                
                tooltip.textContent = score;
                tooltip.className = 'tooltip show';
                
                // Position near the clicked token
                const rect = this.getBoundingClientRect();
                tooltip.style.left = (rect.left + window.pageXOffset) + 'px';
                tooltip.style.top = (rect.bottom + window.pageYOffset + 5) + 'px';
                
                // Auto-hide after 3 seconds
                setTimeout(() => {{
                    tooltip.className = 'tooltip';
                }}, 3000);
                
                e.stopPropagation();
            }});
        }});
        
        // Click anywhere to hide
        document.addEventListener('click', function(e) {{
            if (!e.target.classList.contains('token')) {{
                tooltip.className = 'tooltip';
            }}
        }});
    </script>
</body>
</html>
""")
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html_parts))
    
    print(f"✅ Saved HTML: {output_path}")


def generate_figures(
    dataset: str,
    question_ids: Iterable[str],
    pattern: str,
    attention_dir: Path,
    token_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    use_log_scale: bool = True,
):
    """Generate interactive HTML visualizations for the selected questions."""
    dataset_data = load_dataset(dataset, dataset_dir=dataset_dir)
    question_lookup = build_question_lookup(dataset_data)
    
    question_text_map = {qid: meta["question"] for qid, meta in question_lookup.items()}
    
    for question_id in question_ids:
        meta = question_lookup.get(question_id)
        if not meta:
            print(f"⚠️  Skipping unknown question_id: {question_id}")
            continue
        
        question_text = meta.get("question", "")
        farest_id = meta.get("farest_question_id")
        farest_text = question_text_map.get(farest_id, None)
        doc_id = meta["document_id"]
        
        try:
            tokens = load_tokens(dataset, doc_id, token_dir=token_dir)
            scores = load_attention_file(dataset, question_id, pattern, attention_dir=attention_dir)
        except FileNotFoundError as exc:
            print(f"⚠️  {exc}")
            continue
        
        dataset_output_dir = output_dir / dataset
        filename = f"{sanitize_filename(question_id)}_{pattern}.html"
        output_path = dataset_output_dir / filename
        
        create_html_visualization(
            tokens=tokens,
            scores=scores,
            question_text=question_text,
            farest_text=farest_text,
            dataset=dataset,
            doc_id=doc_id,
            question_id=question_id,
            pattern=pattern,
            output_path=output_path,
            use_log_scale=use_log_scale,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create interactive HTML visualizations for attention scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all questions for Llama3.2-1B with farest attention
  python attention_heatmap_html.py --dataset notice --pattern farest --model llama3.2_1b
  
  # Generate specific question ID for Qwen3-8B
  python attention_heatmap_html.py --dataset paper --pattern baseline --question-id 0_0 --model qwen3_8b
  
  # Generate all questions for a specific document with Qwen3-14B
  python attention_heatmap_html.py --dataset notice --pattern farest --document-id 5 --model qwen3_14b
  
  # Generate multiple specific questions with linear scaling
  python attention_heatmap_html.py --dataset paper --questions 0_0 0_2 1_1 --model llama3.2_1b --no-log-scale
        """
    )
    
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["paper", "notice", "aitqa"],
        help="Dataset to visualize.",
    )
    
    # ✅ NEW: Model selection (must match preprocessing)
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2_1b",
        choices=["llama3.2_1b", "qwen3_8b", "qwen3_14b"],
        help='Model used for preprocessing (MUST match!)',
    )
    
    parser.add_argument(
        "--pattern",
        default="farest",
        choices=["raw", "baseline", "farest"],
        help="Which attention pattern to plot.",
    )
    parser.add_argument(
        "--questions",
        nargs="*",
        help="Question IDs to visualize (e.g., 0_0 1_2). If omitted, plot all questions.",
    )
    parser.add_argument(
        "--document-id",
        type=int,
        help="Filter by specific document ID. If omitted, include all documents.",
    )
    parser.add_argument(
        "--question-id",
        type=str,
        help="Filter by specific question ID (e.g., 0_0). Takes precedence over --questions.",
    )
    parser.add_argument(
        "--no-log-scale",
        action="store_true",
        help="Use linear scaling instead of log-scaling for colors (log-scaling is default).",
    )
    parser.add_argument(
        "--attention-dir",
        type=str,
        default=None,
        help="Override attention directory (default: {model}/data/attention_summary).",
    )
    parser.add_argument(
        "--token-dir",
        type=str,
        default=None,
        help="Override token directory (default: {model}/data/tokens).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Override dataset directory (default: {model}/data/datasets).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: {model}/figures_attention_html).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # ✅ Resolve model name from mapping
    model_name = MODEL_MAPPING.get(args.model, args.model)
    
    # ✅ Setup model-specific directories
    model_base = BASE_DIR / args.model
    
    # Use provided paths or construct defaults with model prefix
    attention_dir = Path(args.attention_dir) if args.attention_dir else model_base / DEFAULT_ATTENTION_DIR
    token_dir = Path(args.token_dir) if args.token_dir else model_base / DEFAULT_TOKEN_DIR
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else model_base / DEFAULT_DATASET_DIR
    output_dir = Path(args.output_dir) if args.output_dir else model_base / DEFAULT_OUTPUT_DIR
    
    print("="*70)
    print("INTERACTIVE ATTENTION HEATMAP - HTML VERSION")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model} ({model_name})")
    print(f"Pattern: {args.pattern}")
    
    print(f"\n📂 Directory structure:")
    print(f"   Attention:  {attention_dir}/")
    print(f"   Tokens:     {token_dir}/")
    print(f"   Dataset:    {dataset_dir}/")
    print(f"   Output:     {output_dir}/")
    
    # ✅ WARNING: Check model consistency
    print(f"\n⚠️  IMPORTANT: Using data from model {args.model}")
    print(f"   Make sure data_reader.py was run with --model {args.model}")
    print()
    
    # Check if directories exist
    if not dataset_dir.exists():
        print(f"❌ ERROR: Dataset directory not found: {dataset_dir}")
        print(f"   Expected directory: {args.model}/data/datasets/")
        print(f"   Make sure you ran data_reader.py with --model {args.model}")
        return
    
    if not attention_dir.exists():
        print(f"❌ ERROR: Attention directory not found: {attention_dir}")
        print(f"   Expected directory: {args.model}/data/attention_summary/")
        print(f"   Make sure you ran data_reader.py with --model {args.model}")
        return
    
    if not token_dir.exists():
        print(f"❌ ERROR: Token directory not found: {token_dir}")
        print(f"   Expected directory: {args.model}/data/tokens/")
        print(f"   Make sure you ran data_reader.py with --model {args.model}")
        return
    
    dataset_data = load_dataset(args.dataset, dataset_dir=dataset_dir)
    question_lookup = build_question_lookup(dataset_data)
    
    use_log_scale = not args.no_log_scale
    
    # Determine which questions to process
    target_questions = []
    
    # Check for conflicting filters (both document-id and question-id)
    if args.document_id is not None and args.question_id:
        meta = question_lookup.get(args.question_id)
        if not meta:
            error_msg = f"Question ID '{args.question_id}' not found in dataset '{args.dataset}'."
            output_path = output_dir / args.dataset / "error_question_not_found.html"
            create_error_html(error_msg, args.dataset, output_path)
            print(f"❌ Question ID '{args.question_id}' not found")
            return
        
        if meta["document_id"] != args.document_id:
            error_msg = f"Question ID '{args.question_id}' belongs to document {meta['document_id']}, " \
                       f"not document {args.document_id}."
            output_path = output_dir / args.dataset / "error_mismatch.html"
            create_error_html(error_msg, args.dataset, output_path)
            print(f"❌ Question/Document ID mismatch")
            return
        
        target_questions = [args.question_id]
    
    # Priority 1: Specific question ID only
    elif args.question_id:
        if args.question_id in question_lookup:
            target_questions = [args.question_id]
        else:
            # Question ID not found - create error page
            error_msg = f"Question ID '{args.question_id}' not found in dataset '{args.dataset}'.\n\n" \
                       f"Available question IDs: {', '.join(sorted(question_lookup.keys())[:10])}..."
            output_path = output_dir / args.dataset / "error_question_not_found.html"
            create_error_html(error_msg, args.dataset, output_path)
            print(f"❌ Question ID '{args.question_id}' not found")
            return
    
    # Priority 2: Filter by document ID only
    elif args.document_id is not None:
        doc_id = args.document_id
        # Find all questions for this document
        for qid, meta in question_lookup.items():
            if meta["document_id"] == doc_id:
                target_questions.append(qid)
        
        if not target_questions:
            # Document ID not found - create error page
            available_docs = set(meta["document_id"] for meta in question_lookup.values())
            error_msg = f"Document ID {doc_id} not found in dataset '{args.dataset}'.\n\n" \
                       f"Available document IDs: {', '.join(map(str, sorted(available_docs)))}"
            output_path = output_dir / args.dataset / "error_document_not_found.html"
            create_error_html(error_msg, args.dataset, output_path)
            print(f"❌ Document ID {doc_id} not found")
            return
        
        target_questions.sort()
    
    # Priority 3: Specific questions list
    elif args.questions:
        target_questions = [str(qid) for qid in args.questions]
        # Check if any questions are invalid
        invalid_questions = [qid for qid in target_questions if qid not in question_lookup]
        if invalid_questions:
            error_msg = f"The following question IDs were not found: {', '.join(invalid_questions)}\n\n" \
                       f"Available question IDs: {', '.join(sorted(question_lookup.keys())[:10])}..."
            output_path = output_dir / args.dataset / "error_questions_not_found.html"
            create_error_html(error_msg, args.dataset, output_path)
            print(f"❌ Invalid question IDs: {', '.join(invalid_questions)}")
            return
    
    # Default: All questions
    else:
        target_questions = list(question_lookup.keys())
        target_questions.sort()
    
    if not target_questions:
        print("❌ No questions to process")
        return
    
    print(f"🔢 Questions to plot: {len(target_questions)}")
    if args.document_id is not None:
        print(f"📄 Filtering by document ID: {args.document_id}")
    if args.question_id:
        print(f"❓ Filtering by question ID: {args.question_id}")
    print(f"📈 Color scaling: {'logarithmic' if use_log_scale else 'linear'}")
    print()
    
    generate_figures(
        dataset=args.dataset,
        question_ids=target_questions,
        pattern=args.pattern,
        attention_dir=attention_dir,
        token_dir=token_dir,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        use_log_scale=use_log_scale,
    )
    
    print("\n" + "="*70)
    print("✅ HEATMAP GENERATION COMPLETE")
    print("="*70)
    print(f"📁 HTML files saved in: {output_dir}/{args.dataset}/")


if __name__ == "__main__":
    main()