#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Reader for Attention-Based Context Reduction
-------------------------------------------------
Handles dataset preprocessing, attention computation, and differential attention.

Usage:
    python data_reader.py --dataset paper
    python data_reader.py --dataset notice --skip-preprocessing
    python data_reader.py --dataset custom_dataset --custom-format
"""

import sys
import os
import warnings
import logging
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import random
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import gc

try:
    from datasets import load_dataset as hf_load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ Warning: 'datasets' library not installed. HuggingFace datasets will not be available.")
    print("   Install with: pip install datasets")

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['ACCELERATE_LOG_LEVEL'] = 'error'

class NoFlushStreamHandler(logging.StreamHandler):
    """Custom handler that doesn't flush to avoid permission errors"""
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

logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('rouge_score').setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.WARNING)
warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

tokenizer = None
model = None
device = None
embedding_model = None

# Model name mapping for easier selection
MODEL_MAPPING = {
    'llama3.2_1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'qwen3_8b': 'Qwen/Qwen3-8B',
    'qwen3_14b': 'Qwen/Qwen3-14B'
}

# Add to DATASET_REGISTRY (around line 85)
DATASET_REGISTRY = {
    # Existing entries...
    'paper': {
        'type': 'legacy',
        'config_path': '../questions/question.json'
    },
    'notice': {
        'type': 'legacy',
        'config_path': '../questions/question.json'
    },
    # Add AIT-QA
    'aitqa': {
        'type': 'local',
        'path': 'raw_data/aitqa_questions.jsonl',
        'custom_converter': 'aitqa',
        'converter_kwargs': {
            'questions_path': 'raw_data/aitqa/aitqa_questions.jsonl',
            'tables_path': 'raw_data/aitqa/aitqa_tables.jsonl'
        }
    },
    'quality': {
        'type': 'huggingface',
        'dataset': 'tasksource/QuALITY',
        'split': 'train',
        'custom_converter': 'quality'
    }
}

# KV cache - now a dictionary mapping chunk hash -> cached KV states
document_chunk_caches = {}
current_document_id = None

# Performance tracking - Track raw, baseline, baseline_diff, and farest separately for both modes
performance_stats = {
    'with_cache': {
        'raw': {  # Pure raw attention computation (used as input by baseline and farest)
            'total_time': 0.0,
            'num_queries': 0,
            'context_encoding_time': 0.0,
            'cache_hits': 0,
            'query_times': []
        },
        'baseline': {  # End-to-end baseline differential
            'total_time': 0.0,
            'num_queries': 0,
            'context_encoding_time': 0.0,  # raw + baseline question
            'cache_hits': 0,
            'baseline_attention_time': 0.0,  # Time for baseline question only
            'differential_time': 0.0  # Time for subtraction
        },
        'farest': {  # End-to-end farest differential
            'total_time': 0.0,
            'num_queries': 0,
            'context_encoding_time': 0.0,  # from raw
            'cache_hits': 0,
            'embedding_time': 0.0,
            'differential_time': 0.0
        }
    },
    'without_cache': {
        'raw': {  # Pure raw attention computation
            'total_time': 0.0,
            'num_queries': 0,
            'query_times': []
        },
        'baseline': {  # End-to-end baseline differential
            'total_time': 0.0,
            'num_queries': 0,
            'baseline_attention_time': 0.0,  # Time for baseline question only
            'differential_time': 0.0
        },
        'farest': {  # End-to-end farest differential
            'total_time': 0.0,
            'num_queries': 0,
            'embedding_time': 0.0,
            'differential_time': 0.0
        }
    }
}

# Supported legacy datasets (have special handling)
LEGACY_DATASETS = ['notice', 'paper']


# ============================================================================
# DATASET SOURCE MANAGEMENT
# ============================================================================

def register_dataset(name: str, source_type: str, **kwargs):
    """
    Register a new dataset in the registry.
    
    Args:
        name: Dataset name
        source_type: 'local', 'huggingface', or 'legacy'
        **kwargs: Additional parameters based on type
            - For 'local': path (required)
            - For 'huggingface': dataset, config, split, custom_converter
            - For 'legacy': config_path
    """
    DATASET_REGISTRY[name] = {'type': source_type, **kwargs}
    print(f"✅ Registered dataset '{name}' (type: {source_type})")
    if kwargs.get('custom_converter'):
        print(f"   Using custom converter: {kwargs['custom_converter']}")

def is_dataset_registered(name: str) -> bool:
    """Check if a dataset is registered."""
    return name in DATASET_REGISTRY

def get_dataset_info(name: str) -> Dict:
    """Get dataset information from registry."""
    if not is_dataset_registered(name):
        raise ValueError(f"Dataset '{name}' is not registered. Available datasets: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]

def load_from_local_file(file_path: str, custom_converter: str = None, **converter_kwargs) -> Dict:
    """
    Load dataset from local JSON file.
    
    Args:
        file_path: Path to local JSON file
        custom_converter: Name of custom converter to use
        **converter_kwargs: Additional arguments for custom converter
    
    Returns:
        Dictionary in standard dataset format
    """
    print(f"📖 Loading from local file: {file_path}")
    
    # Use custom converter if specified
    if custom_converter == 'aitqa':
        return convert_aitqa_dataset(
            questions_path=converter_kwargs.get('questions_path', file_path),
            tables_path=converter_kwargs.get('tables_path')
        )
    elif custom_converter == 'quality':
        # Quality should be loaded from HuggingFace, not local file
        raise ValueError(
            "QuALITY dataset should be loaded from HuggingFace, not local file. "
            "Use --dataset quality directly."
        )
    
    # Default: load standard JSON format
    with open(file_path) as f:
        dataset = json.load(f)
    
    # Validate format
    if 'dataset_name' not in dataset or 'documents' not in dataset:
        raise ValueError("Invalid dataset format. Must contain 'dataset_name' and 'documents' fields.")
    
    print(f"✅ Loaded dataset: {dataset['dataset_name']}")
    print(f"   Documents: {len(dataset['documents'])}")
    
    return dataset

def load_from_huggingface(dataset_name: str, config: str = None, 
                          split: str = 'train', custom_converter: str = None) -> Dict:
    """
    Load dataset from HuggingFace Hub.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., 'tasksource/QuALITY')
        config: Dataset configuration name (optional)
        split: Dataset split ('train', 'validation', 'test')
        custom_converter: Name of custom converter to use
    
    Returns:
        Dictionary in standard dataset format
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets library not available. "
            "Install with: pip install datasets"
        )
    
    print(f"📖 Loading from HuggingFace: {dataset_name}")
    if config:
        print(f"   Config: {config}")
    print(f"   Split: {split}")
    
    # Use custom converter if specified
    if custom_converter == 'quality':
        return convert_quality_dataset(dataset_name, config, split)
    
    # Default: load standard format (for future datasets)
    from datasets import load_dataset as hf_load
    dataset = hf_load(dataset_name, config, split=split)
    
    print(f"✅ Loaded {len(dataset)} examples from HuggingFace")
    
    # Would need conversion logic here for other datasets
    raise NotImplementedError(
        f"Converter for {dataset_name} not implemented. "
        "Please specify custom_converter or implement conversion logic."
    )


def get_or_create_cached_dataset(dataset_name: str, dataset_dir: str) -> Dict:
    """
    Get cached dataset or create it from source.
    
    Args:
        dataset_name: Name of the dataset
        dataset_dir: Directory to store cached datasets
    
    Returns:
        Dictionary in standard dataset format
    """
    cached_file = Path(dataset_dir) / f'{dataset_name}_source.json'
    
    # Check if cached version exists
    if cached_file.exists():
        print(f"📦 Loading cached dataset from: {cached_file}")
        with open(cached_file) as f:
            return json.load(f)
    
    # Load from source
    dataset_info = get_dataset_info(dataset_name)
    
    if dataset_info['type'] == 'huggingface':
        # ADD THIS BLOCK:
        dataset = load_from_huggingface(
            dataset_info['dataset'],
            dataset_info.get('config'),
            dataset_info.get('split', 'train'),
            dataset_info.get('custom_converter')
        )
    elif dataset_info['type'] == 'local':
        dataset = load_from_local_file(
            dataset_info['path'],
            custom_converter=dataset_info.get('custom_converter'),
            **dataset_info.get('converter_kwargs', {})
        )
    elif dataset_info['type'] == 'legacy':
        # Will be handled by load_legacy_dataset
        return None
    else:
        raise ValueError(f"Unknown dataset type: {dataset_info['type']}")
    
    # Cache the dataset
    os.makedirs(dataset_dir, exist_ok=True)
    with open(cached_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"💾 Cached dataset to: {cached_file}")
    
    return dataset

# ============================================================================
# PREPROCESSING STATUS CHECK
# ============================================================================

def check_preprocessing_status(dataset_name: str, dataset: Dict, token_dir: str, 
                               attention_dir: str) -> Tuple[bool, Dict]:
    """
    Check if preprocessing has been completed for this dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset: Dataset dictionary
        token_dir: Directory containing token files
        attention_dir: Directory containing attention files
    
    Returns:
        Tuple of (is_complete, missing_info)
    """
    print(f"\n🔍 Checking preprocessing status for '{dataset_name}'...")
    
    missing_tokens = []
    missing_attention = []
    missing_baseline = []
    
    for doc in dataset['documents']:
        doc_id = doc['document_id']
        
        # Check token file
        token_file = Path(token_dir) / f"{dataset_name}_{doc_id}.json"
        if not token_file.exists():
            missing_tokens.append(doc_id)
        
        # Check attention files for each question
        for question_data in doc['questions']:
            question_id = question_data['question_id']
            attention_file = Path(attention_dir) / f"{dataset_name}_{question_id}.npy"
            if not attention_file.exists():
                missing_attention.append(question_id)
        
        # Check baseline attention file
        baseline_file = Path(attention_dir) / f"{dataset_name}_{doc_id}_baseline.npy"
        if not baseline_file.exists():
            missing_baseline.append(doc_id)
    
    is_complete = (len(missing_tokens) == 0 and 
                  len(missing_attention) == 0 and 
                  len(missing_baseline) == 0)
    
    missing_info = {
        'tokens': missing_tokens,
        'attention': missing_attention,
        'baseline': missing_baseline
    }
    
    if is_complete:
        print(f"✅ Preprocessing is COMPLETE")
        print(f"   - All token files present")
        print(f"   - All attention files present")
        print(f"   - All baseline files present")
    else:
        print(f"⚠️  Preprocessing is INCOMPLETE")
        if missing_tokens:
            print(f"   - Missing {len(missing_tokens)} token files")
        if missing_attention:
            print(f"   - Missing {len(missing_attention)} attention files")
        if missing_baseline:
            print(f"   - Missing {len(missing_baseline)} baseline files")
    
    return is_complete, missing_info

def check_differential_status(dataset_name: str, dataset: Dict, 
                              attention_dir: str, 
                              check_farest: bool = True, 
                              check_baseline: bool = True) -> Tuple[bool, Dict]:
    """
    Check if differential attention computation has been completed.
    
    Args:
        dataset_name: Name of the dataset
        dataset: Dataset dictionary
        attention_dir: Directory containing attention files
        check_farest: Whether to check farest differential
        check_baseline: Whether to check baseline differential
    
    Returns:
        Tuple of (is_complete, missing_info)
    """
    print(f"\n🔍 Checking differential status for '{dataset_name}'...")
    
    missing_farest = []
    missing_baseline = []
    
    for doc in dataset['documents']:
        doc_id = doc['document_id']
        
        for question_data in doc['questions']:
            question_id = question_data['question_id']
            
            # Check farest differential
            if check_farest:
                farest_file = Path(attention_dir) / f"{dataset_name}_{question_id}_farest.npy"
                if not farest_file.exists():
                    missing_farest.append(question_id)
            
            # Check baseline differential
            if check_baseline:
                baseline_file = Path(attention_dir) / f"{dataset_name}_{question_id}_baseline.npy"
                if not baseline_file.exists():
                    missing_baseline.append(question_id)
    
    is_complete = (len(missing_farest) == 0 if check_farest else True) and \
                  (len(missing_baseline) == 0 if check_baseline else True)
    
    missing_info = {
        'farest': missing_farest if check_farest else [],
        'baseline': missing_baseline if check_baseline else []
    }
    
    if is_complete:
        print(f"✅ Differential computation is COMPLETE")
        if check_farest:
            print(f"   - All farest differential files present")
        if check_baseline:
            print(f"   - All baseline differential files present")
    else:
        print(f"⚠️  Differential computation is INCOMPLETE")
        if check_farest and missing_farest:
            print(f"   - Missing {len(missing_farest)} farest differential files")
        if check_baseline and missing_baseline:
            print(f"   - Missing {len(missing_baseline)} baseline differential files")
    
    return is_complete, missing_info

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def initialize_model(model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
    """Initialize the language model and tokenizer."""
    global tokenizer, model, device

    if tokenizer is not None and model is not None:
        print("✅ Model and tokenizer already initialized.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    print(f"📥 Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True
    )

    print(f"📥 Loading model {model_name}...")
    if device.type == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            output_attentions=True,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,
            trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Model and tokenizer initialized successfully.")

def initialize_embedding_model():
    """Initialize sentence embedding model for computing question similarity."""
    global embedding_model
    
    if embedding_model is not None:
        print("✅ Embedding model already initialized.")
        return
    
    print("📥 Loading embedding model for question similarity...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Embedding model initialized successfully.")

# ============================================================================
# KV CACHE MANAGEMENT
# ============================================================================

def clear_document_kv_cache():
    """Clear all KV caches when switching to a new document."""
    global document_chunk_caches, current_document_id
    document_chunk_caches = {}
    current_document_id = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

def get_chunk_hash(chunk_token_ids: List[int]) -> str:
    """Generate hash for chunk tokens."""
    return hashlib.md5(str(chunk_token_ids).encode()).hexdigest()

def encode_document_context(chunk_token_ids: List[int]):
    """Encode document context and return KV cache."""
    global tokenizer, model, device
    
    context_marker_ids = tokenizer.encode("Context: ", add_special_tokens=False)
    context_ids = context_marker_ids + chunk_token_ids
    context_inputs = torch.tensor([context_ids], device=device)
    
    with torch.no_grad():
        outputs = model(context_inputs, use_cache=True, output_attentions=False)
        past_key_values = outputs.past_key_values
    
    return past_key_values

def compute_chunk_attention_with_cache(chunk_token_ids: List[int], question: str, 
                                       use_cache: bool = True,
                                       question_type: str = 'raw') -> np.ndarray:
    """
    Compute attention scores using KV cache if available.
    
    Args:
        chunk_token_ids: List of token IDs for the chunk
        question: Question text
        use_cache: Whether to use KV cache
        question_type: Type of question ('raw', 'baseline_question') for performance tracking
    
    Returns:
        NumPy array of attention scores
    """
    global tokenizer, model, device, document_chunk_caches
    
    start_time = time.time()
    
    context_marker_ids = tokenizer.encode("Context: ", add_special_tokens=False)
    question_marker_ids = tokenizer.encode(f"\nQuestion: {question}\nAnswer:", add_special_tokens=False)
    
    context_start_idx = len(context_marker_ids)
    question_start_idx = len(context_marker_ids) + len(chunk_token_ids)
    
    if use_cache:
        from transformers import DynamicCache
        
        chunk_hash = get_chunk_hash(chunk_token_ids)
        
        if chunk_hash not in document_chunk_caches:
            # Cache miss - encode context
            context_encoding_start = time.time()
            past_key_values = encode_document_context(chunk_token_ids)
            document_chunk_caches[chunk_hash] = past_key_values
            context_encoding_time = time.time() - context_encoding_start
            
            # Track encoding time by question type
            if question_type == 'raw':
                performance_stats['with_cache']['raw']['context_encoding_time'] += context_encoding_time
            elif question_type == 'baseline_question':
                # Baseline question encoding goes to baseline stats
                performance_stats['with_cache']['baseline']['context_encoding_time'] += context_encoding_time
        else:
            # Cache hit!
            if question_type == 'raw':
                performance_stats['with_cache']['raw']['cache_hits'] += 1
            elif question_type == 'baseline_question':
                performance_stats['with_cache']['baseline']['cache_hits'] += 1
        
        original_cache_tuple = document_chunk_caches[chunk_hash]
        
        cache_copy = DynamicCache()
        for layer_idx, (key, value) in enumerate(original_cache_tuple):
            cache_copy.update(
                key.clone(),
                value.clone(),
                layer_idx
            )
        
        question_inputs = torch.tensor([question_marker_ids], device=device)
        
        with torch.no_grad():
            outputs = model(
                question_inputs,
                past_key_values=cache_copy,
                use_cache=True,
                output_attentions=True
            )
            attentions = outputs.attentions
        
        num_question_tokens = len(question_marker_ids)
        
        full_context_scores = torch.zeros(question_start_idx, device=device)
        
        for layer_attention in attentions:
            attention_to_context = layer_attention[0, :, :, :question_start_idx].sum(dim=(0, 1))
            full_context_scores += attention_to_context
        
    else:
        prompt_token_ids = context_marker_ids + chunk_token_ids + question_marker_ids
        inputs = torch.tensor([prompt_token_ids], device=device)
        
        with torch.no_grad():
            outputs = model(inputs, output_attentions=True)
            attentions = outputs.attentions
        
        num_question_tokens = len(prompt_token_ids) - question_start_idx
        
        full_context_scores = torch.zeros(question_start_idx, device=device)
        
        for layer_attention in attentions:
            attention_to_context = layer_attention[0, :, question_start_idx:, :question_start_idx].sum(dim=(0, 1))
            full_context_scores += attention_to_context

    if num_question_tokens <= 0:
        return np.array([])

    if num_question_tokens > 0:
        full_context_scores /= num_question_tokens

    actual_context_scores = full_context_scores[context_start_idx:].cpu().numpy()

    # Track performance by question type
    elapsed = time.time() - start_time
    if use_cache:
        if question_type == 'raw':
            performance_stats['with_cache']['raw']['total_time'] += elapsed
            performance_stats['with_cache']['raw']['num_queries'] += 1
            performance_stats['with_cache']['raw']['query_times'].append(elapsed)
        elif question_type == 'baseline_question':
            # Track baseline question time separately
            performance_stats['with_cache']['baseline']['baseline_attention_time'] += elapsed
    else:
        if question_type == 'raw':
            performance_stats['without_cache']['raw']['total_time'] += elapsed
            performance_stats['without_cache']['raw']['num_queries'] += 1
            performance_stats['without_cache']['raw']['query_times'].append(elapsed)
        elif question_type == 'baseline_question':
            # Track baseline question time separately
            performance_stats['without_cache']['baseline']['baseline_attention_time'] += elapsed

    # Cleanup
    del outputs, attentions
    if use_cache:
        del question_inputs, cache_copy
    else:
        del inputs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return actual_context_scores


def compute_chunk_attention(chunk_token_ids: List[int], question: str) -> np.ndarray:
    """
    Compute attention scores for a single chunk of context (backward compatibility).
    
    Args:
        chunk_token_ids: List of token IDs representing the context chunk
        question: The question text
    
    Returns:
        NumPy array of attention scores (one per token in chunk)
    """
    return compute_chunk_attention_with_cache(chunk_token_ids, question, use_cache=False)

def process_document_question_with_cache(dataset: str, doc_id: int, 
                                        questions_data: List[Dict],
                                        context: str,
                                        token_dir: str, attention_summary_dir: str, 
                                        max_chunk_length: int = 2000,
                                        use_kv_cache: bool = False,
                                        model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
    """
    Process all questions for a document with proper KV cache reuse.
    Now properly marks baseline questions.
    """
    global tokenizer, current_document_id
    if tokenizer is None:
        initialize_model(model_name)

    print(f"⚙️  Processing doc {doc_id} with {len(questions_data)} questions")
    print(f"    KV Cache: {'Enabled' if use_kv_cache else 'Disabled'}")
    
    # Tokenize full context once
    full_context_tokens = tokenizer.encode(context, add_special_tokens=False)
    
    if not full_context_tokens:
        print("⚠️ Warning: Context is empty. Skipping.")
        return
    
    # Save tokens once per document
    import os
    from pathlib import Path
    os.makedirs(token_dir, exist_ok=True)
    token_file = Path(token_dir) / f"{dataset}_{doc_id}.json"
    if not token_file.exists():
        import json
        context_tokens_as_strings = tokenizer.convert_ids_to_tokens(full_context_tokens)
        with open(token_file, 'w') as f:
            json.dump(context_tokens_as_strings, f)
        print(f"📄 Saved context tokens to {token_file}")
    
    # Split into chunks
    chunks = []
    for chunk_start in range(0, len(full_context_tokens), max_chunk_length):
        chunk_end = min(chunk_start + max_chunk_length, len(full_context_tokens))
        chunk_token_ids = full_context_tokens[chunk_start:chunk_end]
        chunks.append({
            'token_ids': chunk_token_ids,
            'start': chunk_start,
            'end': chunk_end
        })
    
    print(f"  📦 Split into {len(chunks)} chunks")
    
    question_attentions = {q['question_id']: [] for q in questions_data}
    chunk_lengths = []
    
    # Process chunks first, then questions (enables cache reuse!)
    for chunk_idx, chunk in enumerate(chunks):
        chunk_token_ids = chunk['token_ids']
        chunk_lengths.append(len(chunk_token_ids))
        
        print(f"\n  📦 Chunk {chunk_idx + 1}/{len(chunks)} (tokens {chunk['start']}:{chunk['end']}, length={len(chunk_token_ids)})")
        
        # Process ALL questions for this chunk before moving to next chunk
        for question_data in questions_data:
            question_id = question_data['question_id']
            question = question_data['question']
            
            # Determine question type for performance tracking
            question_type = question_data.get('type', 'raw')  # 'raw' or 'baseline_question'
            
            try:
                chunk_attention = compute_chunk_attention_with_cache(
                    chunk_token_ids, question, 
                    use_cache=use_kv_cache,
                    question_type=question_type
                )
                question_attentions[question_id].append(chunk_attention)
                
                if chunk_idx == 0:
                    type_label = "BASELINE" if question_type == 'baseline_question' else f"Q{question_data.get('idx', '?')}"
                    print(f"    ✓ {type_label}: {question[:40]}...")
            
            except Exception as e:
                print(f"    ❌ Error processing Q{question_id}: {e}")
                question_attentions[question_id].append(np.zeros(len(chunk_token_ids)))
    
    # Now combine chunks for each question and save
    print(f"\n  💾 Saving attention scores...")
    os.makedirs(attention_summary_dir, exist_ok=True)
    
    max_chunk_len = max(chunk_lengths)
    
    for question_data in questions_data:
        question_id = question_data['question_id']
        chunk_attentions = question_attentions[question_id]
        
        if not chunk_attentions:
            continue
        
        # Apply normalization by chunk length
        normalized_chunks = []
        for i, chunk_attention in enumerate(chunk_attentions):
            scaling_factor = chunk_lengths[i] / max_chunk_len
            normalized_chunk = chunk_attention * scaling_factor
            normalized_chunks.append(normalized_chunk)
        
        full_document_attention = np.concatenate(normalized_chunks)
        
        # Save
        attention_file = Path(attention_summary_dir) / f"{dataset}_{question_id}.npy"
        np.save(attention_file, full_document_attention)
    
    print(f"  ✅ Saved {len(questions_data)} attention files")

def process_document_question(dataset: str, doc_id: int, question_id: str, 
                              question: str, context: str,
                              token_dir: str, attention_summary_dir: str, 
                              max_chunk_length: int = 2000,
                              use_kv_cache: bool = False,
                              model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
    """
    Compute and save attention scores for a document-question pair.
    
    Args:
        dataset: Dataset name
        doc_id: Document ID
        question_id: Unique question identifier
        question: Question text
        context: Document context text
        token_dir: Directory to save tokens
        attention_summary_dir: Directory to save attention scores
        max_chunk_length: Maximum tokens per chunk
        use_kv_cache: Whether to use KV cache
        model_name: Name of the model to use
    """
    global tokenizer
    if tokenizer is None: 
        initialize_model(model_name)

    print(f"⚙️  Processing doc {doc_id}, question_id: {question_id}")
    print(f"    Question: {question[:50]}...")
    print(f"    KV Cache: {'Enabled' if use_kv_cache else 'Disabled'}")
    
    full_context_tokens = tokenizer.encode(context, add_special_tokens=False)

    if not full_context_tokens:
        print("⚠️ Warning: Context is empty. Skipping.")
        return

    all_chunk_attentions = []
    chunk_lengths = []

    # Process each chunk
    for chunk_start in range(0, len(full_context_tokens), max_chunk_length):
        chunk_end = min(chunk_start + max_chunk_length, len(full_context_tokens))
        chunk_token_ids = full_context_tokens[chunk_start:chunk_end]
        chunk_lengths.append(len(chunk_token_ids))
        print(f"  - Chunk tokens {chunk_start} to {chunk_end} (length: {len(chunk_token_ids)})")
        
        try:
            chunk_attention = compute_chunk_attention_with_cache(
                chunk_token_ids, question, use_cache=use_kv_cache
            )
            all_chunk_attentions.append(chunk_attention)
        except Exception as e:
            print(f"❌ Error processing chunk {chunk_start}-{chunk_end}: {e}")
            all_chunk_attentions.append(np.zeros(len(chunk_token_ids)))

    if not all_chunk_attentions:
        return

    # Apply normalization: scale by chunk length relative to max
    max_len = max(chunk_lengths)
    print(f"  📏 Max chunk length: {max_len}")
    
    normalized_chunks = []
    for i, chunk_attention in enumerate(all_chunk_attentions):
        scaling_factor = chunk_lengths[i] / max_len
        normalized_chunk = chunk_attention * scaling_factor
        normalized_chunks.append(normalized_chunk)
        print(f"  - Chunk {i}: length={chunk_lengths[i]}, scale={scaling_factor:.3f}")
    
    full_document_attention = np.concatenate(normalized_chunks)
    print(f"  ✅ Normalized by max chunk length")

    # Save results
    os.makedirs(token_dir, exist_ok=True)
    os.makedirs(attention_summary_dir, exist_ok=True)

    # Save tokens once per document
    token_file = Path(token_dir) / f"{dataset}_{doc_id}.json"
    if not token_file.exists():
        context_tokens_as_strings = tokenizer.convert_ids_to_tokens(full_context_tokens)
        with open(token_file, 'w') as f:
            json.dump(context_tokens_as_strings, f)
        print(f"📄 Saved context tokens to {token_file}")

    # Save attention scores for this question
    attention_summary_file = Path(attention_summary_dir) / f"{dataset}_{question_id}.npy"
    np.save(attention_summary_file, full_document_attention)
    print(f"💾 Saved attention scores to {attention_summary_file}")
    print(f"    Stats: min={np.min(full_document_attention):.6f}, "
          f"max={np.max(full_document_attention):.6f}, "
          f"mean={np.mean(full_document_attention):.6f}")


def save_performance_report(output_path: str):
    """Save performance comparison report with cleaner structure."""
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'with_cache': {
            'raw': {
                'total_queries': performance_stats['with_cache']['raw']['num_queries'],
                'total_time_seconds': performance_stats['with_cache']['raw']['total_time'],
                'context_encoding_time_seconds': performance_stats['with_cache']['raw']['context_encoding_time'],
                'cache_hits': performance_stats['with_cache']['raw']['cache_hits'],
                'avg_time_per_query': (
                    performance_stats['with_cache']['raw']['total_time'] / performance_stats['with_cache']['raw']['num_queries']
                    if performance_stats['with_cache']['raw']['num_queries'] > 0 else 0
                ),
                'query_times': performance_stats['with_cache']['raw']['query_times']
            },
            'baseline': {
                'total_queries': performance_stats['with_cache']['baseline']['num_queries'],
                'total_time_seconds': performance_stats['with_cache']['baseline']['total_time'],
                'context_encoding_time_seconds': performance_stats['with_cache']['baseline']['context_encoding_time'],
                'baseline_attention_time_seconds': performance_stats['with_cache']['baseline']['baseline_attention_time'],
                'differential_time_seconds': performance_stats['with_cache']['baseline']['differential_time'],
                'cache_hits': performance_stats['with_cache']['baseline']['cache_hits'],
                'avg_time_per_query': (
                    performance_stats['with_cache']['baseline']['total_time'] / performance_stats['with_cache']['baseline']['num_queries']
                    if performance_stats['with_cache']['baseline']['num_queries'] > 0 else 0
                )
            },
            'farest': {
                'total_queries': performance_stats['with_cache']['farest']['num_queries'],
                'total_time_seconds': performance_stats['with_cache']['farest']['total_time'],
                'context_encoding_time_seconds': performance_stats['with_cache']['farest']['context_encoding_time'],
                'embedding_time_seconds': performance_stats['with_cache']['farest']['embedding_time'],
                'differential_time_seconds': performance_stats['with_cache']['farest']['differential_time'],
                'cache_hits': performance_stats['with_cache']['farest']['cache_hits'],
                'avg_time_per_query': (
                    performance_stats['with_cache']['farest']['total_time'] / performance_stats['with_cache']['farest']['num_queries']
                    if performance_stats['with_cache']['farest']['num_queries'] > 0 else 0
                )
            }
        },
        'without_cache': {
            'raw': {
                'total_queries': performance_stats['without_cache']['raw']['num_queries'],
                'total_time_seconds': performance_stats['without_cache']['raw']['total_time'],
                'avg_time_per_query': (
                    performance_stats['without_cache']['raw']['total_time'] / performance_stats['without_cache']['raw']['num_queries']
                    if performance_stats['without_cache']['raw']['num_queries'] > 0 else 0
                ),
                'query_times': performance_stats['without_cache']['raw']['query_times']
            },
            'baseline': {
                'total_queries': performance_stats['without_cache']['baseline']['num_queries'],
                'total_time_seconds': performance_stats['without_cache']['baseline']['total_time'],
                'baseline_attention_time_seconds': performance_stats['without_cache']['baseline']['baseline_attention_time'],
                'differential_time_seconds': performance_stats['without_cache']['baseline']['differential_time'],
                'avg_time_per_query': (
                    performance_stats['without_cache']['baseline']['total_time'] / performance_stats['without_cache']['baseline']['num_queries']
                    if performance_stats['without_cache']['baseline']['num_queries'] > 0 else 0
                )
            },
            'farest': {
                'total_queries': performance_stats['without_cache']['farest']['num_queries'],
                'total_time_seconds': performance_stats['without_cache']['farest']['total_time'],
                'embedding_time_seconds': performance_stats['without_cache']['farest']['embedding_time'],
                'differential_time_seconds': performance_stats['without_cache']['farest']['differential_time'],
                'avg_time_per_query': (
                    performance_stats['without_cache']['farest']['total_time'] / performance_stats['without_cache']['farest']['num_queries']
                    if performance_stats['without_cache']['farest']['num_queries'] > 0 else 0
                )
            }
        }
    }
    
    # Calculate speedups for each type
    for qtype in ['raw', 'baseline', 'farest']:
        if (performance_stats['without_cache'][qtype]['num_queries'] > 0 and 
            performance_stats['with_cache'][qtype]['num_queries'] > 0):
            avg_without = report['without_cache'][qtype]['avg_time_per_query']
            avg_with = report['with_cache'][qtype]['avg_time_per_query']
            if avg_with > 0:
                report['with_cache'][qtype]['speedup'] = avg_without / avg_with
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Performance report saved to: {output_path}")
    print(f"\n{'='*60}")
    print("PERFORMANCE BREAKDOWN BY QUESTION TYPE")
    print('='*60)
    
    # Raw questions
    print(f"\n{'='*60}")
    print("📌 RAW ATTENTION (Regular Questions)")
    print('='*60)
    if report['with_cache']['raw']['total_queries'] > 0:
        print(f"\nWith KV Cache:")
        print(f"  Total queries: {report['with_cache']['raw']['total_queries']}")
        print(f"  Cache hits: {report['with_cache']['raw']['cache_hits']}")
        print(f"  Total time: {report['with_cache']['raw']['total_time_seconds']:.2f}s")
        print(f"  Context encoding time: {report['with_cache']['raw']['context_encoding_time_seconds']:.2f}s")
        print(f"  Average per query: {report['with_cache']['raw']['avg_time_per_query']:.3f}s")
        if 'speedup' in report['with_cache']['raw']:
            print(f"  🚀 Speedup: {report['with_cache']['raw']['speedup']:.2f}x")
    
    if report['without_cache']['raw']['total_queries'] > 0:
        print(f"\nWithout KV Cache:")
        print(f"  Total queries: {report['without_cache']['raw']['total_queries']}")
        print(f"  Total time: {report['without_cache']['raw']['total_time_seconds']:.2f}s")
        print(f"  Average per query: {report['without_cache']['raw']['avg_time_per_query']:.3f}s")
    
    # Baseline differential (end-to-end)
    print(f"\n{'='*60}")
    print("📌 BASELINE DIFFERENTIAL (End-to-End)")
    print('='*60)
    if report['with_cache']['baseline']['total_queries'] > 0:
        print(f"\nWith KV Cache:")
        print(f"  Total queries: {report['with_cache']['baseline']['total_queries']}")
        print(f"  Total time: {report['with_cache']['baseline']['total_time_seconds']:.2f}s")
        print(f"  └─ Context encoding (raw + baseline): {report['with_cache']['baseline']['context_encoding_time_seconds']:.2f}s")
        print(f"  └─ Baseline attention: {report['with_cache']['baseline']['baseline_attention_time_seconds']:.2f}s")
        print(f"  └─ Differential computation: {report['with_cache']['baseline']['differential_time_seconds']:.2f}s")
        print(f"  Average per query: {report['with_cache']['baseline']['avg_time_per_query']:.3f}s")
        if 'speedup' in report['with_cache']['baseline']:
            print(f"  🚀 Speedup: {report['with_cache']['baseline']['speedup']:.2f}x")
    
    if report['without_cache']['baseline']['total_queries'] > 0:
        print(f"\nWithout KV Cache:")
        print(f"  Total queries: {report['without_cache']['baseline']['total_queries']}")
        print(f"  Total time: {report['without_cache']['baseline']['total_time_seconds']:.2f}s")
        print(f"  └─ Baseline attention: {report['without_cache']['baseline']['baseline_attention_time_seconds']:.2f}s")
        print(f"  └─ Differential computation: {report['without_cache']['baseline']['differential_time_seconds']:.2f}s")
        print(f"  Average per query: {report['without_cache']['baseline']['avg_time_per_query']:.3f}s")
    
    # Farest differential (end-to-end)
    print(f"\n{'='*60}")
    print("📌 FAREST DIFFERENTIAL (End-to-End)")
    print('='*60)
    if report['with_cache']['farest']['total_queries'] > 0:
        print(f"\nWith KV Cache:")
        print(f"  Total queries: {report['with_cache']['farest']['total_queries']}")
        print(f"  Total time: {report['with_cache']['farest']['total_time_seconds']:.2f}s")
        print(f"  └─ Context encoding (from raw): {report['with_cache']['farest']['context_encoding_time_seconds']:.2f}s")
        print(f"  └─ Embedding computation: {report['with_cache']['farest']['embedding_time_seconds']:.2f}s")
        print(f"  └─ Differential computation: {report['with_cache']['farest']['differential_time_seconds']:.2f}s")
        print(f"  Average per query: {report['with_cache']['farest']['avg_time_per_query']:.3f}s")
        if 'speedup' in report['with_cache']['farest']:
            print(f"  🚀 Speedup: {report['with_cache']['farest']['speedup']:.2f}x")
    
    if report['without_cache']['farest']['total_queries'] > 0:
        print(f"\nWithout KV Cache:")
        print(f"  Total queries: {report['without_cache']['farest']['total_queries']}")
        print(f"  Total time: {report['without_cache']['farest']['total_time_seconds']:.2f}s")
        print(f"  └─ Embedding computation: {report['without_cache']['farest']['embedding_time_seconds']:.2f}s")
        print(f"  └─ Differential computation: {report['without_cache']['farest']['differential_time_seconds']:.2f}s")
        print(f"  Average per query: {report['without_cache']['farest']['avg_time_per_query']:.3f}s")
    
    print('='*60)

# ============================================================================
# QUESTION SIMILARITY COMPUTATION
# ============================================================================

def compute_question_embeddings(questions: List[str]) -> np.ndarray:
    """Compute embeddings for a list of questions."""
    global embedding_model
    if embedding_model is None:
        initialize_embedding_model()
    
    embeddings = embedding_model.encode(questions, show_progress_bar=False)
    return embeddings

def find_farest_questions(questions: List[str], embeddings: np.ndarray = None) -> List[int]:
    """
    Find the farest (most dissimilar) question for each question in the list.
    
    Args:
        questions: List of question texts
        embeddings: Pre-computed embeddings (optional)
    
    Returns:
        List of indices indicating the farest question for each question
    """
    if embeddings is None:
        embeddings = compute_question_embeddings(questions)
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # For each question, find the one with lowest similarity (excluding itself)
    farest_indices = []
    for i in range(len(questions)):
        similarities = similarity_matrix[i].copy()
        similarities[i] = 1.0  # Set self-similarity to max so it's not chosen
        farest_idx = np.argmin(similarities)
        farest_indices.append(int(farest_idx))
    
    return farest_indices

def compute_differential_attention(attention_a: np.ndarray, attention_b: np.ndarray) -> np.ndarray:
    """
    Compute differential attention: attention_a - attention_b
    
    Args:
        attention_a: First attention array
        attention_b: Second attention array
    
    Returns:
        Differential attention array
    """
    if len(attention_a) != len(attention_b):
        min_len = min(len(attention_a), len(attention_b))
        attention_a = attention_a[:min_len]
        attention_b = attention_b[:min_len]
    
    attention_diff = attention_a - attention_b
    return attention_diff

# ============================================================================
# DATASET LOADING
# ============================================================================

def _normalize_question_key(question: str) -> str:
    """Return a normalized key for question lookups."""
    return question.strip().lower() if question else ""

def _canonicalize_path(path: str) -> str:
    """Normalize separators for consistent comparisons."""
    if not path:
        return ""
    canonical = os.path.normpath(str(path)).replace('\\', '/')
    return canonical

def _path_key_variants(path: str) -> set:
    """Generate multiple comparable keys for a path."""
    keys = set()
    canonical = _canonicalize_path(path)
    if not canonical:
        return keys
    basename = os.path.basename(canonical)
    raw_data_split = canonical.split('/raw_data/', 1)
    candidates = [canonical, canonical.lower()]
    if basename:
        candidates.extend([basename, basename.lower()])
    if len(raw_data_split) == 2 and raw_data_split[1]:
        rel = raw_data_split[1]
        candidates.extend([rel, rel.lower()])
    for candidate in candidates:
        if candidate:
            keys.add(candidate)
    return keys

def _build_groundtruth_lookup(dataset_name: str, questions: List[str], grd_dir: Path) -> Tuple[Dict, Dict]:
    """
    Load ground truth JSON files from grd_loc and index them by question and file path.
    
    Returns:
        tuple(dict, dict): (question_lookup, stats)
    """
    lookup: Dict[str, Dict] = {}
    stats = {
        'questions_with_gt': 0,
        'entries_loaded': 0,
        'missing_questions': []
    }
    if not grd_dir.exists():
        print(f"⚠️ Ground truth directory not found: {grd_dir}")
        return lookup, stats
    pattern = f"{dataset_name}_*-location.json"
    available_files = list(grd_dir.glob(pattern))
    if not available_files:
        print(f"⚠️ No ground truth files matching '{pattern}' in {grd_dir}")
    for gt_file in sorted(available_files):
        try:
            with open(gt_file, 'r') as f:
                gt_data = json.load(f)
        except Exception as exc:
            print(f"⚠️ Failed to load ground truth file {gt_file}: {exc}")
            continue
        question_text = gt_data.get('question')
        question_key = _normalize_question_key(question_text)
        entries = gt_data.get('files', [])
        if not question_key:
            print(f"⚠️ Ground truth file {gt_file} missing 'question' field")
            continue
        answers_by_key: Dict[str, int] = {}
        for idx, entry in enumerate(entries):
            for key in _path_key_variants(entry.get('file')):
                if key in answers_by_key and answers_by_key[key] != idx:
                    print(f"⚠️ Duplicate ground truth key '{key}' in {gt_file}; overwriting previous entry.")
                answers_by_key[key] = idx
        lookup[question_key] = {
            'path': gt_file,
            'question': question_text,
            'entries': entries,
            'answers_by_key': answers_by_key
        }
        stats['questions_with_gt'] += 1
        stats['entries_loaded'] += len(entries)
    expected_keys = {_normalize_question_key(q) for q in questions}
    missing = sorted(q for q in questions if _normalize_question_key(q) not in lookup)
    stats['missing_questions'] = missing
    if missing:
        print(f"⚠️ Missing ground truth files for {len(missing)} questions:")
        for q in missing:
            print(f"   - {q}")
    else:
        print(f"✅ Ground truth files located for all {len(questions)} questions.")
    print(f"   → Loaded {stats['entries_loaded']} ground truth entries from {grd_dir}")
    return lookup, stats

def _find_answer_for_document(question_gt_data: Dict, doc_keys: set, doc_id: int,
                              total_docs: int) -> Tuple[str, str]:
    """
    Attempt to find the answer entry index for the current document.
    
    Returns:
        (answer, match_reason)
    """
    answers_by_key = question_gt_data['answers_by_key']
    entries = question_gt_data['entries']
    match_idx = None
    for key in doc_keys:
        if key in answers_by_key:
            match_idx = answers_by_key[key]
            match_reason = 'path-match'
            break
    else:
        match_reason = None
    if match_idx is None and doc_id < len(entries):
        entry_keys = _path_key_variants(entries[doc_id].get('file'))
        if doc_keys & entry_keys:
            match_idx = doc_id
            match_reason = 'path-index'
    if match_idx is None and len(entries) == total_docs:
        match_idx = doc_id
        match_reason = 'index-fallback'
    if match_idx is None:
        return None, None
    answer = entries[match_idx].get('groundtruth')
    return answer, match_reason

def load_legacy_dataset(dataset_name: str, config_path: str, home_dir: str) -> Dict:
    """
    Load legacy datasets (notice, paper, civic) from config file and convert to standard format.
    
    This function:
    1. Reads questions from config
    2. Reads ground truth from separate file (if available)
    3. Loads document content inline (no file_path dependencies)
    4. Formats everything into standard format matching other datasets
    
    Args:
        dataset_name: Name of the dataset ('notice' or 'paper')
        config_path: Path to the config JSON file
        home_dir: Home directory for file path conversion
    
    Returns:
        Dictionary in standard dataset format
    """
    print(f"📖 Loading legacy dataset: {dataset_name}")
    
    with open(config_path) as f:
        data = json.load(f)
    
    file_list = data["datasets"][dataset_name]["list"]
    questions = data["query"][f'{dataset_name.upper()}_QUESTIONS']
    
    print(f"   Found {len(file_list)} documents and {len(questions)} questions")
    
    # Load question-specific ground truths from grd_loc
    questions_dir = Path(config_path).resolve().parent
    grd_dir = questions_dir.parent / 'grd_loc'
    groundtruth_lookup, gt_stats = _build_groundtruth_lookup(dataset_name, questions, grd_dir)
    missing_question_keys = {_normalize_question_key(q) for q in gt_stats['missing_questions']}
    warned_missing_questions = set()
    
    # Convert to standard format with inline content
    documents = []
    
    for doc_id, file_path in enumerate(file_list):
        print(f"\n📄 Processing document {doc_id + 1}/{len(file_list)}")
        
        # Convert file path
        original_path = file_path
        if "/Users/chunwei/research/ZenDB/" in file_path:
            file_path = file_path.replace("/Users/chunwei/research/ZenDB/", f"{home_dir}/")
        
        print(f"   Original: {original_path}")
        print(f"   Converted: {file_path}")
        
        # Load document content inline (no file_path dependency)
        content = None
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    file_content = json.load(f)
                    # Try different field names for content
                    content = file_content.get("symbols", 
                             file_content.get("content", 
                             file_content.get("text", "")))
                print(f"   ✅ Loaded content: {len(content)} characters")
            except Exception as e:
                print(f"   ❌ Error loading document: {e}")
                content = None
        else:
            print(f"   ⚠️ File not found: {file_path}")
        
        if not content:
            print(f"   ⚠️ Skipping document {doc_id} - no content")
            continue
        
        # Extract file name for ground truth lookup
        file_name = os.path.basename(file_path)
        
        # Create questions for this document
        doc_questions = []
        doc_keys = set()
        doc_keys.update(_path_key_variants(original_path))
        doc_keys.update(_path_key_variants(file_path))
        
        for q_idx, question in enumerate(questions):
            question_id = f"{doc_id}_{q_idx}"
            
            # Try to find ground truth answer
            answer = None
            question_key = _normalize_question_key(question)
            question_gt_data = groundtruth_lookup.get(question_key)
            if question_gt_data:
                answer, match_reason = _find_answer_for_document(
                    question_gt_data,
                    doc_keys,
                    doc_id,
                    len(file_list)
                )
                if answer is None:
                    print(f"   ⚠️ No answer found for {question_id} despite available ground truth file.")
            elif question_key not in warned_missing_questions:
                print(f"   ⚠️ Ground truth file missing for question: '{question}'")
                warned_missing_questions.add(question_key)
                missing_question_keys.add(question_key)
            
            doc_questions.append({
                'question_id': question_id,
                'question': question,
                'answer': answer,  # Loaded from ground truth file!
                'farest_question_id': None  # Will be computed during differential
            })
        
        # Store document with inline content (standard format)
        documents.append({
            'document_id': doc_id,
            'content': content,  # ← Inline content, no file_path!
            'questions': doc_questions,
            'metadata': {
                'source': 'legacy',
                'original_path': original_path,
                'dataset': dataset_name
            }
        })
    
    # Count statistics
    total_questions = sum(len(doc['questions']) for doc in documents)
    answers_loaded = sum(1 for doc in documents for q in doc['questions'] if q['answer'] is not None)
    
    print(f"\n📊 Dataset statistics:")
    print(f"   Documents: {len(documents)}")
    print(f"   Total questions: {total_questions}")
    print(f"   Answers loaded: {answers_loaded}/{total_questions}")
    
    if answers_loaded == 0:
        print(f"\n⚠️  WARNING: No ground truth answers found!")
        print(f"   Evaluation will fail without answers.")
        print(f"   Please add ground truth files under: {grd_dir}")
    elif answers_loaded < total_questions:
        print(f"\n⚠️  WARNING: Only {answers_loaded}/{total_questions} answers found!")
        missing = total_questions - answers_loaded
        print(f"   Missing {missing} answers")
    else:
        print(f"\n✅ All answers loaded successfully!")
    
    return {
        'dataset_name': dataset_name,
        'documents': documents,
        'metadata': {
            'source': 'legacy',
            'config_path': config_path,
            'home_dir': home_dir,
            'total_documents': len(documents),
            'total_questions': total_questions,
            'answers_loaded': answers_loaded
        }
    }

def convert_aitqa_dataset(questions_path: str, tables_path: str) -> Dict:
    """
    Convert AIT-QA dataset from JSONL format to standard dataset format.
    
    Args:
        questions_path: Path to aitqa_questions.jsonl
        tables_path: Path to aitqa_tables.jsonl
    
    Returns:
        Dictionary in standard dataset format
    """
    print(f"📖 Converting AIT-QA dataset...")
    print(f"   Questions: {questions_path}")
    print(f"   Tables: {tables_path}")
    
    # Load tables
    tables = {}
    with open(tables_path, 'r') as f:
        for line in f:
            table = json.loads(line.strip())
            tables[table['id']] = table
    
    print(f"   Loaded {len(tables)} tables")
    
    # Load questions and group by table_id
    questions_by_table = {}
    with open(questions_path, 'r') as f:
        for line in f:
            question = json.loads(line.strip())
            table_id = question['table_id']
            if table_id not in questions_by_table:
                questions_by_table[table_id] = []
            questions_by_table[table_id].append(question)
    
    print(f"   Loaded questions for {len(questions_by_table)} tables")
    
    # Convert to standard format
    documents = []
    doc_id = 0
    
    for table_id, table_data in sorted(tables.items()):
        # Format table as readable JSON string for LLM
        table_content = format_table_for_llm(table_data)
        
        # Get questions for this table
        table_questions = questions_by_table.get(table_id, [])
        
        if not table_questions:
            print(f"   ⚠️ No questions found for table {table_id}, skipping")
            continue
        
        # Format questions
        formatted_questions = []
        for q_idx, q in enumerate(table_questions):
            question_id = f"{doc_id}_{q_idx}"
            # Handle multiple answers (take first one)
            answer = q['answers'][0] if q['answers'] else None
            
            formatted_questions.append({
                'question_id': question_id,
                'question': q['question'],
                'answer': answer,
                'farest_question_id': None,  # Will be computed later
                'metadata': {
                    'original_id': q['id'],
                    'type': q.get('type'),
                    'paraphrase_group': q.get('paraphrase_group')
                }
            })
        
        documents.append({
            'document_id': doc_id,
            'content': table_content,
            'questions': formatted_questions,
            'metadata': {
                'source': 'aitqa',
                'table_id': table_id,
                'num_rows': len(table_data['data']),
                'num_columns': len(table_data['column_header'])
            }
        })
        
        doc_id += 1
    
    print(f"✅ Converted {len(documents)} tables with questions")
    
    # Count total questions
    total_questions = sum(len(doc['questions']) for doc in documents)
    answers_loaded = sum(1 for doc in documents for q in doc['questions'] if q['answer'] is not None)
    
    print(f"   Total questions: {total_questions}")
    print(f"   Answers loaded: {answers_loaded}/{total_questions}")
    
    return {
        'dataset_name': 'aitqa',
        'documents': documents,
        'metadata': {
            'source': 'aitqa',
            'total_documents': len(documents),
            'total_questions': total_questions,
            'answers_loaded': answers_loaded
        }
    }

def convert_quality_dataset(dataset_name: str = 'tasksource/QuALITY', 
                            config: str = None, split: str = 'train') -> Dict:
    """
    Convert QuALITY dataset from HuggingFace to standard format.
    
    IMPORTANT: QuALITY uses 1-indexed labels (1, 2, 3, 4), 
               but Python arrays are 0-indexed (0, 1, 2, 3).
               We must subtract 1 from all label values!
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets library not available.")
    
    print(f"📖 Converting QuALITY dataset (split: {split})...")
    

    split = "validation"
    dataset = hf_load_dataset(dataset_name, split=split)
    print(f"   Loaded {len(dataset)} questions")
    
    def extract_article_id(question_unique_id: str) -> str:
        """Extract article identifier from question_unique_id."""
        parts = question_unique_id.split('_')
        if len(parts) >= 2:
            return '_'.join(parts[:2])
        return question_unique_id
    
    articles_dict = {}
    
    for idx, item in enumerate(dataset):
        question_unique_id = item['question_unique_id']
        article_id = extract_article_id(question_unique_id)
        
        if article_id not in articles_dict:
            articles_dict[article_id] = {
                'title': item.get('title', ''),
                'article': item.get('article', ''),
                'questions': []
            }
        
        # Only select difficult
        if item["difficult"] == 1:
            print("yes")
        else:
            continue


        # Label options (A, B, C, D, ...)
        options = item['options']

        options = item['options']
        num_options = len(options)

        num_options = len(options)
        labels = [chr(65 + i) for i in range(num_options)]  # A, B, C, D...
        labeled_options = [f"{label}. {option}" for label, option in zip(labels, options)]
        
        # Format question with options
        question_text = item['question']
        question_with_options = f"{question_text}\n\nOptions:\n" + "\n".join(labeled_options)
        
        # ✅ FIX: QuALITY uses 1-indexed labels, convert to 0-indexed
        gold_label_value = item.get('gold_label', -1)      # 1, 2, 3, or 4
        writer_label_value = item.get('writer_label', -1)  # 1, 2, 3, or 4
        
        # Convert to 0-indexed for array access
        gold_label_idx = gold_label_value - 1      # 0, 1, 2, or 3
        writer_label_idx = writer_label_value - 1  # 0, 1, 2, or 3
        
        # Get answer labels with bounds checking
        if 0 <= gold_label_idx < num_options:
            answer_label = labels[gold_label_idx]
        elif 0 <= writer_label_idx < num_options:
            answer_label = labels[writer_label_idx]
        else:
            answer_label = None
        
        writer_answer = labels[writer_label_idx] if 0 <= writer_label_idx < num_options else None
        gold_answer = labels[gold_label_idx] if 0 <= gold_label_idx < num_options else None
        
        articles_dict[article_id]['questions'].append({
            'question': question_with_options,
            'answer': answer_label,  # Now correctly maps to D when gold_label=4
            'question_unique_id': question_unique_id,
            'writer_label': writer_label_value,    # Store original 1-indexed value
            'writer_answer': writer_answer,
            'gold_label': gold_label_value,        # Store original 1-indexed value
            'gold_answer': gold_answer,
            'validation': item.get('validation', []),
            'speed_validation': item.get('speed_validation', []),
            'difficult': item['difficult']
        })
    
    # Convert to standard format
    documents = []
    doc_id = 0
    
    for article_id, article_data in sorted(articles_dict.items()):
        title = article_data['title'].strip()
        article_text = article_data['article'].strip()
        
        if title:
            content = f"Title: {title}\n\n{article_text}"
        else:
            content = article_text
        
        formatted_questions = []
        for q_idx, q_data in enumerate(article_data['questions']):
            question_id = f"{doc_id}_{q_idx}"
            
            formatted_questions.append({
                'question_id': question_id,
                'question': q_data['question'],
                'answer': q_data['answer'],
                'farest_question_id': None,
                'metadata': {
                    'original_id': q_data['question_unique_id'],
                    'writer_label': q_data['writer_label'],
                    'writer_answer': q_data['writer_answer'],
                    'gold_label': q_data['gold_label'],
                    'gold_answer': q_data['gold_answer'],
                    'validation': q_data['validation'],
                    'speed_validation': q_data['speed_validation'],
                    'difficult': q_data['difficult']
                }
            })
        
        documents.append({
            'document_id': doc_id,
            'content': content,
            'questions': formatted_questions,
            'metadata': {
                'source': 'quality',
                'article_id': article_id,
                'title': title,
                'article_length': len(article_text),
                'num_questions': len(formatted_questions)
            }
        })
        
        doc_id += 1
    
    print(f"✅ Converted {len(documents)} articles with questions")
    
    total_questions = sum(len(doc['questions']) for doc in documents)
    answers_loaded = sum(
        1 for doc in documents 
        for q in doc['questions'] 
        if q['answer'] is not None
    )
    
    print(f"   Total questions: {total_questions}")
    print(f"   Answers loaded: {answers_loaded}/{total_questions}")
    
    if answers_loaded < total_questions:
        missing = total_questions - answers_loaded
        print(f"   ⚠️  Warning: {missing} questions have null answers")
    
    return {
        'dataset_name': 'quality',
        'documents': documents,
        'metadata': {
            'source': 'quality',
            'hf_dataset': dataset_name,
            'split': split,
            'total_documents': len(documents),
            'total_questions': total_questions,
            'answers_loaded': answers_loaded
        }
    }
    
def format_table_for_llm(table_data: Dict) -> str:
    """
    Format table data as compact single-line JSON matching original JSONL format.
    
    Args:
        table_data: Table dictionary with column_header, row_header, and data
    
    Returns:
        Compact single-line JSON string
    """
    # Use exact original keys (not table_id, not column_headers)
    formatted_table = {
        'column_header': table_data['column_header'],  # ← singular, not 'column_headers'
        'row_header': table_data.get('row_header', []), # ← singular, not 'row_headers'
        'data': table_data['data'],
        'id': table_data['id']                          # ← 'id', not 'table_id'
    }
    
    # Compact JSON with no spaces
    return json.dumps(formatted_table, separators=(',', ':'))

def sample_dataset_documents(dataset: Dict, sample_size: int, verbose: bool = False) -> Dict:
    """Return a dataset dictionary sampled down to a fixed number of documents."""
    documents = dataset.get('documents', [])
    total_docs = len(documents)
    
    if total_docs == 0:
        print("⚠️  SAMPLE requested but dataset has no documents.")
        return dataset
    
    if sample_size <= 0:
        print("⚠️  SAMPLE size <= 0; ignoring sampling request.")
        return dataset
    
    if sample_size >= total_docs:
        print(f"ℹ️  SAMPLE size ({sample_size}) >= total docs ({total_docs}); using all documents.")
        return dataset
    
    sampled_docs = random.sample(documents, sample_size)
    sampled_docs.sort(key=lambda doc: doc.get('document_id', 0))
    
    dataset['documents'] = sampled_docs

    random.seed(42)
    sampled_docs = random.sample(documents, sample_size)
    
    metadata = dataset.setdefault('metadata', {})
    original_total = metadata.get('original_total_documents', metadata.get('total_documents', total_docs))
    metadata.update({
        'sampled': True,
        'sample_size': sample_size,
        'original_total_documents': original_total,
        'total_documents': len(sampled_docs),
        'total_questions': sum(len(doc.get('questions', [])) for doc in sampled_docs),
        'answers_loaded': sum(
            1 for doc in sampled_docs for q in doc.get('questions', []) if q.get('answer') is not None
        )
    })
    
    if verbose:
        print(f"🎯 SAMPLE applied: {sample_size}/{total_docs} documents selected.")
    
    return dataset

def save_processed_dataset(dataset: Dict, output_path: str):
    """Save processed dataset to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"✅ Saved processed dataset to {output_path}")

# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================
def run_preprocessing_with_cache(dataset_name: str, dataset: Dict, token_dir: str, 
                                 attention_dir: str, use_kv_cache: bool = False,
                                 model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
    """
    Run preprocessing with proper KV cache support.
    Now marks baseline questions correctly.
    """
    print("\n" + "="*60)
    print(f"STEP 1: PREPROCESSING - Computing attention scores for '{dataset_name}'")
    print(f"Model: {model_name}")
    print(f"KV Cache: {'Enabled' if use_kv_cache else 'Disabled'}")
    print("="*60)
    
    initialize_model(model_name)
    
    total_docs = len(dataset['documents'])
    for doc_idx, doc in enumerate(dataset['documents']):
        doc_id = doc['document_id']
        
        print(f"\n📄 Processing document {doc_idx + 1}/{total_docs} (ID: {doc_id})")
        
        if use_kv_cache:
            clear_document_kv_cache()
            print("  🗑️  Cleared KV cache for new document")
        
        # Load context
        context = None
        if 'content' in doc and doc['content']:
            context = doc['content']
            print(f"   Using inline content ({len(context)} characters)")
        elif 'file_path' in doc:
            file_path = doc['file_path']
            print(f"   Loading from: {file_path}")
            try:
                import json
                with open(file_path, 'r') as f:
                    file_content = json.load(f)
                    context = file_content.get("symbols", 
                             file_content.get("content", 
                             file_content.get("text", "")))
            except Exception as e:
                print(f"❌ Error loading document {doc_id}: {e}")
                continue
        else:
            print(f"❌ No content or file_path provided for document {doc_id}")
            continue
        
        if not context:
            print(f"⚠️ Warning: Empty context for document {doc_id}")
            continue
        
        # Prepare questions list with type markers
        questions_data = []
        for q_idx, question_data in enumerate(doc['questions']):
            questions_data.append({
                'question_id': question_data['question_id'],
                'question': question_data['question'],
                'idx': q_idx,
                'type': 'raw'  # Mark as raw question
            })
        
        # Add baseline question with type marker
        baseline_question_id = f"{doc_id}_baseline"
        questions_data.append({
            'question_id': baseline_question_id,
            'question': "Please repeat the context.",
            'idx': 'baseline',
            'type': 'baseline_question'  # Mark as baseline question
        })
        
        print(f"   Processing {len(doc['questions'])} raw + 1 baseline question...")
        
        # Process all questions for this document
        try:
            process_document_question_with_cache(
                dataset_name, doc_id, questions_data, context,
                token_dir, attention_dir, use_kv_cache=use_kv_cache,
                model_name=model_name
            )
        except Exception as e:
            print(f"❌ Error processing document {doc_id}: {e}")
            import traceback
            traceback.print_exc()
        
        if use_kv_cache:
            clear_document_kv_cache()
    
    print(f"\n✅ PREPROCESSING COMPLETE for '{dataset_name}'")

# ============================================================================
# DIFFERENTIAL ATTENTION PIPELINE
# ============================================================================

def ensure_doc_farest_assignments(doc: Dict, force: bool = False, verbose: bool = False) -> List[str]:
    """
    Ensure each question in the document has a farest_question_id.
    
    Args:
        doc: Document dictionary containing questions
        force: Recompute farest assignments even if already present
        verbose: Whether to print assignments
    
    Returns:
        List of farest question IDs for the document (aligned with question order)
    """
    questions = doc.get('questions', [])
    if not questions:
        return []
    
    question_texts = [q['question'] for q in questions]
    question_ids = [q['question_id'] for q in questions]
    
    needs_compute = force or any(q.get('farest_question_id') is None for q in questions)
    
    if needs_compute:
        embeddings = compute_question_embeddings(question_texts)
        farest_indices = find_farest_questions(question_texts, embeddings)
        for idx, question_data in enumerate(questions):
            farest_idx = farest_indices[idx]
            question_data['farest_question_id'] = question_ids[farest_idx]
    
    farest_ids = [q.get('farest_question_id') for q in questions]
    
    if verbose:
        id_to_idx = {qid: idx for idx, qid in enumerate(question_ids)}
        for idx, (question_id, farest_id) in enumerate(zip(question_ids, farest_ids)):
            if farest_id is None:
                print(f"    Q{idx} ({question_id}) -> Farest: None")
            else:
                farest_idx = id_to_idx.get(farest_id)
                if farest_idx is not None:
                    print(f"    Q{idx} ({question_id}) -> Farest Q{farest_idx} ({farest_id})")
                else:
                    print(f"    Q{idx} ({question_id}) -> Farest ({farest_id})")
    
    return farest_ids

def ensure_farest_assignments(dataset: Dict, force: bool = False, verbose: bool = False):
    """Ensure all documents in the dataset have farest question assignments."""
    for doc in dataset.get('documents', []):
        ensure_doc_farest_assignments(doc, force=force, verbose=verbose)

def run_differential(dataset_name: str, dataset: Dict, attention_dir: str,
                    compute_farest: bool = True, compute_baseline: bool = True,
                    track_mode: str = 'with_cache'):
    """
    Compute differential attention for all questions.
    Now tracks end-to-end timing for both baseline and farest.
    
    Args:
        dataset_name: Name of the dataset
        dataset: Dataset dictionary
        attention_dir: Directory containing attention files
        compute_farest: Whether to compute farest question differential
        compute_baseline: Whether to compute baseline differential
        track_mode: 'with_cache' or 'without_cache' for performance tracking
    """
    print("\n" + "="*60)
    print(f"STEP 2: DIFFERENTIAL - Computing differential attention for '{dataset_name}'")
    print(f"Tracking mode: {track_mode}")
    print("="*60)
    
    if not compute_farest and not compute_baseline:
        print("⚠️ No differential types selected. Skipping.")
        return
    
    # Start timing for end-to-end metrics
    farest_start_time = time.time() if compute_farest else 0
    baseline_start_time = time.time() if compute_baseline else 0
    
    for doc in dataset['documents']:
        doc_id = doc['document_id']
        
        print(f"\n📄 Document {doc_id}:")
        
        # ----------------------------------------------------------------
        # FAREST QUESTION DIFFERENTIAL
        # ----------------------------------------------------------------
        if compute_farest:
            # Time embedding computation
            embedding_start = time.time()
            print(f"\n  🔍 Computing farest questions...")
            farest_ids = ensure_doc_farest_assignments(doc, force=True, verbose=True)
            embedding_elapsed = time.time() - embedding_start
            performance_stats[track_mode]['farest']['embedding_time'] += embedding_elapsed

            # Time differential computation
            differential_start = time.time()
            print(f"\n  📊 Computing farest differential attention...")
            for question_data, farest_question_id in zip(doc['questions'], farest_ids):
                question_id = question_data['question_id']
                if not farest_question_id:
                    print(f"    ⚠️ Missing farest question assignment for {question_id}")
                    continue
                
                attention_file = Path(attention_dir) / f"{dataset_name}_{question_id}.npy"
                farest_attention_file = Path(attention_dir) / f"{dataset_name}_{farest_question_id}.npy"
                
                if not attention_file.exists():
                    print(f"    ⚠️ Missing attention file: {attention_file.name}")
                    continue
                if not farest_attention_file.exists():
                    print(f"    ⚠️ Missing farest attention file: {farest_attention_file.name}")
                    continue
                
                try:
                    attention = np.load(attention_file)
                    farest_attention = np.load(farest_attention_file)
                    
                    diff_attention = compute_differential_attention(attention, farest_attention)
                    
                    diff_file = Path(attention_dir) / f"{dataset_name}_{question_id}_farest.npy"
                    np.save(diff_file, diff_attention)
                    print(f"    ✅ Saved: {diff_file.name}")
                    
                    # Count this as a farest query
                    performance_stats[track_mode]['farest']['num_queries'] += 1
                    
                except Exception as e:
                    print(f"    ❌ Error for {question_id}: {e}")
            
            differential_elapsed = time.time() - differential_start
            performance_stats[track_mode]['farest']['differential_time'] += differential_elapsed
        
        # ----------------------------------------------------------------
        # BASELINE DIFFERENTIAL
        # ----------------------------------------------------------------
        if compute_baseline:
            # Time baseline differential computation
            baseline_diff_start = time.time()
            print(f"\n  🔍 Computing baseline differential...")
            
            baseline_question_id = f"{doc_id}_baseline"
            baseline_attention_file = Path(attention_dir) / f"{dataset_name}_{baseline_question_id}.npy"
            
            if not baseline_attention_file.exists():
                print(f"    ⚠️ Missing baseline attention file for doc {doc_id}")
                continue
            
            try:
                baseline_attention = np.load(baseline_attention_file)
                
                for question_data in doc['questions']:
                    question_id = question_data['question_id']
                    attention_file = Path(attention_dir) / f"{dataset_name}_{question_id}.npy"
                    
                    if not attention_file.exists():
                        print(f"    ⚠️ Missing attention file: {attention_file.name}")
                        continue
                    
                    attention = np.load(attention_file)
                    diff_attention = compute_differential_attention(attention, baseline_attention)
                    
                    diff_file = Path(attention_dir) / f"{dataset_name}_{question_id}_baseline.npy"
                    np.save(diff_file, diff_attention)
                    print(f"    ✅ Saved: {diff_file.name}")
                    
                    # Count this as a baseline query
                    performance_stats[track_mode]['baseline']['num_queries'] += 1
                    
            except Exception as e:
                print(f"    ❌ Error computing baseline differential for doc {doc_id}: {e}")
            
            baseline_diff_elapsed = time.time() - baseline_diff_start
            performance_stats[track_mode]['baseline']['differential_time'] += baseline_diff_elapsed
    
    # Complete farest timing
    if compute_farest:
        # Calculate from components for accurate end-to-end time
        performance_stats[track_mode]['farest']['total_time'] = (
            performance_stats[track_mode]['raw']['total_time'] +
            performance_stats[track_mode]['farest']['embedding_time'] +
            performance_stats[track_mode]['farest']['differential_time']
        )
        
        # Copy encoding time and cache hits from raw (farest reuses raw attention scores)
        if track_mode == 'with_cache':
            performance_stats[track_mode]['farest']['context_encoding_time'] = \
                performance_stats[track_mode]['raw']['context_encoding_time']
            performance_stats[track_mode]['farest']['cache_hits'] = \
                performance_stats[track_mode]['raw']['cache_hits']
    
    # Complete baseline timing
    if compute_baseline:
        # Calculate from components for accurate end-to-end time
        performance_stats[track_mode]['baseline']['total_time'] = (
            performance_stats[track_mode]['raw']['total_time'] +
            performance_stats[track_mode]['baseline']['baseline_attention_time'] +
            performance_stats[track_mode]['baseline']['differential_time']
        )
        
        if track_mode == 'with_cache':
            performance_stats[track_mode]['baseline']['context_encoding_time'] += \
                performance_stats[track_mode]['raw']['context_encoding_time']
            performance_stats[track_mode]['baseline']['cache_hits'] += \
                performance_stats[track_mode]['raw']['cache_hits']
    
    print(f"\n✅ DIFFERENTIAL COMPUTATION COMPLETE for '{dataset_name}'")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Data reader for attention-based context reduction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List registered datasets
  python data_reader.py --list-datasets
  
  # Process pre-registered legacy dataset 'paper'
  python data_reader.py --dataset paper
  
  # Process pre-registered NarrativeQA
  python data_reader.py --dataset narrativeqa
  
  # Register a new local dataset
  python data_reader.py --register my_data --type local --path /path/to/data.json
  
  # Register a new HuggingFace dataset (standard)
  python data_reader.py --register squad --type huggingface --hf-dataset squad
  
  # Register with custom converter
  python data_reader.py --register my_narrativeqa --type huggingface --hf-dataset deepmind/narrativeqa --custom-converter narrativeqa
  
  # Force re-preprocessing even if files exist
  python data_reader.py --dataset paper --force-preprocessing
  
  # Only compute farest differential (skip baseline)
  python data_reader.py --dataset paper --no-baseline
        """
    )
    
    # Main arguments
    parser.add_argument('--dataset', type=str,
                       help='Dataset name (must be registered)')
    
    # Registration arguments
    parser.add_argument('--register', type=str,
                       help='Register a new dataset with this name')
    parser.add_argument('--type', type=str, choices=['local', 'huggingface'],
                       help='Type of dataset to register')
    parser.add_argument('--path', type=str,
                       help='Path to local dataset file (for type=local)')
    parser.add_argument('--hf-dataset', type=str,
                       help='HuggingFace dataset name (for type=huggingface)')
    parser.add_argument('--hf-config', type=str,
                       help='HuggingFace dataset config')
    parser.add_argument('--hf-split', type=str, default='train',
                       help='HuggingFace dataset split (default: train)')
    parser.add_argument('--custom-converter', type=str,
                       help='Name of custom converter function (e.g., narrativeqa)')
    parser.add_argument('--model', type=str, default="llama3.2_1b", choices=["llama3.2_1b", "qwen3_8b", "qwen3_14b"], 
                        help='Model to use for attention computation (default: llama3.2_1b)')
    
    # Directory arguments
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory to store processed data')
    parser.add_argument('--home-dir', type=str, 
                    default='/home/x-xwang65/projects/AttentiveTrim/system',
                       help='Home directory for file path conversion (legacy datasets)')
    parser.add_argument('--sample', action='store_true',
                       help='Sample a subset of the dataset for quick experiments')
    parser.add_argument('--number', type=int, default=0,
                       help='Number of documents to sample when --sample is enabled')
    
    # Processing control
    parser.add_argument('--force-preprocessing', action='store_true',
                       help='Force re-preprocessing even if files exist')
    parser.add_argument('--skip-differential', action='store_true',
                       help='Skip differential computation step')
    parser.add_argument('--no-farest', action='store_true',
                       help='Skip farest question differential')
    parser.add_argument('--no-baseline', action='store_true',
                       help='Skip baseline differential')
    
    # List registered datasets
    parser.add_argument('--list-datasets', action='store_true',
                       help='List all registered datasets')

    # kv cache
    parser.add_argument('--use_cache', action='store_true',
                       help='Use KV cache optimization for faster processing')
    parser.add_argument('--compare-performance', action='store_true',
                       help='Run both with and without cache to compare performance')
    parser.add_argument('--performance-report', type=str, 
                       default='performance_report',
                       help='Path to save performance comparison report')
    
    args = parser.parse_args()
    # Resolve model name from mapping
    model_name = MODEL_MAPPING.get(args.model, args.model)
    
    # Handle listing datasets
    if args.list_datasets:
        print("="*60)
        print("REGISTERED DATASETS")
        print("="*60)
        for name, info in DATASET_REGISTRY.items():
            print(f"\n📊 {name}")
            print(f"   Type: {info['type']}")
            if info['type'] == 'huggingface':
                print(f"   Dataset: {info['dataset']}")
                if info.get('config'):
                    print(f"   Config: {info['config']}")
                print(f"   Split: {info.get('split', 'train')}")
                if info.get('custom_converter'):
                    print(f"   Custom Converter: {info['custom_converter']}")
            elif info['type'] == 'local':
                print(f"   Path: {info['path']}")
            elif info['type'] == 'legacy':
                print(f"   Config: {info['config_path']}")
        return
    
    # Handle registration
    if args.register:
        if not args.type:
            print("❌ ERROR: --type is required when registering a dataset")
            return
        
        if args.type == 'local':
            if not args.path:
                print("❌ ERROR: --path is required for local datasets")
                return
            register_dataset(args.register, 'local', path=args.path)
        elif args.type == 'huggingface':
            if not args.hf_dataset:
                print("❌ ERROR: --hf-dataset is required for HuggingFace datasets")
                return
            register_dataset(args.register, 'huggingface', 
                           dataset=args.hf_dataset,
                           config=args.hf_config,
                           split=args.hf_split,
                           custom_converter=args.custom_converter)
        
        print(f"\n✅ Dataset '{args.register}' registered successfully")
        print("   Run again with --dataset {args.register} to process it")
        return
    
    # Main processing
    if not args.dataset:
        print("❌ ERROR: --dataset is required")
        print("   Use --list-datasets to see available datasets")
        print("   Use --register to add a new dataset")
        return
    
    if args.sample and args.number <= 0:
        print("❌ ERROR: --number must be greater than 0 when using --sample")
        return
    
    print("="*60)
    print("DATA READER FOR ATTENTION-BASED CONTEXT REDUCTION")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model} ({model_name})")
    print(f"Force preprocessing: {args.force_preprocessing}")
    print(f"Skip differential: {args.skip_differential}")
    
    # Setup directories
    base_data_dir = os.path.join(args.model, args.data_dir)
    token_dir = os.path.join(base_data_dir, 'tokens')
    attention_dir = os.path.join(base_data_dir, 'attention_summary')
    dataset_dir = os.path.join(base_data_dir, 'datasets')
    os.makedirs(token_dir, exist_ok=True)
    os.makedirs(attention_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Check if dataset is registered
    if not is_dataset_registered(args.dataset):
        print(f"❌ ERROR: Dataset '{args.dataset}' is not registered")
        print(f"   Available datasets: {list(DATASET_REGISTRY.keys())}")
        print(f"   Use --register to add this dataset")
        return
    
    dataset_info = get_dataset_info(args.dataset)
    print(f"Dataset type: {dataset_info['type']}")
    
    # Load dataset based on type
    if dataset_info['type'] == 'legacy':
        dataset = load_legacy_dataset(args.dataset, dataset_info['config_path'], args.home_dir)
    else:
        # Use cached version or load from source
        dataset = get_or_create_cached_dataset(args.dataset, dataset_dir)
    
    if args.sample:
        dataset = sample_dataset_documents(dataset, args.number, verbose=True)
        print("⚠️  SAMPLE mode active - downstream steps will use only the sampled subset.")
    
    # Save processed dataset format
    dataset_file = os.path.join(dataset_dir, f'{args.dataset}_processed.json')

    # Setup performance report path EARLY
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)
    performance_report_name = os.path.join(reports_dir, f"{args.performance_report}_{args.dataset}_{args.model}.json")

    # Check preprocessing status
    preprocessing_complete, missing_preprocessing = check_preprocessing_status(
        args.dataset, dataset, token_dir, attention_dir
    )
    
    # Run preprocessing if needed
    if args.force_preprocessing or not preprocessing_complete:
        if args.force_preprocessing:
            print("\n🔄 FORCE PREPROCESSING enabled - will recompute all attention scores")
        else:
            print("\n🔄 PREPROCESSING needed - missing files detected")
        
        if args.compare_performance:
            print("\n🔬 PERFORMANCE COMPARISON MODE")
            print("   Running FULL PIPELINE twice: with and without KV cache")
            
            # Setup directories for no-cache run
            data_dir_no_cache_base = args.data_dir.replace('data', 'data_no_cache') if args.data_dir == 'data' else args.data_dir + '_no_cache'
            base_data_dir_no_cache = os.path.join(args.model, data_dir_no_cache_base)
            token_dir_no_cache = os.path.join(base_data_dir_no_cache, 'tokens')
            attention_dir_no_cache = os.path.join(base_data_dir_no_cache, 'attention_summary')
            dataset_dir_no_cache = os.path.join(base_data_dir_no_cache, 'datasets')
            os.makedirs(token_dir_no_cache, exist_ok=True)
            os.makedirs(attention_dir_no_cache, exist_ok=True)
            os.makedirs(dataset_dir_no_cache, exist_ok=True)
            
            # WITH KV CACHE - FULL PIPELINE
            print("\n" + "="*60)
            print("PHASE 2: WITH KV CACHE - FULL PIPELINE")
            print("="*60)
            print(f"   📁 Output directory: {args.data_dir}")
            print(f"   🤖 Model: {args.model} ({model_name})")
            
            # Step 2a: Preprocessing (with cache)
            run_preprocessing_with_cache(args.dataset, dataset, token_dir, 
                                        attention_dir, use_kv_cache=True,
                                        model_name=model_name)
            
            # Step 2b: Differential (with cache)
            if not args.skip_differential:
                print("\n" + "="*60)
                print("PHASE 2: DIFFERENTIAL COMPUTATION (WITH CACHE)")
                print("="*60)
                run_differential(
                    args.dataset, dataset, attention_dir,
                    compute_farest=not args.no_farest,
                    compute_baseline=not args.no_baseline,
                    track_mode='with_cache'  # ← Add this
                )

            # WITHOUT KV CACHE - FULL PIPELINE
            print("\n" + "="*60)
            print("PHASE 1: WITHOUT KV CACHE - FULL PIPELINE")
            print("="*60)
            print(f"   📁 Output directory: {dataset_dir_no_cache}")
            print(f"   🤖 Model: {args.model} ({model_name})")
            
            # Step 1a: Preprocessing (no cache)
            run_preprocessing_with_cache(args.dataset, dataset, token_dir_no_cache, 
                                        attention_dir_no_cache, use_kv_cache=False,
                                        model_name=model_name)
            
            # Step 1b: Differential (no cache)
            if not args.skip_differential:
                print("\n" + "="*60)
                print("PHASE 1: DIFFERENTIAL COMPUTATION (WITHOUT CACHE)")
                print("="*60)
                run_differential(
                    args.dataset, dataset, attention_dir_no_cache,
                    compute_farest=not args.no_farest,
                    compute_baseline=not args.no_baseline,
                    track_mode='without_cache'  # ← Add this
                )
            
            # Save processed dataset to primary location only
            # (Results are identical, cache only affects performance)
            save_processed_dataset(dataset, dataset_file)
            
            # Save performance report
            save_performance_report(performance_report_name)
            
            # Summary
            print(f"\n" + "="*60)
            print("✅ COMPARISON COMPLETE - FULL PIPELINE RUN FOR BOTH MODES")
            print("="*60)
            print(f"\n📁 PRIMARY OUTPUT (for evaluation):")
            print(f"   Directory:  {args.data_dir}/")
            print(f"   Tokens:     {token_dir}")
            print(f"   Attention:  {attention_dir}")
            print(f"   Dataset:    {dataset_file}")
            print(f"\n📁 COMPARISON DATA (WITHOUT CACHE - for performance testing only):")
            print(f"   Directory:  {dataset_dir_no_cache}/")
            print(f"   Tokens:     {token_dir_no_cache}")
            print(f"   Attention:  {attention_dir_no_cache}")
            print(f"   Note:       These files are identical to primary output")
            print(f"               They exist only to verify cache correctness")
            print(f"\n📊 Performance Report: {performance_report_name}")
            print(f"\n💡 IMPORTANT:")
            print(f"   - Use 'data/' directory for downstream evaluation")
            print(f"   - 'data_no_cache/' is for performance verification only")
            print(f"   - Both contain identical results (cache doesn't change output)")
            
        else:
            # Normal single run
            run_preprocessing_with_cache(args.dataset, dataset, token_dir, attention_dir, 
                            use_kv_cache=args.use_cache,
                            model_name=model_name)
            
            if args.use_cache:
                save_performance_report(performance_report_name)
    else:
        print("\n✅ PREPROCESSING already complete - skipping")
        print("   Use --force-preprocessing to recompute")

    # Check differential status (for non-comparison mode)
    if not args.compare_performance and not args.skip_differential:
        differential_complete, missing_differential = check_differential_status(
            args.dataset, dataset, attention_dir,
            check_farest=not args.no_farest,
            check_baseline=not args.no_baseline
        )
        
        # Run differential if needed
        if not differential_complete:
            print("\n🔄 DIFFERENTIAL computation needed")
            run_differential(
                args.dataset, dataset, attention_dir,
                compute_farest=not args.no_farest,
                compute_baseline=not args.no_baseline,
                track_mode='with_cache' if args.use_cache else 'without_cache'  # ← Add this
            )
        else:
            print("\n✅ DIFFERENTIAL computation already complete - skipping")

    # Ensure farest assignments exist even if differential was skipped
    if not args.no_farest:
        ensure_farest_assignments(dataset, force=False, verbose=False)

    # Persist the latest dataset view (for non-comparison mode)
    if not args.compare_performance:
        save_processed_dataset(dataset, dataset_file)

        print("\n" + "="*60)
        print(f"✅ COMPLETE: Dataset '{args.dataset}' is ready for evaluation")
        print("="*60)
        print(f"📁 Tokens: {token_dir}")
        print(f"📁 Attention: {attention_dir}")
        print(f"📁 Dataset: {dataset_file}")
        print("\nNext step: Run unit_window.py to generate evaluation results")

if __name__ == "__main__":
    main()