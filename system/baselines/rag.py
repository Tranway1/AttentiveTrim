#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Baseline for Context Reduction - WITH PERFORMANCE TRACKING
---------------------------------------------------------------

This script implements a pure RAG (Retrieval Augmented Generation) baseline
for question answering over large documents with comprehensive performance tracking.

**RAG Workflow:**

1. **Preprocessing:**
   - Load raw documents from processed dataset files
   - Tokenize each document
   - Save tokens to baseline/data/tokens/
   - Chunk tokens into fixed-size chunks (200 tokens by default)
   - Compute embeddings for all chunks (TRACKED)
   - Cache embeddings to baseline/data/rag_embeddings/

2. **Retrieval & Generation:**
   - For each document, process its associated questions
   - Find the most similar chunks using embedding-based cosine similarity
   - Pick the top K most relevant chunks
   - Reorder chunks by original document position
   - Separate chunks with "..." in context
   - Pass to Gemini 2.5 Flash Lite to generate an answer (TRACKED)
   - Store results in baseline/pred_att/

3. **Performance Reporting:**
   - Track time for embedding each chunk
   - Track time for generating each answer
   - Generate performance report with statistics
   - Save to ../reports/performance_report_{dataset}_rag.json

**Prerequisites:**
- Processed datasets in data/datasets/{dataset}_processed.json
- GEMINI_API_KEY environment variable

**Features:**
- Works with processed dataset structure
- Chunks ordered by original position (not similarity)
- Chunks separated with "..." in prompt
- Outputs saved to baseline/ subdirectory
- Performance reports saved to ../reports/
"""

import sys
import os
import warnings
import logging
import argparse

# Suppress logging
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

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

# Add parent directory to path
from pathlib import Path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple
import numpy as np
import json
import gc
import time
import pytz
from datetime import datetime
import google.generativeai as genai
from transformers import BitsAndBytesConfig
import re
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ==============================================================================
# PATH CONFIGURATION - Since file is now in baseline/ subdirectory
# ==============================================================================

def get_project_root() -> Path:
    """Get the project root directory (parent of baseline/)."""
    return Path(__file__).parent.parent

def get_baseline_dir() -> Path:
    """Get the baseline directory (where this script is located)."""
    return Path(__file__).parent

# Input data paths - read from project root (original location)
DATA_ROOT = get_project_root() / "data"
QUESTIONS_ROOT = get_project_root() / "questions"

# Output paths - write to baseline subdirectory
PRED_ATT_ROOT = get_baseline_dir() / "pred_att"

# Cache paths - in baseline (isolated from other experiments)
TOKEN_CACHE_ROOT = get_baseline_dir() / "data" / "tokens"
EMBEDDING_CACHE_ROOT = get_baseline_dir() / "data" / "rag_embeddings"

# Report paths - write to parent reports directory
REPORTS_ROOT = get_project_root() / "reports"

# Global variables
tokenizer = None
embedding_model = None
current_embedding_model_name = None

AVAILABLE_EMBEDDING_MODELS = [
    'UAE-Large-V1',
    'bflhc/Octen-Embedding-4B',
    'Qwen3-Embedding-8B'
]

# ==============================================================================
# PERFORMANCE TRACKING STRUCTURES
# ==============================================================================

class PerformanceTracker:
    """Track performance metrics for RAG baseline."""
    
    def __init__(self, dataset_name: str, model_name: str = None, top_k: int = None):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.top_k = top_k
        self.start_time = time.time()
        
        # Document-level tracking
        self.total_documents = 0
        self.total_chunks = 0
        
        # Embedding tracking
        self.chunk_embedding_times = []  # List of (doc_id, chunk_idx, time_seconds)
        self.total_embedding_time = 0.0
        
        # Query tracking
        self.query_generation_times = []  # List of (doc_id, question_id, time_seconds)
        self.total_query_time = 0.0
        
    def record_chunk_embedding(self, doc_id: str, chunk_idx: int, time_seconds: float):
        """Record time taken to embed a single chunk."""
        self.chunk_embedding_times.append({
            "doc_id": doc_id,
            "chunk_idx": chunk_idx,
            "time_seconds": time_seconds
        })
        self.total_embedding_time += time_seconds
    
    def record_query_generation(self, doc_id: str, question_id: str, time_seconds: float):
        """Record time taken to generate answer for a query."""
        self.query_generation_times.append({
            "doc_id": doc_id,
            "question_id": question_id,
            "time_seconds": time_seconds
        })
        self.total_query_time += time_seconds
    
    def generate_report(self) -> Dict:
        """Generate performance report with all statistics."""
        total_time = time.time() - self.start_time
        
        # Calculate averages
        avg_time_per_chunk = (self.total_embedding_time / len(self.chunk_embedding_times) 
                              if self.chunk_embedding_times else 0.0)
        avg_time_per_query = (self.total_query_time / len(self.query_generation_times)
                              if self.query_generation_times else 0.0)
        
        report = {
            "timestamp": datetime.now(pytz.timezone('America/New_York')).isoformat(),
            "dataset": self.dataset_name,
            "model": self.model_name,
            "top_k": self.top_k,
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "total_time_seconds": total_time,
            "context_embedding_time_seconds": self.total_embedding_time,
            "query_generation_time_seconds": self.total_query_time,
            "avg_time_per_chunk": avg_time_per_chunk,
            "avg_time_per_query": avg_time_per_query,
            "num_chunks_embedded": len(self.chunk_embedding_times),
            "num_queries_generated": len(self.query_generation_times),
            "chunk_embedding_times": self.chunk_embedding_times,
            "query_generation_times": self.query_generation_times
        }
        
        return report
    
    def save_report(self, output_dir: Path = None):
        """Save performance report to JSON file."""
        if output_dir is None:
            output_dir = REPORTS_ROOT
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = self.generate_report()
        
        # Include model name and top_k in the report filename
        safe_model_name = self.model_name.replace('/', '_').replace('-', '_') if self.model_name else "unknown"
        k_suffix = f"_k{self.top_k}" if self.top_k is not None else ""
        report_file = output_dir / f"performance_report_{self.dataset_name}_{safe_model_name}{k_suffix}_rag.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"📊 PERFORMANCE REPORT SAVED")
        print(f"{'='*60}")
        print(f"📁 Location: {report_file}")
        print(f"📋 Dataset: {self.dataset_name}")
        print(f"🤖 Model: {self.model_name}")
        print(f"🔍 Top-k: {self.top_k}")
        print(f"📄 Documents: {self.total_documents}")
        print(f"📦 Chunks: {self.total_chunks}")
        print(f"⏱️  Total Time: {report['total_time_seconds']:.2f}s")
        print(f"🔢 Embedding Time: {report['context_embedding_time_seconds']:.2f}s")
        print(f"💬 Query Time: {report['query_generation_time_seconds']:.2f}s")
        print(f"📈 Avg per Chunk: {report['avg_time_per_chunk']:.4f}s")
        print(f"📈 Avg per Query: {report['avg_time_per_query']:.4f}s")
        print(f"{'='*60}")
        
        return report_file

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def load_processed_dataset(dataset_name: str, dataset_dir: str = None) -> Dict:
    """
    Load processed dataset file containing ground truth.
    
    Args:
        dataset_name: Name of the dataset
        dataset_dir: Directory containing processed dataset files
    
    Returns:
        Dictionary containing the full dataset
    """
    if dataset_dir is None:
        dataset_dir = DATA_ROOT / 'datasets'
    
    processed_file = Path(dataset_dir) / f'{dataset_name}_processed.json'
    
    if not processed_file.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_file}")
    
    print(f"📖 Loading dataset from: {processed_file}")
    
    with open(processed_file) as f:
        dataset = json.load(f)
    
    num_docs = len(dataset.get('documents', []))
    print(f"✅ Loaded dataset with {num_docs} documents")
    
    return dataset

# ==============================================================================
# SECTION 1: INITIALIZATION
# ==============================================================================

def clear_gpu_memory():
    """Aggressively clear GPU memory cache"""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        time.sleep(2)

def initialize_tokenizer(model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
    """Initialize tokenizer for token processing."""
    global tokenizer
    
    if tokenizer is not None:
        print("✅ Tokenizer already initialized")
        return
    
    print(f"🔄 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        cache_dir="/tmp/huggingface-cache", 
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Tokenizer initialized")


def initialize_embedding_model(model_name: str):
    global embedding_model, current_embedding_model_name
    
    if current_embedding_model_name == model_name and embedding_model is not None:
        print(f"✅ Embedding model '{model_name}' already initialized")
        return
    
    # Clear any existing model from memory
    if embedding_model is not None:
        print(f"🔄 Clearing previous model from memory...")
        del embedding_model
        embedding_model = None
        clear_gpu_memory()
    
    model_mapping = {
        'UAE-Large-V1': 'WhereIsAI/UAE-Large-V1',
        'bflhc/Octen-Embedding-4B': 'bflhc/Octen-Embedding-4B',
        'Qwen3-Embedding-8B': 'Qwen/Qwen3-Embedding-8B',
        'GritLM-7B': 'GritLM/GritLM-7B'
    }
    
    # Models that require 8-bit quantization
    quantized_models = ['Qwen3-Embedding-8B', 'GritLM-7B']
    
    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {list(model_mapping.keys())}")
    
    hf_model_name = model_mapping[model_name]
    use_quantization = model_name in quantized_models
    
    print(f"🔄 Loading embedding model: {hf_model_name}")
    if use_quantization:
        print(f"   Using 8-bit quantization to reduce memory usage")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {device}")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_free = total_memory - memory_allocated
            print(f"   GPU Memory Total: {total_memory:.2f} GB")
            print(f"   GPU Memory Free: {memory_free:.2f} GB")
        
        # 8-bit quantization for large models
        if use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            embedding_model = SentenceTransformer(
                hf_model_name,
                cache_folder="/tmp/huggingface-cache",
                trust_remote_code=True,
                model_kwargs={
                    'quantization_config': quantization_config,
                    'device_map': 'auto'
                }
            )
        # Standard loading for smaller models
        else:
            embedding_model = SentenceTransformer(
                hf_model_name,
                cache_folder="/tmp/huggingface-cache",
                trust_remote_code=True,
                device=device
            )
        
        current_embedding_model_name = model_name
        print(f"✅ Embedding model loaded successfully!")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"   GPU Memory After Loading - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        import traceback
        traceback.print_exc()
        raise


def initialize_gemini():
    """Initialize Gemini API for answer generation."""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ Gemini API configured")
    else:
        print("⚠️ Warning: GEMINI_API_KEY not found in environment variables")


# ==============================================================================
# SECTION 2: CHUNKING AND EMBEDDING
# ==============================================================================

def chunk_tokens(tokens: List[str], chunk_size: int = 200, overlap: int = 50) -> List[Tuple[int, int, List[str]]]:
    """
    Split tokens into overlapping chunks.
    
    Args:
        tokens: List of token strings
        chunk_size: Target size for each chunk (default: 200)
        overlap: Number of overlapping tokens between chunks (default: 50)
    
    Returns:
        List of tuples (start_idx, end_idx, chunk_tokens)
    """
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = tokens[start:end]
        chunks.append((start, end, chunk))
        
        # Move to next chunk with overlap
        start = end - overlap if end < len(tokens) else end
        
        if start >= len(tokens):
            break
    
    print(f"  📦 Created {len(chunks)} chunks from {len(tokens)} tokens")
    return chunks


def embed_chunks(chunks: List[Tuple[int, int, List[str]]], 
                 doc_id: str = None, 
                 tracker: PerformanceTracker = None) -> np.ndarray:
    """
    Create embeddings for all chunks with per-chunk timing.
    
    Args:
        chunks: List of (start_idx, end_idx, chunk_tokens) tuples
        doc_id: Document ID for tracking purposes
        tracker: PerformanceTracker instance to record timings
    
    Returns:
        NumPy array of embeddings (num_chunks x embedding_dim)
    """
    global embedding_model, tokenizer, current_embedding_model_name
    
    if embedding_model is None:
        raise ValueError("Embedding model not initialized")
    
    if tokenizer is None:
        initialize_tokenizer()
    
    print(f"  🔄 Creating embeddings for {len(chunks)} chunks...")
    
    # Convert token lists to text
    chunk_texts = []
    for start, end, chunk_tokens in chunks:
        token_ids = tokenizer.convert_tokens_to_ids(chunk_tokens)
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        # Clean text to avoid issues with multimodal models
        text = text.strip()
        
        # Skip empty chunks
        if not text:
            text = "[EMPTY]"
        
        chunk_texts.append(text)
    
    # Create embeddings in batches with per-chunk timing
    batch_size = 8
    all_embeddings = []
    
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i + batch_size]
        
        try:
            # Time the batch embedding
            batch_start_time = time.time()
            embeddings = embedding_model.encode(
                batch,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            batch_end_time = time.time()
            
            all_embeddings.append(embeddings.cpu().numpy())
            
            # Record timing for each chunk in the batch
            if tracker is not None and doc_id is not None:
                batch_time = batch_end_time - batch_start_time
                time_per_chunk = batch_time / len(batch)
                for j in range(len(batch)):
                    chunk_idx = i + j
                    tracker.record_chunk_embedding(doc_id, chunk_idx, time_per_chunk)
            
        except Exception as e:
            print(f"  ⚠️  Error encoding batch {i//batch_size + 1}: {e}")
            print(f"  🔄 Retrying with individual chunks...")
            
            # Retry individual chunks in this batch
            for j, text in enumerate(batch):
                try:
                    chunk_start_time = time.time()
                    embedding = embedding_model.encode(
                        [text],
                        convert_to_tensor=True,
                        show_progress_bar=False
                    )
                    chunk_end_time = time.time()
                    all_embeddings.append(embedding.cpu().numpy())
                    
                    # Record timing for this chunk
                    if tracker is not None and doc_id is not None:
                        chunk_time = chunk_end_time - chunk_start_time
                        tracker.record_chunk_embedding(doc_id, i + j, chunk_time)
                        
                except Exception as e2:
                    print(f"  ❌ Failed to encode chunk {i+j}: {e2}")
                    print(f"     Text preview: {text[:100]}...")
                    # Create a zero embedding as fallback
                    dummy_embedding = np.zeros((1, embedding_model.get_sentence_embedding_dimension()))
                    all_embeddings.append(dummy_embedding)
    
    embeddings_array = np.vstack(all_embeddings)
    print(f"  ✅ Created embeddings with shape: {embeddings_array.shape}")
    
    return embeddings_array

def retrieve_top_k_chunks(question: str, 
                          chunks: List[Tuple[int, int, List[str]]], 
                          chunk_embeddings: np.ndarray,
                          top_k: int = 5) -> List[Tuple[int, int, List[str], float]]:
    """
    Retrieve top-k most relevant chunks for a question using cosine similarity.
    
    Uses sentence_transformers.util.cos_sim for robust similarity computation.
    
    Args:
        question: The question to answer
        chunks: List of (start_idx, end_idx, chunk_tokens) tuples
        chunk_embeddings: Embeddings for all chunks (numpy array)
        top_k: Number of chunks to retrieve (default: 5)
    
    Returns:
        List of (start_idx, end_idx, chunk_tokens, similarity_score) tuples
    """
    global embedding_model, current_embedding_model_name
    
    if embedding_model is None:
        raise ValueError("Embedding model not initialized")
    
    # Embed the question
    question_embedding = embedding_model.encode(
        question,
        convert_to_tensor=True,
        show_progress_bar=False
    )
    
    # Get the device where the model's embeddings are
    device = question_embedding.device
    
    # Convert chunk embeddings to tensor on the SAME device
    chunk_embeddings_tensor = torch.from_numpy(chunk_embeddings).to(device)
    
    # Compute cosine similarities using sentence_transformers util
    similarities = util.cos_sim(question_embedding, chunk_embeddings_tensor)[0]
    
    # Convert to numpy for indexing
    similarities = similarities.cpu().numpy()
    
    # Get top-k indices
    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return chunks with similarity scores
    retrieved = []
    for idx in top_indices:
        start, end, chunk_tokens = chunks[idx]
        score = float(similarities[idx])
        retrieved.append((start, end, chunk_tokens, score))
        print(f"    - Chunk {idx}: tokens {start}-{end}, similarity={score:.4f}")
    
    return retrieved


def reorder_chunks_by_position(retrieved_chunks: List[Tuple[int, int, List[str], float]]) -> List[Tuple[int, int, List[str], float]]:
    """
    Reorder retrieved chunks by their original position in the document.
    
    Args:
        retrieved_chunks: List of (start_idx, end_idx, chunk_tokens, similarity_score)
    
    Returns:
        Same list but sorted by start_idx (original document order)
    """
    return sorted(retrieved_chunks, key=lambda x: x[0])  # Sort by start_idx


# ==============================================================================
# SECTION 3: ANSWER GENERATION
# ==============================================================================

def generate_answer_with_rationale(context: str, question: str) -> tuple[str, str]:
    """
    Generate answer from context using Gemini 2.5 Flash Lite.
    
    Modified to indicate context is a snippet with chunks separated by "..."
    
    Args:
        context: The selected context text (with "..." separators)
        question: The question to answer
        
    Returns:
        Tuple of (answer, rationale)
    """
    try:
        if isinstance(context, list):
            context_text = "\n...\n".join(context)
        else:
            context_text = context

        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        prompt = f"""You are provided with excerpts from a document. The excerpts are separated by "..." and represent only relevant snippets from the full document, not the complete text.

Based on these excerpts, answer the question concisely and accurately.

Document Excerpts:
{context_text}

Question: {question}

Answer:"""
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=150,
            )
        )
        
        # Extract answer
        answer = response.text.strip()
        
        # Create rationale
        num_words = len(context_text.split())
        num_excerpts = context_text.count("\n...\n") + 1 if "\n...\n" in context_text else 1
        rationale = f"Generated answer using Gemini 2.5 Flash Lite from {num_excerpts} document excerpts ({num_words} words total)."

        return answer if answer else "None", rationale
        
    except Exception as e:
        print(f"❌ Error during Gemini API call: {e}")
        return "None", f"Error occurred: {str(e)}"


# ==============================================================================
# SECTION 4: PREPROCESSING - LOAD AND PREPARE EMBEDDINGS
# ==============================================================================

def preprocess_documents_and_compute_embeddings(dataset: str,
                                               embedding_model_name: str,
                                               chunk_size: int = 200,
                                               chunk_overlap: int = 50,
                                               dataset_dir: str = None,
                                               tracker: PerformanceTracker = None):
    """
    Preprocess documents: load from processed dataset, tokenize, chunk, and compute embeddings.
    
    This function:
    1. Loads documents directly from {dataset}_processed.json
    2. Tokenizes each document's content field
    3. Saves tokens to baseline/data/tokens/
    4. Chunks the tokens
    5. Computes embeddings for all chunks (with timing tracking)
    6. Caches embeddings to baseline/data/rag_embeddings/
    
    Args:
        dataset: Dataset name
        embedding_model_name: Embedding model to use
        chunk_size: Tokens per chunk (default: 200)
        chunk_overlap: Overlap between chunks (default: 50)
        dataset_dir: Directory containing processed dataset files
        tracker: PerformanceTracker instance for recording timings
    """
    print("\n" + "="*60)
    print("PREPROCESSING - Load Documents & Compute Embeddings")
    print("="*60)
    print(f"📋 Dataset: {dataset}")
    print(f"🤖 Embedding model: {embedding_model_name}")
    print(f"📦 Chunk size: {chunk_size} tokens")
    print("="*60)
    
    # Initialize
    initialize_tokenizer()
    initialize_embedding_model(embedding_model_name)
    
    # Load processed dataset
    if dataset_dir is None:
        dataset_dir = DATA_ROOT / 'datasets'
    
    processed_file = Path(dataset_dir) / f'{dataset}_processed.json'
    if not processed_file.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_file}")
    
    print(f"\n📖 Loading processed dataset: {processed_file}")
    with open(processed_file) as f:
        dataset_data = json.load(f)
    
    documents = dataset_data.get('documents', [])
    print(f"✅ Loaded {len(documents)} documents")
    
    # Update tracker
    if tracker is not None:
        tracker.total_documents = len(documents)
    
    # Setup directories - cache in baseline/data/
    token_dir = TOKEN_CACHE_ROOT
    token_dir.mkdir(parents=True, exist_ok=True)
    
    safe_model_name = embedding_model_name.replace('/', '_')
    embedding_cache_dir = EMBEDDING_CACHE_ROOT / dataset / f"{safe_model_name}_chunk{chunk_size}"
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Token cache: {token_dir}")
    print(f"📁 Embedding cache: {embedding_cache_dir}")
    
    total_processed = 0
    total_skipped = 0
    
    for doc_idx, document in enumerate(documents):
        doc_id = document.get('document_id')
        content = document.get('content', '')
        
        print(f"\n{'─'*60}")
        print(f"📄 Processing document {doc_idx + 1}/{len(documents)}: ID={doc_id}")
        print(f"{'─'*60}")
        
        token_file = token_dir / f"{dataset}_{doc_id}.json"
        embeddings_file = embedding_cache_dir / f"{dataset}_{doc_id}_embeddings.npy"
        chunks_file = embedding_cache_dir / f"{dataset}_{doc_id}_chunks.json"
        
        # Check if embeddings already exist
        if embeddings_file.exists() and chunks_file.exists():
            print(f"  ✅ Embeddings already cached. Skipping.")
            total_skipped += 1
            
            # Still count chunks for tracker
            if tracker is not None:
                with open(chunks_file, 'r') as f:
                    chunks_metadata = json.load(f)
                tracker.total_chunks += len(chunks_metadata)
            continue
        
        try:
            if not content:
                print(f"  ⚠️  Empty content in document")
                total_skipped += 1
                continue
            
            print(f"  ✅ Document loaded ({len(content)} characters)")
            
            # STEP 1: Tokenize document
            print(f"  🔄 Tokenizing document...")
            global tokenizer
            tokens = tokenizer.encode(content, add_special_tokens=False)
            tokens_as_strings = tokenizer.convert_ids_to_tokens(tokens)
            
            print(f"  ✅ Tokenized: {len(tokens)} tokens")
            
            # STEP 2: Save tokens to file
            if not token_file.exists():
                with open(token_file, 'w') as f:
                    json.dump(tokens_as_strings, f)
                print(f"  💾 Saved tokens to: {token_file.name}")
            
            # STEP 3: Chunk the tokens
            print(f"  📦 Chunking into {chunk_size}-token chunks...")
            chunks = chunk_tokens(tokens_as_strings, chunk_size=chunk_size, overlap=chunk_overlap)
            
            # Update tracker
            if tracker is not None:
                tracker.total_chunks += len(chunks)
            
            # STEP 4: Compute embeddings with timing
            print(f"  🔄 Computing embeddings for {len(chunks)} chunks...")
            chunk_embeddings = embed_chunks(chunks, doc_id=doc_id, tracker=tracker)
            
            # STEP 5: Cache embeddings
            np.save(embeddings_file, chunk_embeddings)
            
            # Save chunks metadata as JSON
            chunks_serializable = [
                {"start": start, "end": end, "num_tokens": len(tokens_list)}
                for start, end, tokens_list in chunks
            ]
            with open(chunks_file, 'w') as f:
                json.dump(chunks_serializable, f)
            
            print(f"  💾 Cached embeddings to: {embeddings_file.name}")
            print(f"  💾 Cached chunks to: {chunks_file.name}")
            
            total_processed += 1
            
        except Exception as e:
            print(f"  ❌ Error processing document: {e}")
            import traceback
            traceback.print_exc()
            total_skipped += 1
    
    print(f"\n{'='*60}")
    print(f"🎯 PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Processed: {total_processed}")
    print(f"⚠️  Skipped: {total_skipped}")
    print(f"📁 Tokens saved in: {token_dir}")
    print(f"📁 Embeddings saved in: {embedding_cache_dir}")
    print(f"{'='*60}")

# ==============================================================================
# SECTION 5: RAG PIPELINE
# ==============================================================================

def generate_rag_results(dataset: str, 
                        data: dict = None,
                        budget_map: dict = None,
                        embedding_model_name: str = 'UAE-Large-V1',
                        chunk_size: int = 200,
                        chunk_overlap: int = 50,
                        top_k_chunks: int = 5,
                        dataset_dir: str = None,
                        tracker: PerformanceTracker = None):
    """
    Generate RAG baseline results with performance tracking.
    
    MODIFIED to work with processed dataset files:
    - If data and budget_map are None, loads from {dataset}_processed.json
    - For each document, processes only its associated questions
    - Reorders retrieved chunks by original position
    - Separates chunks with "..." in context
    - Tracks query generation time
    
    Reads datasets from project root, writes outputs to baseline/.
    
    Args:
        dataset: Dataset name (e.g., 'paper', 'notice')
        data: Configuration dictionary (deprecated, use None for processed datasets)
        budget_map: Dictionary with questions (deprecated, use None for processed datasets)
        embedding_model_name: Embedding model to use
        chunk_size: Tokens per chunk (default: 200)
        chunk_overlap: Overlap between chunks (default: 50)
        top_k_chunks: Number of chunks to retrieve (default: 5)
        dataset_dir: Directory containing processed dataset files
        tracker: PerformanceTracker instance for recording timings
    """
    print("\n" + "="*60)
    print("RAG BASELINE - Retrieval Augmented Generation")
    print("="*60)
    print(f"📋 Dataset: {dataset}")
    print(f"🤖 Embedding model: {embedding_model_name}")
    print(f"📦 Chunk size: {chunk_size} tokens (overlap: {chunk_overlap})")
    print(f"🔍 Top-k retrieval: {top_k_chunks} chunks")
    print(f"🔄 Chunk ordering: Original document order")
    print(f"📝 Chunk separation: '...' between excerpts")
    print("="*60)
    
    # Initialize
    initialize_tokenizer()
    initialize_embedding_model(embedding_model_name)
    initialize_gemini()
    
    # MODIFIED: Load from processed dataset (project root)
    if data is None or budget_map is None:
        if dataset_dir is None:
            dataset_dir = DATA_ROOT / 'datasets'
        
        processed_file = Path(dataset_dir) / f'{dataset}_processed.json'
        if not processed_file.exists():
            raise FileNotFoundError(f"Processed dataset not found: {processed_file}")
        
        print(f"\n📖 Loading processed dataset: {processed_file}")
        with open(processed_file) as f:
            dataset_data = json.load(f)
        
        documents = dataset_data.get('documents', [])
        print(f"✅ Loaded {len(documents)} documents")
        
        use_processed = True
    else:
        # Legacy mode
        print(f"\n⚠️  Using legacy data structure (not recommended)")
        file_list = data["datasets"][dataset]["list"]
        questions = list(budget_map.keys())
        use_processed = False
    
    # Setup directories - outputs in baseline/
    pred_att_dir = PRED_ATT_ROOT
    dataset_pred_dir = pred_att_dir / dataset
    dataset_pred_dir.mkdir(parents=True, exist_ok=True)
    
    token_dir = TOKEN_CACHE_ROOT
    safe_model_name = embedding_model_name.replace('/', '_')
    embedding_cache_dir = EMBEDDING_CACHE_ROOT / dataset / f"{safe_model_name}_chunk{chunk_size}"
    
    total_processed = 0
    total_skipped = 0
    total_gemini_calls = 0
    
    # Process by document -> questions
    if use_processed:
        total_questions = sum(len(doc.get('questions', [])) for doc in documents)
        print(f"\n📊 Total documents: {len(documents)}")
        print(f"📊 Total questions: {total_questions}")
        
        for doc_idx, document in enumerate(documents):
            doc_id = document.get('document_id')
            doc_questions = document.get('questions', [])
            
            if not doc_questions:
                print(f"\n⚠️  Document {doc_id}: No questions, skipping")
                continue
            
            print(f"\n{'='*60}")
            print(f"Document {doc_idx + 1}/{len(documents)}: ID={doc_id}")
            print(f"Questions for this document: {len(doc_questions)}")
            print(f"{'='*60}")
            
            # Load cached tokens and embeddings from baseline/data/
            token_file = token_dir / f"{dataset}_{doc_id}.json"
            embeddings_file = embedding_cache_dir / f"{dataset}_{doc_id}_embeddings.npy"
            chunks_file = embedding_cache_dir / f"{dataset}_{doc_id}_chunks.json"
            
            if not token_file.exists():
                print(f"  ⚠️  Token file not found: {token_file}")
                total_skipped += len(doc_questions)
                continue
            
            if not embeddings_file.exists() or not chunks_file.exists():
                print(f"  ⚠️  Embeddings not cached. Run preprocessing first!")
                total_skipped += len(doc_questions)
                continue
            
            # Load tokens
            with open(token_file, 'r') as f:
                tokens = json.load(f)
            
            # Load embeddings and chunks
            chunk_embeddings = np.load(embeddings_file)
            with open(chunks_file, 'r') as f:
                chunks_metadata = json.load(f)
            
            # Reconstruct chunks
            chunks = []
            for chunk_meta in chunks_metadata:
                start = chunk_meta["start"]
                end = chunk_meta["end"]
                chunk_tokens = tokens[start:end]
                chunks.append((start, end, chunk_tokens))
            
            print(f"  ✅ Loaded {len(tokens)} tokens, {len(chunks)} chunks")
            
            # Process each question for this document
            for q_idx, question_obj in enumerate(doc_questions):
                question = question_obj.get('question', '')
                question_id = question_obj.get('question_id', '')
                
                if not question:
                    continue
                
                print(f"\n  📝 Question {q_idx + 1}/{len(doc_questions)}: {question[:60]}...")
                
                # Save results to baseline/pred_att/
                safe_question = question.replace("?", "").replace("/", "_").replace(" ", "_")
                safe_model = embedding_model_name.replace('/', '-').replace('_', '-')
                results_file = dataset_pred_dir / f"results-rag-{safe_model}-top{top_k_chunks}-{dataset}_doc{doc_id}_q{question_id}.json"
                
                if results_file.exists():
                    print(f"    ✅ Results exist, skipping")
                    continue
                
                try:
                    # Start timing for this query
                    query_start_time = time.time()
                    
                    # Retrieve top-k chunks
                    retrieved_chunks = retrieve_top_k_chunks(
                        question, 
                        chunks, 
                        chunk_embeddings, 
                        top_k=top_k_chunks
                    )
                    
                    # MODIFIED: Reorder by original position
                    retrieved_chunks = reorder_chunks_by_position(retrieved_chunks)
                    print(f"    ✅ Retrieved & reordered {len(retrieved_chunks)} chunks")
                    
                    # Prepare context with "..." separators
                    retrieved_texts = []
                    chunk_token_counts = []
                    total_tokens_retrieved = 0
                    
                    for start, end, chunk_tokens, score in retrieved_chunks:
                        token_ids = tokenizer.convert_tokens_to_ids(chunk_tokens)
                        text = tokenizer.decode(token_ids)
                        retrieved_texts.append(text)
                        chunk_token_count = len(chunk_tokens)
                        chunk_token_counts.append(chunk_token_count)
                        total_tokens_retrieved += chunk_token_count
                    
                    # MODIFIED: Use "..." to separate chunks
                    combined_context = "\n...\n".join(retrieved_texts)
                    
                    # Token metrics
                    question_tokens = len(tokenizer.encode(question, add_special_tokens=False))
                    full_prompt = f"""You are provided with excerpts from a document. The excerpts are separated by "..." and represent only relevant snippets from the full document, not the complete text.

Based on these excerpts, answer the question concisely and accurately.

Document Excerpts:
{combined_context}

Question: {question}

Answer:"""
                    prompt_tokens = len(tokenizer.encode(full_prompt, add_special_tokens=False))
                    
                    # Generate answer
                    answer, rationale = generate_answer_with_rationale(combined_context, question)
                    total_gemini_calls += 1
                    
                    answer_tokens = len(tokenizer.encode(answer, add_special_tokens=False)) if answer != "None" else 0
                    
                    # End timing for this query
                    query_end_time = time.time()
                    query_duration = query_end_time - query_start_time
                    
                    # Record query timing
                    if tracker is not None:
                        tracker.record_query_generation(doc_id, question_id, query_duration)
                    
                    # Save result
                    result = {
                        "question": question,
                        "question_id": question_id,
                        "document_id": doc_id,
                        "method": "RAG",
                        "embedding_model": embedding_model_name,
                        "chunk_size": chunk_size,
                        "top_k_chunks": top_k_chunks,
                        "chunk_ordering": "original_position",
                        "chunk_separator": "...",
                        "dataset": dataset,
                        "result": answer,
                        "rationale": rationale,
                        "selected_context": retrieved_texts,
                        "combined_context": combined_context,
                        "token_usage": {
                            "document_total_tokens": len(tokens),
                            "chunks_retrieved": len(retrieved_texts),
                            "chunk_token_counts": chunk_token_counts,
                            "context_tokens": total_tokens_retrieved,
                            "question_tokens": question_tokens,
                            "prompt_tokens": prompt_tokens,
                            "answer_tokens": answer_tokens,
                            "total_tokens_used": prompt_tokens + answer_tokens
                        },
                        "duration": query_duration
                    }
                    
                    with open(results_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    print(f"    ✅ Saved: {results_file.name}")
                    print(f"    ⏱️  Time: {query_duration:.2f}s, Tokens: {total_tokens_retrieved}")
                    total_processed += 1
                    
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    total_skipped += 1
    
    else:
        # OLD PATH: Legacy processing
        print("\n⚠️  Using legacy processing mode - consider migrating to processed datasets")
    
    print(f"\n{'='*60}")
    print(f"🎯 RAG COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Processed: {total_processed}")
    print(f"⚠️  Skipped: {total_skipped}")
    print(f"🤖 Gemini calls: {total_gemini_calls}")
    print(f"📁 Results saved to: {dataset_pred_dir}")
    print(f"{'='*60}")


# ==============================================================================
# SECTION 6: MAIN PIPELINE
# ==============================================================================

def _run_single_rag_pipeline(dataset: str = 'paper',
                             generate_results: bool = True,
                             embedding_model_name: str = 'UAE-Large-V1',
                             chunk_size: int = 200,
                             top_k: int = 5,
                             tracker: PerformanceTracker = None):
    """Internal helper that runs the pipeline for a single embedding model."""
    print("\n" + "="*60)
    print("RAG BASELINE PIPELINE")
    print("="*60)
    print(f"📋 Dataset: {dataset}")
    print(f"🤖 Embedding: {embedding_model_name}")
    print(f"📦 Chunk size: {chunk_size}")
    print(f"🔍 Top-k: {top_k}")
    print("="*60)
    
    if generate_results:
        # Check if embeddings exist in baseline/data/
        safe_model_name = embedding_model_name.replace('/', '_')
        embedding_cache_dir = EMBEDDING_CACHE_ROOT / dataset / f"{safe_model_name}_chunk{chunk_size}"
        
        needs_preprocessing = False
        
        # Load processed dataset to get document count
        try:
            dataset_data = load_processed_dataset(dataset)
            num_documents = len(dataset_data.get('documents', []))
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return
        
        if not embedding_cache_dir.exists():
            print(f"\n⚠️  Embedding cache directory not found: {embedding_cache_dir}")
            print(f"   Will run preprocessing to compute embeddings...")
            needs_preprocessing = True
        else:
            # Check if embedding files exist for all documents
            existing_embeddings = 0
            
            for doc_idx in range(num_documents):
                embeddings_file = embedding_cache_dir / f"{dataset}_{doc_idx}_embeddings.npy"
                chunks_file = embedding_cache_dir / f"{dataset}_{doc_idx}_chunks.json"
                
                if embeddings_file.exists() and chunks_file.exists():
                    existing_embeddings += 1
            
            if existing_embeddings == 0:
                print(f"\n⚠️  No embedding files found in {embedding_cache_dir}")
                print(f"   Will run preprocessing to compute embeddings...")
                needs_preprocessing = True
            else:
                print(f"\n✅ Found {existing_embeddings}/{num_documents} cached embeddings")
                if existing_embeddings < num_documents:
                    print(f"   Will preprocess missing {num_documents - existing_embeddings} documents...")
                    needs_preprocessing = True
        
        # Run preprocessing if needed
        if needs_preprocessing:
            print(f"\n{'='*60}")
            print(f"STEP 1: PREPROCESSING")
            print(f"{'='*60}")
            preprocess_documents_and_compute_embeddings(
                dataset=dataset,
                embedding_model_name=embedding_model_name,
                chunk_size=chunk_size,
                chunk_overlap=50,
                tracker=tracker
            )
        else:
            print(f"\n✅ All embeddings already cached. Skipping preprocessing.")
            
            # Still need to count chunks for tracker
            if tracker is not None:
                tracker.total_documents = num_documents
                for doc_idx in range(num_documents):
                    chunks_file = embedding_cache_dir / f"{dataset}_{doc_idx}_chunks.json"
                    if chunks_file.exists():
                        with open(chunks_file, 'r') as f:
                            chunks_metadata = json.load(f)
                        tracker.total_chunks += len(chunks_metadata)
        
        # STEP 2: Generate results
        generate_rag_results(
            dataset=dataset,
            data=None,  # Use processed dataset
            budget_map=None,  # Use processed dataset
            embedding_model_name=embedding_model_name,
            chunk_size=chunk_size,
            top_k_chunks=top_k,
            tracker=tracker
        )
    else:
        print("\n⚠️ Generation disabled (`generate_results=False`). Skipping preprocessing/generation step.")
    
    print("\n" + "="*60)
    print(f"✅ PIPELINE COMPLETE")
    print(f"🕐 {datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %I:%M %p EDT')}")
    print("="*60)


def run_rag_pipeline(dataset: str = 'paper',
                     generate_results: bool = True,
                     embedding_model_name: str = 'UAE-Large-V1',
                     chunk_size: int = 200,
                     top_k: int = 5):
    """
    Public entry point for the RAG pipeline. Supports running multiple embedding models by
    passing `embedding_model_name='all'` or a list of model names.
    
    Generates performance reports for each (dataset, model) combination.
    """
    if isinstance(embedding_model_name, str):
        if embedding_model_name.lower() == 'all':
            model_list = AVAILABLE_EMBEDDING_MODELS
        else:
            model_list = [embedding_model_name]
    else:
        model_list = list(embedding_model_name)

    for idx, model in enumerate(model_list, 1):
        print("\n" + "=" * 80)
        print(f"🚀 Running RAG pipeline ({idx}/{len(model_list)}) with embedding model: {model}")
        print("=" * 80)
        
        # Create performance tracker for this (dataset, model) combination
        tracker = PerformanceTracker(dataset, model)
        
        _run_single_rag_pipeline(
            dataset=dataset,
            generate_results=generate_results,
            embedding_model_name=model,
            chunk_size=chunk_size,
            top_k=top_k,
            tracker=tracker
        )
        
        # Save performance report after this model is processed
        if generate_results:
            tracker.save_report()

# ==============================================================================
# SECTION 7: MAIN
# ==============================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Baseline for Context Reduction - WITH PERFORMANCE TRACKING
---------------------------------------------------------------

This script implements a pure RAG (Retrieval Augmented Generation) baseline
for question answering over large documents with comprehensive performance tracking.

**RAG Workflow:**

1. **Preprocessing:**
   - Load raw documents from processed dataset files
   - Tokenize each document
   - Save tokens to baseline/data/tokens/
   - Chunk tokens into fixed-size chunks (200 tokens by default)
   - Compute embeddings for all chunks (TRACKED)
   - Cache embeddings to baseline/data/rag_embeddings/

2. **Retrieval & Generation:**
   - For each document, process its associated questions
   - Find the most similar chunks using embedding-based cosine similarity
   - Pick the top K most relevant chunks
   - Reorder chunks by original document position
   - Separate chunks with "..." in context
   - Pass to Gemini 2.5 Flash Lite to generate an answer (TRACKED)
   - Store results in baseline/pred_att/

3. **Performance Reporting:**
   - Track time for embedding each chunk
   - Track time for generating each answer
   - Generate performance report with statistics
   - Save to ../reports/performance_report_{dataset}_rag.json

**Prerequisites:**
- Processed datasets in data/datasets/{dataset}_processed.json
- GEMINI_API_KEY environment variable

**Features:**
- Works with processed dataset structure
- Chunks ordered by original position (not similarity)
- Chunks separated with "..." in prompt
- Outputs saved to baseline/ subdirectory
- Performance reports saved to ../reports/
"""

import sys
import os
import warnings
import logging

# Suppress logging
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

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

# Add parent directory to path
from pathlib import Path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple
import numpy as np
import json
import gc
import time
import pytz
from datetime import datetime
import google.generativeai as genai
from transformers import BitsAndBytesConfig
import re
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ==============================================================================
# PATH CONFIGURATION - Since file is now in baseline/ subdirectory
# ==============================================================================

def get_project_root() -> Path:
    """Get the project root directory (parent of baseline/)."""
    return Path(__file__).parent.parent

def get_baseline_dir() -> Path:
    """Get the baseline directory (where this script is located)."""
    return Path(__file__).parent

# Input data paths - read from project root (original location)
DATA_ROOT = get_project_root() / "data"
QUESTIONS_ROOT = get_project_root() / "questions"

# Output paths - write to baseline subdirectory
PRED_ATT_ROOT = get_baseline_dir() / "pred_att"

# Cache paths - in baseline (isolated from other experiments)
TOKEN_CACHE_ROOT = get_baseline_dir() / "data" / "tokens"
EMBEDDING_CACHE_ROOT = get_baseline_dir() / "data" / "rag_embeddings"

# Report paths - write to parent reports directory
REPORTS_ROOT = get_project_root() / "reports"

# Global variables
tokenizer = None
embedding_model = None
current_embedding_model_name = None

AVAILABLE_EMBEDDING_MODELS = [
    'UAE-Large-V1',
    'bflhc/Octen-Embedding-4B',
    'Qwen3-Embedding-8B'
]

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def load_processed_dataset(dataset_name: str, dataset_dir: str = None) -> Dict:
    """
    Load processed dataset file containing ground truth.
    
    Args:
        dataset_name: Name of the dataset
        dataset_dir: Directory containing processed dataset files
    
    Returns:
        Dictionary containing the full dataset
    """
    if dataset_dir is None:
        dataset_dir = DATA_ROOT / 'datasets'
    
    processed_file = Path(dataset_dir) / f'{dataset_name}_processed.json'
    
    if not processed_file.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_file}")
    
    print(f"📖 Loading dataset from: {processed_file}")
    
    with open(processed_file) as f:
        dataset = json.load(f)
    
    num_docs = len(dataset.get('documents', []))
    print(f"✅ Loaded dataset with {num_docs} documents")
    
    return dataset

# ==============================================================================
# SECTION 1: INITIALIZATION
# ==============================================================================

def clear_gpu_memory():
    """Aggressively clear GPU memory cache"""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        time.sleep(2)

def initialize_tokenizer(model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
    """Initialize tokenizer for token processing."""
    global tokenizer
    
    if tokenizer is not None:
        print("✅ Tokenizer already initialized")
        return
    
    print(f"🔄 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        cache_dir="/tmp/huggingface-cache", 
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Tokenizer initialized")


def initialize_embedding_model(model_name: str):
    global embedding_model, current_embedding_model_name
    
    if current_embedding_model_name == model_name and embedding_model is not None:
        print(f"✅ Embedding model '{model_name}' already initialized")
        return
    
    # Clear any existing model from memory
    if embedding_model is not None:
        print(f"🔄 Clearing previous model from memory...")
        del embedding_model
        embedding_model = None
        clear_gpu_memory()
    
    model_mapping = {
        'UAE-Large-V1': 'WhereIsAI/UAE-Large-V1',
        'bflhc/Octen-Embedding-4B': 'bflhc/Octen-Embedding-4B',
        'Qwen3-Embedding-8B': 'Qwen/Qwen3-Embedding-8B',
        'GritLM-7B': 'GritLM/GritLM-7B'
    }
    
    # Models that require 8-bit quantization
    quantized_models = ['Qwen3-Embedding-8B', 'GritLM-7B']
    
    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {list(model_mapping.keys())}")
    
    hf_model_name = model_mapping[model_name]
    use_quantization = model_name in quantized_models
    
    print(f"🔄 Loading embedding model: {hf_model_name}")
    if use_quantization:
        print(f"   Using 8-bit quantization to reduce memory usage")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {device}")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_free = total_memory - memory_allocated
            print(f"   GPU Memory Total: {total_memory:.2f} GB")
            print(f"   GPU Memory Free: {memory_free:.2f} GB")
        
        # 8-bit quantization for large models
        if use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            embedding_model = SentenceTransformer(
                hf_model_name,
                cache_folder="/tmp/huggingface-cache",
                trust_remote_code=True,
                model_kwargs={
                    'quantization_config': quantization_config,
                    'device_map': 'auto'
                }
            )
        # Standard loading for smaller models
        else:
            embedding_model = SentenceTransformer(
                hf_model_name,
                cache_folder="/tmp/huggingface-cache",
                trust_remote_code=True,
                device=device
            )
        
        current_embedding_model_name = model_name
        print(f"✅ Embedding model loaded successfully!")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"   GPU Memory After Loading - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        import traceback
        traceback.print_exc()
        raise


def initialize_gemini():
    """Initialize Gemini API for answer generation."""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ Gemini API configured")
    else:
        print("⚠️ Warning: GEMINI_API_KEY not found in environment variables")


# ==============================================================================
# SECTION 2: CHUNKING AND EMBEDDING
# ==============================================================================

def chunk_tokens(tokens: List[str], chunk_size: int = 200, overlap: int = 50) -> List[Tuple[int, int, List[str]]]:
    """
    Split tokens into overlapping chunks.
    
    Args:
        tokens: List of token strings
        chunk_size: Target size for each chunk (default: 200)
        overlap: Number of overlapping tokens between chunks (default: 50)
    
    Returns:
        List of tuples (start_idx, end_idx, chunk_tokens)
    """
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = tokens[start:end]
        chunks.append((start, end, chunk))
        
        # Move to next chunk with overlap
        start = end - overlap if end < len(tokens) else end
        
        if start >= len(tokens):
            break
    
    print(f"  📦 Created {len(chunks)} chunks from {len(tokens)} tokens")
    return chunks


def embed_chunks(chunks: List[Tuple[int, int, List[str]]], 
                 doc_id: str = None, 
                 tracker: PerformanceTracker = None) -> np.ndarray:
    """
    Create embeddings for all chunks with per-chunk timing.
    
    Args:
        chunks: List of (start_idx, end_idx, chunk_tokens) tuples
        doc_id: Document ID for tracking purposes
        tracker: PerformanceTracker instance to record timings
    
    Returns:
        NumPy array of embeddings (num_chunks x embedding_dim)
    """
    global embedding_model, tokenizer, current_embedding_model_name
    
    if embedding_model is None:
        raise ValueError("Embedding model not initialized")
    
    if tokenizer is None:
        initialize_tokenizer()
    
    print(f"  🔄 Creating embeddings for {len(chunks)} chunks...")
    
    # Convert token lists to text
    chunk_texts = []
    for start, end, chunk_tokens in chunks:
        token_ids = tokenizer.convert_tokens_to_ids(chunk_tokens)
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        # Clean text to avoid issues with multimodal models
        text = text.strip()
        
        # Skip empty chunks
        if not text:
            text = "[EMPTY]"
        
        chunk_texts.append(text)
    
    # Create embeddings in batches with per-chunk timing
    batch_size = 8
    all_embeddings = []
    
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i + batch_size]
        
        try:
            # Time the batch embedding
            batch_start_time = time.time()
            embeddings = embedding_model.encode(
                batch,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            batch_end_time = time.time()
            
            all_embeddings.append(embeddings.cpu().numpy())
            
            # Record timing for each chunk in the batch
            if tracker is not None and doc_id is not None:
                batch_time = batch_end_time - batch_start_time
                time_per_chunk = batch_time / len(batch)
                for j in range(len(batch)):
                    chunk_idx = i + j
                    tracker.record_chunk_embedding(doc_id, chunk_idx, time_per_chunk)
            
        except Exception as e:
            print(f"  ⚠️  Error encoding batch {i//batch_size + 1}: {e}")
            print(f"  🔄 Retrying with individual chunks...")
            
            # Retry individual chunks in this batch
            for j, text in enumerate(batch):
                try:
                    chunk_start_time = time.time()
                    embedding = embedding_model.encode(
                        [text],
                        convert_to_tensor=True,
                        show_progress_bar=False
                    )
                    chunk_end_time = time.time()
                    all_embeddings.append(embedding.cpu().numpy())
                    
                    # Record timing for this chunk
                    if tracker is not None and doc_id is not None:
                        chunk_time = chunk_end_time - chunk_start_time
                        tracker.record_chunk_embedding(doc_id, i + j, chunk_time)
                        
                except Exception as e2:
                    print(f"  ❌ Failed to encode chunk {i+j}: {e2}")
                    print(f"     Text preview: {text[:100]}...")
                    # Create a zero embedding as fallback
                    dummy_embedding = np.zeros((1, embedding_model.get_sentence_embedding_dimension()))
                    all_embeddings.append(dummy_embedding)
    
    embeddings_array = np.vstack(all_embeddings)
    print(f"  ✅ Created embeddings with shape: {embeddings_array.shape}")
    
    return embeddings_array

def retrieve_top_k_chunks(question: str, 
                          chunks: List[Tuple[int, int, List[str]]], 
                          chunk_embeddings: np.ndarray,
                          top_k: int = 5) -> List[Tuple[int, int, List[str], float]]:
    """
    Retrieve top-k most relevant chunks for a question using cosine similarity.
    
    Uses sentence_transformers.util.cos_sim for robust similarity computation.
    
    Args:
        question: The question to answer
        chunks: List of (start_idx, end_idx, chunk_tokens) tuples
        chunk_embeddings: Embeddings for all chunks (numpy array)
        top_k: Number of chunks to retrieve (default: 5)
    
    Returns:
        List of (start_idx, end_idx, chunk_tokens, similarity_score) tuples
    """
    global embedding_model, current_embedding_model_name
    
    if embedding_model is None:
        raise ValueError("Embedding model not initialized")
    
    # Embed the question
    question_embedding = embedding_model.encode(
        question,
        convert_to_tensor=True,
        show_progress_bar=False
    )
    
    # Get the device where the model's embeddings are
    device = question_embedding.device
    
    # Convert chunk embeddings to tensor on the SAME device
    chunk_embeddings_tensor = torch.from_numpy(chunk_embeddings).to(device)
    
    # Compute cosine similarities using sentence_transformers util
    similarities = util.cos_sim(question_embedding, chunk_embeddings_tensor)[0]
    
    # Convert to numpy for indexing
    similarities = similarities.cpu().numpy()
    
    # Get top-k indices
    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return chunks with similarity scores
    retrieved = []
    for idx in top_indices:
        start, end, chunk_tokens = chunks[idx]
        score = float(similarities[idx])
        retrieved.append((start, end, chunk_tokens, score))
        print(f"    - Chunk {idx}: tokens {start}-{end}, similarity={score:.4f}")
    
    return retrieved


def reorder_chunks_by_position(retrieved_chunks: List[Tuple[int, int, List[str], float]]) -> List[Tuple[int, int, List[str], float]]:
    """
    Reorder retrieved chunks by their original position in the document.
    
    Args:
        retrieved_chunks: List of (start_idx, end_idx, chunk_tokens, similarity_score)
    
    Returns:
        Same list but sorted by start_idx (original document order)
    """
    return sorted(retrieved_chunks, key=lambda x: x[0])  # Sort by start_idx


# ==============================================================================
# SECTION 3: ANSWER GENERATION
# ==============================================================================

def generate_answer_with_rationale(context: str, question: str) -> tuple[str, str]:
    """
    Generate answer from context using Gemini 2.5 Flash Lite.
    
    Modified to indicate context is a snippet with chunks separated by "..."
    
    Args:
        context: The selected context text (with "..." separators)
        question: The question to answer
        
    Returns:
        Tuple of (answer, rationale)
    """
    try:
        if isinstance(context, list):
            context_text = "\n...\n".join(context)
        else:
            context_text = context

        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        prompt = f"""You are provided with excerpts from a document. The excerpts are separated by "..." and represent only relevant snippets from the full document, not the complete text.

Based on these excerpts, answer the question concisely and accurately.

Document Excerpts:
{context_text}

Question: {question}

Answer:"""
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=150,
            )
        )
        
        # Extract answer
        answer = response.text.strip()
        
        # Create rationale
        num_words = len(context_text.split())
        num_excerpts = context_text.count("\n...\n") + 1 if "\n...\n" in context_text else 1
        rationale = f"Generated answer using Gemini 2.5 Flash Lite from {num_excerpts} document excerpts ({num_words} words total)."

        return answer if answer else "None", rationale
        
    except Exception as e:
        print(f"❌ Error during Gemini API call: {e}")
        return "None", f"Error occurred: {str(e)}"


# ==============================================================================
# SECTION 4: PREPROCESSING - LOAD AND PREPARE EMBEDDINGS
# ==============================================================================

def preprocess_documents_and_compute_embeddings(dataset: str,
                                               embedding_model_name: str,
                                               chunk_size: int = 200,
                                               chunk_overlap: int = 50,
                                               dataset_dir: str = None,
                                               tracker: PerformanceTracker = None):
    """
    Preprocess documents: load from processed dataset, tokenize, chunk, and compute embeddings.
    
    This function:
    1. Loads documents directly from {dataset}_processed.json
    2. Tokenizes each document's content field
    3. Saves tokens to baseline/data/tokens/
    4. Chunks the tokens
    5. Computes embeddings for all chunks (with timing tracking)
    6. Caches embeddings to baseline/data/rag_embeddings/
    
    Args:
        dataset: Dataset name
        embedding_model_name: Embedding model to use
        chunk_size: Tokens per chunk (default: 200)
        chunk_overlap: Overlap between chunks (default: 50)
        dataset_dir: Directory containing processed dataset files
        tracker: PerformanceTracker instance for recording timings
    """
    print("\n" + "="*60)
    print("PREPROCESSING - Load Documents & Compute Embeddings")
    print("="*60)
    print(f"📋 Dataset: {dataset}")
    print(f"🤖 Embedding model: {embedding_model_name}")
    print(f"📦 Chunk size: {chunk_size} tokens")
    print("="*60)
    
    # Initialize
    initialize_tokenizer()
    initialize_embedding_model(embedding_model_name)
    
    # Load processed dataset
    if dataset_dir is None:
        dataset_dir = DATA_ROOT / 'datasets'
    
    processed_file = Path(dataset_dir) / f'{dataset}_processed.json'
    if not processed_file.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_file}")
    
    print(f"\n📖 Loading processed dataset: {processed_file}")
    with open(processed_file) as f:
        dataset_data = json.load(f)
    
    documents = dataset_data.get('documents', [])
    print(f"✅ Loaded {len(documents)} documents")
    
    # Update tracker
    if tracker is not None:
        tracker.total_documents = len(documents)
    
    # Setup directories - cache in baseline/data/
    token_dir = TOKEN_CACHE_ROOT
    token_dir.mkdir(parents=True, exist_ok=True)
    
    safe_model_name = embedding_model_name.replace('/', '_')
    embedding_cache_dir = EMBEDDING_CACHE_ROOT / dataset / f"{safe_model_name}_chunk{chunk_size}"
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Token cache: {token_dir}")
    print(f"📁 Embedding cache: {embedding_cache_dir}")
    
    total_processed = 0
    total_skipped = 0
    
    for doc_idx, document in enumerate(documents):
        doc_id = document.get('document_id')
        content = document.get('content', '')
        
        print(f"\n{'─'*60}")
        print(f"📄 Processing document {doc_idx + 1}/{len(documents)}: ID={doc_id}")
        print(f"{'─'*60}")
        
        token_file = token_dir / f"{dataset}_{doc_id}.json"
        embeddings_file = embedding_cache_dir / f"{dataset}_{doc_id}_embeddings.npy"
        chunks_file = embedding_cache_dir / f"{dataset}_{doc_id}_chunks.json"
        
        # Check if embeddings already exist
        if embeddings_file.exists() and chunks_file.exists():
            print(f"  ✅ Embeddings already cached. Skipping.")
            total_skipped += 1
            
            # Still count chunks for tracker
            if tracker is not None:
                with open(chunks_file, 'r') as f:
                    chunks_metadata = json.load(f)
                tracker.total_chunks += len(chunks_metadata)
            continue
        
        try:
            if not content:
                print(f"  ⚠️  Empty content in document")
                total_skipped += 1
                continue
            
            print(f"  ✅ Document loaded ({len(content)} characters)")
            
            # STEP 1: Tokenize document
            print(f"  🔄 Tokenizing document...")
            global tokenizer
            tokens = tokenizer.encode(content, add_special_tokens=False)
            tokens_as_strings = tokenizer.convert_ids_to_tokens(tokens)
            
            print(f"  ✅ Tokenized: {len(tokens)} tokens")
            
            # STEP 2: Save tokens to file
            if not token_file.exists():
                with open(token_file, 'w') as f:
                    json.dump(tokens_as_strings, f)
                print(f"  💾 Saved tokens to: {token_file.name}")
            
            # STEP 3: Chunk the tokens
            print(f"  📦 Chunking into {chunk_size}-token chunks...")
            chunks = chunk_tokens(tokens_as_strings, chunk_size=chunk_size, overlap=chunk_overlap)
            
            # Update tracker
            if tracker is not None:
                tracker.total_chunks += len(chunks)
            
            # STEP 4: Compute embeddings with timing
            print(f"  🔄 Computing embeddings for {len(chunks)} chunks...")
            chunk_embeddings = embed_chunks(chunks, doc_id=doc_id, tracker=tracker)
            
            # STEP 5: Cache embeddings
            np.save(embeddings_file, chunk_embeddings)
            
            # Save chunks metadata as JSON
            chunks_serializable = [
                {"start": start, "end": end, "num_tokens": len(tokens_list)}
                for start, end, tokens_list in chunks
            ]
            with open(chunks_file, 'w') as f:
                json.dump(chunks_serializable, f)
            
            print(f"  💾 Cached embeddings to: {embeddings_file.name}")
            print(f"  💾 Cached chunks to: {chunks_file.name}")
            
            total_processed += 1
            
        except Exception as e:
            print(f"  ❌ Error processing document: {e}")
            import traceback
            traceback.print_exc()
            total_skipped += 1
    
    print(f"\n{'='*60}")
    print(f"🎯 PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Processed: {total_processed}")
    print(f"⚠️  Skipped: {total_skipped}")
    print(f"📁 Tokens saved in: {token_dir}")
    print(f"📁 Embeddings saved in: {embedding_cache_dir}")
    print(f"{'='*60}")

# ==============================================================================
# SECTION 5: RAG PIPELINE
# ==============================================================================

def generate_rag_results(dataset: str, 
                        data: dict = None,
                        budget_map: dict = None,
                        embedding_model_name: str = 'UAE-Large-V1',
                        chunk_size: int = 200,
                        chunk_overlap: int = 50,
                        top_k_chunks: int = 5,
                        dataset_dir: str = None,
                        tracker: PerformanceTracker = None):
    """
    Generate RAG baseline results with performance tracking.
    
    MODIFIED to work with processed dataset files:
    - If data and budget_map are None, loads from {dataset}_processed.json
    - For each document, processes only its associated questions
    - Reorders retrieved chunks by original position
    - Separates chunks with "..." in context
    - Tracks query generation time
    
    Reads datasets from project root, writes outputs to baseline/.
    
    Args:
        dataset: Dataset name (e.g., 'paper', 'notice')
        data: Configuration dictionary (deprecated, use None for processed datasets)
        budget_map: Dictionary with questions (deprecated, use None for processed datasets)
        embedding_model_name: Embedding model to use
        chunk_size: Tokens per chunk (default: 200)
        chunk_overlap: Overlap between chunks (default: 50)
        top_k_chunks: Number of chunks to retrieve (default: 5)
        dataset_dir: Directory containing processed dataset files
        tracker: PerformanceTracker instance for recording timings
    """
    print("\n" + "="*60)
    print("RAG BASELINE - Retrieval Augmented Generation")
    print("="*60)
    print(f"📋 Dataset: {dataset}")
    print(f"🤖 Embedding model: {embedding_model_name}")
    print(f"📦 Chunk size: {chunk_size} tokens (overlap: {chunk_overlap})")
    print(f"🔍 Top-k retrieval: {top_k_chunks} chunks")
    print(f"🔄 Chunk ordering: Original document order")
    print(f"📝 Chunk separation: '...' between excerpts")
    print("="*60)
    
    # Initialize
    initialize_tokenizer()
    initialize_embedding_model(embedding_model_name)
    initialize_gemini()
    
    # MODIFIED: Load from processed dataset (project root)
    if data is None or budget_map is None:
        if dataset_dir is None:
            dataset_dir = DATA_ROOT / 'datasets'
        
        processed_file = Path(dataset_dir) / f'{dataset}_processed.json'
        if not processed_file.exists():
            raise FileNotFoundError(f"Processed dataset not found: {processed_file}")
        
        print(f"\n📖 Loading processed dataset: {processed_file}")
        with open(processed_file) as f:
            dataset_data = json.load(f)
        
        documents = dataset_data.get('documents', [])
        print(f"✅ Loaded {len(documents)} documents")
        
        use_processed = True
    else:
        # Legacy mode
        print(f"\n⚠️  Using legacy data structure (not recommended)")
        file_list = data["datasets"][dataset]["list"]
        questions = list(budget_map.keys())
        use_processed = False
    
    # Setup directories - outputs in baseline/
    pred_att_dir = PRED_ATT_ROOT
    dataset_pred_dir = pred_att_dir / dataset
    dataset_pred_dir.mkdir(parents=True, exist_ok=True)
    
    token_dir = TOKEN_CACHE_ROOT
    safe_model_name = embedding_model_name.replace('/', '_')
    embedding_cache_dir = EMBEDDING_CACHE_ROOT / dataset / f"{safe_model_name}_chunk{chunk_size}"
    
    total_processed = 0
    total_skipped = 0
    total_gemini_calls = 0
    
    # Process by document -> questions
    if use_processed:
        total_questions = sum(len(doc.get('questions', [])) for doc in documents)
        print(f"\n📊 Total documents: {len(documents)}")
        print(f"📊 Total questions: {total_questions}")
        
        for doc_idx, document in enumerate(documents):
            doc_id = document.get('document_id')
            doc_questions = document.get('questions', [])
            
            if not doc_questions:
                print(f"\n⚠️  Document {doc_id}: No questions, skipping")
                continue
            
            print(f"\n{'='*60}")
            print(f"Document {doc_idx + 1}/{len(documents)}: ID={doc_id}")
            print(f"Questions for this document: {len(doc_questions)}")
            print(f"{'='*60}")
            
            # Load cached tokens and embeddings from baseline/data/
            token_file = token_dir / f"{dataset}_{doc_id}.json"
            embeddings_file = embedding_cache_dir / f"{dataset}_{doc_id}_embeddings.npy"
            chunks_file = embedding_cache_dir / f"{dataset}_{doc_id}_chunks.json"
            
            if not token_file.exists():
                print(f"  ⚠️  Token file not found: {token_file}")
                total_skipped += len(doc_questions)
                continue
            
            if not embeddings_file.exists() or not chunks_file.exists():
                print(f"  ⚠️  Embeddings not cached. Run preprocessing first!")
                total_skipped += len(doc_questions)
                continue
            
            # Load tokens
            with open(token_file, 'r') as f:
                tokens = json.load(f)
            
            # Load embeddings and chunks
            chunk_embeddings = np.load(embeddings_file)
            with open(chunks_file, 'r') as f:
                chunks_metadata = json.load(f)
            
            # Reconstruct chunks
            chunks = []
            for chunk_meta in chunks_metadata:
                start = chunk_meta["start"]
                end = chunk_meta["end"]
                chunk_tokens = tokens[start:end]
                chunks.append((start, end, chunk_tokens))
            
            print(f"  ✅ Loaded {len(tokens)} tokens, {len(chunks)} chunks")
            
            # Process each question for this document
            for q_idx, question_obj in enumerate(doc_questions):
                question = question_obj.get('question', '')
                question_id = question_obj.get('question_id', '')
                
                if not question:
                    continue
                
                print(f"\n  📝 Question {q_idx + 1}/{len(doc_questions)}: {question[:60]}...")
                
                # Save results to baseline/pred_att/
                safe_question = question.replace("?", "").replace("/", "_").replace(" ", "_")
                safe_model = embedding_model_name.replace('/', '-').replace('_', '-')
                results_file = dataset_pred_dir / f"results-rag-{safe_model}-top{top_k_chunks}-{dataset}_doc{doc_id}_q{question_id}.json"
                
                if results_file.exists():
                    print(f"    ✅ Results exist, skipping")
                    continue
                
                try:
                    # Start timing for this query
                    query_start_time = time.time()
                    
                    # Retrieve top-k chunks
                    retrieved_chunks = retrieve_top_k_chunks(
                        question, 
                        chunks, 
                        chunk_embeddings, 
                        top_k=top_k_chunks
                    )
                    
                    # MODIFIED: Reorder by original position
                    retrieved_chunks = reorder_chunks_by_position(retrieved_chunks)
                    print(f"    ✅ Retrieved & reordered {len(retrieved_chunks)} chunks")
                    
                    # Prepare context with "..." separators
                    retrieved_texts = []
                    chunk_token_counts = []
                    total_tokens_retrieved = 0
                    
                    for start, end, chunk_tokens, score in retrieved_chunks:
                        token_ids = tokenizer.convert_tokens_to_ids(chunk_tokens)
                        text = tokenizer.decode(token_ids)
                        retrieved_texts.append(text)
                        chunk_token_count = len(chunk_tokens)
                        chunk_token_counts.append(chunk_token_count)
                        total_tokens_retrieved += chunk_token_count
                    
                    # MODIFIED: Use "..." to separate chunks
                    combined_context = "\n...\n".join(retrieved_texts)
                    
                    # Token metrics
                    question_tokens = len(tokenizer.encode(question, add_special_tokens=False))
                    full_prompt = f"""You are provided with excerpts from a document. The excerpts are separated by "..." and represent only relevant snippets from the full document, not the complete text.

Based on these excerpts, answer the question concisely and accurately.

Document Excerpts:
{combined_context}

Question: {question}

Answer:"""
                    prompt_tokens = len(tokenizer.encode(full_prompt, add_special_tokens=False))
                    
                    # Generate answer
                    answer, rationale = generate_answer_with_rationale(combined_context, question)
                    total_gemini_calls += 1
                    
                    answer_tokens = len(tokenizer.encode(answer, add_special_tokens=False)) if answer != "None" else 0
                    
                    # End timing for this query
                    query_end_time = time.time()
                    query_duration = query_end_time - query_start_time
                    
                    # Record query timing
                    if tracker is not None:
                        tracker.record_query_generation(doc_id, question_id, query_duration)
                    
                    # Save result
                    result = {
                        "question": question,
                        "question_id": question_id,
                        "document_id": doc_id,
                        "method": "RAG",
                        "embedding_model": embedding_model_name,
                        "chunk_size": chunk_size,
                        "top_k_chunks": top_k_chunks,
                        "chunk_ordering": "original_position",
                        "chunk_separator": "...",
                        "dataset": dataset,
                        "result": answer,
                        "rationale": rationale,
                        "selected_context": retrieved_texts,
                        "combined_context": combined_context,
                        "token_usage": {
                            "document_total_tokens": len(tokens),
                            "chunks_retrieved": len(retrieved_texts),
                            "chunk_token_counts": chunk_token_counts,
                            "context_tokens": total_tokens_retrieved,
                            "question_tokens": question_tokens,
                            "prompt_tokens": prompt_tokens,
                            "answer_tokens": answer_tokens,
                            "total_tokens_used": prompt_tokens + answer_tokens
                        },
                        "duration": query_duration
                    }
                    
                    with open(results_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    print(f"    ✅ Saved: {results_file.name}")
                    print(f"    ⏱️  Time: {query_duration:.2f}s, Tokens: {total_tokens_retrieved}")
                    total_processed += 1
                    
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    total_skipped += 1
    
    else:
        # OLD PATH: Legacy processing
        print("\n⚠️  Using legacy processing mode - consider migrating to processed datasets")
    
    print(f"\n{'='*60}")
    print(f"🎯 RAG COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Processed: {total_processed}")
    print(f"⚠️  Skipped: {total_skipped}")
    print(f"🤖 Gemini calls: {total_gemini_calls}")
    print(f"📁 Results saved to: {dataset_pred_dir}")
    print(f"{'='*60}")


# ==============================================================================
# SECTION 6: MAIN PIPELINE
# ==============================================================================

def _run_single_rag_pipeline(dataset: str = 'paper',
                             generate_results: bool = True,
                             embedding_model_name: str = 'UAE-Large-V1',
                             chunk_size: int = 200,
                             top_k: int = 5,
                             tracker: PerformanceTracker = None):
    """Internal helper that runs the pipeline for a single embedding model."""
    print("\n" + "="*60)
    print("RAG BASELINE PIPELINE")
    print("="*60)
    print(f"📋 Dataset: {dataset}")
    print(f"🤖 Embedding: {embedding_model_name}")
    print(f"📦 Chunk size: {chunk_size}")
    print(f"🔍 Top-k: {top_k}")
    print("="*60)
    
    if generate_results:
        # Check if embeddings exist in baseline/data/
        safe_model_name = embedding_model_name.replace('/', '_')
        embedding_cache_dir = EMBEDDING_CACHE_ROOT / dataset / f"{safe_model_name}_chunk{chunk_size}"
        
        needs_preprocessing = False
        
        # Load processed dataset to get document count
        try:
            dataset_data = load_processed_dataset(dataset)
            num_documents = len(dataset_data.get('documents', []))
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return
        
        if not embedding_cache_dir.exists():
            print(f"\n⚠️  Embedding cache directory not found: {embedding_cache_dir}")
            print(f"   Will run preprocessing to compute embeddings...")
            needs_preprocessing = True
        else:
            # Check if embedding files exist for all documents
            existing_embeddings = 0
            
            for doc_idx in range(num_documents):
                embeddings_file = embedding_cache_dir / f"{dataset}_{doc_idx}_embeddings.npy"
                chunks_file = embedding_cache_dir / f"{dataset}_{doc_idx}_chunks.json"
                
                if embeddings_file.exists() and chunks_file.exists():
                    existing_embeddings += 1
            
            if existing_embeddings == 0:
                print(f"\n⚠️  No embedding files found in {embedding_cache_dir}")
                print(f"   Will run preprocessing to compute embeddings...")
                needs_preprocessing = True
            else:
                print(f"\n✅ Found {existing_embeddings}/{num_documents} cached embeddings")
                if existing_embeddings < num_documents:
                    print(f"   Will preprocess missing {num_documents - existing_embeddings} documents...")
                    needs_preprocessing = True
        
        # Run preprocessing if needed
        if needs_preprocessing:
            print(f"\n{'='*60}")
            print(f"STEP 1: PREPROCESSING")
            print(f"{'='*60}")
            preprocess_documents_and_compute_embeddings(
                dataset=dataset,
                embedding_model_name=embedding_model_name,
                chunk_size=chunk_size,
                chunk_overlap=50,
                tracker=tracker
            )
        else:
            print(f"\n✅ All embeddings already cached. Skipping preprocessing.")
            
            # Still need to count chunks for tracker
            if tracker is not None:
                tracker.total_documents = num_documents
                for doc_idx in range(num_documents):
                    chunks_file = embedding_cache_dir / f"{dataset}_{doc_idx}_chunks.json"
                    if chunks_file.exists():
                        with open(chunks_file, 'r') as f:
                            chunks_metadata = json.load(f)
                        tracker.total_chunks += len(chunks_metadata)
        
        # STEP 2: Generate results
        generate_rag_results(
            dataset=dataset,
            data=None,  # Use processed dataset
            budget_map=None,  # Use processed dataset
            embedding_model_name=embedding_model_name,
            chunk_size=chunk_size,
            top_k_chunks=top_k,
            tracker=tracker
        )
    else:
        print("\n⚠️ Generation disabled (`generate_results=False`). Skipping preprocessing/generation step.")
    
    print("\n" + "="*60)
    print(f"✅ PIPELINE COMPLETE")
    print(f"🕐 {datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %I:%M %p EDT')}")
    print("="*60)


def run_rag_pipeline(dataset: str = 'paper',
                     generate_results: bool = True,
                     embedding_model_name: str = 'UAE-Large-V1',
                     chunk_size: int = 200,
                     top_k: int = 5):
    """
    Public entry point for the RAG pipeline. Supports running multiple embedding models by
    passing `embedding_model_name='all'` or a list of model names.
    
    Generates performance reports for each (dataset, model) combination.
    """
    if isinstance(embedding_model_name, str):
        if embedding_model_name.lower() == 'all':
            model_list = AVAILABLE_EMBEDDING_MODELS
        else:
            model_list = [embedding_model_name]
    else:
        model_list = list(embedding_model_name)

    for idx, model in enumerate(model_list, 1):
        print("\n" + "=" * 80)
        print(f"🚀 Running RAG pipeline ({idx}/{len(model_list)}) with embedding model: {model}")
        print("=" * 80)
        
        # Create performance tracker for this (dataset, model) combination
        tracker = PerformanceTracker(dataset, model)
        
        _run_single_rag_pipeline(
            dataset=dataset,
            generate_results=generate_results,
            embedding_model_name=model,
            chunk_size=chunk_size,
            top_k=top_k,
            tracker=tracker
        )
        
        # Save performance report after this model is processed
        if generate_results:
            tracker.save_report()

# ==============================================================================
# SECTION 7: MAIN
# ==============================================================================

def parse_arguments():
    """Parse command-line arguments for flexible pipeline configuration."""
    parser = argparse.ArgumentParser(
        description='RAG Baseline with Performance Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments (default)
  python rag.py
  
  # Run only paper dataset with all models and all top_k values
  python rag.py --datasets paper
  
  # Run both datasets with specific model
  python rag.py --datasets paper notice --models UAE-Large-V1
  
  # Run with specific top_k values
  python rag.py --datasets paper --top-k 3 5
  
  # Run single experiment: paper dataset, UAE model, top_k=5
  python rag.py --datasets paper --models UAE-Large-V1 --top-k 5
  
  # Run with multiple specific models
  python rag.py --models UAE-Large-V1 Qwen3-Embedding-8B --top-k 5 7
        """
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['paper', 'notice', 'all'],
        default=['all'],
        help='Dataset(s) to process. Use "all" for both paper and notice (default: all)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['all'],
        help=f'Embedding model(s) to use. Available: {", ".join(AVAILABLE_EMBEDDING_MODELS)}. Use "all" for all models (default: all)'
    )
    
    parser.add_argument(
        '--top-k',
        nargs='+',
        type=int,
        default=[1, 3, 5, 7, 10],
        help='Top-k value(s) for chunk retrieval (default: 1 3 5 7 10)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=200,
        help='Chunk size in tokens (default: 200)'
    )
    
    parser.add_argument(
        '--no-generate',
        action='store_true',
        help='Skip result generation (only show configuration)'
    )
    
    return parser.parse_args()


def validate_and_prepare_config(args):
    """Validate arguments and prepare configuration."""
    # Process datasets
    if 'all' in args.datasets:
        datasets = ['paper', 'notice']
    else:
        datasets = args.datasets
    
    # Process models
    if 'all' in args.models:
        models = AVAILABLE_EMBEDDING_MODELS
    else:
        # Validate model names
        invalid_models = [m for m in args.models if m not in AVAILABLE_EMBEDDING_MODELS]
        if invalid_models:
            print(f"❌ Error: Invalid model(s): {', '.join(invalid_models)}")
            print(f"Available models: {', '.join(AVAILABLE_EMBEDDING_MODELS)}")
            sys.exit(1)
        models = args.models
    
    # Validate top_k values
    if any(k < 1 for k in args.top_k):
        print(f"❌ Error: top_k values must be positive integers")
        sys.exit(1)
    
    top_k_values = sorted(args.top_k)
    
    config = {
        'datasets': datasets,
        'models': models,
        'top_k_values': top_k_values,
        'chunk_size': args.chunk_size,
        'generate_results': not args.no_generate
    }
    
    return config


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    config = validate_and_prepare_config(args)
    
    # Extract configuration
    DATASETS = config['datasets']
    EMBEDDING_MODELS = config['models']
    TOP_K_VALUES = config['top_k_values']
    CHUNK_SIZE = config['chunk_size']
    GENERATE_RESULTS = config['generate_results']
    
    # ==========================================================================
    # RAG BASELINE - COMPREHENSIVE EVALUATION WITH PERFORMANCE TRACKING
    # ==========================================================================
    
    print("\n" + "="*80)
    print("RAG BASELINE - RESULT GENERATION WITH PERFORMANCE TRACKING")
    print("="*80)
    print(f"📋 Datasets: {', '.join(DATASETS)}")
    print(f"🔍 Top-k values: {TOP_K_VALUES}")
    print(f"🤖 Embedding models: {len(EMBEDDING_MODELS)}")
    for model in EMBEDDING_MODELS:
        print(f"   - {model}")
    print(f"📦 Chunk size: {CHUNK_SIZE} tokens")
    print(f"🎯 Total experiments: {len(DATASETS)} × {len(TOP_K_VALUES)} × {len(EMBEDDING_MODELS)} = {len(DATASETS) * len(TOP_K_VALUES) * len(EMBEDDING_MODELS)}")
    print(f"📊 Performance reports: One per dataset-model-k combination")
    print(f"📁 Reports directory: {REPORTS_ROOT}")
    print(f"🔄 Generate results: {'Yes' if GENERATE_RESULTS else 'No (dry run)'}")
    print("="*80)
    
    if not GENERATE_RESULTS:
        print("\n⚠️  DRY RUN MODE - No results will be generated")
        print("Remove --no-generate flag to run experiments")
        sys.exit(0)
    
    experiment_count = 0
    total_experiments = len(DATASETS) * len(TOP_K_VALUES) * len(EMBEDDING_MODELS)
    
    # Outer loop: datasets
    for dataset_idx, dataset in enumerate(DATASETS, 1):
        print(f"\n{'#'*80}")
        print(f"# DATASET {dataset_idx}/{len(DATASETS)}: {dataset.upper()}")
        print(f"{'#'*80}")
        
        # Middle loop: embedding models
        for model_idx, model_name in enumerate(EMBEDDING_MODELS, 1):
            print(f"\n{'='*80}")
            print(f"MODEL {model_idx}/{len(EMBEDDING_MODELS)}: {model_name}")
            print(f"{'='*80}")
            
            # Inner loop: top_k values
            for k_idx, top_k in enumerate(TOP_K_VALUES, 1):
                experiment_count += 1
                
                print(f"\n{'─'*80}")
                print(f"⚡ EXPERIMENT {experiment_count}/{total_experiments}")
                print(f"   Dataset: {dataset} | Model: {model_name} | top_k: {top_k}")
                print(f"{'─'*80}")
                
                # Create a performance tracker for each (dataset, model, k) combination
                tracker = PerformanceTracker(dataset, model_name, top_k)
                
                try:
                    # Use the (dataset, model, k)-specific tracker
                    _run_single_rag_pipeline(
                        dataset=dataset,
                        generate_results=GENERATE_RESULTS,
                        embedding_model_name=model_name,
                        chunk_size=CHUNK_SIZE,
                        top_k=top_k,
                        tracker=tracker
                    )
                    
                    print(f"\n✅ Experiment {experiment_count}/{total_experiments} completed successfully!")
                    
                    # Save performance report for this (dataset, model, k) combination
                    if GENERATE_RESULTS:
                        tracker.save_report()
                    
                except Exception as e:
                    print(f"\n❌ ERROR in Experiment {experiment_count}/{total_experiments}:")
                    print(f"   Dataset: {dataset}, Model: {model_name}, top_k: {top_k}")
                    print(f"   Error: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"\n⚠️  Continuing to next experiment...")
                
                print(f"\n{'─'*80}")
                print(f"Progress: {experiment_count}/{total_experiments} ({100*experiment_count/total_experiments:.1f}%)")
                print(f"{'─'*80}")
    
    print(f"\n{'#'*80}")
    print(f"# 🎉 ALL EXPERIMENTS COMPLETE!")
    print(f"{'#'*80}")
    print(f"✅ Completed: {experiment_count}/{total_experiments} experiments")
    print(f"📊 Datasets tested: {', '.join(DATASETS)}")
    print(f"🔍 Top-k values: {TOP_K_VALUES}")
    print(f"🤖 Models tested: {len(EMBEDDING_MODELS)}")
    print(f"📁 Results directory: baseline/pred_att/")
    print(f"📊 Performance reports: {REPORTS_ROOT}")
    print(f"   - {len(DATASETS) * len(EMBEDDING_MODELS) * len(TOP_K_VALUES)} report files (one per dataset-model-k combo)")
    print(f"🕐 Completed at: {datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %I:%M %p EDT')}")
    print(f"{'#'*80}")