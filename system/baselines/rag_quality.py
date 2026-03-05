#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Baseline for Quality Dataset - WITH PERFORMANCE TRACKING
------------------------------------------------------------

This script implements a RAG baseline for the QuALITY dataset where
documents and questions are stored in a single JSON file.

**Key Features:**
- Reads from system/data/datasets/quality_processed.json
- Outputs to system/baselines/ subdirectories
- Uses UNIFIED RAG generation code (same as rag.py)
- Reorders chunks by original document position
- Separates chunks with "..." in context
- Uses Gemini 2.5 Flash Lite for answer generation
- Tracks performance metrics (embedding time, query generation time)
- Generates performance reports in ../reports/

**RAG Workflow:**
1. Load quality_processed.json from system/data/datasets/
2. Tokenize each document's content
3. Chunk tokens into fixed-size chunks (200 tokens)
4. Compute embeddings for all chunks (cached in baselines/data/) - TRACKED
5. For each question: retrieve top-k chunks, reorder by position, generate answer - TRACKED
6. Save results to baselines/pred_att/quality/
7. Generate performance report to ../reports/performance_report_quality_rag.json
"""

import sys
import os
import warnings
import logging

# Suppress logging
os.environ['PYTHONUNBUFFLED'] = '1'
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
# PATH CONFIGURATION - Since file is now in baselines/ subdirectory
# ==============================================================================

def get_project_root() -> Path:
    """Get the project root directory (parent of baselines/)."""
    return Path(__file__).parent.parent

def get_baseline_dir() -> Path:
    """Get the baseline directory (where this script is located)."""
    return Path(__file__).parent

# Input data paths - read from project root (original location)
DATA_ROOT = get_project_root() / "llama3.2_1b" / "data"

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
# PERFORMANCE TRACKING
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
# INITIALIZATION
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
    
    quantized_models = ['Qwen3-Embedding-8B', 'GritLM-7B']
    
    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model: {model_name}")
    
    hf_model_name = model_mapping[model_name]
    use_quantization = model_name in quantized_models
    
    print(f"🔄 Loading embedding model: {hf_model_name}")
    if use_quantization:
        print(f"   Using 8-bit quantization")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {device}")
        
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
        else:
            embedding_model = SentenceTransformer(
                hf_model_name,
                cache_folder="/tmp/huggingface-cache",
                trust_remote_code=True,
                device=device
            )
        
        current_embedding_model_name = model_name
        print(f"✅ Embedding model loaded successfully!")
        
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
        print("⚠️ Warning: GEMINI_API_KEY not found")

# ==============================================================================
# CHUNKING AND EMBEDDING (WITH PERFORMANCE TRACKING)
# ==============================================================================

def chunk_tokens(tokens: List[str], chunk_size: int = 200, overlap: int = 50) -> List[Tuple[int, int, List[str]]]:
    """Split tokens into overlapping chunks."""
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = tokens[start:end]
        chunks.append((start, end, chunk))
        
        start = end - overlap if end < len(tokens) else end
        
        if start >= len(tokens):
            break
    
    print(f"  📦 Created {len(chunks)} chunks from {len(tokens)} tokens")
    return chunks

def embed_chunks(chunks: List[Tuple[int, int, List[str]]], 
                 doc_id: str = None,
                 tracker: PerformanceTracker = None) -> np.ndarray:
    """Create embeddings for all chunks with per-chunk timing."""
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
    """Retrieve top-k most relevant chunks."""
    global embedding_model, current_embedding_model_name
    
    if embedding_model is None:
        raise ValueError("Embedding model not initialized")
    
    question_embedding = embedding_model.encode(
        question,
        convert_to_tensor=True,
        show_progress_bar=False
    )
    
    device = question_embedding.device
    chunk_embeddings_tensor = torch.from_numpy(chunk_embeddings).to(device)
    similarities = util.cos_sim(question_embedding, chunk_embeddings_tensor)[0]
    similarities = similarities.cpu().numpy()
    
    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
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
    """
    return sorted(retrieved_chunks, key=lambda x: x[0])  # Sort by start_idx

# ==============================================================================
# ANSWER GENERATION
# ==============================================================================

def generate_answer_with_rationale(context: str, question: str) -> tuple[str, str]:
    """
    Generate answer from context using Gemini 2.5 Flash Lite.
    """
    try:
        if isinstance(context, list):
            context_text = "\n...\n".join(context)
        else:
            context_text = context

        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        prompt = f"""You are provided with excerpts from a document. The excerpts are separated by "..." and represent only relevant snippets from the full document, not the complete text.

Based on these excerpts, answer the question concisely and accurately.

Document Excerpts:
{context_text}

Question: {question}

Answer:"""
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=150,
            )
        )
        
        answer = response.text.strip()
        
        # Update rationale to mention excerpts
        num_words = len(context_text.split())
        num_excerpts = context_text.count("\n...\n") + 1 if "\n...\n" in context_text else 1
        rationale = f"Generated answer using Gemini 2.5 Flash Lite from {num_excerpts} document excerpts ({num_words} words total)."

        return answer if answer else "None", rationale
        
    except Exception as e:
        print(f"❌ Error during Gemini API call: {e}")
        return "None", f"Error occurred: {str(e)}"

# ==============================================================================
# QUALITY DATASET SPECIFIC: PREPROCESSING (WITH PERFORMANCE TRACKING)
# ==============================================================================

def preprocess_quality_dataset(dataset_file: str = None,
                               embedding_model_name: str = 'UAE-Large-V1',
                               chunk_size: int = 200,
                               chunk_overlap: int = 50,
                               tracker: PerformanceTracker = None):
    """
    Preprocess Quality dataset: load documents, tokenize, chunk, compute embeddings.
    
    Args:
        dataset_file: Path to quality_processed.json (defaults to system/data/datasets/)
        embedding_model_name: Embedding model to use
        chunk_size: Tokens per chunk
        chunk_overlap: Overlap between chunks
        tracker: PerformanceTracker instance for recording timings
    """
    # Use default path if not provided
    if dataset_file is None:
        dataset_file = DATA_ROOT / 'datasets' / 'quality_processed.json'
    
    print("\n" + "="*60)
    print("PREPROCESSING - Quality Dataset")
    print("="*60)
    print(f"📋 Dataset file: {dataset_file}")
    print(f"🤖 Embedding model: {embedding_model_name}")
    print(f"📦 Chunk size: {chunk_size} tokens")
    print("="*60)
    
    # Initialize
    initialize_tokenizer()
    initialize_embedding_model(embedding_model_name)
    
    # Setup directories in baseline/
    token_dir = TOKEN_CACHE_ROOT
    token_dir.mkdir(parents=True, exist_ok=True)
    
    safe_model_name = embedding_model_name.replace('/', '_')
    embedding_cache_dir = EMBEDDING_CACHE_ROOT / "quality" / f"{safe_model_name}_chunk{chunk_size}"
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Token cache: {token_dir}")
    print(f"📁 Embedding cache: {embedding_cache_dir}")
    
    # Load quality dataset
    print(f"\n📂 Loading dataset from: {dataset_file}")
    with open(dataset_file, 'r') as f:
        quality_data = json.load(f)
    
    documents = quality_data.get('documents', [])
    print(f"✅ Loaded {len(documents)} documents")
    
    # Update tracker
    if tracker is not None:
        tracker.total_documents = len(documents)
    
    total_processed = 0
    total_skipped = 0
    
    for doc in documents:
        doc_id = doc.get('document_id')
        content = doc.get('content', '')
        
        print(f"\n{'─'*60}")
        print(f"📄 Processing document {doc_id}")
        print(f"{'─'*60}")
        
        token_file = token_dir / f"quality_{doc_id}.json"
        embeddings_file = embedding_cache_dir / f"quality_{doc_id}_embeddings.npy"
        chunks_file = embedding_cache_dir / f"quality_{doc_id}_chunks.json"
        
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
                print(f"  ⚠️  Empty content in document {doc_id}")
                total_skipped += 1
                continue
            
            print(f"  ✅ Loaded document ({len(content)} characters)")
            
            # Tokenize document
            print(f"  🔄 Tokenizing document...")
            global tokenizer
            tokens = tokenizer.encode(content, add_special_tokens=False)
            tokens_as_strings = tokenizer.convert_ids_to_tokens(tokens)
            
            print(f"  ✅ Tokenized: {len(tokens)} tokens")
            
            # Save tokens
            if not token_file.exists():
                with open(token_file, 'w') as f:
                    json.dump(tokens_as_strings, f)
                print(f"  💾 Saved tokens to: {token_file.name}")
            
            # Chunk the tokens
            print(f"  📦 Chunking into {chunk_size}-token chunks...")
            chunks = chunk_tokens(tokens_as_strings, chunk_size=chunk_size, overlap=chunk_overlap)
            
            # Update tracker
            if tracker is not None:
                tracker.total_chunks += len(chunks)
            
            # Compute embeddings with timing
            print(f"  🔄 Computing embeddings for {len(chunks)} chunks...")
            chunk_embeddings = embed_chunks(chunks, doc_id=doc_id, tracker=tracker)
            
            # Cache embeddings
            np.save(embeddings_file, chunk_embeddings)
            
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
# QUALITY DATASET SPECIFIC: RAG PIPELINE (WITH PERFORMANCE TRACKING)
# ==============================================================================

def generate_quality_rag_results(dataset_file: str = None,
                                 embedding_model_name: str = 'UAE-Large-V1',
                                 chunk_size: int = 200,
                                 chunk_overlap: int = 50,
                                 top_k_chunks: int = 5,
                                 tracker: PerformanceTracker = None):
    """
    Generate RAG results for Quality dataset with performance tracking.
    
    Args:
        dataset_file: Path to quality_processed.json
        embedding_model_name: Embedding model to use
        chunk_size: Tokens per chunk
        chunk_overlap: Overlap between chunks
        top_k_chunks: Number of top chunks to retrieve
        tracker: PerformanceTracker instance for recording timings
    """
    # Use default path if not provided
    if dataset_file is None:
        dataset_file = DATA_ROOT  / 'datasets' / 'quality_processed.json'
    
    print("\n" + "="*60)
    print("RAG BASELINE - Quality Dataset")
    print("="*60)
    print(f"📋 Dataset file: {dataset_file}")
    print(f"🤖 Embedding model: {embedding_model_name}")
    print(f"📦 Chunk size: {chunk_size} tokens")
    print(f"🔍 Top-k retrieval: {top_k_chunks} chunks")
    print(f"🔄 Chunk ordering: Original document order")
    print(f"📝 Chunk separation: '...' between excerpts")
    print("="*60)
    
    # Initialize
    print("\n🔧 Initializing components...")
    initialize_tokenizer()
    initialize_embedding_model(embedding_model_name)
    initialize_gemini()
    
    # Load quality dataset
    print(f"\n📂 Loading dataset...")
    with open(dataset_file, 'r') as f:
        quality_data = json.load(f)
    
    documents = quality_data.get('documents', [])
    print(f"✅ Loaded {len(documents)} documents")
    
    # Extract all unique questions
    all_questions = set()
    doc_question_map = {}
    
    for doc in documents:
        doc_id = doc.get('document_id')
        doc_question_map[doc_id] = {}
        
        for q in doc.get('questions', []):
            question_text = q.get('question', '')
            all_questions.add(question_text)
            doc_question_map[doc_id][question_text] = {
                'question_id': q.get('question_id'),
                'answer': q.get('answer', '')
            }
    
    questions = sorted(list(all_questions))
    print(f"✅ Found {len(questions)} unique questions")
    
    # Setup directories in baseline/
    pred_att_dir = PRED_ATT_ROOT
    dataset_pred_dir = pred_att_dir / "quality"
    dataset_pred_dir.mkdir(parents=True, exist_ok=True)
    
    token_dir = TOKEN_CACHE_ROOT
    
    safe_model_name = embedding_model_name.replace('/', '_')
    embedding_cache_dir = EMBEDDING_CACHE_ROOT / "quality" / f"{safe_model_name}_chunk{chunk_size}"
    
    print(f"\n📁 Directories:")
    print(f"   Embedding cache: {embedding_cache_dir}")
    print(f"   Results output: {dataset_pred_dir}")
    
    total_gemini_calls = 0
    
    for query_idx, question in enumerate(questions):
        print(f"\n{'='*60}")
        print(f"Question {query_idx + 1}/{len(questions)}: '{question[:50]}...'")
        print(f"{'='*60}")
        
        # Sanitize question for filename
        safe_question = question.lower()
        safe_question = re.sub(r'[^\w\s-]', '', safe_question)
        safe_question = re.sub(r'[-\s]+', '_', safe_question)
        safe_question = safe_question.strip('_')
        if len(safe_question) > 80:
            safe_question = safe_question[:80].rstrip('_')
        if not safe_question:
            safe_question = f"question_{query_idx}"
        
        safe_model = embedding_model_name.replace('/', '-').replace('_', '-')
        results_file = dataset_pred_dir / f"results-rag-{safe_model}-top{top_k_chunks}-quality_{safe_question}.json"
        
        print(f"📝 Output file: {results_file.name}")
        
        if results_file.exists():
            print(f"✅ Results file already exists. Skipping.")
            continue
        
        results = {
            "question": question,
            "method": "RAG",
            "embedding_model": embedding_model_name,
            "chunk_size": chunk_size,
            "top_k_chunks": top_k_chunks,
            "chunk_ordering": "original_position",
            "chunk_separator": "...",
            "dataset": "quality",
            "files": []
        }
        
        # Process each document that has this question
        for doc in documents:
            doc_id = doc.get('document_id')
            
            # Check if this document has this question
            if question not in doc_question_map.get(doc_id, {}):
                continue
            
            # Get question_id for tracking
            question_id = doc_question_map[doc_id][question]['question_id']
            
            token_file = token_dir / f"quality_{doc_id}.json"
            
            if not token_file.exists():
                print(f"⚠️  Document {doc_id}: Missing token file")
                continue
            
            try:
                # Start timing for this query
                query_start_time = time.time()
                
                print(f"\n📄 Processing document {doc_id}")
                
                # Load tokens
                with open(token_file, 'r') as f:
                    tokens = json.load(f)
                print(f"  ✅ Loaded {len(tokens)} tokens")
                
                # Load cached embeddings
                embeddings_file = embedding_cache_dir / f"quality_{doc_id}_embeddings.npy"
                chunks_file = embedding_cache_dir / f"quality_{doc_id}_chunks.json"
                
                if embeddings_file.exists() and chunks_file.exists():
                    print(f"  📂 Loading cached embeddings...")
                    
                    chunk_embeddings = np.load(embeddings_file)
                    
                    with open(chunks_file, 'r') as f:
                        chunks_metadata = json.load(f)
                    
                    chunks = []
                    for chunk_meta in chunks_metadata:
                        start = chunk_meta["start"]
                        end = chunk_meta["end"]
                        chunk_tokens = tokens[start:end]
                        chunks.append((start, end, chunk_tokens))
                    
                    print(f"  ✅ Loaded {len(chunks)} chunks with embeddings")
                    
                else:
                    print(f"  ⚠️  Embeddings not cached. Run preprocessing first!")
                    continue
                
                # === UNIFIED RAG GENERATION CODE ===
                
                # Retrieve top-k chunks
                print(f"  🔍 Retrieving top-{top_k_chunks} chunks...")
                retrieved_chunks = retrieve_top_k_chunks(
                    question, 
                    chunks, 
                    chunk_embeddings, 
                    top_k=top_k_chunks
                )
                
                # Reorder by original position
                retrieved_chunks = reorder_chunks_by_position(retrieved_chunks)
                print(f"  ✅ Retrieved & reordered {len(retrieved_chunks)} chunks")
                
                # Prepare context with "..." separators
                retrieved_texts = []
                chunk_token_counts = []
                total_tokens_retrieved = 0
                
                for start, end, chunk_tokens, score in retrieved_chunks:
                    token_ids = tokenizer.convert_tokens_to_ids(chunk_tokens)
                    text = tokenizer.decode(token_ids, skip_special_tokens=True)
                    retrieved_texts.append(text)
                    chunk_token_count = len(chunk_tokens)
                    chunk_token_counts.append(chunk_token_count)
                    total_tokens_retrieved += chunk_token_count
                
                # Use "..." to separate chunks
                combined_context = "\n...\n".join(retrieved_texts)
                
                # Calculate token metrics
                question_tokens = len(tokenizer.encode(question, add_special_tokens=False))
                
                full_prompt = f"""You are provided with excerpts from a document. The excerpts are separated by "..." and represent only relevant snippets from the full document, not the complete text.

Based on these excerpts, answer the question concisely and accurately.

Document Excerpts:
{combined_context}

Question: {question}

Answer:"""
                prompt_tokens = len(tokenizer.encode(full_prompt, add_special_tokens=False))
                
                print(f"  ✅ Retrieved {len(retrieved_texts)} chunks ({total_tokens_retrieved} tokens)")
                
                # Generate answer
                print(f"  🤖 Calling Gemini API...")
                answer, rationale = generate_answer_with_rationale(combined_context, question)
                total_gemini_calls += 1
                
                answer_tokens = len(tokenizer.encode(answer, add_special_tokens=False)) if answer != "None" else 0
                
                print(f"  ✅ Answer: {answer[:100]}...")
                
                # End timing for this query
                query_end_time = time.time()
                query_duration = query_end_time - query_start_time
                
                # Record query timing
                if tracker is not None:
                    tracker.record_query_generation(doc_id, question_id, query_duration)
                
                # === END UNIFIED RAG GENERATION CODE ===
                
                # Store result
                results["files"].append({
                    "file": str(doc_id),
                    "result": answer,
                    "rationale": rationale,
                    "ratio": 0.0,
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
                    "tokens_extracted": total_tokens_retrieved,
                    "duration": query_duration
                })
                
                print(f"  ⏱️  Time: {query_duration:.2f}s")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Save results
        if results["files"]:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            total_context_tokens = sum(f["token_usage"]["context_tokens"] for f in results["files"])
            avg_context_tokens = total_context_tokens / len(results["files"])
            
            print(f"\n{'='*60}")
            print(f"✅ Saved: {results_file.name}")
            print(f"📊 Processed: {len(results['files'])} documents")
            print(f"🤖 Gemini calls: {total_gemini_calls}")
            print(f"📈 Average context: {avg_context_tokens:.1f} tokens")
            print(f"{'='*60}")
        else:
            print(f"\n⚠️  No documents processed for this question")
    
    print(f"\n{'='*60}")
    print(f"🎯 RAG COMPLETE")
    print(f"{'='*60}")
    print(f"🤖 Total Gemini calls: {total_gemini_calls}")
    print(f"📁 Results saved to: {dataset_pred_dir}")
    print(f"{'='*60}")

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_quality_rag_pipeline(generate_results: bool = True,
                             embedding_model_name: str = 'UAE-Large-V1',
                             chunk_size: int = 200,
                             top_k: int = 5,
                             dataset_file: str = None,
                             tracker: PerformanceTracker = None):
    """
    Run complete RAG pipeline for Quality dataset with performance tracking.
    
    Args:
        generate_results: Whether to generate RAG results
        embedding_model_name: Embedding model to use or 'all' for all models
        chunk_size: Tokens per chunk (default: 200)
        top_k: Number of chunks to retrieve
        dataset_file: Path to quality_processed.json
        tracker: PerformanceTracker instance
    """
    # Use default path if not provided
    if dataset_file is None:
        dataset_file = DATA_ROOT / 'datasets' / 'quality_processed.json'
    
    if isinstance(embedding_model_name, str):
        if embedding_model_name.lower() == 'all':
            model_list = AVAILABLE_EMBEDDING_MODELS
        else:
            model_list = [embedding_model_name]
    else:
        model_list = list(embedding_model_name)

    for idx, model in enumerate(model_list, 1):
        print("\n" + "=" * 80)
        print(f"🚀 RAG Pipeline ({idx}/{len(model_list)}): {model}")
        print("=" * 80)
        
        if generate_results:
            # Check if preprocessing needed
            safe_model_name = model.replace('/', '_')
            embedding_cache_dir = EMBEDDING_CACHE_ROOT / "quality" / f"{safe_model_name}_chunk{chunk_size}"
            
            needs_preprocessing = False
            
            if not embedding_cache_dir.exists():
                print(f"\n⚠️  Embeddings not found. Will preprocess.")
                needs_preprocessing = True
            else:
                # Check for existing embeddings
                with open(dataset_file, 'r') as f:
                    quality_data = json.load(f)
                documents = quality_data.get('documents', [])
                
                existing = sum(1 for doc in documents 
                              if (embedding_cache_dir / f"quality_{doc['document_id']}_embeddings.npy").exists())
                
                if existing == 0:
                    print(f"\n⚠️  No embeddings found. Will preprocess.")
                    needs_preprocessing = True
                elif existing < len(documents):
                    print(f"\n⚠️  Only {existing}/{len(documents)} embeddings found. Will preprocess missing.")
                    needs_preprocessing = True
                else:
                    print(f"\n✅ All {len(documents)} embeddings cached.")
                    
                    # Still need to count chunks for tracker
                    if tracker is not None:
                        tracker.total_documents = len(documents)
                        for doc in documents:
                            doc_id = doc.get('document_id')
                            chunks_file = embedding_cache_dir / f"quality_{doc_id}_chunks.json"
                            if chunks_file.exists():
                                with open(chunks_file, 'r') as f:
                                    chunks_metadata = json.load(f)
                                tracker.total_chunks += len(chunks_metadata)
            
            if needs_preprocessing:
                print(f"\n{'='*60}")
                print(f"STEP 1: PREPROCESSING")
                print(f"{'='*60}")
                preprocess_quality_dataset(
                    dataset_file=dataset_file,
                    embedding_model_name=model,
                    chunk_size=chunk_size,
                    chunk_overlap=50,
                    tracker=tracker
                )
            
            continue
            # STEP 2: Generate results
            print(f"\n{'='*60}")
            print(f"STEP 2: GENERATING RESULTS")
            print(f"{'='*60}")
            generate_quality_rag_results(
                dataset_file=dataset_file,
                embedding_model_name=model,
                chunk_size=chunk_size,
                top_k_chunks=top_k,
                tracker=tracker
            )
        
        print("\n" + "="*60)
        print(f"✅ PIPELINE COMPLETE: {model}")
        print("="*60)

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # Quality Dataset RAG Evaluation with Performance Tracking
    # Run across multiple top_k values and embedding models
    
    # TOP_K_VALUES = [1, 3, 5, 7, 10]
    TOP_K_VALUES = [1]
    CHUNK_SIZE = 200
    DATASET_FILE = DATA_ROOT / 'datasets' / 'quality_processed.json'
    GENERATE_RESULTS = True
    
    print("\n" + "="*80)
    print("QUALITY DATASET - RAG BASELINE EVALUATION WITH PERFORMANCE TRACKING")
    print("="*80)
    print(f"📋 Dataset: {DATASET_FILE}")
    print(f"🔍 Top-k values: {TOP_K_VALUES}")
    print(f"🤖 Models: {len(AVAILABLE_EMBEDDING_MODELS)}")
    for model in AVAILABLE_EMBEDDING_MODELS:
        print(f"   - {model}")
    print(f"📦 Chunk size: {CHUNK_SIZE}")
    print(f"🔄 Chunk ordering: Original document order")
    print(f"📝 Chunk separator: '...'")
    print(f"🎯 Total experiments: {len(TOP_K_VALUES)} × {len(AVAILABLE_EMBEDDING_MODELS)} = {len(TOP_K_VALUES) * len(AVAILABLE_EMBEDDING_MODELS)}")
    print(f"📊 Performance reports: One per model-k combination")
    print(f"📁 Reports directory: {REPORTS_ROOT}")
    print("="*80)
    
    experiment_count = 0
    total_experiments = len(TOP_K_VALUES) * len(AVAILABLE_EMBEDDING_MODELS)
    
    for model_idx, model_name in enumerate(AVAILABLE_EMBEDDING_MODELS, 1):
        print(f"\n{'#'*80}")
        print(f"# MODEL {model_idx}/{len(AVAILABLE_EMBEDDING_MODELS)}: {model_name}")
        print(f"{'#'*80}")
        
        for k_idx, top_k in enumerate(TOP_K_VALUES, 1):
            experiment_count += 1
            
            print(f"\n{'─'*80}")
            print(f"⚡ EXPERIMENT {experiment_count}/{total_experiments}")
            print(f"   Model: {model_name} | top_k: {top_k}")
            print(f"{'─'*80}")
            
            # Create a performance tracker for each (model, k) combination
            tracker = PerformanceTracker("quality", model_name, top_k)
            
            try:
                run_quality_rag_pipeline(
                    generate_results=GENERATE_RESULTS,
                    embedding_model_name=model_name,
                    chunk_size=CHUNK_SIZE,
                    top_k=top_k,
                    dataset_file=DATASET_FILE,
                    tracker=tracker
                )
                
                print(f"\n✅ Experiment {experiment_count}/{total_experiments} completed!")
                
                # Save performance report for this (model, k) combination
                if GENERATE_RESULTS:
                    tracker.save_report()
                
            except Exception as e:
                print(f"\n❌ ERROR in Experiment {experiment_count}/{total_experiments}:")
                print(f"   Model: {model_name}, top_k: {top_k}")
                print(f"   Error: {e}")
                import traceback
                traceback.print_exc()
                print(f"\n⚠️  Continuing...")
            
            print(f"\n{'─'*80}")
            print(f"Progress: {experiment_count}/{total_experiments} ({100*experiment_count/total_experiments:.1f}%)")
            print(f"{'─'*80}")
    
    print(f"\n{'#'*80}")
    print(f"# 🎉 ALL EXPERIMENTS COMPLETE!")
    print(f"{'#'*80}")
    print(f"✅ Completed: {experiment_count}/{total_experiments}")
    print(f"🔍 Top-k values: {TOP_K_VALUES}")
    print(f"🤖 Models: {len(AVAILABLE_EMBEDDING_MODELS)}")
    print(f"📁 Results: {PRED_ATT_ROOT / 'quality'}")
    print(f"📊 Performance reports: {REPORTS_ROOT}")
    print(f"   - {len(AVAILABLE_EMBEDDING_MODELS) * len(TOP_K_VALUES)} report files (one per model-k combo)")
    print(f"🕐 Completed: {datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %I:%M %p EDT')}")
    print(f"{'#'*80}")