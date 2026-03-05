#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aitqa Baseline: RAG with Row-level Retrieval
------------------------------------
Retrieves relevant table rows based on question embeddings and passes
them to LLM for answer generation.

Usage:
    # Run all embedding models
    python baselines/aitqa_baseline_rag.py --embedding_model all --top_k 5
    
    # Run specific embedding model
    python baselines/aitqa_baseline_rag.py --embedding_model UAE-Large-V1 --top_k 5
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import google.generativeai as genai
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================

AVAILABLE_EMBEDDING_MODELS = [
    'UAE-Large-V1',
    'bflhc-Octen-Embedding-4B',
    'Qwen3-Embedding-8B'
]

# Global variables
embedding_model = None
current_embedding_model_name = None

# ============================================================================
# INITIALIZATION
# ============================================================================

def clear_gpu_memory():
    """Aggressively clear GPU memory cache"""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        time.sleep(1)

def initialize_gemini():
    """Initialize Gemini API."""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ Gemini API configured")
    else:
        print("⚠️ Warning: GEMINI_API_KEY not found in environment variables")
        print("   Set it with: export GEMINI_API_KEY='your-api-key'")
        sys.exit(1)

def initialize_tokenizer(model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
    """Initialize tokenizer for token counting."""
    print(f"📥 Loading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir="/tmp/huggingface-cache", trust_remote_code=True
        )
        print("✅ Tokenizer initialized successfully.")
        return tokenizer
    except Exception as e:
        print(f"⚠️ Tokenizer load failed: {e}. Token counts will be inaccurate.")
        return None

def initialize_embedding_model(model_name: str):
    """Initialize embedding model using SentenceTransformer."""
    global embedding_model, current_embedding_model_name
    
    if current_embedding_model_name == model_name and embedding_model is not None:
        print(f"✅ Embedding model '{model_name}' already initialized")
        device = embedding_model.device if hasattr(embedding_model, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return None, embedding_model, device
    
    # Clear any existing model from memory
    if embedding_model is not None:
        print(f"🔄 Clearing previous model from memory...")
        del embedding_model
        embedding_model = None
        clear_gpu_memory()
    
    # Model mapping
    model_mapping = {
        'UAE-Large-V1': 'WhereIsAI/UAE-Large-V1',
        'bflhc-Octen-Embedding-4B': 'bflhc/Octen-Embedding-4B',
        'Qwen3-Embedding-8B': 'Qwen/Qwen3-Embedding-8B'
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {list(model_mapping.keys())}")
    
    hf_model_name = model_mapping[model_name]
    
    print(f"📥 Loading embedding model: {hf_model_name}")
    
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
        
        return None, embedding_model, device
        
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def cleanup_embedding_model():
    """Clean up embedding model to free memory."""
    global embedding_model
    print(f"🧹 Cleaning up embedding model...")
    if embedding_model is not None:
        del embedding_model
        embedding_model = None
    clear_gpu_memory()
    print(f"✅ Embedding model cleaned up")

# ============================================================================
# EMBEDDING CACHE FUNCTIONS
# ============================================================================

def load_embeddings(dataset_name: str, doc_id: int, embedding_model_name: str, expected_count: int) -> Optional[np.ndarray]:
    """
    Load cached embeddings from disk.
    
    Args:
        dataset_name: Name of dataset
        doc_id: Document ID
        embedding_model_name: Name of embedding model
        expected_count: Expected number of chunks
    
    Returns:
        Numpy array of embeddings, or None if not found or count mismatch
    """
    cache_dir = Path(f"data/rag_embeddings/{dataset_name}/{embedding_model_name}")
    cache_file = cache_dir / f"doc_{doc_id}_embeddings.npz"
    
    if not cache_file.exists():
        return None
    
    try:
        data = np.load(cache_file)
        embeddings = data['embeddings']
        num_chunks = data['num_chunks']
        
        # Verify count matches
        if num_chunks != expected_count:
            print(f"   ⚠️ Cached embedding count mismatch: expected {expected_count}, got {num_chunks}")
            return None
        
        print(f"   ✅ Loaded cached embeddings: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        print(f"   ⚠️ Error loading cached embeddings: {e}")
        return None

def save_embeddings(chunks: List[Dict], embeddings: np.ndarray, dataset_name: str, doc_id: int, embedding_model_name: str):
    """
    Save embeddings to disk cache.
    
    Args:
        chunks: List of chunks
        embeddings: Numpy array of embeddings
        dataset_name: Name of dataset
        doc_id: Document ID
        embedding_model_name: Name of embedding model
    """
    cache_dir = Path(f"data/rag_embeddings/{dataset_name}/{embedding_model_name}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"doc_{doc_id}_embeddings.npz"
    
    try:
        np.savez_compressed(
            cache_file,
            embeddings=embeddings,
            num_chunks=len(chunks),
            embedding_model=embedding_model_name
        )
        print(f"   💾 Saved embeddings to cache: {cache_file.name}")
    except Exception as e:
        print(f"   ⚠️ Error saving embeddings: {e}")

# ============================================================================
# EMBEDDING AND RETRIEVAL
# ============================================================================

def get_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Generate embeddings for a list of texts using SentenceTransformer."""
    global embedding_model
    
    if embedding_model is None:
        raise ValueError("Embedding model not initialized")
    
    try:
        embeddings = embedding_model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        return embeddings.cpu().numpy()
        
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()
        raise

def get_or_compute_chunk_embeddings(chunks: List[Dict], dataset_name: str, doc_id: int,
                                    embedding_model_name: str) -> np.ndarray:
    """
    Load existing chunk embeddings or compute new ones.
    
    Args:
        chunks: List of row chunks
        dataset_name: Name of dataset
        doc_id: Document ID
        embedding_model_name: Name of embedding model
    
    Returns:
        Numpy array of chunk embeddings
    """
    # Try to load existing embeddings
    chunk_embeddings = load_embeddings(dataset_name, doc_id, embedding_model_name, len(chunks))
    
    if chunk_embeddings is not None:
        return chunk_embeddings
    
    # Compute new embeddings
    print(f"🔄 Computing embeddings for {len(chunks)} chunks...")
    chunk_texts = [
        format_row_for_embedding(
            chunk['column_headers'],
            chunk['row_header'],
            chunk['row_data']
        ) 
        for chunk in chunks
    ]
    
    chunk_embeddings = get_embeddings(chunk_texts)
    
    # Save for future use
    save_embeddings(chunks, chunk_embeddings, dataset_name, doc_id, embedding_model_name)
    
    return chunk_embeddings

def retrieve_top_k_rows(question: str, chunks: List[Dict], chunk_embeddings: np.ndarray,
                       top_k: int = 5) -> List[Dict]:
    """
    Retrieve top-k most relevant rows based on question.
    
    Args:
        question: The question text
        chunks: List of row chunks
        chunk_embeddings: Pre-computed chunk embeddings
        top_k: Number of rows to retrieve
    
    Returns:
        List of top-k chunks sorted by relevance
    """
    global embedding_model
    
    if embedding_model is None:
        raise ValueError("Embedding model not initialized")
    
    # Get question embedding
    question_embedding = embedding_model.encode(
        question,
        convert_to_tensor=True,
        show_progress_bar=False
    )
    
    device = question_embedding.device
    
    # Convert chunk embeddings to tensor on same device
    chunk_embeddings_tensor = torch.from_numpy(chunk_embeddings).to(device)
    
    # Compute cosine similarities
    similarities = util.cos_sim(question_embedding, chunk_embeddings_tensor)[0]
    
    # Convert to numpy for indexing
    similarities = similarities.cpu().numpy()
    
    # Get top-k indices
    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return top chunks with scores
    top_chunks = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk['similarity_score'] = float(similarities[idx])
        top_chunks.append(chunk)
    
    return top_chunks

# ============================================================================
# TABLE PARSING
# ============================================================================

def parse_table_content(content_str: str) -> Dict:
    """
    Parse table content from JSON string.
    
    Args:
        content_str: JSON string containing table data
    
    Returns:
        Dictionary with parsed table structure
    """
    try:
        table_data = json.loads(content_str)
        return {
            'column_headers': table_data.get('column_header', []),
            'row_headers': table_data.get('row_header', []),
            'data': table_data.get('data', []),
            'id': table_data.get('id', '')
        }
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing table JSON: {e}")
        return None

def create_row_chunks(table_data: Dict) -> List[Dict]:
    """
    Create row-level chunks from table data.
    Each row becomes one chunk.
    
    Args:
        table_data: Parsed table data
    
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    
    column_headers = table_data['column_headers']
    row_headers = table_data.get('row_headers', [])
    data_rows = table_data['data']
    table_id = table_data.get('id', '')
    
    # Create one chunk per row
    for row_idx, row_data in enumerate(data_rows):
        # Get row header if available
        row_header = row_headers[row_idx] if row_idx < len(row_headers) else []
        
        # Create chunk in table JSON format
        chunk_json = {
            'column_header': column_headers,
            'row_header': [row_header] if row_header else [],
            'data': [row_data],
            'id': table_id
        }
        
        chunk = {
            'row_idx': row_idx,
            'chunk_json_str': json.dumps(chunk_json),
            'column_headers': column_headers,
            'row_header': row_header,
            'row_data': row_data
        }
        
        chunks.append(chunk)
    
    print(f"  📦 Created {len(chunks)} row chunks")
    return chunks

def format_row_for_embedding(column_headers: List, row_header, row_data: List) -> str:
    """
    Format a single row into text for embedding.
    
    Args:
        column_headers: List of column headers
        row_header: Row header (can be list or string)
        row_data: List of data values for this row
    
    Returns:
        Formatted text string
    """
    # Format row header
    if isinstance(row_header, list):
        row_header_str = ' '.join(str(h).strip() for h in row_header)
    else:
        row_header_str = str(row_header) if row_header else ""
    
    # Format column headers
    formatted_headers = []
    for col_header in column_headers:
        if isinstance(col_header, list):
            merged = ' '.join(str(h).strip() for h in col_header)
            formatted_headers.append(merged.strip())
        else:
            formatted_headers.append(str(col_header).strip())
    
    # Create text
    text_parts = []
    
    if row_header_str:
        text_parts.append(f"Row: {row_header_str}")
    
    col_val_pairs = []
    for col_header, value in zip(formatted_headers, row_data):
        col_val_pairs.append(f"{col_header}: {value}")
    
    text_parts.append(f"Columns: {', '.join(col_val_pairs)}")
    
    return " | ".join(text_parts)

# ============================================================================
# ANSWER GENERATION
# ============================================================================

def format_table_for_llm(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into readable text format.
    Rows are renumbered sequentially (Row 1, Row 2, Row 3...).
    
    Args:
        chunks: List of retrieved chunks
    
    Returns:
        Formatted text representation
    """
    if not chunks:
        return ""
    
    # Get structure from first chunk
    first_chunk_data = json.loads(chunks[0]['chunk_json_str'])
    column_headers = first_chunk_data['column_header']
    
    # Format column headers
    formatted_column_headers = []
    for col_header in column_headers:
        if isinstance(col_header, list):
            merged_header = ' '.join(str(h).strip() for h in col_header)
            merged_header = ' '.join(merged_header.split())
            formatted_column_headers.append(merged_header)
        else:
            formatted_column_headers.append(str(col_header))
    
    # Format each retrieved row with sequential numbering
    formatted_rows = []
    for sequential_idx, chunk in enumerate(chunks, 1):
        chunk_data = json.loads(chunk['chunk_json_str'])
        row_headers = chunk_data.get('row_header', [])
        row_data_list = chunk_data.get('data', [])
        
        if not row_data_list:
            continue
            
        row_data = row_data_list[0]
        
        # Start with row header if available
        if row_headers and row_headers[0]:
            if isinstance(row_headers[0], list):
                row_header = ' '.join(str(h).strip() for h in row_headers[0])
                row_header = ' '.join(row_header.split())
            else:
                row_header = str(row_headers[0])
            row_text = f"Row: {row_header}\n"
        else:
            row_text = f"Row {sequential_idx}\n"
        
        # Add column values as bullet points
        for j, col_header in enumerate(formatted_column_headers):
            if j < len(row_data):
                row_text += f"- {col_header}: {row_data[j]}\n"
        
        formatted_rows.append(row_text)
    
    return "\n".join(formatted_rows)

def generate_answer(retrieved_chunks: List[Dict], question: str, count_tokenizer) -> tuple:
    """
    Generate an answer using Gemini API with RAG-retrieved context.
    
    Args:
        retrieved_chunks: List of retrieved chunks with metadata
        question: The question to answer
        count_tokenizer: Tokenizer for counting tokens
    
    Returns:
        Tuple of (answer, generation_time, token_count)
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Format chunks into readable text
        formatted_context = format_table_for_llm(retrieved_chunks)
        
        # Same prompt as attention method
        prompt = f"""Based on the following table, answer the question concisely.

IMPORTANT:
- The table content provided is complete.
- Column headers define the meaning of each value.
- Row headers define what each row represents.
- The entity and time period mentioned in the question are already resolved and do not need to be verified against the table.
- Your task is ONLY to extract the value from the table that answers the question.
- Do not introduce new information beyond the table values.

Table:
{formatted_context}

Question:
{question}

Answer:"""
        
        # Count tokens
        token_count = 0
        if count_tokenizer:
            token_count = len(count_tokenizer.encode(prompt))
        
        # Generate response
        start_time = time.time()
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=150,
            )
        )
        generation_time = time.time() - start_time
        
        # Extract answer
        answer = response.text.strip()
        
        return answer if answer else "None", generation_time, token_count
        
    except Exception as e:
        print(f"❌ Error during Gemini API call: {e}")
        return f"Error: {str(e)}", 0.0, 0

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_aitqa_rag(top_k: int = 5, embedding_model: str = 'UAE-Large-V1'):
    """
    Process aitqa dataset with RAG baseline (row-level retrieval).
    
    Args:
        top_k: Number of rows to retrieve
        embedding_model: Name of embedding model to use
    """
    print("="*60)
    print("aitqa BASELINE: RAG WITH ROW-LEVEL RETRIEVAL")
    print("="*60)
    print(f"Top-K: {top_k}")
    print(f"Embedding Model: {embedding_model}")
    print("="*60)
    
    # Initialize
    initialize_gemini()
    count_tokenizer = initialize_tokenizer()
    _, _, _ = initialize_embedding_model(embedding_model)
    
    # Setup paths
    data_file = Path("../qwen3_8b/data/datasets/aitqa_processed.json")
    output_dir = Path(f"pred_att/aitqa/top{top_k}/{embedding_model}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\n📖 Loading dataset from: {data_file}")
    if not data_file.exists():
        print(f"❌ ERROR: Dataset file not found: {data_file}")
        sys.exit(1)
    
    with open(data_file, 'r') as f:
        dataset = json.load(f)
    
    dataset_name = dataset.get('dataset_name', 'aitqa')
    documents = dataset['documents']
    
    print(f"✅ Loaded dataset: {dataset_name}")
    print(f"   Total documents: {len(documents)}\n")
    
    # Process each document
    total_docs = len(documents)
    total_questions = 0
    
    for doc_idx, doc in enumerate(documents, 1):
        doc_id = doc['document_id']
        questions = doc['questions']
        
        print(f"\n{'='*60}")
        print(f"Processing Document {doc_idx}/{total_docs} (ID: {doc_id})")
        print(f"Questions: {len(questions)}")
        print(f"{'='*60}")
        
        # Check if results already exist
        results_file = output_dir / f"results-{dataset_name}-{doc_id}.json"
        if results_file.exists():
            print(f"✅ Results file already exists: {results_file.name}")
            print(f"   Skipping document {doc_id}")
            total_questions += len(questions)
            continue
        
        # Parse table content
        content_str = doc.get('content', '')
        if not content_str:
            print(f"⚠️ Warning: Empty content for document {doc_id}")
            continue
        
        table_data = parse_table_content(content_str)
        if not table_data:
            print(f"❌ Failed to parse table for document {doc_id}")
            continue
        
        # Create row chunks
        print(f"\n📊 Creating row chunks...")
        chunks = create_row_chunks(table_data)
        
        # Show sample chunk (only for first document)
        if chunks and doc_idx == 1:
            print(f"\n   Sample chunk text for embedding:")
            sample_text = format_row_for_embedding(
                chunks[0]['column_headers'],
                chunks[0]['row_header'],
                chunks[0]['row_data']
            )
            print(f"   {sample_text[:200]}...")
        
        # Get or compute chunk embeddings (FIXED: only 4 parameters)
        chunk_embeddings = get_or_compute_chunk_embeddings(
            chunks, dataset_name, doc_id, embedding_model
        )
        
        # Process all questions for this document
        results = []
        
        for q_idx, question_data in enumerate(questions, 1):
            question_id = question_data['question_id']
            question = question_data['question']
            
            print(f"\n🔄 Question {q_idx}/{len(questions)}: {question[:60]}...")
            
            file_start_time = time.time()
            
            # Retrieve top-k relevant rows (FIXED: only 4 parameters)
            print(f"   🔍 Retrieving top-{top_k} rows...")
            retrieval_start = time.time()
            top_chunks = retrieve_top_k_rows(
                question, chunks, chunk_embeddings, top_k
            )
            retrieval_time = time.time() - retrieval_start
            
            # Show retrieved rows
            print(f"   Retrieved rows (by similarity):")
            for rank, chunk in enumerate(top_chunks, 1):
                print(f"     {rank}. Row {chunk['row_idx']} (score: {chunk['similarity_score']:.4f})")
            
            # Generate answer
            answer, generation_time, token_count = generate_answer(
                top_chunks, question, count_tokenizer
            )
            
            file_total_time = time.time() - file_start_time
            
            # Create result entry
            result = {
                "dataset": dataset_name,
                "document_id": doc_id,
                "question_id": question_id,
                "question": question,
                "result": answer,
                "rationale": f"Retrieved top-{top_k} rows using {embedding_model} embeddings and generated answer with Gemini.",
                "timing": {
                    "total": round(file_total_time, 4),
                    "retrieval": round(retrieval_time, 4),
                    "generate_answer": round(generation_time, 4)
                },
                "total_tokens": token_count,
                "rag_metadata": {
                    "retrieved_rows": [chunk['row_idx'] for chunk in top_chunks],
                    "similarity_scores": [chunk['similarity_score'] for chunk in top_chunks],
                    "embedding_model": embedding_model,
                    "top_k": top_k
                }
            }
            
            results.append(result)
            
            print(f"   ✅ Answer: {answer[:80]}...")
            print(f"   ⏱️  Time: {file_total_time:.3f}s")
            
            total_questions += 1
        
        # Save all results for this document
        output_data = {
            "dataset": dataset_name,
            "document_id": doc_id,
            "total_questions": len(results),
            "config": {
                "method": "rag_baseline",
                "top_k": top_k,
                "embedding_model": embedding_model
            },
            "results": results
        }
        
        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Saved {len(results)} results to: {results_file.name}")
    
    # Clean up embedding model (FIXED: no parameters)
    cleanup_embedding_model()
    
    # Final summary
    print("\n" + "="*60)
    print("✅ PROCESSING COMPLETE")
    print("="*60)
    print(f"Embedding Model: {embedding_model}")
    print(f"Total documents processed: {total_docs}")
    print(f"Total questions answered: {total_questions}")
    print(f"Results directory: {output_dir}")
    print(f"Embeddings directory: data/rag_embeddings/{dataset_name}/{embedding_model}/")
    print("="*60)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='aitqa RAG Baseline with Row-level Retrieval')
    parser.add_argument('--top_k', type=int, default=5, help='Number of rows to retrieve (default: 5)')
    parser.add_argument('--embedding_model', type=str, default='UAE-Large-V1', 
                       help=f'Embedding model to use. Options: {", ".join(AVAILABLE_EMBEDDING_MODELS + ["all"])}. Use "all" to run all models sequentially.')
    
    args = parser.parse_args()
    
    # Determine which models to run
    if args.embedding_model.lower() == 'all':
        models_to_run = AVAILABLE_EMBEDDING_MODELS
        print(f"\n🚀 Running ALL embedding models: {models_to_run}\n")
    else:
        if args.embedding_model not in AVAILABLE_EMBEDDING_MODELS:
            print(f"❌ Error: Unknown embedding model '{args.embedding_model}'")
            print(f"   Available models: {', '.join(AVAILABLE_EMBEDDING_MODELS)}")
            print(f"   Or use 'all' to run all models sequentially")
            sys.exit(1)
        models_to_run = [args.embedding_model]
    
    # Run for each model
    for model_idx, model_name in enumerate(models_to_run, 1):
        if len(models_to_run) > 1:
            print("\n" + "="*80)
            print(f"🔄 RUNNING MODEL {model_idx}/{len(models_to_run)}: {model_name}")
            print("="*80 + "\n")
        
        try:
            process_aitqa_rag(top_k=args.top_k, embedding_model=model_name)
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error processing model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            if len(models_to_run) > 1:
                print(f"\n⚠️ Continuing with next model...")
                continue
            else:
                sys.exit(1)
        
        if len(models_to_run) > 1 and model_idx < len(models_to_run):
            print(f"\n⏭️  Moving to next model...\n")
    
    # Final summary for all models
    if len(models_to_run) > 1:
        print("\n" + "="*80)
        print("✅ ALL MODELS PROCESSING COMPLETE")
        print("="*80)
        print(f"Models run: {', '.join(models_to_run)}")
        print(f"Top-K: {args.top_k}")
        print(f"Results base directory: pred_att/aitqa/top{args.top_k}/")
        for model_name in models_to_run:
            print(f"  - {model_name}/")
        print("="*80)

if __name__ == "__main__":
    main()