#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aitqa Baseline: Full Context
------------------------------------
Passes complete article context with questions directly to LLM for answer generation.

Usage:
    python baselines/aitqa_baseline.py
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List
import google.generativeai as genai
from transformers import AutoTokenizer

# ============================================================================
# INITIALIZATION
# ============================================================================

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

# ============================================================================
# ANSWER GENERATION
# ============================================================================

def generate_answer(context_text: str, question: str) -> tuple:
    """
    Generate an answer from the full text context using Gemini API.
    
    Args:
        context_text: Full article text
        question: The question to answer
    
    Returns:
        Tuple of (answer, generation_time)
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        # Updated prompt to refer to 'Context' instead of 'Table'
        prompt = f"""Based on the following table, answer the question concisely.

IMPORTANT:
- The table content provided is complete.
- Column headers define the meaning of each value.
- Row headers define what each row represents.
- The entity and time period mentioned in the question are already resolved and do not need to be verified against the table.
- Your task is ONLY to extract the value from the table that answers the question.
- Do not introduce new information beyond the table values.

Table:
{context_text}

Question:
{question}

Answer:"""
        

        print("Prompt: \n", prompt)

        # Generate response
        start_time = time.time()
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,  # Deterministic output
                max_output_tokens=150,
            )
        )
        generation_time = time.time() - start_time
        
        # Extract answer
        answer = response.text.strip()

        print(answer)
        
        # Debug print
        print(f"   Prompt size (approx chars): {len(prompt)}")
        
        return answer if answer else "None", generation_time
        
    except Exception as e:
        print(f"❌ Error during Gemini API call: {e}")
        return f"Error: {str(e)}", 0.0

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_aitqa_baseline():
    """
    Process aitqa dataset with full context baseline.
    """
    print("="*60)
    print("aitqa BASELINE: FULL CONTEXT")
    print("="*60)
    
    # Initialize
    initialize_gemini()
    tokenizer = initialize_tokenizer()
    
    # Setup paths
    # UPDATED: Path to the aitqa processed file
    data_file = Path("../llama3.2_1b/data/datasets/aitqa_processed.json")
    output_dir = Path("pred_att/aitqa_full_table")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\n📖 Loading dataset from: {data_file}")
    if not data_file.exists():
        print(f"❌ ERROR: Dataset file not found: {data_file}")
        print("   Please run data_reader.py first to generate the processed dataset.")
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
        
        # Get content directly
        # UPDATED: Use content directly as string (it is not a table)
        context_text = doc.get('content', '')
        
        if not context_text:
            print(f"⚠️ Warning: Empty content for document {doc_id}")
            context_text = ""

        # Count tokens if tokenizer is available
        total_tokens = 0
        if tokenizer:
            full_context_preview = f"Context:\n{context_text}\n\nQuestion: [PLACEHOLDER]"
            total_tokens = len(tokenizer.encode(full_context_preview))
            print(f"\n📊 Context loaded ({total_tokens} tokens)")
        else:
            print(f"\n📊 Context loaded ({len(context_text)} chars)")
            
        print(f"Preview:\n{context_text[:200]}...\n")
        
        # Process all questions for this document
        results = []
        
        for q_idx, question_data in enumerate(questions, 1):
            question_id = question_data['question_id']
            question = question_data['question']
            
            print(f"\n🔄 Question {q_idx}/{len(questions)}: {question[:60]}...")
            
            file_start_time = time.time()
            
            # Generate answer
            answer, generation_time = generate_answer(context_text, question)
            
            file_total_time = time.time() - file_start_time
            
            # Create result entry
            result = {
                "dataset": dataset_name,
                "document_id": doc_id,
                "question_id": question_id,
                "question": question,
                "result": answer,
                "rationale": f"Generated answer using Gemini API from full text context of {len(context_text.split())} words.",
                "timing": {
                    "total": round(file_total_time, 4),
                    "generate_answer": round(generation_time, 4)
                },
                "total_tokens": total_tokens
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
            "results": results
        }
        
        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Saved {len(results)} results to: {results_file.name}")
    
    # Final summary
    print("\n" + "="*60)
    print("✅ PROCESSING COMPLETE")
    print("="*60)
    print(f"Total documents processed: {total_docs}")
    print(f"Total questions answered: {total_questions}")
    print(f"Results directory: {output_dir}")
    print("="*60)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    try:
        process_aitqa_baseline()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()