#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit Window for Attention-Based Context Selection (Table Version)
-----------------------------------------------------------------
Uses pre-computed attention scores to select table rows and generate answers.

This version is specifically designed for table datasets (AITQA) where the
system selects the top k rows based on average attention scores.

Key behavior:
- Computes average attention score for each table row (including row header + data)
- Ranks rows by attention
- Returns top k rows (highest attention)
- If number of rows <= k, returns all rows
- Row headers (when present) are included in attention computation

Usage:
    # Generate results using raw attention, selecting top 3 rows
    python unit_window_aitqa.py --dataset aitqa --use-raw --top-k 3 --model llama3.2_1b
    
    # Generate results using farest differential, selecting top 5 rows with Qwen3-8B
    python unit_window_aitqa.py --dataset aitqa --use-farest --top-k 5 --model qwen3_8b
    
    # Use all attention types with top 2 rows with Qwen3-14B
    python unit_window_aitqa.py --dataset aitqa --use-raw --use-farest --use-baseline --top-k 2 --model qwen3_14b
    
    # Test mode: only process document at index 1
    python unit_window_aitqa.py --dataset aitqa --use-raw --top-k 3 --test-doc-idx 1 --model llama3.2_1b
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
# TABLE ROW-BASED CONTEXT SELECTION
# ============================================================================

def parse_table_and_map_tokens(table_content: str, tokens: List[str]) -> Dict:
    """
    Parse table JSON and create mapping from tokens to table rows.
    
    Strategy:
    1. Parse JSON to extract row_header and data arrays
    2. Find where row_header array starts and ends in token sequence
    3. Find where data array starts and ends in token sequence
    4. Parse each array to identify individual row headers and data rows
    5. Match row_header[i] with data[i] by index
    
    This approach maintains the correct correspondence between headers and data.
    
    Args:
        table_content: JSON string containing table data
        tokens: List of tokens
        
    Returns:
        Dictionary with row information and token mappings
    """
    global tokenizer
    if tokenizer is None:
        initialize_tokenizer()
    
    try:
        table_data = json.loads(table_content)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing table JSON: {e}")
        return None
    
    column_headers = table_data.get('column_header', [])
    row_headers = table_data.get('row_header', [])
    data_rows = table_data.get('data', [])
    
    if not data_rows:
        print("⚠️ No data rows found in table")
        return None
    
    num_rows = len(data_rows)
    num_tokens = len(tokens)
    
    print(f"    - Table structure: {num_rows} data rows, {num_tokens} tokens")
    print(f"    - Column headers: {len(column_headers)} columns")
    print(f"    - Row headers: {len(row_headers)} row headers")
    print(f"    - First data row: {data_rows[0]}")
    
    # Step 1: Decode tokens to get full text
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    full_text = tokenizer.decode(token_ids)
    
    print(f"    - Full text length: {len(full_text)} characters")
    
    # Step 2: Build character-to-token mapping
    char_to_token = {}
    current_char_pos = 0
    
    for token_idx, token in enumerate(tokens):
        token_id = tokenizer.convert_tokens_to_ids([token])
        token_text = tokenizer.decode(token_id)
        token_length = len(token_text)
        
        for i in range(token_length):
            char_pos = current_char_pos + i
            if char_pos < len(full_text):
                char_to_token[char_pos] = token_idx
        
        current_char_pos += token_length
    
    print(f"    - Built character-to-token mapping for {len(char_to_token)} characters")
    
    # Step 3: Find row_header array in full text (if exists)
    header_mappings = []
    if row_headers:
        # Create JSON string for the entire row_header array
        row_header_array_str = json.dumps(row_headers, separators=(',', ':'))
        row_header_array_start = full_text.find(row_header_array_str)
        
        if row_header_array_start != -1:
            print(f"    - Found row_header array at char position {row_header_array_start}")
            
            # Now find each individual row header within this array
            current_search_pos = row_header_array_start
            for row_idx, row_header in enumerate(row_headers):
                row_header_str = json.dumps(row_header, separators=(',', ':'))
                char_start = full_text.find(row_header_str, current_search_pos)
                
                if char_start != -1:
                    char_end = char_start + len(row_header_str)
                    
                    # Map to tokens
                    start_token = char_to_token.get(char_start, -1)
                    if start_token == -1:
                        for pos in range(char_start, -1, -1):
                            if pos in char_to_token:
                                start_token = char_to_token[pos]
                                break
                    
                    end_token = char_to_token.get(char_end - 1, -1)
                    if end_token == -1:
                        for pos in range(char_end - 1, len(full_text)):
                            if pos in char_to_token:
                                end_token = char_to_token[pos] + 1
                                break
                    else:
                        end_token = end_token + 1
                    
                    header_mappings.append({
                        'row_idx': row_idx,
                        'header': row_header,
                        'char_start': char_start,
                        'char_end': char_end,
                        'start_token': start_token,
                        'end_token': end_token
                    })
                    
                    print(f"    - Row header {row_idx}: chars [{char_start}:{char_end}], tokens [{start_token}:{end_token}]")
                    current_search_pos = char_end
                else:
                    print(f"    ⚠️ Row header {row_idx}: Could not locate")
                    header_mappings.append({
                        'row_idx': row_idx,
                        'header': row_header,
                        'start_token': -1,
                        'end_token': -1
                    })
        else:
            print(f"    ⚠️ Could not locate row_header array in text")
    
    # Step 4: Find data array in full text
    data_mappings = []
    data_array_str = json.dumps(data_rows, separators=(',', ':'))
    data_array_start = full_text.find(data_array_str)
    
    if data_array_start != -1:
        print(f"    - Found data array at char position {data_array_start}")
        
        # Now find each individual data row within this array
        current_search_pos = data_array_start
        for row_idx, row_data in enumerate(data_rows):
            row_data_str = json.dumps(row_data, separators=(',', ':'))
            char_start = full_text.find(row_data_str, current_search_pos)
            
            if char_start != -1:
                char_end = char_start + len(row_data_str)
                
                # Map to tokens
                start_token = char_to_token.get(char_start, -1)
                if start_token == -1:
                    for pos in range(char_start, -1, -1):
                        if pos in char_to_token:
                            start_token = char_to_token[pos]
                            break
                
                end_token = char_to_token.get(char_end - 1, -1)
                if end_token == -1:
                    for pos in range(char_end - 1, len(full_text)):
                        if pos in char_to_token:
                            end_token = char_to_token[pos] + 1
                            break
                else:
                    end_token = end_token + 1
                
                data_mappings.append({
                    'row_idx': row_idx,
                    'data': row_data,
                    'char_start': char_start,
                    'char_end': char_end,
                    'start_token': start_token,
                    'end_token': end_token
                })
                
                print(f"    - Row data {row_idx}: chars [{char_start}:{char_end}], tokens [{start_token}:{end_token}]")
                current_search_pos = char_end
            else:
                print(f"    ⚠️ Row data {row_idx}: Could not locate")
                data_mappings.append({
                    'row_idx': row_idx,
                    'data': row_data,
                    'start_token': -1,
                    'end_token': -1
                })
    else:
        print(f"    ⚠️ Could not locate data array in text")
        return None
    
    # Step 5: Combine header and data mappings by index
    row_mappings = []
    for row_idx in range(num_rows):
        # Get data mapping (required)
        if row_idx >= len(data_mappings):
            print(f"    ⚠️ Missing data mapping for row {row_idx}")
            continue
        
        data_map = data_mappings[row_idx]
        
        # Get header mapping (optional)
        header_map = None
        if row_idx < len(header_mappings):
            header_map = header_mappings[row_idx]
        
        # Combine
        row_mapping = {
            'row_idx': row_idx,
            'row_data': data_map['data'],
            'row_header': header_map['header'] if header_map else [],
            'start_token': data_map['start_token'],
            'end_token': data_map['end_token'],
            'num_tokens': data_map['end_token'] - data_map['start_token'] if data_map['start_token'] != -1 else 0
        }
        
        # Add header token info if available
        if header_map and header_map['start_token'] != -1:
            row_mapping['start_token_header'] = header_map['start_token']
            row_mapping['end_token_header'] = header_map['end_token']
            row_mapping['num_tokens_header'] = header_map['end_token'] - header_map['start_token']
        else:
            row_mapping['start_token_header'] = -1
            row_mapping['end_token_header'] = -1
            row_mapping['num_tokens_header'] = 0
        
        row_mappings.append(row_mapping)
        
        # VERIFICATION
        print(f"\n    === VERIFICATION: Row {row_idx} Mapping ===")
        print(f"    Row data: {row_mapping['row_data']}")
        print(f"    Row header: {row_mapping['row_header']}")
        print(f"    Data tokens: [{row_mapping['start_token']}:{row_mapping['end_token']}] ({row_mapping['num_tokens']} tokens)")
        if row_mapping['start_token_header'] != -1:
            print(f"    Header tokens: [{row_mapping['start_token_header']}:{row_mapping['end_token_header']}] ({row_mapping['num_tokens_header']} tokens)")
            
            # Decode and verify
            header_tokens = tokens[row_mapping['start_token_header']:row_mapping['end_token_header']]
            header_token_ids = tokenizer.convert_tokens_to_ids(header_tokens)
            decoded_header = tokenizer.decode(header_token_ids)
            print(f"    Decoded header: '{decoded_header}'")
        
        data_tokens = tokens[row_mapping['start_token']:row_mapping['end_token']]
        data_token_ids = tokenizer.convert_tokens_to_ids(data_tokens)
        decoded_data = tokenizer.decode(data_token_ids)
        print(f"    Decoded data: '{decoded_data}'")
        print(f"    {'='*70}\n")
    
    # Check if we successfully mapped all rows
    if any(r['start_token'] == -1 for r in row_mappings):
        print("    ⚠️ Some rows could not be mapped")
        return None
    
    return {
        'column_headers': column_headers,
        'row_headers': row_headers,
        'data_rows': data_rows,
        'row_mappings': row_mappings,
        'num_rows': num_rows,
        'num_tokens': num_tokens
    }

def use_uniform_distribution(data_rows: List, num_tokens: int, 
                            column_headers: List = None, row_headers: List = None) -> Dict:
    """
    Fallback: uniformly distribute tokens across rows.
    This is only used when token mapping completely fails.
    
    Args:
        data_rows: List of data rows
        num_tokens: Total number of tokens
        column_headers: Column headers (optional)
        row_headers: Row headers (optional)
        
    Returns:
        Dictionary with row mappings using uniform distribution
    """
    print("    ⚠️ WARNING: Using uniform distribution fallback - results may be inaccurate")
    
    num_rows = len(data_rows)
    tokens_per_row = num_tokens // num_rows
    
    print(f"    - Uniform distribution: ~{tokens_per_row} tokens per row")
    
    row_mappings = []
    for row_idx in range(num_rows):
        start_idx = row_idx * tokens_per_row
        end_idx = (row_idx + 1) * tokens_per_row if row_idx < num_rows - 1 else num_tokens
        
        row_mappings.append({
            'row_idx': row_idx,
            'row_data': data_rows[row_idx],
            'row_header': row_headers[row_idx] if (row_headers and row_idx < len(row_headers)) else [],
            'start_token': start_idx,
            'end_token': end_idx,
            'start_token_header': -1,  # Not available in uniform distribution
            'end_token_header': -1,    # Not available in uniform distribution
            'num_tokens': end_idx - start_idx,
            'num_tokens_header': 0
        })
    
    return {
        'column_headers': column_headers or [],
        'row_headers': row_headers or [],
        'data_rows': data_rows,
        'row_mappings': row_mappings,
        'num_rows': num_rows,
        'num_tokens': num_tokens
    }

def apply_budget_to_table_attention(attention_scores: np.ndarray, tokens: List[str], 
                                   table_content: str, top_k: int = 1) -> Tuple[List[List[str]], List[str], Dict]:
    """
    Selects the top k rows from a table based on attention scores.
    
    Strategy:
    1. Parse table JSON to identify rows and row headers
    2. Map tokens to rows and their headers
    3. Compute average attention score per row (including both header and data tokens)
    4. Select the top k rows with highest average attention
    5. If num_rows <= k, select all rows
    
    Args:
        attention_scores: Attention scores for each token
        tokens: List of tokens
        table_content: JSON string containing table data
        top_k: Number of top rows to select (default: 1)
    
    Returns:
        Tuple of (list_of_token_lists, list_of_decoded_texts, table_info_dict)
    """
    global tokenizer
    if tokenizer is None:
        initialize_tokenizer()
    
    if attention_scores.size == 0 or not tokens:
        return [[]], [""], {}
    
    # Parse table and get row mappings
    table_info = parse_table_and_map_tokens(table_content, tokens)
    
    if table_info is None:
        print("    ⚠️ Could not parse table, returning empty result")
        return [[]], [""], {}
    
    row_mappings = table_info['row_mappings']
    num_rows = table_info['num_rows']
    column_headers = table_info['column_headers']
    row_headers = table_info['row_headers']
    
    # Compute average attention for each row
    row_attentions = []
    for row_info in row_mappings:
        start_idx_data = row_info['start_token']
        end_idx_data = row_info['end_token']
        start_idx_header = row_info.get('start_token_header', -1)
        end_idx_header = row_info.get('end_token_header', -1)
        
        # Collect attention scores from both row header and data
        attention_scores_list = []
        
        # Add data row attention scores
        data_attention_scores = attention_scores[start_idx_data:end_idx_data]
        if len(data_attention_scores) > 0:
            attention_scores_list.extend(data_attention_scores)
        
        # Add row header attention scores (if exists)
        if start_idx_header != -1 and end_idx_header != -1:
            header_attention_scores = attention_scores[start_idx_header:end_idx_header]
            if len(header_attention_scores) > 0:
                attention_scores_list.extend(header_attention_scores)
        
        # Compute average attention across both header and data
        avg_attention = np.mean(attention_scores_list) if len(attention_scores_list) > 0 else 0.0
        
        row_attentions.append({
            'row_idx': row_info['row_idx'],
            'row_data': row_info['row_data'],
            'row_header': row_info.get('row_header', []),
            'start_token': start_idx_data,
            'end_token': end_idx_data,
            'start_token_header': start_idx_header,
            'end_token_header': end_idx_header,
            'num_tokens': end_idx_data - start_idx_data,
            'num_tokens_header': (end_idx_header - start_idx_header) if start_idx_header != -1 else 0,
            'avg_attention': avg_attention
        })
        
        header_info = ""
        if start_idx_header != -1 and end_idx_header != -1:
            header_info = f", header tokens [{start_idx_header}:{end_idx_header}]"
        
        print(f"    - Row {row_info['row_idx']}: data tokens [{start_idx_data}:{end_idx_data}]{header_info}, "
              f"avg_attention={avg_attention:.6f} (computed across {len(attention_scores_list)} tokens)")
    
    # Sort by attention score (highest first)
    row_attentions.sort(key=lambda x: x['avg_attention'], reverse=True)
    
    # Determine how many rows to select
    rows_to_select = min(top_k, num_rows)
    
    if num_rows <= top_k:
        print(f"    - Number of rows ({num_rows}) <= k ({top_k}), selecting all {num_rows} rows")
    else:
        print(f"    - Selecting top {rows_to_select} rows out of {num_rows} rows")
    
    # Select top k rows (or all rows if num_rows <= k)
    selected_rows = row_attentions[:rows_to_select]
    
    # Collect all selected rows information
    selected_row_indices = []
    selected_row_headers = []
    selected_row_data = []
    all_selected_tokens = []
    all_selected_texts = []
    total_tokens_data = 0
    total_tokens_header = 0
    
    for i, row in enumerate(selected_rows):
        row_idx = row['row_idx']
        selected_row_indices.append(row_idx)
        selected_row_data.append(row['row_data'])
        
        # Get row header if it exists
        if row_headers and len(row_headers) > row_idx:
            selected_row_headers.append(row_headers[row_idx])
        else:
            selected_row_headers.append([])
        
        # Extract tokens and decode
        row_tokens = tokens[row['start_token']:row['end_token']]
        token_ids = tokenizer.convert_tokens_to_ids(row_tokens)
        row_text = tokenizer.decode(token_ids)
        
        all_selected_tokens.append(row_tokens)
        all_selected_texts.append(row_text)
        
        total_tokens_data += len(row_tokens)
        total_tokens_header += row['num_tokens_header']
        
        print(f"    - Selected row {row_idx} (rank {i+1}): {len(row_tokens)} tokens, "
              f"attention={row['avg_attention']:.6f}")
    
    total_selected_tokens = total_tokens_data + total_tokens_header
    print(f"    - Total selected: {total_selected_tokens} tokens "
          f"({total_selected_tokens/len(tokens)*100:.1f}%)")
    
    # VERIFICATION: Print selected rows details
    print(f"\n    === VERIFICATION: Selected Rows ===")
    for i, row_idx in enumerate(selected_row_indices):
        row = selected_rows[i]
        print(f"    Rank {i+1}: Row {row_idx}")
        print(f"      Row data: {row['row_data']}")
        if selected_row_headers[i]:
            print(f"      Row header: {selected_row_headers[i]}")
        else:
            print(f"      Row header: (none)")
        print(f"      Token range (data): [{row['start_token']}:{row['end_token']}]")
        print(f"      Number of data tokens: {row['num_tokens']}")
        if row.get('start_token_header', -1) != -1:
            print(f"      Token range (header): [{row['start_token_header']}:{row['end_token_header']}]")
            print(f"      Number of header tokens: {row['num_tokens_header']}")
        print(f"      Average attention: {row['avg_attention']:.6f}")
        print(f"      Decoded text: '{all_selected_texts[i][:100]}{'...' if len(all_selected_texts[i]) > 100 else ''}'")
        print()
    print(f"    {'='*70}\n")
    
    # Return all selected rows
    return all_selected_tokens, all_selected_texts, {
        'column_headers': column_headers,
        'row_headers': row_headers,
        'selected_row_indices': selected_row_indices,
        'selected_row_headers': selected_row_headers,
        'selected_row_data': selected_row_data,
        'num_tokens_header': total_tokens_header,
        'num_tokens_data': total_tokens_data,
        'num_rows_selected': len(selected_rows),
        'total_rows': num_rows
    }

# ============================================================================
# ANSWER GENERATION
# ============================================================================

def generate_answer_with_rationale(context: str, question: str, 
                                   table_info: Dict = None,
                                   doc_id: str = None) -> Tuple[str, str]:
    """
    Generate an answer from the provided (reduced) context using Gemini API.
    
    Args:
        context: The selected context text (can be a list or string)
        question: The question to answer
        table_info: Dictionary with column_headers, selected_row_headers, and selected_row_data
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
        
        # Format the context in readable text format with column headers and selected rows
        if table_info and 'column_headers' in table_info and 'selected_row_data' in table_info:
            column_headers = table_info['column_headers']
            selected_row_data = table_info['selected_row_data']
            selected_row_headers = table_info.get('selected_row_headers', [])
            
            # Format column headers: merge list elements into readable strings
            formatted_column_headers = []
            for col_header in column_headers:
                if isinstance(col_header, list):
                    # Join list elements with space, strip trailing commas/spaces
                    merged_header = ' '.join(str(h).strip() for h in col_header)
                    # Clean up any double spaces
                    merged_header = ' '.join(merged_header.split())
                    formatted_column_headers.append(merged_header)
                else:
                    formatted_column_headers.append(str(col_header))
            
            # Format as text: "Row: <header>" followed by "- <column>: <value>"
            formatted_rows = []
            for i, row_data in enumerate(selected_row_data):
                # Start with row header if available
                if selected_row_headers and i < len(selected_row_headers) and selected_row_headers[i]:
                    # Row header might be a list or a single value
                    if isinstance(selected_row_headers[i], list):
                        row_header = ' '.join(str(h).strip() for h in selected_row_headers[i])
                        row_header = ' '.join(row_header.split())  # Clean up double spaces
                    else:
                        row_header = str(selected_row_headers[i])
                    row_text = f"Row: {row_header}\n"
                else:
                    row_text = f"Row {i+1}\n"
                
                # Add column values as bullet points
                for j, col_header in enumerate(formatted_column_headers):
                    if j < len(row_data):
                        row_text += f"- {col_header}: {row_data[j]}\n"
                
                formatted_rows.append(row_text)
            
            formatted_context = "\n".join(formatted_rows)
        else:
            formatted_context = context_text
        
        model = genai.GenerativeModel('gemini-2.5-flash-lite')

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

Answer:
"""
        print(prompt)

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
        num_rows = table_info.get('num_rows_selected', 1) if table_info else 1
        rationale = f"Generated answer using Gemini API from {num_rows} selected table row(s) with column headers and row headers."

        return answer if answer else "None", rationale
        
    except Exception as e:
        print(f"❌ Error during Gemini API call: {e}")
        return "None", f"Error occurred: {str(e)}"

# ============================================================================
# RESULT GENERATION PIPELINES
# ============================================================================

def generate_results_for_attention_type(dataset_name: str, dataset: Dict, 
                                       attention_type: str,
                                       token_dir: str, attention_dir: str, 
                                       output_dir: str,
                                       top_k: int = 1,
                                       test_doc_idx: int = None):
    """
    Generate evaluation results for a specific attention type.
    
    Args:
        dataset_name: Name of the dataset
        dataset: Dataset dictionary
        attention_type: One of 'raw', 'farest', 'baseline'
        token_dir: Directory containing token files
        attention_dir: Directory containing attention files
        output_dir: Directory to save results
        top_k: Number of top rows to select
        test_doc_idx: If provided, only process document at this index (testing mode)
    """
    print(f"\n{'='*60}")
    print(f"GENERATING RESULTS: {attention_type.upper()} attention (top-{top_k})")
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
    
    # Filter documents if in test mode
    documents_to_process = dataset['documents']
    if test_doc_idx is not None:
        if test_doc_idx < 0 or test_doc_idx >= len(documents_to_process):
            print(f"❌ ERROR: Invalid document index {test_doc_idx}")
            print(f"   Valid range: 0 to {len(documents_to_process) - 1}")
            return
        
        documents_to_process = [documents_to_process[test_doc_idx]]
        print(f"\n🧪 TEST MODE: Processing only document at index {test_doc_idx}")
        print(f"   Document ID: {documents_to_process[0]['document_id']}")
    
    # Process each document (table)
    for doc in documents_to_process:
        doc_id = doc['document_id']
        table_content = doc['content']  # JSON string of table
        
        # Load tokens once per document
        token_file = Path(token_dir) / f"{dataset_name}_{doc_id}.json"
        if not token_file.exists():
            print(f"⚠️ Missing token file for document {doc_id}")
            total_skipped += len(doc['questions'])
            continue
        
        with open(token_file, 'r') as f:
            tokens = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"📊 Table {doc_id}")
        print(f"{'='*60}")
        
        # VERIFICATION: Show table structure
        try:
            table_data = json.loads(table_content)
            print(f"\n    === TABLE STRUCTURE ===")
            print(f"    Document ID: {doc_id}")
            print(f"    Column headers: {len(table_data.get('column_header', []))} columns")
            if table_data.get('column_header'):
                print(f"    Headers: {table_data['column_header'][:3]}{'...' if len(table_data['column_header']) > 3 else ''}")
            print(f"    Row headers: {len(table_data.get('row_header', []))} row headers")
            if table_data.get('row_header'):
                print(f"    Sample row headers: {table_data['row_header'][:2]}{'...' if len(table_data['row_header']) > 2 else ''}")
            print(f"    Data rows: {len(table_data.get('data', []))} rows")
            if table_data.get('data'):
                print(f"    Sample row 0: {table_data['data'][0]}")
                if len(table_data['data']) > 1:
                    print(f"    Sample row 1: {table_data['data'][1]}")
            print(f"    {'='*60}\n")
        except json.JSONDecodeError:
            print(f"    ⚠️ Warning: Could not parse table JSON")
        
        # Process each question
        for question_data in doc['questions']:
            question_id = question_data['question_id']
            question = question_data['question']
            
            print(f"\n🔄 Processing: Q: {question[:60]}...")
            
            # Create results filename with truncated question text and top_k indicator
            safe_question = question.replace("?", "").replace("/", "_").replace(" ", "_")
            max_question_length = 80
            if len(safe_question) > max_question_length:
                safe_question = safe_question[:max_question_length]
            results_file = dataset_output_dir / f"results-{attention_type}-top{top_k}-{dataset_name}_{doc_id}_{safe_question}.json"
            
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
                
                # Apply selection to get top k rows
                budget_start = time.time()
                selected_token_lists, selected_texts, table_info = apply_budget_to_table_attention(
                    attention_scores, tokens, table_content, top_k=top_k
                )
                total_tokens_extracted = sum(len(token_list) for token_list in selected_token_lists)
                budget_time = time.time() - budget_start
                
                # Generate answer
                generate_start = time.time()
                answer, rationale = generate_answer_with_rationale(
                    selected_texts, 
                    question,
                    table_info=table_info,
                    doc_id=doc_id
                )
                print(f"    - Generated answer: {answer}")
                generate_time = time.time() - generate_start
                
                file_total_time = time.time() - file_start_time
                
                # Create timing dictionary
                timing_dict = {
                    "total": file_total_time,
                    "load_attention": load_time,
                    "apply_selection": budget_time,
                    "generate_answer": generate_time
                }
                
                # Save results
                result = {
                    "dataset": dataset_name,
                    "document_id": doc_id,
                    "question_id": question_id,
                    "question": question,
                    "attention_type": attention_type,
                    "top_k": top_k,
                    "use_kv_cache": False,
                    "result": answer,
                    "rationale": rationale,
                    "timing": timing_dict,
                    "tokens_extracted_data": table_info.get('num_tokens_data', 0),
                    "tokens_extracted_header": table_info.get('num_tokens_header', 0),
                    "tokens_extracted": total_tokens_extracted,
                    "total_tokens": len(tokens),
                    "num_rows_selected": table_info.get('num_rows_selected', 0),
                    "total_rows": table_info.get('total_rows', 0),
                    "selected_row_indices": table_info.get('selected_row_indices', []),
                    "selected_row_headers": table_info.get('selected_row_headers', []),
                    "selected_row_data": table_info.get('selected_row_data', []),
                    "column_headers": table_info.get('column_headers', [])
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
    
    print(f"\n📊 {attention_type.upper()} (top-{top_k}) SUMMARY:")
    print(f"   ✅ Processed: {total_processed}")
    print(f"   ⚠️  Skipped: {total_skipped}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unit window for table attention-based row selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate results using raw attention, selecting top 3 rows with Llama
  python unit_window_aitqa.py --dataset aitqa --use-raw --top-k 3 --model llama3.2_1b
  
  # Generate results using farest differential, selecting top 5 rows with Qwen3-8B
  python unit_window_aitqa.py --dataset aitqa --use-farest --top-k 5 --model qwen3_8b
  
  # Generate results using all attention types with top 2 rows with Qwen3-14B
  python unit_window_aitqa.py --dataset aitqa --use-raw --use-farest --use-baseline --top-k 2 --model qwen3_14b
  
  # Test mode: only process document at index 1 (second document)
  python unit_window_aitqa.py --dataset aitqa --use-raw --top-k 3 --test-doc-idx 1 --model llama3.2_1b

Note: If the number of rows in a table is <= k, all rows will be selected.
        """
    )
    
    parser.add_argument('--dataset', type=str, default='aitqa',
                       help='Dataset name (default: aitqa)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing processed data')
    parser.add_argument('--output-dir', type=str, default='pred_att',
                       help='Directory to save evaluation results')
    
    # ✅ UPDATED: Model selection with full names (must match preprocessing)
    parser.add_argument('--model', type=str,
                       default="llama3.2_1b",
                       choices=["llama3.2_1b", "qwen3_8b", "qwen3_14b"],
                       help='Model to use for tokenization (MUST match preprocessing!)')
    
    # Top-k selection
    parser.add_argument('--top-k', type=int, default=1,
                       help='Number of top rows to select (default: 1). If num_rows <= k, all rows are selected.')
    
    # Attention type selection
    parser.add_argument('--use-raw', action='store_true',
                       help='Use raw attention scores')
    parser.add_argument('--use-farest', action='store_true',
                       help='Use farest question differential attention')
    parser.add_argument('--use-baseline', action='store_true',
                       help='Use baseline differential attention')
    
    # Testing mode
    parser.add_argument('--test-doc-idx', type=int, default=None,
                       help='Test mode: only process document at this index (0-based)')
    
    args = parser.parse_args()
    
    # ✅ Resolve model name from mapping
    model_name = MODEL_MAPPING.get(args.model, args.model)
    
    print("="*60)
    print("UNIT WINDOW FOR TABLE - TOP K ROW SELECTION")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model} ({model_name})")
    print(f"Selection: Top {args.top_k} row(s) (or all rows if num_rows <= {args.top_k})")
    if args.test_doc_idx is not None:
        print(f"🧪 TEST MODE: Only processing document at index {args.test_doc_idx}")
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
    
    # Validate top_k
    if args.top_k < 1:
        print("\n❌ ERROR: --top-k must be at least 1")
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
    print(f"   Dataset file: {dataset_file}")
    
    # Show first document info
    if dataset.get('documents'):
        first_doc = dataset['documents'][0]
        print(f"\n   First document preview:")
        print(f"     - Document ID: {first_doc.get('document_id', 'N/A')}")
        print(f"     - Questions: {len(first_doc.get('questions', []))}")
        print(f"     - Content length: {len(first_doc.get('content', ''))} characters")
    
    # ✅ Initialize tokenizer with the correct model
    print(f"\n📥 Initializing tokenizer for {model_name}...")
    initialize_tokenizer(model_name)
    
    # Generate results for each selected attention type
    if args.use_raw:
        generate_results_for_attention_type(
            args.dataset, dataset, 'raw',
            token_dir, attention_dir, base_output_dir,
            top_k=args.top_k,
            test_doc_idx=args.test_doc_idx
        )
    
    if args.use_farest:
        generate_results_for_attention_type(
            args.dataset, dataset, 'farest',
            token_dir, attention_dir, base_output_dir,
            top_k=args.top_k,
            test_doc_idx=args.test_doc_idx
        )
    
    if args.use_baseline:
        generate_results_for_attention_type(
            args.dataset, dataset, 'baseline',
            token_dir, attention_dir, base_output_dir,
            top_k=args.top_k,
            test_doc_idx=args.test_doc_idx
        )
    
    print("\n" + "="*60)
    print(f"✅ COMPLETE: All results generated for '{args.dataset}' (top-{args.top_k})")
    print("="*60)
    print(f"📁 Results saved in: {os.path.join(base_output_dir, args.dataset)}")

if __name__ == "__main__":
    main()