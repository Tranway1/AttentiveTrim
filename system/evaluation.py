#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation Script for Attention-Based Context Reduction
Loads individual result files (one per document) and aggregates them by query.
Uses imported evaluation functions from src module.
Matches ground truth using doc_id from processed dataset.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys
import warnings
import re
from collections import defaultdict
import os

warnings.filterwarnings('ignore')
try:
    os.getcwd()
except (FileNotFoundError, OSError):
    os.chdir('/tmp')

# Import for embedding evaluation
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    print("✅ Embedding libraries initialized")
except ImportError as e:
    print(f"⚠️  Warning: Could not import embedding dependencies: {e}")

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

# Model name mapping (must match data_reader.py and unit_window.py)
MODEL_MAPPING = {
    'llama3.2_1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'qwen3_8b': 'Qwen/Qwen3-8B',
    'qwen3_14b': 'Qwen/Qwen3-14B'
}

# ============================================================================
# DATASET-SPECIFIC PROMPTS
# ============================================================================

def get_evaluation_prompt(dataset_name: str, question: str, ground_truth: str, prediction: str) -> str:
    """
    Returns the appropriate evaluation prompt based on dataset.
    
    Args:
        dataset_name: Name of the dataset ('paper', 'notice', 'quality')
        question: The question being evaluated
        ground_truth: The ground truth answer
        prediction: The predicted answer
    
    Returns:
        Formatted prompt string
    """
    
    if dataset_name == 'paper':
        return f"""You are evaluating whether a predicted answer correctly answers the question.

STEP 1: IDENTIFY QUESTION TYPE

Examine the question carefully to determine what's being asked:

TYPE A - FACTUAL LISTS (strict matching required):
- Questions asking for multiple specific items
  * "What is/are the authors" 
  * "What are the keywords"
  * Look for plural nouns asking for enumerable items

TYPE B - FACTUAL IDENTIFICATION (core entity must match):
- Questions asking to identify one specific thing
  * "What is the venue" (conference/journal name)
  * "What is the publication year"
  * "What is the title"
  * "What is the artifact" (specific tool/system)

TYPE C - SEMANTIC/CONCEPTUAL (meaning-based evaluation):
- Questions asking for interpretative or conceptual content
  * "What is the main contribution"
  * "What are the research questions"
  * "What is the domain" (field of study)
  * "What are the findings"
  * "What is the methodology" or "type of study"
  * Any question asking for summary, explanation, or interpretation

STEP 2: APPLY APPROPRIATE RULES

FOR TYPE A - FACTUAL LISTS (authors, keywords):
- ALL items must be present for high scores
- Missing even ONE item means score 0-4 (INCORRECT)
- Only score 9-10 if the list is COMPLETE
- Author names can omit affiliations/institutions (those are supplementary)
- Order doesn't matter, but completeness does
- Examples:
  * Authors GT: "Alice, Bob, Carol, David"
  * Prediction: "Alice, Bob, Carol" receive score 0-4 (missing David)
  * Prediction: "Bob, Alice, David, Carol" should receive score 9-10 (all present, different order OK)

FOR TYPE B - FACTUAL IDENTIFICATION (venue, title, year):
- Focus on whether the CORE ENTITY is correctly identified
- Exact wording not required, but the specific entity must match
- Supplementary details (dates, locations, full citations) are OPTIONAL
- Examples:
  * Venue GT: "CHI'16, May 07-12 2016, San Jose, CA, USA"
  * Prediction: "CHI'16" should receive score 9-10 (core venue identified)
  * Prediction: "CHI" should receive score 7-8 (close but missing year)
  * Title GT: "Understanding Personal Tracking"
  * Prediction: "Understanding Personal Tracking" should receive score 9-10 (exact match)
  * Prediction: "Personal Tracking" should receive score 5-6 (incomplete, missing key words)

FOR TYPE C - SEMANTIC/CONCEPTUAL (contribution, domain, methodology):
- Focus on whether KEY CONCEPTS and MEANING are captured
- Different wording is acceptable if semantically equivalent
- Prediction can be shorter or more concise than ground truth
- Score based on conceptual overlap, not word matching
- Missing minor details is acceptable if main ideas are present

Scoring for SEMANTIC questions:
9-10: Captures the core concept(s) accurately, semantically equivalent even if worded differently
7-8: Captures most key concepts with minor omissions or slight inaccuracies
5-6: Captures some concepts but missing significant elements or partially misunderstood
3-4: Largely incorrect understanding or missing most key concepts
0-2: Completely wrong or irrelevant

Examples for SEMANTIC questions:
  * Main Contribution GT: "We present a novel attention-based method for context reduction that achieves 40% better accuracy than baselines while using 50% fewer tokens"
  * Prediction: "An attention-based context reduction approach that improves accuracy and reduces token usage" should receive score 9-10 (captures key concepts)
  * Prediction: "A method for context reduction using attention mechanisms" should receive score 7-8 (captures method but missing performance claims)
  * Prediction: "Improving accuracy in NLP tasks" should receive score 3-4 (too vague, missing key concepts)

  * Domain GT: "Human-Computer Interaction (HCI), specifically focusing on Computer Supported Cooperative Work (CSCW) within the context of team-based online games"
  * Prediction: "Human-Computer Interaction (HCI)" should receive score 7-8 (captures primary domain, missing specifics is acceptable)
  * Prediction: "CSCW and online gaming" should receive score 7-8 (captures key sub-domains)
  * Prediction: "HCI, CSCW, online games" should receive score 9-10 (captures all key elements)

  * Methodology GT: "longitudinal participant-driven photo elicitation study"
  * Prediction: "Photo elicitation study with participants over time" should receive score 9-10 (semantically captures all key elements)
  * Prediction: "Participant-driven photo elicitation" should receive score 7-8 (missing longitudinal aspect)
  * Prediction: "Qualitative study" should receive score 3-4 (too generic, missing key methodological details)

STEP 3: SCORE

Question Type: [Determine if TYPE A, B, or C]
Question: {question}
Ground Truth: {ground_truth}
Prediction: {prediction}

Scoring Scale (0-10):
9-10: Complete and correct (factual items complete, entities identified, or concepts captured)
7-8: Mostly correct with minor issues
5-6: Partially correct but missing important information
3-4: Incomplete or mostly wrong
0-2: Completely wrong or irrelevant

Output ONLY a number 0-10:"""
    
    elif dataset_name == 'notice':
        return f"""You are evaluating whether a predicted answer correctly answers the question.

Compare the prediction to the ground truth answer:
- For multiple items (like violation items): All items must be present
- For single facts: The core information must match
- Format variations are acceptable (spacing, punctuation, abbreviations)
- For company names: The main business name must match (legal suffixes like "Inc.", "Company", "LLC" are optional)
- For regulation sections: § 195.446 = Section 195.446
- For legal citations: "U.S.C." = "United States Code"

Score 0-10:
- 10: Perfect or nearly perfect match
- 7-9: Correct but minor formatting differences or missing optional details
- 4-6: Partially correct but missing important information
- 0-3: Wrong or mostly wrong

Question: {question}
Ground Truth: {ground_truth}
Prediction: {prediction}

Output ONLY a number 0-10:"""
    
    elif dataset_name == 'quality':
        return f"""Question: {question}

Model's answer: {prediction}

Ground truth: {ground_truth}

Does the model's answer match ground truth {ground_truth}?

The model might express its choice in various formats (works for A, B, C, or D):
- Just the letter: "A" or "B" or "C" or "D"
- Boxed: "$\\boxed{{A}}$" or "$\\boxed{{B}}$" etc.
- Bold: "**A**" or "**B**" or "**C**" or "**D**"
- Statements: "The answer is A", "Therefore B", "I choose C"
- Restating the option content word-for-word

Look for the model's FINAL answer. Ignore intermediate analysis of options.

Output 10 if the model selected option {ground_truth}, otherwise 0.

Output:"""
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets: paper, notice, quality")

# ============================================================================
# LLM API CLIENTS
# ============================================================================

class LLMClient:
    """Base class for LLM clients."""
    
    def evaluate(self, prompt: str) -> tuple:
        """
        Evaluate using the LLM.
        
        Args:
            prompt: The evaluation prompt
            
        Returns:
            Tuple of (score: float, rationale: str)
        """
        raise NotImplementedError

class GeminiClient(LLMClient):
    """Google Gemini API client."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite"):
        import google.generativeai as genai
        self.api_key = api_key
        self.model_name = model
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        print(f"✅ Gemini client initialized with model: {model}")
    
    def evaluate(self, prompt: str) -> tuple:
        """Evaluate using Gemini."""
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # For quality dataset: expect just a single number (10 or 0)
            # For paper dataset: expect number on last line
            if response_text.isdigit():
                score = float(response_text)
            else:
                # Try to extract first number from response
                score_match = re.search(r'(\d+)', response_text)
                if score_match:
                    score = float(score_match.group(1))
                else:
                    print(f"⚠️  Warning: Could not parse score from response")
                    print(f"Response: {response_text}")
                    score = 0
            
            score = max(0, min(10, score))  # Clamp to 0-10
            return score, response_text
        except Exception as e:
            print(f"❌ Error calling Gemini API: {e}")
            return 0, f"Error: {str(e)}"

class GPT4Client(LLMClient):
    """OpenAI GPT-4 API client."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.api_key = api_key
        self.model_name = model
        self.client = OpenAI(api_key=api_key)
        print(f"✅ GPT-4 client initialized with model: {model}")
    
    def evaluate(self, prompt: str) -> tuple:
        """Evaluate using GPT-4."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_completion_tokens=100  # Fixed for newer GPT-4 models
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # For quality dataset: expect just a single number (10 or 0)
            # For paper dataset: expect number on last line
            if response_text.isdigit():
                score = float(response_text)
            else:
                # Try to extract first number from response
                score_match = re.search(r'(\d+)', response_text)
                if score_match:
                    score = float(score_match.group(1))
                else:
                    print(f"⚠️  Warning: Could not parse score from response")
                    print(f"Response: {response_text}")
                    score = 0
            
            score = max(0, min(10, score))  # Clamp to 0-10
            return score, response_text
        except Exception as e:
            print(f"❌ Error calling GPT-4 API: {e}")
            return 0, f"Error: {str(e)}"

def create_llm_client(model_name: str) -> LLMClient:
    """
    Create an LLM client based on model name.
    
    Args:
        model_name: Either 'gemini' or 'gpt4'
    
    Returns:
        LLM client instance
    """
    if model_name == 'gemini':
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        return GeminiClient(api_key)
    
    elif model_name == 'gpt4':
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return GPT4Client(api_key)
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported models: gemini, gpt4")

# ============================================================================
# FILENAME UTILITIES
# ============================================================================

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

# ============================================================================
# DATASET LOADING
# ============================================================================

def load_processed_dataset(dataset_name: str, dataset_dir: str = 'data/datasets') -> Dict:
    """
    Load processed dataset file containing ground truth.
    
    Args:
        dataset_name: Name of the dataset
        dataset_dir: Directory containing processed dataset files
    
    Returns:
        Dictionary containing the full dataset
    """
    processed_file = Path(dataset_dir) / f'{dataset_name}_processed.json'
    
    if not processed_file.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_file}")
    
    print(f"📖 Loading dataset from: {processed_file}")
    
    with open(processed_file) as f:
        dataset = json.load(f)
    
    num_docs = len(dataset.get('documents', []))
    print(f"✅ Loaded dataset with {num_docs} documents")
    
    return dataset

# ============================================================================
# GROUND TRUTH EXTRACTION (IN-MEMORY)
# ============================================================================

def build_groundtruth_for_question(dataset: Dict, question: str) -> Dict:
    """
    Builds ground truth structure for a specific question IN MEMORY.
    Uses the ORIGINAL file path from the dataset metadata.
    
    Args:
        dataset: Processed dataset dictionary
        question: The question text
    
    Returns:
        Dictionary with question and ground truth for all matching documents
    """
    groundtruth = {
        'question': question,
        'files': []
    }
    
    # Find all documents that have this question
    for doc in dataset.get('documents', []):
        doc_id = doc['document_id']
        
        # Get the original file path from metadata
        original_path = doc.get('metadata', {}).get('original_path', '')
        if not original_path:
            # Fallback: construct from document_id
            original_path = f"doc_{doc_id}"
        
        for q in doc.get('questions', []):
            if q['question'] == question:
                answer = q.get('answer', '')
                
                groundtruth['files'].append({
                    'file': original_path,
                    'groundtruth': answer
                })
                break
    
    print(f"📝 Built ground truth with {len(groundtruth['files'])} documents")
    
    return groundtruth

# ============================================================================
# RESULT LOADING AND AGGREGATION
# ============================================================================

def load_and_group_results(dataset_name: str, attention_type: str, budget: float,
                           results_dir: str = 'pred_att') -> Dict[str, List[Dict]]:
    """
    Loads individual result files and groups them by question.
    Each input file contains one document's result.
    Returns a dict mapping question text to list of all documents for that question.
    """
    results_path = Path(results_dir) / dataset_name
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")
    
    # Pattern to match individual result files
    pattern = f"results-{attention_type}-{budget:.3f}-{dataset_name}_*.json"
    result_files = list(results_path.glob(pattern))
    
    if not result_files:
        raise FileNotFoundError(f"No result files found matching: {pattern}")
    
    print(f"📂 Found {len(result_files)} individual result files")
    
    # Group by question text
    grouped = defaultdict(list)
    
    for file_path in sorted(result_files):
        with open(file_path) as f:
            result = json.load(f)
            question = result.get('question', '')
            grouped[question].append(result)
    
    print(f"📋 Grouped into {len(grouped)} unique questions")
    for question, docs in grouped.items():
        print(f"   '{question[:60]}...': {len(docs)} documents")
    
    return dict(grouped)

def aggregate_results_to_dict(results_list: List[Dict], question: str,
                               processed_dataset: Dict) -> Dict:
    """
    Aggregates all document results for a query into a single dictionary.
    Uses ORIGINAL file paths from dataset metadata to match with ground truth.
    
    Args:
        results_list: List of ALL document results for this query
        question: Question text
        processed_dataset: The processed dataset with metadata
    
    Returns:
        Dictionary with aggregated results
    """
    print(f"\n📦 Aggregating {len(results_list)} documents")
    
    # Create a mapping from document_id to original_path
    doc_id_to_path = {}
    for doc in processed_dataset.get('documents', []):
        doc_id = doc['document_id']
        original_path = doc.get('metadata', {}).get('original_path', f'doc_{doc_id}')
        doc_id_to_path[str(doc_id)] = original_path
    
    # Create aggregated results structure
    aggregated = {
        'question': question,
        'files': []
    }
    
    for result in results_list:
        doc_id = str(result.get('document_id'))
        predicted = result.get('result', '')
        tokens_extracted = result.get('tokens_extracted')
        total_tokens = result.get('total_tokens')
        
        # Use original file path to match with ground truth
        original_path = doc_id_to_path.get(doc_id, f'doc_{doc_id}')
        
        aggregated['files'].append({
            'file': original_path,
            'result': predicted,
            'tokens_extracted': tokens_extracted,
            'total_tokens': total_tokens,
        })
    
    print(f"✅ Aggregated results with {len(aggregated['files'])} documents")
    
    return aggregated

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_results_llm(results_dict: Dict, groundtruth_dict: Dict, 
                         acc_file: Path, dataset_name: str, llm_client: LLMClient) -> None:
    """
    LLM-based evaluation using direct API calls with dataset-specific prompts.
    
    Args:
        results_dict: Dictionary with results
        groundtruth_dict: Dictionary with ground truth
        acc_file: Path to save evaluation output
        dataset_name: Name of the dataset (for prompt selection)
        llm_client: LLM client instance
    """
    groundtruth_data = groundtruth_dict["files"]
    acc_res = {"question": groundtruth_dict["question"], "files": []}
    question = groundtruth_dict["question"]
    
    print("Question:", question)
    
    total_budget = 0
    # Note: Budget calculation logic from original
    if any(keyword in str(acc_file) for keyword in ["fallback", "sentence", "section"]):
        for entry in results_dict["files"]:
            if "budget" in entry:
                total_budget += entry["budget"]
    
    cnt = 0
    total_score = 0
    
    for result in results_dict["files"]:
        file = result["file"]
        test = result["result"]
        tokens_extracted = result.get("tokens_extracted")  
        total_tokens = result.get("total_tokens")          
        ground_truth = next((item["groundtruth"] for item in groundtruth_data 
                            if item["file"] == file), None)
        
        if ground_truth:
            # Get dataset-specific prompt
            evaluation_prompt = get_evaluation_prompt(
                dataset_name=dataset_name,
                question=question,
                ground_truth=ground_truth,
                prediction=test
            )
            
            # Run LLM evaluation
            score, rationale = llm_client.evaluate(evaluation_prompt)
            
            # Count as match if score >= 7
            match = score >= 7
            if match:
                cnt += 1
            total_score += score
            
            print(f"file: {file}, groundtruth: {ground_truth[:100]}..., result: {test[:100]}..., score: {score}, match: {match}")
            
            acc_res["files"].append({
                "file": file,
                "groundtruth": ground_truth,
                "result": test,
                "score": score,
                "match": match,
                "rationale": rationale,
                "tokens_extracted": tokens_extracted, 
                "total_tokens": total_tokens,          
            })
    
    acc_res["total_matches"] = cnt
    acc_res["total_files"] = len(results_dict["files"])
    acc_res["average_score"] = total_score / len(results_dict["files"]) if results_dict["files"] else 0
    if total_budget > 0:
        acc_res["total_avg_budget"] = total_budget / len(results_dict["files"])
    
    # Save output file
    acc_file.parent.mkdir(parents=True, exist_ok=True)
    with open(acc_file, "w") as f:
        json.dump(acc_res, f, indent=4)
    
    print(f"Total matches (score >= 9): {cnt} out of {len(results_dict['files'])}")
    print(f"Average score: {acc_res['average_score']:.2f}")
    if total_budget > 0:
        print(f"Total avg budget: {total_budget/len(results_dict['files'])}")

def evaluate_results_embedding(results_dict: Dict, groundtruth_dict: Dict,
                                acc_file: Path, embedding_model: SentenceTransformer) -> None:
    """
    Embedding-based evaluation using cosine similarity.
    
    Args:
        results_dict: Dictionary with results
        groundtruth_dict: Dictionary with ground truth
        acc_file: Path to save evaluation output
        embedding_model: Pre-loaded sentence transformer model
    """
    groundtruth_data = groundtruth_dict["files"]
    acc_res = {"question": groundtruth_dict["question"], "files": []}
    question = groundtruth_dict["question"]
    
    print("Question:", question)
    
    total_similarity = 0
    cnt = 0
    
    for result in results_dict["files"]:
        file = result["file"]
        test = result["result"]
        tokens_extracted = result.get("tokens_extracted")
        total_tokens = result.get("total_tokens")

        ground_truth = next((item["groundtruth"] for item in groundtruth_data 
                            if item["file"] == file), None)
        
        if ground_truth:
            # Compute embeddings
            embeddings = embedding_model.encode([ground_truth, test])
            
            # Calculate cosine similarity
            cos_sim = cosine_similarity(
                [embeddings[0]],
                [embeddings[1]]
            )[0][0]
            
            # Clamp to [0, 1]
            similarity = max(0.0, min(1.0, float(cos_sim)))
            total_similarity += similarity
            
            # Log results
            print(f"file: {file}, "
                  f"groundtruth: {ground_truth[:100]}..., "
                  f"result: {test[:100]}..., "
                  f"similarity: {similarity:.4f}")
            
            # Append results to acc_res
            acc_res["files"].append({
                "file": file,
                "groundtruth": ground_truth,
                "result": test,
                "similarity": similarity,
                "tokens_extracted": tokens_extracted,
                "total_tokens": total_tokens,
            })
    
    acc_res["total_matches"] = 0  # Not applicable for embedding
    acc_res["total_files"] = len(results_dict["files"])
    acc_res["average_similarity"] = total_similarity / len(results_dict["files"]) if results_dict["files"] else 0
    
    # Save output file
    acc_file.parent.mkdir(parents=True, exist_ok=True)
    with open(acc_file, "w") as f:
        json.dump(acc_res, f, indent=4)
    
    print(f"Average similarity: {acc_res['average_similarity']:.4f}")

# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate attention-based context reduction results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Llama3.2-1B results with Gemini
  python evaluate.py --dataset paper --attention-type raw --budget 0.05 --model llama3.2_1b --method llm --llm-model gemini
  
  # Evaluate Qwen3-8B results with GPT-4
  python evaluate.py --dataset paper --attention-type farest --budget 0.10 --model qwen3_8b --method llm --llm-model gpt4
  
  # Evaluate all budgets for Qwen3-14B with embedding similarity
  python evaluate.py --dataset paper --attention-type baseline --all-budgets --model qwen3_14b --method embedding
  
  # Evaluate with both methods
  python evaluate.py --dataset paper --attention-type raw --budget 0.05 0.10 --model llama3.2_1b --method both
        """
    )
    
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (paper, notice, quality)')
    parser.add_argument('--attention-type', type=str, required=True,
                       choices=['raw', 'farest', 'baseline', 'diff-far', 'diff-baseline'], 
                       help='Type of attention used')
    parser.add_argument('--budget', nargs='+', type=str, help='One or more budget values')
    parser.add_argument('--all-budgets', action='store_true', help='Evaluate all budgets')
    
    # ✅ NEW: Model selection (must match preprocessing and inference)
    parser.add_argument('--model', type=str,
                       default="llama3.2_1b",
                       choices=["llama3.2_1b", "qwen3_8b", "qwen3_14b"],
                       help='Model used for preprocessing and inference (MUST match!)')
    
    parser.add_argument('--method', type=str, default='both',
                       choices=['llm', 'embedding', 'both'], help="Evaluation method")
    parser.add_argument('--llm-model', type=str, default='gemini',
                       choices=['gemini', 'gpt4'], help="LLM model for evaluation (gemini or gpt4)")
    parser.add_argument('--results-dir', type=str, default='pred_att', help='Directory containing results')
    parser.add_argument('--dataset-dir', type=str, default='data/datasets', help='Directory containing processed dataset files')
    parser.add_argument('--output-dir', type=str, default='evaluation', help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # ✅ Resolve model name from mapping
    model_name = MODEL_MAPPING.get(args.model, args.model)
    
    evaluate_all = args.all_budgets
    
    # Then replace the budget parsing logic:
    if not args.budget and not args.all_budgets:
        print("❌ ERROR: Either --budget or --all-budgets must be specified")
        return

    evaluate_all = args.all_budgets

    if evaluate_all:
        budgets_to_evaluate = None  # Will be determined from files
    else:
        # Parse budget values
        budgets_to_evaluate = []
        for budget_str in args.budget:
            try:
                budget_val = float(budget_str)
                budgets_to_evaluate.append(budget_val)
            except ValueError:
                print(f"❌ ERROR: Invalid budget value '{budget_str}'")
                return
        
        if not budgets_to_evaluate:
            print("❌ ERROR: No valid budget values provided")
            return
    
    print("\n" + "="*60)
    print("EVALUATION PIPELINE (IN-MEMORY, NO GROUNDTRUTH FILES)")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model} ({model_name})")
    print(f"Attention Type: {args.attention_type}")
    print(f"Evaluation Method: {args.method}")
    if args.method in ['llm', 'both']:
        print(f"LLM Model: {args.llm_model}")
    
    # ✅ WARNING: Check model consistency
    print(f"\n⚠️  IMPORTANT: Ensure preprocessing and inference used the same model!")
    print(f"   Current model: {args.model} ({model_name})")
    print()
    
    # Initialize LLM client if needed
    llm_client = None
    if args.method in ['llm', 'both']:
        try:
            llm_client = create_llm_client(args.llm_model)
        except Exception as e:
            print(f"❌ Error initializing LLM client: {e}")
            return
    
    # Load embedding model if needed
    embedding_model = None
    if args.method in ['embedding', 'both']:
        try:
            print("📥 Loading Qwen3-Embedding-0.6B model...")
            embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            return
    
    # ✅ Setup directories with model prefix
    base_data_dir = os.path.join(args.model, args.dataset_dir)
    dataset_dir = os.path.join(base_data_dir, '')  # Datasets are in model/data/datasets/
    
    base_results_dir = os.path.join(args.model, args.results_dir)
    base_output_dir = os.path.join(args.model, args.output_dir)
    
    print(f"📂 Directory structure:")
    print(f"   Dataset:     {dataset_dir}")
    print(f"   Results:     {base_results_dir}/")
    print(f"   Output:      {base_output_dir}/")
    print()
    
    # Load processed dataset once (contains ground truth)
    try:
        dataset = load_processed_dataset(args.dataset, dataset_dir)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"   Expected file: {os.path.join(dataset_dir, args.dataset)}_processed.json")
        print(f"   Make sure you ran data_reader.py with --model {args.model}")
        return
    
    # Determine budgets to evaluate
    if evaluate_all:
        results_path = Path(base_results_dir) / args.dataset
        if not results_path.exists():
            print(f"❌ ERROR: Results directory not found: {results_path}")
            print(f"   Expected directory: {args.model}/pred_att/{args.dataset}/")
            print(f"   Make sure you ran unit_window.py with --model {args.model}")
            return
            
        pattern = f"results-{args.attention_type}-*-{args.dataset}_*.json"
        result_files = list(results_path.glob(pattern))
        
        budgets = set()
        for file_path in result_files:
            parts = file_path.name.split('-')
            if len(parts) >= 3:
                try:
                    budget = float(parts[2])
                    budgets.add(budget)
                except ValueError:
                    continue
        
        budgets = sorted(list(budgets))
        print(f"Found budgets: {budgets}")
    else:
        budgets = sorted(budgets_to_evaluate)
        print(f"Evaluating budgets: {budgets}")
    
    output_root = Path(base_output_dir)
    
    # Track statistics
    total_questions = 0
    successful_llm = 0
    successful_embedding = 0
    failed_questions = []
    
    for budget in budgets:
        print(f"\n{'='*60}")
        print(f"EVALUATING BUDGET: {budget:.3f}")
        print(f"{'='*60}")
        
        try:
            # Load and group all individual result files by question
            grouped_results = load_and_group_results(args.dataset, args.attention_type, budget, base_results_dir)
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            continue
        
        dataset_dir_output = output_root / args.dataset
        dataset_dir_output.mkdir(parents=True, exist_ok=True)
        
        # Process each question
        for question, doc_results in grouped_results.items():
            total_questions += 1
            safe_question = sanitize_question_for_filename(question)
            
            print(f"\n{'─'*60}")
            print(f"QUERY {total_questions}: {question[:80]}...")
            print(f"{'─'*60}")
            print(f"Processing {len(doc_results)} documents")
            
            # Aggregate results IN MEMORY
            try:
                aggregated_results = aggregate_results_to_dict(
                    doc_results, question, dataset
                )
            except Exception as e:
                print(f"❌ Error aggregating results: {e}")
                import traceback
                traceback.print_exc()
                failed_questions.append((question, "aggregation", str(e)))
                continue
            
            # Build ground truth IN MEMORY (no file creation)
            try:
                groundtruth_data = build_groundtruth_for_question(dataset, question)
            except Exception as e:
                print(f"❌ Error building ground truth: {e}")
                import traceback
                traceback.print_exc()
                failed_questions.append((question, "groundtruth", str(e)))
                continue
            
            # Run evaluations with in-memory data
            if args.method in ['llm', 'both']:
                llm_output_file = dataset_dir_output / f"{args.attention_type}_budget_{budget:.3f}_{safe_question}_{args.llm_model}.json"
                try:
                    print("\n" + "="*60)
                    print(f"LLM EVALUATION ({args.llm_model.upper()})")
                    print("="*60)
                    evaluate_results_llm(aggregated_results, groundtruth_data, llm_output_file, args.dataset, llm_client)
                    successful_llm += 1
                    print(f"✅ LLM evaluation completed successfully")
                except Exception as e:
                    print(f"❌ Error in LLM evaluation: {e}")
                    import traceback
                    traceback.print_exc()
                    failed_questions.append((question, "llm_eval", str(e)))
            
            if args.method in ['embedding', 'both']:
                embedding_output_file = dataset_dir_output / f"{args.attention_type}_budget_{budget:.3f}_{safe_question}_embedding.json"
                try:
                    print("\n" + "="*60)
                    print("EMBEDDING EVALUATION")
                    print("="*60)
                    evaluate_results_embedding(aggregated_results, groundtruth_data, embedding_output_file, embedding_model)
                    successful_embedding += 1
                    print("✅ Embedding evaluation completed successfully")
                except Exception as e:
                    print(f"❌ Error in embedding evaluation: {e}")
                    import traceback
                    traceback.print_exc()
                    failed_questions.append((question, "embedding_eval", str(e)))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    print(f"Total questions processed: {total_questions}")
    if args.method in ['llm', 'both']:
        print(f"Successful LLM evaluations: {successful_llm}/{total_questions}")
    if args.method in ['embedding', 'both']:
        print(f"Successful embedding evaluations: {successful_embedding}/{total_questions}")
    
    if failed_questions:
        print(f"\n⚠️  Failed evaluations: {len(failed_questions)}")
        for question, stage, error in failed_questions[:5]:  # Show first 5
            print(f"   - {question[:50]}... (failed at {stage})")
        if len(failed_questions) > 5:
            print(f"   ... and {len(failed_questions) - 5} more")
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE")
    print("="*60)
    print(f"📁 Results saved in: {base_output_dir}/{args.dataset}/")
    print(f"\n💡 Directory structure:")
    print(f"   {args.model}/")
    print(f"   ├── data/              (input)")
    print(f"   │   └── datasets/")
    print(f"   ├── pred_att/          (results from unit_window.py)")
    print(f"   │   └── {args.dataset}/")
    print(f"   └── evaluation/        (evaluation output)")
    print(f"       └── {args.dataset}/")
    print("\n📌 No ground truth files created - all processing done in memory")

if __name__ == "__main__":
    main()