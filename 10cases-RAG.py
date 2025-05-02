import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm
import time
import os
import re 
from Bio import Entrez

Entrez.email = "tungtranthanh258@gmail.com" 

MODEL_ID_INTERNIST = "internistai/base-7b-v0.2"
# We only need the LLM now, not a separate embedding model for the KB

DATA_PATH = 'task2data.jsonl'
OUTPUT_CSV_PATH_PUBMED_RAG = "nejm_poc_pubmed_rag_outputs.csv" # New output file
OUTPUT_DIR_PUBMED_RAG = "nejm_outputs_pubmed_rag" # New output dir

# RAG Parameters
MAX_PUBMED_RESULTS = 3 # Number of abstracts to fetch
PUBMED_QUERY_SUFFIX = " AND (review[ptyp] OR systematic review[ptyp] OR clinical trial[ptyp])" # Optional: Focus search

# Prompt Template (Modified for RAG)
PROMPT_TEMPLATE_RAG = """You are an expert clinical assistant AI. Based on the following retrieved PubMed abstract context and clinical case presentation, please provide the top 3 most likely differential diagnoses. For each diagnosis, provide a brief justification citing key evidence primarily from the **case presentation**, potentially referencing the retrieved context if directly relevant and supportive. Ensure your reasoning is clear and concise.

Retrieved PubMed Context:
{retrieved_context}

Clinical Case Presentation:
{case_presentation}

Top 3 Differential Diagnoses with Justification:
1. """

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.1

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR_PUBMED_RAG, exist_ok=True)

def load_model_and_tokenizer(model_id):
    """Loads a model and tokenizer with 8-bit quantization."""
    print(f"Loading LLM: {model_id}...")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token for {model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        print(f"{model_id} loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading {model_id}: {e}")
        raise

# *** NEW FUNCTION: Fetch PubMed Abstracts ***
def fetch_pubmed_abstracts(query, max_results=3):
    """Fetches abstracts from PubMed based on a query term."""
    print(f"    Fetching PubMed abstracts for query: '{query}'...")
    full_query = query + PUBMED_QUERY_SUFFIX 
    try:
        # Search PubMed
        handle_search = Entrez.esearch(db="pubmed", term=full_query, retmax=str(max_results), sort="relevance")
        search_results = Entrez.read(handle_search)
        handle_search.close()
        id_list = search_results["IdList"]

        if not id_list:
            print("    No relevant PubMed IDs found.")
            return "No relevant PubMed abstracts found."

        # Fetch Abstracts
        handle_fetch = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="text")
        abstracts_text = handle_fetch.read()
        handle_fetch.close()

        raw_abstracts = abstracts_text.strip().split('\n\n')
        
        formatted_abstracts = []
        current_abstract_num = 1
        for i, abstract in enumerate(raw_abstracts):
            clean_abstract = re.sub(r'\s+', ' ', abstract).strip() # Clean whitespace
            if len(clean_abstract) > 50: # Filter out potentially empty lines or headers
                 formatted_abstracts.append(f"Context {current_abstract_num}: {clean_abstract}")
                 current_abstract_num += 1
            if current_abstract_num > max_results: break

        if not formatted_abstracts:
             print("    Could not parse abstracts from fetched text.")
             return "Could not parse relevant PubMed abstracts."
             
        context_str = "\n\n".join(formatted_abstracts)
        print(f"    Retrieved {len(formatted_abstracts)} abstracts.")
        return context_str

    except Exception as e:
        print(f"    Error fetching PubMed data: {e}")
        # Avoid halting the whole script on API errors
        return f"Error fetching PubMed context: {e}"

# *** MODIFIED FUNCTION: PubMed RAG Generation ***
def generate_diagnosis_pubmed_rag(model, tokenizer, case_presentation, pubmed_query):
    """Generates diagnosis using PubMed RAG."""
    retrieval_start_time = time.time()
    retrieved_context = fetch_pubmed_abstracts(pubmed_query, MAX_PUBMED_RESULTS)
    retrieval_time = time.time() - retrieval_start_time
    print(f"    PubMed retrieval time: {retrieval_time:.2f}s")


    prompt = PROMPT_TEMPLATE_RAG.format(
        retrieved_context=retrieved_context,
        case_presentation=case_presentation
    )

    # Adjust input length calculation
    input_max_length = (model.config.max_position_embeddings
                        if hasattr(model.config, 'max_position_embeddings')
                        else 4096) - MAX_NEW_TOKENS # Default context length

    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    if encoded_prompt.input_ids.shape[1] > input_max_length:
        print(f"    Warning: Combined prompt length might exceed model capacity. Truncating.")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=input_max_length
    ).to(model.device)


    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "pad_token_id": tokenizer.eos_token_id
    }

    output_text = ""
    generation_start_time = time.time()
    try:
        with torch.no_grad():
             outputs = model.generate(**inputs, **generation_kwargs)

        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        output_text = "1. " + output_text.strip()

    except Exception as e:
        print(f"  Error during generation: {e}")
        output_text = f"GENERATION_ERROR: {e}"
    finally:
        generation_time = time.time() - generation_start_time
        total_latency = retrieval_time + generation_time

    return output_text, total_latency

# --- Main Execution ---

# 1. Load Data (Same as before)
print(f"Loading NEJM data from: {DATA_PATH}")
nejm_cases = []
try:
    with open(DATA_PATH, 'r') as f:
        for line in f:
            try:
                nejm_cases.append(json.loads(line))
            except json.JSONDecodeError as e: print(f"Skipping line: {e}")
    df_cases = pd.DataFrame(nejm_cases)
    print(f"Loaded {len(df_cases)} NEJM cases.")
    if df_cases.empty: raise FileNotFoundError("No cases loaded.")
except Exception as e:
    print(f"Error loading NEJM data: {e}")
    exit()

# 2. Load InternistAI LLM
model_internist, tokenizer_internist = load_model_and_tokenizer(MODEL_ID_INTERNIST)

# 3. Run PubMed RAG Inference Loop
print("\n--- Starting NEJM PubMed RAG PoC Evaluation ---")
results_pubmed_rag_list = []


NCBI_REQUEST_DELAY = 0.4

for index, row in tqdm(df_cases.iterrows(), total=len(df_cases), desc="Processing NEJM Cases with PubMed RAG"):
    case_name = row['casename']
    presentation = row['presentation']
    final_diagnosis = row['final']

    print(f"\nProcessing Case {index + 1}/{len(df_cases)}: {case_name} (Query: '{final_diagnosis}')")

    # Generate with RAG-Enhanced InternistAI-7B using PubMed
    print(f"  Generating RAG response with {MODEL_ID_INTERNIST}...")
    internist_rag_output, internist_rag_latency = generate_diagnosis_pubmed_rag(
        model_internist,
        tokenizer_internist,
        presentation,
        final_diagnosis
    )
    print(f"  PubMed RAG {MODEL_ID_INTERNIST} Latency: {internist_rag_latency:.2f}s")

    # Store results
    results_pubmed_rag_list.append({
        "casename": case_name,
        "final_diagnosis": final_diagnosis,
        "internistai_7b_pubmed_rag_output_raw": internist_rag_output,
        "internistai_7b_pubmed_rag_latency_s": internist_rag_latency,
    })

    # Save individual outputs
    case_output_filename = os.path.join(OUTPUT_DIR_PUBMED_RAG, f"case_{index+1}_{case_name.replace(' ', '_').replace(':', '')}_pubmed_rag.txt")
    with open(case_output_filename, 'w', encoding='utf-8') as f:
        f.write(f"--- CASE: {case_name} ---\n")
        f.write(f"--- FINAL DIAGNOSIS (Used as Query): {final_diagnosis} ---\n\n")
        f.write(f"--- PubMed RAG {MODEL_ID_INTERNIST} Output (Latency: {internist_rag_latency:.2f}s) ---\n")
        f.write(internist_rag_output + "\n")
    print(f"  Saved individual PubMed RAG results to {case_output_filename}")

    # Add delay to respect NCBI guidelines
    time.sleep(NCBI_REQUEST_DELAY)


# 4. Save Aggregate PubMed RAG Results
print("\n--- PubMed RAG Evaluation Complete ---")
df_results_pubmed_rag = pd.DataFrame(results_pubmed_rag_list)
try:
    df_results_pubmed_rag.to_csv(OUTPUT_CSV_PATH_PUBMED_RAG, index=False, encoding='utf-8')
    print(f"Aggregate PubMed RAG results saved to: {OUTPUT_CSV_PATH_PUBMED_RAG}")
except Exception as e:
    print(f"Error saving PubMed RAG results to CSV: {e}")

print("\n--- PubMed RAG Script Finished ---")