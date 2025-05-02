import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm
import time
import os

MODEL_ID_INTERNIST = "internistai/base-7b-v0.2"
MODEL_ID_BIOMISTRAL = "BioMistral/BioMistral-7B"

DATA_PATH = 'task2data.jsonl'
OUTPUT_CSV_PATH = "nejm_poc_raw_outputs.csv"
OUTPUT_DIR = "nejm_outputs"

# Prompt Template
PROMPT_TEMPLATE = """You are an expert clinical assistant AI. Based on the following clinical case presentation, please provide the top 3 most likely differential diagnoses. For each diagnosis, provide a brief justification citing key evidence from the text. Ensure your reasoning is clear and concise.

Clinical Case Presentation:
{case_presentation}

Top 3 Differential Diagnoses with Justification:
1. """

MAX_NEW_TOKENS = 512 # Max tokens to generate for the answer
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.1

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model_and_tokenizer(model_id):
    """Loads a model and tokenizer with 8-bit quantization."""
    print(f"Loading {model_id}...")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 if available, else float16
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token for {model_id}")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto", # Automatically distributes layers
            trust_remote_code=True
        )
        model.eval() # Set model to evaluation mode
        print(f"{model_id} loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading {model_id}: {e}")
        print("Please ensure you have accepted license terms on Hugging Face if required.")
        raise

def generate_diagnosis(model, tokenizer, case_presentation):
    """Generates diagnosis for a given case using the specified model."""
    prompt = PROMPT_TEMPLATE.format(case_presentation=case_presentation)
    input_max_length = (model.config.max_position_embeddings
                        if hasattr(model.config, 'max_position_embeddings')
                        else 4096) - MAX_NEW_TOKENS

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
    start_time = time.time()
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
        end_time = time.time()
        latency = end_time - start_time

    return output_text, latency

print(f"Loading NEJM data from: {DATA_PATH}")
nejm_cases = []
try:
    with open(DATA_PATH, 'r') as f:
        for line in f:
            try:
                nejm_cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping line due to error: {e}\nContent: {line.strip()}")
    df_cases = pd.DataFrame(nejm_cases)
    print(f"Loaded {len(df_cases)} NEJM cases.")
    if df_cases.empty:
        print("Error: No cases loaded. Check data path and file format.")
        exit()
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()


model_internist, tokenizer_internist = load_model_and_tokenizer(MODEL_ID_INTERNIST)
model_biomistral, tokenizer_biomistral = load_model_and_tokenizer(MODEL_ID_BIOMISTRAL)

print("\n--- Starting NEJM PoC Evaluation ---")
results_list = []

for index, row in tqdm(df_cases.iterrows(), total=len(df_cases), desc="Processing NEJM Cases"):
    case_name = row['casename']
    presentation = row['presentation']
    final_diagnosis = row['final']

    print(f"\nProcessing Case {index + 1}/{len(df_cases)}: {case_name}")

    # Generate with InternistAI-7B
    print(f"  Generating with {MODEL_ID_INTERNIST}...")
    internist_output, internist_latency = generate_diagnosis(model_internist, tokenizer_internist, presentation)
    print(f"  {MODEL_ID_INTERNIST} Latency: {internist_latency:.2f}s")

    # Generate with BioMistral-7B
    print(f"  Generating with {MODEL_ID_BIOMISTRAL}...")
    biomistral_output, biomistral_latency = generate_diagnosis(model_biomistral, tokenizer_biomistral, presentation)
    print(f"  {MODEL_ID_BIOMISTRAL} Latency: {biomistral_latency:.2f}s")

    # Store results
    results_list.append({
        "casename": case_name,
        "final_diagnosis": final_diagnosis,
        "internistai_7b_output_raw": internist_output,
        "biomistral_7b_output_raw": biomistral_output,
        "internistai_7b_latency_s": internist_latency,
        "biomistral_7b_latency_s": biomistral_latency
    })

    # Save individual outputs for easier review/evidence
    case_output_filename = os.path.join(OUTPUT_DIR, f"case_{index+1}_{case_name.replace(' ', '_').replace(':', '')}.txt")
    with open(case_output_filename, 'w', encoding='utf-8') as f:
        f.write(f"--- CASE: {case_name} ---\n")
        f.write(f"--- FINAL DIAGNOSIS: {final_diagnosis} ---\n\n")
        f.write(f"--- {MODEL_ID_INTERNIST} Output (Latency: {internist_latency:.2f}s) ---\n")
        f.write(internist_output + "\n\n")
        f.write(f"--- {MODEL_ID_BIOMISTRAL} Output (Latency: {biomistral_latency:.2f}s) ---\n")
        f.write(biomistral_output + "\n")
    print(f"  Saved individual results to {case_output_filename}")

print("\n--- Evaluation Complete ---")
df_results = pd.DataFrame(results_list)
try:
    df_results.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
    print(f"Aggregate raw results saved to: {OUTPUT_CSV_PATH}")
except Exception as e:
    print(f"Error saving results to CSV: {e}")


print("\n--- Script Finished ---")