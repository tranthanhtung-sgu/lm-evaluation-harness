# COMP6011 Task 2: Medical LLM Evaluation

This repository contains the code, configuration, and results for the evaluation of open-source medical Large Language Models as part of the COMP6011 Advanced AI Research Topics assignment.

## Evaluation Overview

The evaluation consists of two main parts:

1.  **Standardized Benchmarking:** Using the `lm-evaluation-harness` framework to evaluate four selected models (BioMistral-7B, Meditron-7B, InternistAI-7B, Asclepius-13B) on key medical benchmarks (MedQA, PubMedQA, MedMCQA, MMLU subsets).
2.  **Qualitative Proof-of-Concept (PoC):** Running a custom script (`10cases.py`) to evaluate the top-performing models from the benchmarking phase (InternistAI-7B and BioMistral-7B) on 10 complex NEJM clinical case studies provided in `task2data.jsonl`.

## Reproducing the Results

Follow these steps to set up the environment and reproduce the evaluations performed for this report.

### 1. Prerequisites

*   **Operating System:** Ubuntu (Tested on Ubuntu 24.04 LTS)
*   **GPU:** NVIDIA GPU with CUDA support and >= 24GB VRAM (Tested on NVIDIA RTX 3090).
*   **NVIDIA Drivers:** Compatible NVIDIA drivers installed.
*   **Python:** Python 3.8+ (Python 3.10 recommended). Managed via `conda` or `venv`.
*   **Git:** Required for cloning the repository.
*   **Internet Connection:** Required for downloading models and datasets.

### 2. Setup Environment

1.  **Clone this Repository:**
    ```bash
    git clone https://github.com/tranthanhtung-sgu/lm-evaluation-harness.git # Or your specific fork URL
    cd lm-evaluation-harness # Or your repo directory name
    ```

2.  **Create and Activate Virtual Environment:**
    *   Using `conda`:
        ```bash
        conda create -n lm-eval-task2 python=3.10 -y
        conda activate lm-eval-task2
        ```
    *   Using `venv`:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Dependencies:** Install all required packages using the provided `requirements.txt` file. This includes `lm-evaluation-harness`, `transformers`, `torch`, `accelerate`, `bitsandbytes`, etc.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The base repository seems to be a fork of lm-evaluation-harness. If you haven't explicitly generated the `requirements.txt` from your *active* environment after installing everything, it's safer to run `pip install -e .` first, then `pip install transformers torch accelerate bitsandbytes pandas tqdm sentencepiece huggingface_hub`)*

### 3. Hugging Face Authentication (Important)

Several models evaluated (Meditron-7B, InternistAI-7B, Asclepius-13B, potentially based on Llama) require accepting license terms on their Hugging Face model pages and may require authentication via a Hugging Face access token.

1.  **Login via CLI:**
    ```bash
    huggingface-cli login
    ```
    Follow the prompts and paste your Hugging Face access token (ensure it has at least 'read' permissions).

2.  **Accept License Terms:** Manually visit the Hugging Face pages for the models listed below and accept any license terms if you haven't already:
    *   `epfl-llm/meditron-7b`
    *   `internistai/base-7b-v0.2`
    *   `starmpcc/Asclepius-13B` (or its base model if applicable)

### 4. Running Standardized Benchmarks (`lm-evaluation-harness`)

These commands replicate the quantitative benchmarking reported in Section 4.1 of the report. They evaluate the models using 8-bit quantization on the specified tasks. The results generated these runs are stored in the `results/` directory.

**Run each model's evaluation separately:**

*   **BioMistral-7B (`BioMistral/BioMistral-7B`)**
    ```bash
    # 5-shot tasks
    lm_eval --model hf \
        --model_args pretrained=BioMistral/BioMistral-7B,load_in_8bit=True,trust_remote_code=True \
        --tasks medqa_4options,medmcqa,mmlu_professional_medicine,mmlu_clinical_knowledge,mmlu_anatomy,mmlu_college_biology,mmlu_medical_genetics \
        --num_fewshot 5 \
        --device cuda:0 \
        --batch_size auto \
        --output_path results/biomistral_7b_5shot_eval.json

    # 0-shot task (Note: ran 5-shot in provided JSON, use num_fewshot 5 to replicate)
    lm_eval --model hf \
        --model_args pretrained=BioMistral/BioMistral-7B,load_in_8bit=True,trust_remote_code=True \
        --tasks pubmedqa \
        --num_fewshot 5 \
        --device cuda:0 \
        --batch_size auto \
        --output_path results/biomistral_7b_pubmedqa_5shot_eval.json
    ```

*   **Meditron-7B (`epfl-llm/meditron-7b`)**
    ```bash
    # 5-shot tasks
    lm_eval --model hf \
        --model_args pretrained=epfl-llm/meditron-7b,load_in_8bit=True,trust_remote_code=True \
        --tasks medqa_4options,medmcqa,mmlu_professional_medicine,mmlu_clinical_knowledge,mmlu_anatomy,mmlu_college_biology,mmlu_medical_genetics,pubmedqa \
        --num_fewshot 5 \
        --device cuda:0 \
        --batch_size auto \
        --output_path results/meditron_7b_5shot_eval.json
    ```

*   **InternistAI-7B (`internistai/base-7b-v0.2`)**
    ```bash
    # 5-shot tasks
    lm_eval --model hf \
        --model_args pretrained=internistai/base-7b-v0.2,load_in_8bit=True,trust_remote_code=True \
        --tasks medqa_4options,medmcqa,mmlu_professional_medicine,mmlu_clinical_knowledge,mmlu_anatomy,mmlu_college_biology,mmlu_medical_genetics,pubmedqa \
        --num_fewshot 5 \
        --device cuda:0 \
        --batch_size auto \
        --output_path results/internistai_7b_5shot_eval.json
    ```

*   **Asclepius-13B (`starmpcc/Asclepius-13B`)**
    ```bash
    # 5-shot tasks
    lm_eval --model hf \
        --model_args pretrained=starmpcc/Asclepius-13B,load_in_8bit=True,trust_remote_code=True \
        --tasks medqa_4options,medmcqa,mmlu_professional_medicine,mmlu_clinical_knowledge,mmlu_anatomy,mmlu_college_biology,mmlu_medical_genetics,pubmedqa \
        --num_fewshot 5 \
        --device cuda:0 \
        --batch_size auto \
        --output_path results/asclepius_13b_5shot_eval.json
    ```

**Notes:**
*   Ensure the `results/` directory exists before running (`mkdir results`).
*   These evaluations can take several hours per model.
*   Monitor GPU memory usage (`nvidia-smi`). If errors occur, try setting `--batch_size 1`.
*   The JSON files provided in the repository (`results_*.json`) are the outputs of these commands.

### 5. Running the NEJM Qualitative Proof-of-Concept (PoC)

This replicates the qualitative evaluation reported in Section 4.2 of the report, using the custom script `10cases.py`.

1.  **Ensure Data File:** Verify that `task2data.jsonl` (containing the 10 NEJM cases) is present in the root directory of the repository.
2.  **Run the Script:** Execute the Python script. This script will load the selected models (InternistAI-7B and BioMistral-7B), process each case from `task2data.jsonl`, generate diagnostic suggestions, measure latency, and save the outputs.
    ```bash
    python 10cases.py
    ```
3.  **Check Outputs:**
    *   An aggregated CSV file named `nejm_poc_raw_outputs.csv` will be created/overwritten in the root directory, containing the raw text outputs and latencies for both models for all 10 cases.
    *   Individual text files for each case, containing the outputs from both models, will be saved in the `nejm_outputs/` directory.

### 6. Expected Outputs and Verification

After running the steps above, you can verify the results against the report:

*   **Standardized Benchmarks:** Compare the accuracy values in the generated JSON files within the `results/` directory against those presented in Table 4 (or your updated results table) in the report. Minor variations due to package versions or hardware specifics are possible but should be small.
*   **NEJM PoC Outputs:** Examine the raw text in `nejm_poc_raw_outputs.csv` and the individual files in `nejm_outputs/`. These form the basis for the qualitative analysis (Top-1/Top-3 accuracy, reasoning scores) presented in Section 4.2 and Table 5 of the report. The latencies recorded in the CSV should match those reported.

This setup allows for the reproduction of the core quantitative and qualitative evaluations presented in the accompanying report.