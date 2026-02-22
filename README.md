# Child Malnutrition Assistant: Fine-Tuning a Domain LLM with LoRA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pauline12ish34/summative_fine-tuning_LLM/blob/main/Child_Malnutrition_Assistant.ipynb)

Direct Colab link: https://colab.research.google.com/github/pauline12ish34/summative_fine-tuning_LLM/blob/main/Child_Malnutrition_Assistant.ipynb

A domain-specific assistant fine-tuned to provide evidence-based guidance on child malnutrition diagnosis, treatment, prevention, and care. The workflow is optimized for Google Colab and demonstrates parameter-efficient fine-tuning using LoRA.

---

## Quick Start (Colab)

1. Open the notebook using the badge or direct link above.
2. Runtime -> Change runtime type -> GPU (T4 or P100).
3. Run the setup cell to install dependencies.
4. Run all cells to reproduce training, evaluation, and the Gradio UI.

---

## Demo Video

 **[Watch the demo video here]** ← *Insert YouTube link*

The video walkthrough covers:
- Dataset preparation and preprocessing
- Model fine-tuning with LoRA
- Hyperparameter experiments and results
- Live Gradio UI demonstration with sample questions

---

## Project Structure


- [Child_Malnutrition_Assistant_FineTuning.ipynb](Child_Malnutrition_Assistant_FineTuning.ipynb) -  fine-tuning notebook
- [malnutrition_dataset_final.jsonl](malnutrition_dataset_final.jsonl) - Dataset (JSONL)
- [requirements.txt](requirements.txt) - Local dependencies

---

## Problem Definition and Domain Alignment
At least 33% of Rwandan children age 6-59 months are stunted (short for their age), 1% are wasted (thin for their height), 8% are underweight (thin for their age), and 6% are overweight (heavy for their height). Minimum acceptable diet: Only 22% of children age 6-23 months were fed a minimum acceptable diet during the previous day. Anemia: 37% of children age 6-59 months and 13% of women age 15-49 are anemic.(Health, Nutrition and Food Security | National Institute of Statistics of Rwanda) .

Child malnutrition remains a critical global health challenge especially in rwandan district nyabihu. This assistant targets caregivers, community health workers, and students who need quick access to accurate guidance on:

- Early signs and symptoms of malnutrition
- Emergency interventions and referral criteria
- Therapeutic feeding protocols
- Prevention strategies and balanced diet planning

The model is trained on a curated, domain-specific dataset to reduce generic or unsafe responses common to general-purpose models.

---

## Dataset and Preprocessing

- **Dataset**: 135 custom-curated Q&A pairs in JSONL format
- **Source**: Created using WHO/UNICEF child malnutrition guidelines combined with AI-assisted generation and human validation
- **File**: [malnutrition_dataset_final.jsonl](malnutrition_dataset_final.jsonl)
- **Schema support**: `messages`, or `question`/`answer`, or `instruction`/`response`

**Preprocessing steps:**

- Unicode NFKC normalization and whitespace cleanup
- Consistent prompt formatting:
	- `### Question: ...` and `### Answer: ...`
- Token length analysis to justify `MAX_SEQ_LENGTH = 512`

---

## Model and Training Configuration

- Base model: TinyLlama-1.1B-Chat-v1.0
- Fine-tuning method: LoRA (PEFT)
- Quantization: 4-bit NF4 for memory efficiency
- Optimizer: paged AdamW 8-bit
- Seed: 42
- Evaluation samples: 20

Hyperparameters explored:

| Config | Learning Rate | Batch Size | Grad Accum | Epochs | LoRA Rank | LoRA Alpha |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2e-4 | 4 | 4 | 2 | 16 | 32 |
| low_lr | 1e-4 | 4 | 4 | 2 | 16 | 32 |
| batch2 | 2e-4 | 2 | 8 | 2 | 16 | 32 |
| higher_rank | 2e-4 | 4 | 4 | 2 | 32 | 64 |

---

## Metrics and Evaluation

Metrics used:

- BLEU
- ROUGE-1 / ROUGE-2 / ROUGE-L
- Token-level F1
- Perplexity (from eval loss)

Baseline metrics are computed before fine-tuning and compared to each experiment.

---

##  Colab Execution Results

Results below are from running the notebook on Google Colab (T4 GPU):

| Configuration | Learning Rate | Batch Size | BLEU | BLEU Improvement | ROUGE-L | Perplexity | GPU Memory | Time (min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batch2 (best) | 2e-4 | 2 | 0.011960 | +38.7% | 0.114646 | 7.589 | 3.68 GB | 1.71 |
| baseline | 2e-4 | 4 | 0.010946 | +26.9% | 0.121840 | 7.563 | 3.09 GB | 1.72 |
| low_lr | 1e-4 | 4 | 0.000000 | -100.0% | 0.104786 | 8.580 | 3.44 GB | 1.65 |
| higher_rank | 2e-4 | 4 | 0.000000 | -100.0% | 0.109733 | 6.924 | 3.93 GB | 1.65 |

Best configuration: `batch2` with +38.7% BLEU improvement over baseline.

---

## Gradio UI

The notebook includes a Gradio interface for interactive testing. It loads the best model checkpoint from Google Drive and provides controls for temperature and response length.

---

## Reproducibility Notes

- Set a fixed seed (42)
- Notebook uses a fixed evaluation subset size (20)
- Training outputs are saved to Google Drive under `/content/drive/My Drive/`

---

## References

- [Health, Nutrition and Food Security | National Institute of Statistics of Rwanda](https://www.alpha.statistics.gov.rw/statistical-publications/health-nutrition-and-food-security)

## Usage Notes and Safety

This assistant is for educational use only. It does not replace professional medical advice. For any medical concerns, consult qualified healthcare professionals.

