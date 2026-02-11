# QA Model Experiments

Collection of NLP experiments around question answering and model ablations: zero-shot and fine-tuned BoolQ classifiers, attention/residual modifications to DistilRoBERTa, sentiment regression on SST, and a dense-retrieval + generation pipeline for SQuAD.

## Features
- Zero-shot BoolQ with DistilGPT-2 and fine-tuning DistilGPT-2/DistilRoBERTa using Hugging Face `Trainer`.
- Ablation variants: shared Q=K=V attention and no-residual layers for DistilRoBERTa; compares accuracy/F1 against a random-weight baseline.
- Sentiment regression on SST using a Roberta encoder head for continuous scores.
- Retrieval-augmented QA demo: encodes 500 SQuAD contexts with `all-MiniLM-L6-v2`, retrieves via cosine similarity, and generates answers with Phi-3-mini.

## Tech Stack
- Python 3.9+
- Hugging Face Transformers & Datasets, PyTorch, sentence-transformers, scikit-learn, matplotlib
- (Phi-3-mini generation benefits from a GPU with bfloat16 support)

## Setup
```bash
python -m venv .venv
. .venv/Scripts/activate
pip install torch transformers datasets sentence-transformers scikit-learn matplotlib
```
Large models will download automatically on first run. Ensure you have enough GPU/CPU memory (BoolQ fine-tuning can run on a mid-range GPU; Phi-3-mini prefers =16GB VRAM or will fall back to CPU very slowly).

## Usage
Zero-shot + fine-tuning on BoolQ:
```bash
python boolq_zero_shot_and_finetune.py
```
Outputs: classification reports, accuracy, loss curves (`loss_curve.png`, `loss_curve2.png`), and saved fine-tuned models.

Ablations & SST regression experiments:
```bash
python roberta_ablation_and_sst_regression.py
```
Outputs: eval metrics printed to console; note the script currently defines `compute_metrics` after first use—run as-is to mirror the original lab, or move the function higher before training.

Retrieval + generation demo:
```bash
python squad_retrieval_phi3_generation.py
```
Outputs: retrieval accuracy over 500 SQuAD contexts and sample generated answers for correct/incorrect retrievals.

## Folder Structure
- `boolq_zero_shot_and_finetune.py` — zero-shot & fine-tune DistilGPT-2/DistilRoBERTa on BoolQ
- `roberta_ablation_and_sst_regression.py` — attention/residual ablations + BoolQ fine-tuning; SST regression head
- `squad_retrieval_phi3_generation.py` — dense retrieval with MiniLM + Phi-3 generation

## Possible Improvements
- Refactor into modules (`data.py`, `models.py`, `train.py`) with CLI flags and config files.
- Move shared utilities (tokenization, metrics) out of notebooks and fix ordering bugs (`compute_metrics` should be defined before use).
- Add experiment tracking (Weights & Biases or MLflow) and checkpoint saving for ablation runs.
- Provide deterministic seeds, smaller debug subsets, and unit tests for retrieval scoring and generation prompts.
- Introduce evaluation loops for generation quality (e.g., ROUGE/BLEU) and automated hyperparameter sweeps.
