# The Limits of Large Language Model Reasoning: A Causal Hierarchy Approach

Code for MATH3001 prroject at the University of Leeds: systematically mutating a canonical confounding scenario (the “firefighter paradox”) and evaluating whether several large language models answer in line with causal structure, under single-agent and multi-agent debate conditions.

LLM causal-reasoning benchmark — programmatic mutations of the firefighter paradox; 505 scored single-agent runs + 20 debate transcripts.*

## Methodology (summary)

The core idea is a **programmatic mutation framework**: the original firefighter story (Level 1) is varied at **Level 2** via surface rewrites into many domains, and at **Level 3** via structural mutations (e.g. sign flip with adverse interaction, explicit mediator, reversed causal edge, and a small numerical *do*-calculus style item). **Five models** are evaluated (GPT-5.4 via OpenAI; GPT-OSS 120B, Llama 3.3 70B, Llama 3.1 8B, and Qwen 3 32B via Groq), yielding **505 single-agent scored responses** (101 problems × 5 models). Each response receives a **two-signal score**: **correctness** (binary match to the designated answer) and a **reasoning / keyword signal** (binary, based on occurrence of problem-specific causal vocabulary). A **multi-agent debate** stress test (naive vs structural analyst, then a judge) is run on a subset of Level-3 problems to see whether answers remain stable under adversarial pressure.

## Repository contents

- `run_experiments.py` — Problem generation, API calls, automated scoring, aggregate tables, chart export, and optional multi-agent debate pipeline.
- `all_results.json` — **505** scored single-agent model responses (101 problems × 5 models), archived from the dissertation run.
- `debate_results.json` — **20** multi-agent debate transcripts with judge verdicts and scores.
- `requirements.txt` — Python dependencies for reproduction.
- `.env.example` — Template for API keys (copy to `.env` locally; never commit real keys).
- `.gitignore` — Excludes secrets, caches, partial result dumps, and sensitive filename patterns.

## How to reproduce

1. **Python 3.10+** recommended.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set **environment variables** (or use a local `.env` file loaded by your shell — do not commit `.env`):

   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export GROQ_API_KEY="your-groq-key"
   ```

4. **Run the full pipeline** (calls all models on all problems; long-running and incurs API usage):

   ```bash
   python run_experiments.py
   ```

   Other modes:

   - `python run_experiments.py --mutations` — Print generated problem previews only (no API calls).
   - `python run_experiments.py --stats` — Recompute tables and charts from existing `all_results.json`.
   - `python run_experiments.py --charts` — Regenerate figures from `all_results.json` only.
   - `python run_experiments.py --debate` — Skip the main sweep; run only the debate experiment (still requires API keys).

## Required Python packages

`openai`, `groq`, `numpy`, `scipy`, `matplotlib` (see `requirements.txt` for install).

## Note on reproducibility

Even at **temperature 0**, provider APIs do not guarantee **bitwise-identical** completions across time. Re-running the pipeline may yield **small shifts** in exact scores or extracted answers, but the **qualitative patterns** reported in the dissertation should be stable. For archival comparison, prefer the committed `all_results.json` / `debate_results.json` from the dissertation run.

## Citation

```bibtex
@mastersthesis{goldschmidt2026limits,
  author  = {Goldschmidt, Jake},
  title   = {The Limits of Large Language Model Reasoning: A Causal Hierarchy Approach},
  school  = {University of Leeds},
  year    = {2026},
  note    = {MATH3001 dissertation}
}
```

## Author

**Jake Goldschmidt**, supervised by **Dr Luisa Cutillo** and **Mike Croucher**, School of Mathematics, **University of Leeds**.
