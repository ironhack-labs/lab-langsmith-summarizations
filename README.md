# Summarization Evaluation with LangSmith

Evaluates three LLMs on news article summarization using LangSmith to trace experiments, log outputs, and compare results against human-written references. → [Notebook](lab_langsmith_summarizations.ipynb)

## Why tracing matters in LLM pipelines

Without observability, you're guessing. LLM outputs are non-deterministic — two models can produce fluent, confident summaries that differ wildly in semantic accuracy. LangSmith solves the "it looked fine" problem by logging every run as a named experiment with inputs, outputs, and evaluation scores in one place. This makes A/B testing models reproducible and auditable rather than eyeballed.

## Key findings

**Llama 3.1-8b (Groq) outperformed both fine-tuned HuggingFace models** on semantic similarity to human-written summaries, achieving the lowest average embedding cosine distance across 10 CNN/DailyMail articles — despite `facebook/bart-large-cnn` and `sshleifer/distilbart-cnn-12-6` being explicitly fine-tuned on this same dataset.

This is a useful signal: task-specific fine-tuning on older architectures doesn't automatically beat a capable general-purpose model, especially when the evaluation metric is semantic meaning rather than surface-level overlap.

A secondary finding: the original notebook was built against APIs that have since been deprecated or removed (LangChain's `run_on_dataset`, OpenAI embeddings, `t5-base` via HF Hub). Keeping evaluation infrastructure current is non-trivial maintenance work.

## Stack

| Component | Tool |
|---|---|
| Evaluation & tracing | LangSmith (`langsmith.evaluate`) |
| Embedding similarity | `sentence-transformers/all-MiniLM-L6-v2` (cosine distance) |
| Model 1 — base | `facebook/bart-large-cnn` via HuggingFace Inference API |
| Model 2 — fine-tuned | `sshleifer/distilbart-cnn-12-6` via HuggingFace Inference API |
| Model 3 — open source LLM | `llama-3.1-8b-instant` via Groq API |
| Dataset | CNN/DailyMail via HuggingFace Datasets |

## How to run

Add the following to `.env`:
```
LANGCHAIN_API_KEY=your_langsmith_key
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_key
```

Then run the notebook top to bottom. Results appear as named experiments in your LangSmith dashboard.
