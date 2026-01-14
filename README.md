\# QLoRA Calculus Tutor



This project fine-tunes a large language model using \*\*QLoRA\*\* to act as a

calculus tutor.



\## What this project does

\- Generates synthetic calculus instruction–response data (Alpaca format)

\- Fine-tunes a base LLM using QLoRA adapters

\- Publishes the trained adapter to Hugging Face



\## Key Technologies

\- PyTorch

\- Hugging Face Transformers

\- PEFT / QLoRA

\- Accelerate



\## Repository Contents

\- `scripts/` – data generation, cleaning, and training scripts

\- `.gitignore` – excludes datasets and model artifacts



The trained LoRA adapter is published separately on Hugging Face.



