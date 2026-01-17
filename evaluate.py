import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_REPO = "ynomula/calculus-qlora-tutor"
TEST_FILE = "data/test.jsonl"

def load_model(use_lora: bool):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if use_lora:
        model = PeftModel.from_pretrained(model, ADAPTER_REPO)

    model.eval()
    return tokenizer, model

def generate(tokenizer, model, instruction):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.2,
            do_sample=False
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def run_evaluation(use_lora: bool, output_file: str):
    tokenizer, model = load_model(use_lora)

    results = []

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            instruction = sample["instruction"]
            expected = sample["response"]

            prediction = generate(tokenizer, model, instruction)

            results.append({
                "instruction": instruction,
                "expected": expected,
                "model_output": prediction
            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    print("Running base model evaluation...")
    run_evaluation(use_lora=False, output_file="outputs_base.json")

    print("Running fine-tuned model evaluation...")
    run_evaluation(use_lora=True, output_file="outputs_finetuned.json")

    print("Evaluation complete.")
