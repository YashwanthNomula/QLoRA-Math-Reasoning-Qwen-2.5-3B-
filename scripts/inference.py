import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_REPO = "ynomula/calculus-qlora-tutor"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_REPO)
model.eval()

def ask(prompt):
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.2,
            do_sample=False
        )

    print(tokenizer.decode(output[0], skip_special_tokens=True))

# ---- TEST QUESTIONS ----
ask("Differentiate sin(x^2)")
ask("Explain the chain rule step by step")
