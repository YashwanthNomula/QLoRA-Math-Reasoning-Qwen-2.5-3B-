import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_PATH = "data/calculus_alpaca.json"
OUTPUT_DIR = "outputs/lora_adapter"

# 1️⃣ Quantization config (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# 2️⃣ Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

# 3️⃣ LoRA adapter
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 4️⃣ Load dataset
dataset = load_dataset("json", data_files=DATA_PATH)

def tokenize(example):
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )

    tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    # RESPONSE-ONLY LOSS
    response_start = prompt.index("### Response:")
    response_tokens = tokenizer(
        prompt[response_start:],
        truncation=True,
        max_length=512,
        padding="max_length"
    )["input_ids"]

    labels = [-100] * len(tokens["input_ids"])
    labels[-len(response_tokens):] = response_tokens

    tokens["labels"] = labels
    return tokens

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

# 5️⃣ Training config
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
