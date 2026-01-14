import json
import os
import time
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OUTPUT_PATH = "data/calculus_alpaca.json"
TARGET_SAMPLES = 600   # enough for QLoRA

SYSTEM_PROMPT = """You are a calculus tutor.
Generate clear, step-by-step solutions.
Return ONLY plain text.
No markdown lists, no JSON, no bullet points.
"""

PROMPT_TEMPLATE = """Create a calculus problem and solve it.

Requirements:
- Undergraduate calculus
- Step-by-step explanation
- Clear reasoning

Return in this exact format:

Instruction: <question>
Response: <solution>
"""

data = []

for _ in tqdm(range(TARGET_SAMPLES)):
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT_TEMPLATE},
            ],
            max_output_tokens=500,
        )

        # ✅ THIS IS THE KEY LINE
        text = response.output_text.strip()

        if "Instruction:" not in text or "Response:" not in text:
            continue

        instr = text.split("Instruction:", 1)[1].split("Response:", 1)[0].strip()
        resp = text.split("Response:", 1)[1].strip()

        if len(instr) < 10 or len(resp) < 50:
            continue

        data.append({
            "instruction": instr,
            "input": "",
            "output": resp
        })

        time.sleep(0.3)  # gentle rate limiting

    except Exception as e:
        print("Skipped due to error:", e)
        time.sleep(2)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Saved {len(data)} clean samples")
