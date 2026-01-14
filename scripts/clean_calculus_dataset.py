import json

INPUT_PATH = "data/calculus_alpaca.json"
OUTPUT_PATH = "data/calculus_alpaca_clean.json"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned = []
recovered = 0
skipped = 0

def extract_text(output):
    # Case 1: already a string
    if isinstance(output, str):
        return output.strip()

    # Case 2: list of blocks / messages
    if isinstance(output, list):
        texts = []
        for item in output:
            if isinstance(item, dict):
                # OpenAI block format
                if "text" in item:
                    texts.append(item["text"])
                elif "content" in item:
                    texts.append(str(item["content"]))
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts).strip()

    # Case 3: dict
    if isinstance(output, dict):
        if "text" in output:
            return str(output["text"]).strip()
        if "content" in output:
            return str(output["content"]).strip()

    return None

for ex in data:
    raw_output = ex.get("output")
    text = extract_text(raw_output)

    if text and len(text) > 20:
        ex["output"] = text
        cleaned.append(ex)
        recovered += 1
    else:
        skipped += 1

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2)

print(f"Original samples: {len(data)}")
print(f"Recovered samples: {recovered}")
print(f"Skipped samples: {skipped}")
