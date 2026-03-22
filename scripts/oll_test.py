import os
import json
import subprocess
import requests
from datetime import datetime

OLLAMA_API = "http://localhost:11434/api/generate"
JSON_LOG_DIR = "json_log"


# getting local Ollama models
def get_ollama_models():
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    lines = result.stdout.strip().split("\n")[1:]  # skip header
    models = [line.split()[0] for line in lines if line]
    return models


# load latest prediction and ground truth files
def get_latest_files():
    files = os.listdir(JSON_LOG_DIR)

    pred_files = sorted([f for f in files if f.startswith("prediction_")])
    gt_files = sorted([f for f in files if f.startswith("ground_truth_")])

    return (
        os.path.join(JSON_LOG_DIR, pred_files[-1]),
        os.path.join(JSON_LOG_DIR, gt_files[-1]),
    )


# query Ollama model
def query_model(model, prompt):
    response = requests.post(
        OLLAMA_API,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


# extract reasoning, solution, and label from model response
def extract_response_parts(text):
    reasoning = ""
    solution = ""
    label = "UNKNOWN"
    
    # Extract REASONING section
    if "REASONING:" in text:
        reasoning_start = text.index("REASONING:") + len("REASONING:")
        reasoning_end = text.find("SOLUTION:") if "SOLUTION:" in text else text.find("LABEL:")
        reasoning = text[reasoning_start:reasoning_end].strip() if reasoning_end != -1 else text[reasoning_start:].strip()
    
    # Extract SOLUTION section
    if "SOLUTION:" in text:
        solution_start = text.index("SOLUTION:") + len("SOLUTION:")
        solution_end = text.find("LABEL:")
        solution = text[solution_start:solution_end].strip() if solution_end != -1 else text[solution_start:].strip()
    
    # Extract LABEL
    if "LABEL:" in text:
        label_start = text.index("LABEL:") + len("LABEL:")
        label_text = text[label_start:].split("\n")[0].strip()
        label = extract_label(label_text)
    
    return {
        "reasoning": reasoning,
        "solution": solution,
        "label": label
    }


# extract label from model response
def extract_label(text):
    text = text.lower()

    if "dos" in text:
        return "DoS Slowhttptest"
    if "attack" in text:
        return "ATTACK"
    if "benign" in text or "normal" in text:
        return "BENIGN"

    return "UNKNOWN"




# build prompt for model
def build_prompt(pred_json):
    return f"""
You are a cybersecurity expert.

Analyze the following network traffic summary produced by an XGBoost classifier.

Provide your response in the following format:

REASONING:
[Analyze what is happening in the traffic, whether it is an attack, and what type of attack if applicable]

SOLUTION:
[Provide specific recommendations or mitigation strategies for this traffic pattern]

LABEL: <label>

JSON:
{json.dumps(pred_json, indent=2)}
"""

# run evaluation
def evaluate(models, pred_json, ground_truth):
    results = []

    true_label = ground_truth["most_common_true_label"]

    for model in models:
        print(f"\n--- Testing model: {model} ---")

        prompt = build_prompt(pred_json)
        response = query_model(model, prompt)

        response_parts = extract_response_parts(response)
        predicted_label = response_parts["label"]
        reasoning = response_parts["reasoning"]
        solution = response_parts["solution"]

        correct = predicted_label == true_label

        results.append({
            "model": model,
            "predicted": predicted_label,
            "true": true_label,
            "correct": correct,
            "reasoning": reasoning,
            "solution": solution
        })

        print(f"Predicted: {predicted_label}")
        print(f"True: {true_label}")
        print(f"Correct: {correct}")
        print("\n--- Reasoning ---")
        print(reasoning)
        print("\n--- Solution ---")
        print(solution)

    return results


# main
def main():
    models = get_ollama_models()

    print("\nAvailable models:")
    for i, m in enumerate(models):
        print(f"{i}: {m}")

    choice = input("\nSelect model index or 'all': ")

    if choice == "all":
        selected_models = models
    else:
        selected_models = [models[int(choice)]]

    pred_file, gt_file = get_latest_files()

    with open(pred_file) as f:
        pred_json = json.load(f)

    with open(gt_file) as f:
        ground_truth = json.load(f)

    results = evaluate(selected_models, pred_json, ground_truth)

    # summary of results
    correct = sum(r["correct"] for r in results)
    total = len(results)

    print("\n--- Summary ---")
    print(f"Accuracy: {correct}/{total} = {correct/total:.2f}")

    # saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"{JSON_LOG_DIR}/evaluation_{timestamp}.json"

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()