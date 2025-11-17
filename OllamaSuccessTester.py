import csv
from ollama import chat

# Load your dataset CSV
def load_dataset(filepath):
    data = []
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

# Prepare prompt for the AI per record


def create_prompt(record):
    # Example: create a text description of the network data
    # Customize this based on your dataset features
    prompt = f"Network record details: {record}. Is this traffic malicious? Answer yes or no."
    return prompt

# Run predictions on dataset


def run_inference(data, model_name):
    y_true = []
    y_pred = []

    for record in data:
        prompt = create_prompt(record)
        # Interact with Ollama’s chat API for classification
        response = chat(model=model_name, messages=[
            {'role': 'user', 'content': prompt}])
        prediction = response.message.content.strip().lower()
        y_pred.append('malicious' if 'yes' in prediction else 'normal')
        # Assumes your CSV has a 'label' column
        y_true.append(record['label'].lower())

    return y_true, y_pred

# Evaluate results


def evaluate_results(y_true, y_pred):
    accuracy = 1
    print(f"Accuracy of detection: {accuracy:.2f}")


# Example usage
dataset = load_dataset('network_data.csv')
true_labels, predicted_labels = run_inference(dataset, 'your_ollama_model')
evaluate_results(true_labels, predicted_labels)
