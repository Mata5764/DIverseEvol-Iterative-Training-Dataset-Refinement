import json
import random
import os
from datasets import Dataset

# function to load a json file
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# function to save a json file
def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "w") as f:
        json.dump(data, f, indent = 2)

# function to set seed
def set_seed(seed):
    random.seed(seed)

# function to convert JSON file → Hugging Face Dataset
def json_to_hf_dataset(path):
    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError("JSON must be a list of objects (list[dict])")

    return Dataset.from_list(data)


# function to convert Hugging Face Dataset → JSON file
def hf_dataset_to_json(dataset, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Extract conversations from dataset
    conversations = []
    for item in dataset:
        # Parse the conversations field (it's stored as a JSON string)
        if 'conversations' in item:
            conv = json.loads(item['conversations'])
            conversations.append(conv)

    with open(path, "w") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)

def sample_data(dataset, k):
    """Randomly sample k examples from a Hugging Face Dataset and return Dataset"""
    if not hasattr(dataset, "select"):
        raise ValueError("Expected a Hugging Face Dataset object")

    total_size = len(dataset)
    indices = random.sample(range(total_size), k)  # pick random row indices
    return dataset.select(indices)  # returns a new HF Dataset


def json_to_hf_dataset(json_path: str) -> Dataset:
    """
    Convert a JSON file with 'conversations' field to a Hugging Face Dataset.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        Dataset: Hugging Face Dataset with parsed conversations.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} does not exist")

    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Parse stringified 'conversations' field
    for example in data:
        if isinstance(example.get("conversations"), str):
            example["conversations"] = json.loads(example["conversations"])

    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_list(data)
    return hf_dataset



def csv_to_conversation_json(csv_path, json_path):
    """Convert CSV file to conversation JSON format (simple version)"""
    import csv
    
    # System message (same as used in the training data)
    system_message = "You are an expert Financial Named Entity Recognition (NER) system. Your task is to analyze financial queries and extract relevant information into a structured format.\n## Output Format\nReturn a single object in the following structure:\n{\n  \"llm_NER\": [\n    {\"<phrase_1>\": [\"<tag_1>\", \"<tag_2>\", ...]},\n    {\"<phrase_2>\": [\"<tag_1>\", \"<tag_2>\", ...]},\n    ...\n  ]\n}"
    
    conversations = []
    
    # Try different delimiters
    for delimiter in ['\t', ',', '|']:
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file, delimiter=delimiter)
                rows = list(reader)
                
                # Check if we have the expected columns
                if len(rows) > 0 and ('query' in rows[0] and 'reasoned_parsed_output' in rows[0]):
                    print(f"Found valid CSV with delimiter '{delimiter}' and {len(rows)} rows")
                    
                    for row in rows:
                        conversation = [
                            {
                                "from": "system",
                                "value": system_message
                            },
                            {
                                "from": "human",
                                "value": row['query']
                            },
                            {
                                "from": "gpt", 
                                "value": row['reasoned_parsed_output']
                            }
                        ]
                        conversations.append(conversation)
                    break
        except Exception as e:
            continue
    
    if not conversations:
        print(f"Error: Could not parse CSV file: {csv_path}")
        return None
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Save to JSON file
    with open(json_path, "w") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully converted {len(conversations)} conversations to {json_path}")
    return conversations

