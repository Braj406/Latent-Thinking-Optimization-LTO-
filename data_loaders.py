import json
import gzip
import requests

def load_humaneval_dataset():
    """Fetches and decompresses the HumanEval dataset from GitHub."""
    print("Downloading HumanEval dataset...")
    url = "https://github.com/openai/human-eval/raw/refs/heads/master/data/HumanEval.jsonl.gz"
    
    response = requests.get(url)
    decompressed = gzip.decompress(response.content)
    
    # Converts the binary text into a readable string, removes whitespace, and loads each string into a dictionary
    dataset = [json.loads(line) for line in decompressed.decode('utf-8').strip().split('\n')]
    
    print(f"Successfully loaded {len(dataset)} HumanEval problems.")
    return dataset

def load_mbpp_dataset():
    """Fetches the MBPP dataset from the Google Research repository."""
    print("Downloading MBPP dataset...")
    url = "https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/mbpp/mbpp.jsonl"
    
    response = requests.get(url)
    
    # Parses the raw text line-by-line into a list of dictionaries
    dataset = [json.loads(line) for line in response.text.strip().split('\n')]
    
    print(f"Successfully loaded {len(dataset)} MBPP problems.")
    return dataset
