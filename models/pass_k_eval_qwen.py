import torch
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from google.colab import drive
import sys

# 1. Mount Drive & Add Src to Path
print("Mounting Google Drive...")
drive.mount('/content/drive', force_remount=True)
sys.path.append('/content/drive/MyDrive/Latent Thinking Optimization/Qwen Model')

# 2. Import the metric from your newly refactored repo
from pass_k import calculate_pass_at_k

# 3. Define Paths
humaneval_path = '/content/drive/MyDrive/chumaneval_lto_dataset_qwenpt'
mbpp_path = '/content/drive/MyDrive/cqwenmbpp_lto_dataset_RESCUED.pt'

print("Loading datasets...")
try:
    humaneval_data = torch.load(humaneval_path)
    mbpp_data = torch.load(mbpp_path)
    combined_data = humaneval_data + mbpp_data
    print(f"✓ Combined datasets: {len(humaneval_data)} (HumanEval) + {len(mbpp_data)} (MBPP) = {len(combined_data)} total samples.")
except Exception as e:
    print(f"Error loading data: {e}")
    combined_data = []

if combined_data:
    # EVALUATION HELPER FUNCTION
    def run_evaluation(data_subset, report_title):
        task_results = defaultdict(list)
        for item in data_subset:
            task_results[item['task_id']].append(item['label'].item())

        sample_counts = [len(res) for res in task_results.values()]
        if not sample_counts:
            return
            
        n_samples = max(set(sample_counts), key=sample_counts.count)
        k_values = [k for k in [1, 2, 3, 5, 10] if k <= n_samples]
        if n_samples not in k_values:
            k_values.append(n_samples)

        pass_at_k_scores = {k: [] for k in k_values}

        for task_id, results in task_results.items():
            n = len(results)
            c = int(sum(results))
            if n >= n_samples:
                for k in k_values:
                    pass_at_k_scores[k].append(calculate_pass_at_k(n, c, k))

        # Print Report
        print("\n" + "═" * 60)
        print(f"{report_title} (Evaluated {len(task_results)} tasks)")
        print("═" * 60)
        for k in k_values:
            if pass_at_k_scores[k]:
                print(f"  Pass@{k}: {np.mean(pass_at_k_scores[k]) * 100:.2f}%")
        print("═" * 60)
        
    # RUN 1: MASTER EVALUATION (ALL DATA)
    run_evaluation(combined_data, "MASTER PASS@K METRICS on Qwen Trajectories")


    # RUN 2: TEST SET EVALUATION (15% SPLIT)
    all_task_ids = [str(item['task_id']) for item in combined_data]
    unique_task_ids = sorted(list(set(all_task_ids)))

    # 70% Train, 30% Temp
    train_tasks, temp_tasks = train_test_split(unique_task_ids, test_size=0.3, random_state=42)
    # Split Temp into 15% Val, 15% Test
    val_tasks, test_tasks = train_test_split(temp_tasks, test_size=0.5, random_state=42)

    # Filter for the 15% test set
    test_data = [item for item in combined_data if str(item['task_id']) in test_tasks]
    
    run_evaluation(test_data, "TEST SET PASS@K METRICS")
