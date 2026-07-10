import torch
import os

def print_dataset_stats(dataset_path):
    """
    Loads a generic .pt dataset and prints summary statistics.
    Accepts the exact path to the dataset file.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Could not find the file at '{dataset_path}'.")
        print("Check if the script has saved its first checkpoint yet!")
        return

    try:
        # Load the current dataset
        dataset = torch.load(dataset_path)

        total_trajectories = len(dataset)
        unique_problems = len(set(sample['task_id'] for sample in dataset))

        # Count labels
        correct_count = sum(1 for sample in dataset if sample['label'].item() == 1.0)
        incorrect_count = sum(1 for sample in dataset if sample['label'].item() == 0.0)

        # Calculate success rate
        success_rate = (correct_count / total_trajectories * 100) if total_trajectories > 0 else 0

        print("=" * 50)
        print(f"DATASET STATS: {os.path.basename(dataset_path)}")
        print("=" * 50)
        print(f"Unique Problems Processed: {unique_problems}")
        print(f"Total Trajectories Evaluated: {total_trajectories}")
        print("-" * 50)
        print(f"Correct Trajectories (Label 1.0):   {correct_count}")
        print(f"Incorrect Trajectories (Label 0.0): {incorrect_count}")
        print(f"Overall Model Success Rate:        {success_rate:.2f}%")
        print("=" * 50)

    except Exception as e:
        print(f"An error occurred while evaluating {dataset_path}: {e}")

# Example Usage:
# print_dataset_stats('/content/drive/MyDrive/cdeepseekmbpp_lto_dataset.pt')
# print_dataset_stats('/content/drive/MyDrive/chumaneval_lto_dataset_deepseek.pt')
