import torch
import numpy as np
from sklearn.model_selection import train_test_split
from google.colab import drive
import sys

# 1. Mount Drive & Add Repo to Path
print("Mounting Google Drive...")
drive.mount('/content/drive', force_remount=True)
sys.path.append('/content/drive/MyDrive/Latent-Thinking-Optimization-LTO-')

# 2. Import from your modular repository
from training.classifiers import DeepSeekLatentClassifier
from training.trainer import LatentClassifierTrainer
from lto_algorithms import conduct_rejection_sampling

# 3. Load Trajectory Data
print("Loading trajectory data...")
humaneval_file = '/content/drive/MyDrive/chumaneval_lto_dataset.pt'
mbpp_file = '/content/drive/MyDrive/cdeepseek_lto_dataset.pt'

humaneval_data = torch.load(humaneval_file)
mbpp_data = torch.load(mbpp_file)
all_data = humaneval_data + mbpp_data
print(f"✓ Total samples: {len(all_data)}")

# 4. Format for Classifier
trajectories_list, labels_list = [], []
for item in all_data:
    combined_trajectory = (item['hidden_states_think'] + item['hidden_states_code']) / 2 
    trajectories_list.append(combined_trajectory)
    labels_list.append(item['label'].item() if isinstance(item['label'], torch.Tensor) else item['label'])

trajectories = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32) for t in trajectories_list])
labels = torch.tensor(labels_list, dtype=torch.float32)

# 5. Split Data (70/15/15)
indices = np.arange(len(labels))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels.numpy())
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx].numpy())

train_tensors, train_labels = trajectories[train_idx], labels[train_idx]
val_tensors, val_labels = trajectories[val_idx], labels[val_idx]
test_tensors, test_labels = trajectories[test_idx], labels[test_idx]

# 6. Initialize & Train DeepSeek LRM
device = 'cuda' if torch.cuda.is_available() else 'cpu'
detected_num_layers = train_tensors.shape[1]
detected_hidden_dim = train_tensors.shape[2]

model = DeepSeekLatentClassifier(hidden_dim=detected_hidden_dim, num_layers=detected_num_layers, dropout=0.3)
trainer = LatentClassifierTrainer(model, device=device)

trainer.train(
    train_tensors, train_labels,
    val_tensors, val_labels,
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-3
)

# 7. Evaluate
results = trainer.evaluate(test_tensors, test_labels)

# (Optional) Save your trained model weights here
