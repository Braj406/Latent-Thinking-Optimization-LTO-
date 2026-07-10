# Latent Thinking Optimization (LTO)

Optimize code generation in Large Language Models by analyzing and classifying latent trajectories. This research repository implements a novel pipeline for extracting hidden state sequences from coding models, evaluating baseline performance using Pass@k metrics, and training a Latent Reward Model (LRM) to improve code correctness via Rejection Sampling. This expands on the research conducted by ninglab.

## Overview

LTO provides an end-to-end pipeline for:
- **Extracting** hidden state trajectories from LLM coding models during the generation process
- **Evaluating** baseline performance on standard coding benchmarks (HumanEval, MBPP)
- **Training** binary classifiers on latent representations to predict code correctness without execution
- **Optimizing** code selection using Rejection Sampling via the trained Latent Reward Model

This approach bridges the gap between model interpretability and practical code generation improvement.

## Repository Structure

```
Latent-Thinking-Optimization-LTO-/
│
├── data_extraction/               # Dataset generation and trajectory extraction
│   ├── dataset_stats.py           # Utility to print summary stats of generated .pt files
│   ├── Deepseek/                  
│   │   ├── HumanEvalDeepseek.py   # Generates trajectories for HumanEval (DeepSeek)
│   │   └── MBPPDeepseek.py        # Generates trajectories for MBPP (DeepSeek)
│   └── Qwen/                      
│       ├── HumanEvalQwen.py       # Generates trajectories for HumanEval (Qwen)
│       └── MBPPQwen.py            # Generates trajectories for MBPP (Qwen)
│
├── models/                        # Model training and baseline evaluation
│   ├── classifiers.py             # PyTorch nn.Modules (LSTM & MLP classifiers)
│   ├── DeepSeek_LRM_Trainer.py    # Training loop for DeepSeek Latent Reward Model
│   ├── pass_k_eval_deepseek.py    # Baseline Pass@k metrics for DeepSeek
│   ├── pass_k_eval_qwen.py        # Baseline Pass@k metrics for Qwen
│   ├── Qwen_LRM_Training.py       # Training loop for Qwen Latent Reward Model
│   └── trainer.py                 # Core LatentClassifierTrainer class
│
├── data_loaders.py                # HumanEval and MBPP dataset fetchers
├── execution_utils.py             # Code extraction and isolated execution
├── lto_algorithms.py              # Rejection Sampling implementation
├── metrics.py                     # Latent space analysis (Entropy, Intrinsic Dimension, Pass@k)
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

## Prerequisites

- Python 3.8 or higher
- GPU support recommended for model training (CUDA 11.0+)
- ~10 GB disk space for trajectory datasets

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Latent-Thinking-Optimization-LTO.git
   cd Latent-Thinking-Optimization-LTO-
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

## Datasets

This pipeline evaluates models on two primary coding benchmarks:

| Dataset | Source | Problems | Language |
|---------|--------|----------|----------|
| **HumanEval** | OpenAI | 164 | Python |
| **MBPP** | Google Research | 974 | Python |

> **Note on Trajectory Data:** Generated hidden states and evaluation tensors are saved as `.pt` files. Due to GitHub's file size limits, these datasets should be stored locally or in cloud storage (Google Drive, AWS S3, etc.).

## Pipeline Overview

### Stage 1: Data Extraction

The scripts in `data_extraction/` generate trajectory datasets by:
1. Prompting models (DeepSeek, Qwen) with multi-shot prompts and step-by-step thinking
2. Extracting intermediate hidden states across all neural layers during "thinking" and "coding" phases
3. Executing generated code in isolated environments to assign correctness labels (1.0 or 0.0)

**Run data extraction:**
```bash
# For HumanEval with Qwen
python data_extraction/Qwen/HumanEvalQwen.py

# For MBPP with DeepSeek
python data_extraction/Deepseek/MBPPDeepseek.py

# View dataset statistics
python data_extraction/dataset_stats.py
```

### Stage 2: Baseline Evaluation

The Pass@k metric establishes performance baselines by measuring how often the correct solution appears in the top k generated samples.

**Run baseline evaluation:**
```bash
# Qwen baseline
python models/pass_k_eval_qwen.py

# DeepSeek baseline
python models/pass_k_eval_deepseek.py
```

### Stage 3: Latent Reward Model (LRM) Training

Train binary classifiers on latent trajectories to predict code correctness without execution.

**Architecture Overview:**
- **Qwen Pipeline:** Global average pooling with MLP classifier
- **DeepSeek Pipeline:** LSTM-based sequence model for temporal feature evolution
  - Handles h0 layer dynamically for precise layer alignment

**Train models:**
```bash
# Qwen LRM
python models/Qwen_LRM_Training.py

# DeepSeek LRM
python models/DeepSeek_LRM_Trainer.py
```

### Stage 4: Latent Thinking Optimization

Apply Acceptance-Rejection Sampling to select high-quality code candidates based on learned latent representations.

**Run LTO optimization:**
```bash
python lto_algorithms.py
```

This stage routes candidate trajectories through the trained LRM, selecting the most promising code and significantly boosting the true Pass@1 metric.

## Usage Example

```python
from data_loaders import fetch_humaneval
from models.classifiers import LSTMLatentClassifier
from lto_algorithms import rejection_sampling

# Load benchmark
problems = fetch_humaneval()

# Generate trajectories and load LRM
lrm = LSTMLatentClassifier(hidden_dim=256)
lrm.load_state_dict(torch.load('checkpoints/deepseek_lrm.pt'))

# Apply LTO optimization
optimized_codes = rejection_sampling(
    candidate_trajectories=trajectories,
    lrm=lrm,
    k=10  # Select from top 10 candidates
)
```

## Key Metrics

The pipeline measures performance using:

- **Pass@k:** Probability that at least one of k samples is correct
- **Entropy:** Uncertainty measure in latent representations
- **Intrinsic Dimension:** Effective dimensionality of latent space
- **Correctness Prediction Accuracy:** LRM classification performance

Detailed metrics are computed in `metrics.py`.

## Configuration

Modify hyperparameters in training scripts:

```python
# Example: Adjust training configuration
CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 50,
    'hidden_dim': 256,
    'dropout': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `.pt` files not found | Run data extraction scripts first or check file paths |
| CUDA out of memory | Reduce batch size or use gradient checkpointing |
| Import errors | Verify all dependencies: `pip install -r requirements.txt` |
| Slow execution | Use GPU for training; CPU inference is acceptable |

## Citation

If you use LTO in your research, please cite:

```bibtex
@misc{lto2024,
  title={Latent Thinking Optimization: Optimizing Code Generation via Latent Reward Models},
  author={Your Name},
  year={2024},
  note={Clark Summer Research Program}
}
```

## Acknowledgements

This research and software pipeline was developed as part of the **Clark Summer Research Program**. We gratefully acknowledge the open-source ML community and the models used (DeepSeek, Qwen). Some parts of the code have been written by ninglab. 

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Last Updated:** July 2024  
**Status:** Active Development
