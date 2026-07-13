"""
LLM Evaluation with Latent Quality Metrics Visualization

This script evaluates LLM coding solutions across the HumanEval dataset,
tracks latent space metrics (entropy, effective rank, anisotropy, intrinsic dimension),
and generates visualizations comparing correct vs incorrect trajectories.
"""

import math
import torch
import random
import gc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_loaders import load_humaneval_dataset
from execution_utils import extract_clean_code, evaluate_humaneval
from metrics import (
    compute_matrix_entropy,
    calculate_effective_rank,
    calculate_anisotropy,
    calculate_intrinsic_dimension
)


class LLMEvaluationGrapher:
    """Evaluates LLM code generation and generates latent quality metric graphs."""

    def __init__(self, model_name="Qwen/Qwen2.5-Coder-3B-Instruct", device="cuda"):
        """Initialize model, tokenizer, and data structures."""
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.device = device
        print("Model loaded successfully!")

        self.dataset = load_humaneval_dataset()

        self.system_instruction = (
            "You are an expert Python software engineer. Your task is to write complete, optimized, "
            "and functional Python code that resolves the specific problem described by the user. "
            "Analyze the provided coding context, constraints, and requirements carefully. "
            "You must output ONLY valid, executable Python source code. "
            "Do not include any conversational filler, introductory remarks, or concluding explanations. "
            "Do not include any inline comments, block comments, or hashtags (#) within the code code block. "
            "Do not wrap your response in markdown code blocks or backticks. "
            "Do not write any comments. "
            "Ensure your code strictly adheres to the provided function signatures and can be dynamically executed as-is."
        )

        # Metric tracking
        self.global_metric_tracker = {
            "correct": {"entropy": [], "erank": [], "anisotropy": [], "id": []},
            "incorrect": {"entropy": [], "erank": [], "anisotropy": [], "id": []}
        }

        # Trajectory storage
        self.saved_3d_trajectories = {}

        # Prompt templates for diverse context generation
        self.rephrase_templates = [
            # Academic & Algorithmic Complexity Focus
            """Task ID: {task_id}

You are competing in the final round of an elite algorithmic programming contest.

Assume the hidden test suite is specifically designed to break naive solutions through adversarial inputs, pathological edge cases, and worst-case complexity scenarios.

Prioritize:
- asymptotic efficiency
- elimination of redundant work
- concise implementation
- direct algorithmic insight

Avoid over-engineering and abstraction layers.

Problem:

{instruction}

Return only executable Python source code.""",

            # Enterprise Production & System Integration Focus
            """Task ID: {task_id}

Your task is not merely to solve the problem.

Your primary objective is to infer the hidden tests that the evaluator is most likely using and construct a solution that survives them.

Before committing to an implementation, internally search for:
- off-by-one failures
- empty inputs
- duplicate values
- boundary conditions
- unusual ordering
- degenerate structures

Treat every specification ambiguity as a potential hidden test.

Problem:

{instruction}

Output only the final Python source code.""",

            # Formal Mathematical & Logic-Driven Focus
            """Task ID: {task_id}

You are a Python implementation specialist.

Favor Python-native reasoning rather than language-agnostic algorithm design.

Leverage:
- iterator behavior
- slicing semantics
- dictionary guarantees
- set operations
- generator expressions
- standard-library primitives

Seek the most natural Python solution rather than the most textbook algorithm.

Problem:

{instruction}

Return only raw Python code."""
        ]

    def evaluate_single_problem(self, problem_index, samples_per_problem=3):
        """Evaluate a single problem with multiple semantic contexts."""
        current_problem = self.dataset[problem_index]
        task_identifier = current_problem["task_id"]
        task_instruction = current_problem["prompt"]
        hidden_test_cases = current_problem["test"]
        target_function_name = current_problem["entry_point"]

        if problem_index not in self.saved_3d_trajectories:
            self.saved_3d_trajectories[problem_index] = {}

        # Generate samples with different contextual framings
        for sample_idx in range(samples_per_problem):
            current_template = self.rephrase_templates[sample_idx % len(self.rephrase_templates)]

            # Format the prompt
            formatted_task_request = current_template.format(
                task_id=task_identifier,
                instruction=task_instruction
            )

            # Build conversation
            conversation_messages = [
                {"role": "system", "content": self.system_instruction},
                {"role": "user", "content": formatted_task_request}
            ]

            # Tokenize and generate
            templated_prompt = self.tokenizer.apply_chat_template(
                conversation_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer(templated_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generation_outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=250,
                    temperature=0.7,
                    do_sample=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )

            # Decode generated code
            generated_code_string = self.tokenizer.batch_decode(
                generation_outputs.sequences,
                skip_special_tokens=True
            )[0]

            # Extract clean code
            generated_code_string = extract_clean_code(generated_code_string, target_function_name)

            # Evaluate
            passed_all_tests = evaluate_humaneval(
                generated_code_string,
                hidden_test_cases,
                target_function_name
            )

            total_model_layers = len(generation_outputs.hidden_states[0])
            trajectory_status_key = "correct" if passed_all_tests else "incorrect"

            self.saved_3d_trajectories[problem_index][sample_idx] = {
                "status": trajectory_status_key,
                "path": []
            }

            # Initialize metric storage if empty
            if not self.global_metric_tracker[trajectory_status_key]["entropy"]:
                for metric_name in self.global_metric_tracker[trajectory_status_key]:
                    self.global_metric_tracker[trajectory_status_key][metric_name] = (
                        [[] for _ in range(total_model_layers)]
                    )

            # Extract metrics from each layer
            for layer_index in range(total_model_layers):
                tokens_at_layer = [
                    generation_outputs.hidden_states[step_index][layer_index][0, -1, :]
                    for step_index in range(len(generation_outputs.hidden_states))
                ]
                raw_layer_representation_matrix = torch.stack(tokens_at_layer)

                # Normalize
                normalized_layer_representation_matrix = self.model.model.norm(raw_layer_representation_matrix)

                # Compress to 1D center
                layer_semantic_center_vector = (
                    normalized_layer_representation_matrix.mean(dim=0).detach().cpu().numpy()
                )
                self.saved_3d_trajectories[problem_index][sample_idx]["path"].append(
                    layer_semantic_center_vector
                )

                # Calculate metrics
                self.global_metric_tracker[trajectory_status_key]["entropy"][layer_index].append(
                    compute_matrix_entropy(normalized_layer_representation_matrix)
                )
                self.global_metric_tracker[trajectory_status_key]["erank"][layer_index].append(
                    calculate_effective_rank(normalized_layer_representation_matrix)
                )
                self.global_metric_tracker[trajectory_status_key]["anisotropy"][layer_index].append(
                    calculate_anisotropy(normalized_layer_representation_matrix)
                )
                self.global_metric_tracker[trajectory_status_key]["id"][layer_index].append(
                    calculate_intrinsic_dimension(normalized_layer_representation_matrix)
                )

            # Cleanup memory
            del generation_outputs, tokens_at_layer, raw_layer_representation_matrix, normalized_layer_representation_matrix
            torch.cuda.empty_cache()
            gc.collect()

    def run_evaluation(self, total_problems=10, samples_per_problem=3):
        """Run full evaluation across problems."""
        print(f"Starting inference across {total_problems} problems with {samples_per_problem} samples each...")

        random_problem_indices = random.sample(range(len(self.dataset)), min(total_problems, len(self.dataset)))

        for problem_index in tqdm(random_problem_indices):
            self.evaluate_single_problem(problem_index, samples_per_problem)

        print("Inference and metric collection complete!")

    def compute_averaged_metrics(self):
        """Compute layer-wise averaged metrics."""
        averaged_metrics_for_plotting = {
            "correct": {"entropy": [], "erank": [], "anisotropy": [], "id": []},
            "incorrect": {"entropy": [], "erank": [], "anisotropy": [], "id": []}
        }

        total_model_layers = (
            len(self.global_metric_tracker["correct"]["entropy"])
            if self.global_metric_tracker["correct"]["entropy"]
            else len(self.global_metric_tracker["incorrect"]["entropy"])
        )

        for trajectory_status_key in ["correct", "incorrect"]:
            for metric_name in ["entropy", "erank", "anisotropy", "id"]:
                for layer_index in range(total_model_layers):
                    collected_values = self.global_metric_tracker[trajectory_status_key][metric_name][layer_index]

                    if len(collected_values) > 0:
                        averaged_metrics_for_plotting[trajectory_status_key][metric_name].append(
                            np.mean(collected_values)
                        )
                    else:
                        averaged_metrics_for_plotting[trajectory_status_key][metric_name].append(None)

        return averaged_metrics_for_plotting, total_model_layers

    def generate_graphs(self, output_path="llm_latent_quality_metrics.png"):
        """Generate 4-panel metric visualization."""
        averaged_metrics_for_plotting, total_model_layers = self.compute_averaged_metrics()

        layers_x_axis = range(1, total_model_layers + 1)

        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

        fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))

        color_correct_path, color_incorrect_path = "#4A90E2", "#D0021B"
        metrics_keys = ["entropy", "erank", "anisotropy", "id"]
        graph_titles = ["Entropy", "Effective Rank", "Anisotropy", "Intrinsic Dimension"]

        for index, current_metric_key in enumerate(metrics_keys):
            if any(value is not None for value in averaged_metrics_for_plotting["correct"][current_metric_key]):
                axes[index].plot(
                    layers_x_axis,
                    averaged_metrics_for_plotting["correct"][current_metric_key],
                    color=color_correct_path,
                    marker='o',
                    linewidth=2,
                    markersize=5,
                    label="Correct Trajectory"
                )

            if any(value is not None for value in averaged_metrics_for_plotting["incorrect"][current_metric_key]):
                axes[index].plot(
                    layers_x_axis,
                    averaged_metrics_for_plotting["incorrect"][current_metric_key],
                    color=color_incorrect_path,
                    marker='s',
                    linewidth=2,
                    markersize=5,
                    label="Incorrect Trajectory"
                )

            axes[index].set_ylabel(graph_titles[index], fontsize=13, fontweight='bold')
            axes[index].set_xlabel("Model Layers", fontsize=11)
            axes[index].set_xticks(range(1, total_model_layers + 1, max(1, total_model_layers // 5)))
            axes[index].tick_params(labelsize=10)
            axes[index].grid(True, linestyle='--', alpha=0.6)

        # Extract legend from first subplot with handles
        legend_handles, legend_labels = [], []
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend_handles, legend_labels = handles, labels
                break

        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=2,
            fontsize=12,
            frameon=True
        )

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Graphs saved to {output_path}")
        plt.show()


if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
    TOTAL_PROBLEMS = 10
    SAMPLES_PER_PROBLEM = 3
    OUTPUT_PATH = "llm_latent_quality_metrics.png"

    # Run evaluation
    grapher = LLMEvaluationGrapher(model_name=MODEL_NAME)
    grapher.run_evaluation(total_problems=TOTAL_PROBLEMS, samples_per_problem=SAMPLES_PER_PROBLEM)
    grapher.generate_graphs(output_path=OUTPUT_PATH)
