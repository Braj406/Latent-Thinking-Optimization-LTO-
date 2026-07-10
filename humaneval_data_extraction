import os
import sys
import random
import torch
import gc
import shutil
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from google.colab import drive

# 1. Mount Drive & Add your scripts folder to the system path
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Update this path to wherever your .py files live in your Drive!
sys.path.append('/content/drive/MyDrive/Latent Thinking Optimization/Qwen Model')

# 2. Import your custom modules
from data_loaders import load_humaneval_dataset
from execution_utils import extract_clean_code, evaluate_humaneval, TimeoutException
from metrics import compute_matrix_entropy, calculate_effective_rank, calculate_anisotropy, calculate_intrinsic_dimension

# 3. Load Model
print("Loading model and tokenizer...")
model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
print("Model Loaded Successfully!")
  
# Fetches the database using your new data_loaders.py function
dataset = load_humaneval_dataset()

system_instruction = (
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

destination_path = '/content/drive/MyDrive/humaneval_lto_dataset_qwenpt'
local_temp_path = '/content/humaneval_lto_temp.pt'

problem_batch_size = 2
samples_per_problem = 3
total_problems = len(dataset)

if os.path.exists(destination_path):
    shutil.copy2(destination_path, local_temp_path)
    lrm_tensor_dataset = torch.load(local_temp_path)
    completed_samples = len(lrm_tensor_dataset)
    start_index = (completed_samples // samples_per_problem)
    start_index = start_index - (start_index % problem_batch_size)
    print(f"\n Found existing dataset! Resuming at Problem {start_index} / {total_problems}.")
else:
    lrm_tensor_dataset = []
    start_index = 0
    print(f"\n Starting fresh HumanEval 2D extraction from Problem 0.")

@torch.inference_mode()
def run_colab_humaneval_extraction():
    for batch_start_idx in tqdm(range(start_index, total_problems, problem_batch_size)):
        current_problems = dataset[batch_start_idx : batch_start_idx + problem_batch_size]
        batch_prompts = []
        batch_metadata = []

        for problem in current_problems:
            task_id = problem["task_id"]
            prompt_text = problem["prompt"]
            valid_examples = [ex for ex in dataset if ex["task_id"] != task_id]
            random_example = random.choice(valid_examples)
            
            formatted_task = (
                f"You are an expert Python engineer. Please complete the function by first thinking step-by-step, then providing the code.\n\n"
                f"--- EXAMPLE ---\nPrompt:\n```python\n{random_example['prompt']}```\n\n"
                f"Solution:\n<think>\nLet's break down the problem and write the body of the function efficiently.\n</think>\n"
                f"```python\n{random_example['prompt']}{random_example['canonical_solution'].strip()}\n```\n"
                f"--- END EXAMPLE ---\n\nNow, solve the following problem:\n\nPrompt:\n```python\n{prompt_text}```\n\n"
                f"CRITICAL Constraints:\n1. You MUST include the original function signature exactly as provided.\n"
                f"2. You must output ONLY valid Python code inside a single ```python block after your thinking phase."
            )

            messages = [{"role": "user", "content": formatted_task}]
            generation_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            generation_prompt += "<think>\n"

            for _ in range(samples_per_problem):
                batch_prompts.append(generation_prompt)
                batch_metadata.append(problem)

        model_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        input_length = model_inputs.input_ids.shape[1]

        generation_outputs = model.generate(
            **model_inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_sequences = generation_outputs.sequences[:, input_length:]
        total_layers = len(generation_outputs.hidden_states[0])

        for batch_offset in range(len(batch_prompts)):
            raw_tokens = generated_sequences[batch_offset]
            raw_string = tokenizer.decode(raw_tokens, skip_special_tokens=False)

            current_meta = batch_metadata[batch_offset]
            task_id = current_meta["task_id"]
            entry_point = current_meta["entry_point"]
            test_string = current_meta["test"]

            # REFACTORED: Utilizing your new execution_utils.py file
            clean_code = extract_clean_code(raw_string, entry_point=entry_point)
            passed_all_tests = evaluate_humaneval(clean_code, test_string, entry_point)

            think_end_idx = 0
            think_token_id = tokenizer.encode("</think>", add_special_tokens=False)
            if len(think_token_id) == 1 and think_token_id[0] in raw_tokens:
                think_end_idx = (raw_tokens == think_token_id[0]).nonzero(as_tuple=True)[0][0].item()
            else:
                if "</think>" in raw_string:
                    think_string = raw_string.split("</think>")[0] + "</think>"
                    think_end_idx = len(tokenizer.encode(think_string, add_special_tokens=False))
                else:
                    think_end_idx = len(raw_tokens) // 2

            actual_seq_len = (raw_tokens != tokenizer.pad_token_id).sum().item()
            think_end_idx = min(think_end_idx, actual_seq_len - 1)

            trajectory_think = []
            trajectory_code = []

            for layer_index in range(total_layers):
                think_tokens = [generation_outputs.hidden_states[step][layer_index][batch_offset, -1, :] for step in range(think_end_idx)]
                if think_tokens:
                    raw_think_matrix = torch.stack(think_tokens)
                    norm_think = model.model.norm(raw_think_matrix)
                    trajectory_think.append(norm_think.mean(dim=0).detach().cpu().numpy())

                code_tokens = [generation_outputs.hidden_states[step][layer_index][batch_offset, -1, :] for step in range(think_end_idx, actual_seq_len)]
                if code_tokens:
                    raw_code_matrix = torch.stack(code_tokens)
                    norm_code = model.model.norm(raw_code_matrix)
                    trajectory_code.append(norm_code.mean(dim=0).detach().cpu().numpy())

            binary_label_flag = 1.0 if passed_all_tests else 0.0

            lrm_tensor_dataset.append({
                "hidden_states_think": torch.tensor(np.array(trajectory_think), dtype=torch.float32) if trajectory_think else torch.empty(0),
                "hidden_states_code": torch.tensor(np.array(trajectory_code), dtype=torch.float32) if trajectory_code else torch.empty(0),
                "label": torch.tensor(binary_label_flag, dtype=torch.float32),
                "task_id": task_id
            })

        del generation_outputs
        del model_inputs
        del generated_sequences
        torch.cuda.empty_cache()
        gc.collect()

        torch.save(lrm_tensor_dataset, local_temp_path)
        shutil.copy2(local_temp_path, destination_path)

if __name__ == "__main__":
    run_colab_humaneval_extraction()
    print(f"\n Inference complete! Saved trajectories permanently to: {destination_path}")
