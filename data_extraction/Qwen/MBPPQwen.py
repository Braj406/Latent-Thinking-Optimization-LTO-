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

print("Mounting Google Drive...")
drive.mount('/content/drive', force_remount=True)

# Update this path to wherever your .py files live in your Drive!
sys.path.append('/content/drive/MyDrive/Latent Thinking Optimization/Qwen Model')

from data_loaders import load_mbpp_dataset
from execution_utils import extract_clean_code, evaluate_mbpp, TimeoutException
from metrics import compute_matrix_entropy, calculate_effective_rank, calculate_anisotropy, calculate_intrinsic_dimension

print("Loading model and tokenizer...")
model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to("cuda")
print("Model Loaded Successfully!")

# Fetches the database using your new data_loaders.py function
mbpp_dataset = load_mbpp_dataset()
total_problems = len(mbpp_dataset)

destination_path = '/content/drive/MyDrive/qwenmbpp_lto_dataset.pt'
local_temp_path = '/content/qwen_lto_temp.pt'

problem_batch_size = 3
samples_per_problem = 3

if os.path.exists(destination_path):
    shutil.copy2(destination_path, local_temp_path)
    lrm_tensor_dataset = torch.load(local_temp_path)
    completed_samples = len(lrm_tensor_dataset)
    start_index = (completed_samples // samples_per_problem)
    start_index = start_index - (start_index % problem_batch_size)
    print(f"\nFound existing dataset! Resuming at Problem {start_index} / {total_problems}.")
else:
    lrm_tensor_dataset = []
    start_index = 0
    print(f"\nStarting fresh extraction from Problem 0.")

@torch.inference_mode()
def run_colab_extraction():
    for batch_start_idx in tqdm(range(start_index, total_problems, problem_batch_size)):
        current_problems = mbpp_dataset[batch_start_idx : batch_start_idx + problem_batch_size]
        batch_prompts = []
        batch_metadata = [] 

        for current_problem in current_problems:
            task_id = current_problem["task_id"]
            original_text = current_problem["text"]
            hidden_tests = current_problem["test_list"]

            if not hidden_tests:
                continue

            guiding_test = hidden_tests[0]
            valid_examples = [ex for ex in mbpp_dataset if ex["task_id"] != task_id]

            for _ in range(samples_per_problem):
                random_example = random.choice(valid_examples)
                
                formatted_task = (
                    f"You are an expert Python engineer. Please solve the problem by first thinking step-by-step, then providing the code.\n\n"
                    f"--- EXAMPLE ---\nProblem: {random_example['text']}\n\nSolution:\n<think>\nLet's break down the problem and write a highly optimized Python function to solve it efficiently.\n</think>\n"
                    f"```python\n{random_example['code'].strip()}\n```\n--- END EXAMPLE ---\n\n"
                    f"Now, solve the following problem:\n\nProblem: {original_text}\nRequired Target Test: {guiding_test}\n\n"
                    f"CRITICAL Constraints:\n1. You MUST name your function exactly as it appears in the Required Target Test.\n"
                    f"2. You MUST accept the exact number of arguments shown in the test.\n3. Your logic MUST result in the exact output shown in the test.\n"
                    f"4. You must output ONLY valid Python code inside a single ```python block after your thinking phase."
                )

                messages = [{"role": "user", "content": formatted_task}]
                generation_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                generation_prompt += "<think>\n"

                batch_prompts.append(generation_prompt)
                batch_metadata.append({
                    "task_id": task_id,
                    "guiding_test": guiding_test,
                    "hidden_tests": hidden_tests
                })

        if not batch_prompts:
            continue

        model_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        input_length = model_inputs.input_ids.shape[1]

        generation_outputs = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_sequences = generation_outputs.sequences[:, input_length:]
        total_layers = len(generation_outputs.hidden_states[0])
        
        first_step_shape = generation_outputs.hidden_states[0][0].shape
        gen_step_offset = 1 if first_step_shape[1] > 1 else 0
        max_recorded_steps = len(generation_outputs.hidden_states) - gen_step_offset

        for batch_offset in range(len(batch_prompts)):
            meta = batch_metadata[batch_offset]
            raw_tokens = generated_sequences[batch_offset]
            raw_string = tokenizer.decode(raw_tokens, skip_special_tokens=False)

            # REFACTORED: Utilizing your new execution_utils.py file
            clean_code = extract_clean_code(raw_string)
            passed_all_tests = evaluate_mbpp(clean_code, meta["hidden_tests"], meta["guiding_test"])

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
            safe_actual_seq_len = min(actual_seq_len, max_recorded_steps)
            safe_think_end_idx = min(think_end_idx, max_recorded_steps)

            trajectory_think = []
            trajectory_code = []

            for layer_index in range(total_layers):
                think_tokens = [generation_outputs.hidden_states[step + gen_step_offset][layer_index][batch_offset, -1, :] for step in range(safe_think_end_idx)]
                if think_tokens:
                    raw_think_matrix = torch.stack(think_tokens)
                    norm_think = model.model.norm(raw_think_matrix)
                    trajectory_think.append(norm_think.mean(dim=0).detach().cpu().numpy())

                code_tokens = [generation_outputs.hidden_states[step + gen_step_offset][layer_index][batch_offset, -1, :] for step in range(safe_think_end_idx, safe_actual_seq_len)]
                if code_tokens:
                    raw_code_matrix = torch.stack(code_tokens)
                    norm_code = model.model.norm(raw_code_matrix)
                    trajectory_code.append(norm_code.mean(dim=0).detach().cpu().numpy())

            binary_label_flag = 1.0 if passed_all_tests else 0.0

            lrm_tensor_dataset.append({
                "hidden_states_think": torch.tensor(np.array(trajectory_think), dtype=torch.float32) if trajectory_think else torch.empty(0),
                "hidden_states_code": torch.tensor(np.array(trajectory_code), dtype=torch.float32) if trajectory_code else torch.empty(0),
                "label": torch.tensor(binary_label_flag, dtype=torch.float32),
                "task_id": meta["task_id"]
            })

        del generation_outputs
        del model_inputs
        del generated_sequences
        torch.cuda.empty_cache()
        gc.collect()

        torch.save(lrm_tensor_dataset, local_temp_path)
        shutil.copy2(local_temp_path, destination_path)

if __name__ == "__main__":
    run_colab_extraction()
    print(f"\nInference complete! Saved trajectories permanently to: {destination_path}")
