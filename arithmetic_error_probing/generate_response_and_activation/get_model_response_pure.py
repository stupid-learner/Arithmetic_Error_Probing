from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import csv
import os
from huggingface_hub import login
import sys
import random
from tqdm import tqdm

n_shots = 2
batch_size = 8  # Process in batches for better GPU utilization
arithmetic_mode = "sum"
if arithmetic_mode == "sum":
    arithmetic_operator = "+"
    arithmetic_function = lambda a,b: a+b
elif arithmetic_mode == "difference":
    arithmetic_operator = "-"
    arithmetic_function = lambda a,b: a-b
elif arithmetic_mode == "product":
    arithmetic_operator = "*"
    arithmetic_function = lambda a,b: a*b


login(token=os.environ.get('HUGGINGFACE_HUB_TOKEN'))

model_name = "microsoft/Phi-3-mini-4k-instruct"
#model_name = "Qwen/Qwen3-1.7B"
#model_name = "meta-llama/Llama-3.2-3B-Instruct"
#model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
#model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id


def create_question(i, j):
    return f"""Calculate the {arithmetic_mode} of the following two numbers:
  First number: {i}
  Second number: {j}"""


def create_shot(i, j, index):
    system_message = f"You are a helpful assistant that calculates the {arithmetic_mode} of two numbers. Always provide your answer in the format <<x{arithmetic_operator}y=z>> where x is the first number, y is the second number, and z is their {arithmetic_mode}. Do not provide any additional explanation."
    message = []
    if index == 0:
        message.append({"role": "user", "content": system_message + "\n" + create_question(i, j)})
    else:
        message.append({"role": "user", "content": create_question(i, j)})
    message.append({"role": "assistant", "content": f"<<{i}{arithmetic_operator}{j}={arithmetic_function(i,j)}>>"})
    return message


correct_answer = 0
not_number = 0
data_list = []
first_example_printed = False  # Flag to track if we've already printed the first example

range_low = int(sys.argv[1])
range_high = int(sys.argv[2])

# Generate operands list for few-shot examples
operands_list = []
for i in range(100, 1000):
    for j in range(100, 1000):
        operands_list.append((i, j))

random.seed(42)
random.shuffle(operands_list)

# Generate all test pairs
all_test_pairs = []
for i in range(range_low, range_high):
    for j in range(100, 1000):
        all_test_pairs.append((i, j))

print(f"Total tests: {len(all_test_pairs)}")

# Process in batches with tqdm showing overall progress
for batch_start in tqdm(range(0, len(all_test_pairs), batch_size), desc="Processing batches"):
    batch = all_test_pairs[batch_start:batch_start + batch_size]
    batch_prompts = []
    batch_prompt_lengths = []
    batch_true_answers = []
    batch_i_j_pairs = []
    
    # Prepare all prompts in the batch
    for i, j in batch:
        messages = []
        shots_index = 0
        while len(messages) < 2*n_shots:
            if (i, j) != operands_list[shots_index]:
                messages += create_shot(operands_list[shots_index][0], operands_list[shots_index][1], len(messages))
            shots_index += 1
        system_message = f"You are a helpful assistant that calculates the {arithmetic_mode} of two numbers. Always provide your answer in the format <<x{arithmetic_operator}y=z>> where x is the first number, y is the second number, and z is their {arithmetic_mode}. Do not provide any additional explanation."
        if n_shots != 0: 
            messages.append({"role": "user", "content": create_question(i, j)})
        else:
            messages.append({"role": "user", "content": system_message + "\n" + create_question(i, j)})
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        ) + "<<"
        
        # Print the first prompt for debugging
        if not first_example_printed and len(batch_prompts) == 0:
            print("\n" + "="*50)
            print("FIRST EXAMPLE INPUT:")
            print("="*50)
            print(prompt)
            print("="*50 + "\n")
        
        # Record prompt length for later output slicing
        prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_length = prompt_tokens.input_ids.shape[1]
        
        batch_prompts.append(prompt)
        batch_prompt_lengths.append(prompt_length)
        batch_true_answers.append(i + j)
        batch_i_j_pairs.append((i, j))
    
    # Tokenize the batch
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")
    
    # Generate outputs for the entire batch
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=20 + len(str(max(batch_true_answers))),
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Process each output
    for idx, (output_ids, prompt_length, (i, j), true_answer) in enumerate(
            zip(outputs, batch_prompt_lengths, batch_i_j_pairs, batch_true_answers)):
        # Get the model's generated part
        full_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        output_text = tokenizer.decode(output_ids[prompt_length:], skip_special_tokens=False)
        
        # Print the first output for debugging
        if not first_example_printed and idx == 0:
            print("FIRST EXAMPLE FULL OUTPUT:")
            print("="*50)
            print(full_output)
            print("="*50)
            print("\nGENERATED PART (after prompt):")
            print("="*50)
            print(output_text)
            print("="*50)
            print("\nEXPECTED ANSWER:", true_answer)
            print("="*50 + "\n")
            first_example_printed = True
        
        data_list.append((i, j, output_text))
        
        # Check if the answer is correct
        try:
            # Using the same indexing logic as the original code
            extracted_answer = output_text[5:].split("\n")[0]  # Skip the first 5 chars and take until newline
            if int(extracted_answer) == true_answer:
                correct_answer += 1
        except:
            not_number += 1

# Save results
folder = model_name.split("/")[-1] + f"_{n_shots}_shots_3_digit_{arithmetic_mode}_output"
os.makedirs(folder, exist_ok=True)

file_name = f'data_{range_low}_to_{range_high}'
with open(f"{folder}/{file_name}", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data_list)

print(f'Data saved to {file_name}')
print(f"Correct answers: {correct_answer}")
print(f"Not an integer: {not_number}")