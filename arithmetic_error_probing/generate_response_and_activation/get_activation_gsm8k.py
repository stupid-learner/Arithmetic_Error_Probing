import torch
import os
from general_ps_utils import ModelAndTokenizer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import sys
import gc
import re
import json

dataset_index = int(sys.argv[1])
model_name = sys.argv[2]

login(token=os.environ.get('HUGGINGFACE_HUB_TOKEN'))

input_folder = "gsm8k/model_response"
output_folder = f"gsm8k/{model_name.split('/')[-1]}_activation"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load results file
results_file = f"{input_folder}/{model_name.split('/')[-1]}_evaluation_results_gen_all_{dataset_index}.json"
print(f"Loading results from {results_file}")
with open(results_file, 'r') as f:
    results = json.load(f)

# Load model and tokenizer
print(f"Loading model {model_name}")
mt = ModelAndTokenizer(
    model_name=model_name,
    use_4bit=False,
    device='cuda'
)
tokenizer = mt.tokenizer

# Process each variant
for variant_index in range(dataset_index, dataset_index+10):
    print(f"Processing variant {variant_index}")
    dataset = results[str(variant_index)]["results"]
    num_to_hidden = {}
    
    # Print example incorrect equations for debugging
    if variant_index == dataset_index:  # Only print examples for the first variant
        print("\n=== Example Incorrect Equations ===")
        
        # Find examples with incorrect equations
        incorrect_examples = []
        for idx, example in enumerate(dataset):
            if example.get('is_incorrect_equation', False) == True:
                incorrect_examples.append((idx, example))
                if len(incorrect_examples) >= 3:
                    break
        
        # If we don't have enough incorrect examples, add some correct ones
        if len(incorrect_examples) < 3:
            for idx, example in enumerate(dataset):
                if example.get('is_incorrect_equation', False) == False and 'selected_equation' in example and example['selected_equation']:
                    incorrect_examples.append((idx, example))
                    if len(incorrect_examples) >= 3:
                        break
        
        # Print the examples
        for i, (idx, example) in enumerate(incorrect_examples):
            is_incorrect = example.get('is_incorrect_equation', False)
            print(f"Example {i+1} (index {idx}, is_incorrect={is_incorrect}):")
            print(f"  Question: {example['question'][:100]}...")
            print(f"  Response: {example['model_response'][:100]}...")
            print(f"  Equation Count: {example.get('equation_count', 'N/A')}")
            print(f"  Selected Equation Index: {example.get('selected_equation_index', 'N/A')}")
            if 'selected_equation' in example and example['selected_equation']:
                print(f"  Selected Equation: {example['selected_equation']}")
                print(f"  Selected Equation Equal Pos: {example.get('selected_equation_equal_pos', 'N/A')}")
            print()
    
    # Process each data point
    for data_index in tqdm(range(len(dataset))):
        i = dataset[data_index]
        prompt = i["prompt"]
        model_response = i["model_response"]
        
        # Get equation information directly from the results
        equal_pos = i.get("selected_equation_equal_pos", -1)
        is_incorrect = i.get("is_incorrect_equation", False)
        selected_equation = i.get("selected_equation", "")
        
        # Extract left and right sides of the equation if available
        left, right = "", ""
        if selected_equation and '=' in selected_equation:
            parts = selected_equation.split('=', 1)
            left = parts[0].strip()
            right = parts[1].strip() if len(parts) > 1 else ""
        
        # If no equal sign found, skip this data point
        if equal_pos == -1:
            print(f"No equal sign found in response for variant {variant_index}, data {data_index}")
            continue
            
        # Calculate the text up to the equal sign
        prompt_until_equal_sign = prompt + model_response[:equal_pos+1]
        
        # Print details about incorrect equations for the first few examples
        if is_incorrect and data_index < 10 and variant_index == dataset_index:
            print(f"\n=== Processing Incorrect Equation Example {data_index} ===")
            print(f"Selected equation: {left}={right}")
            print(f"Equal sign position: {equal_pos}")
            print(f"Text up to equal sign: ...{prompt_until_equal_sign[-50:]}")
        
        # Encode the text up to the equal sign
        x = tokenizer.encode(prompt_until_equal_sign, return_tensors='pt', add_special_tokens=False)
        x = x.to(mt.model.device)
        
        # Get hidden states at the equal sign position
        hidden_states = mt.model(x, output_hidden_states=True).hidden_states
        
        # Extract the last token's hidden states
        new_hidden_states = tuple(tensor[:, -1:].clone() for tensor in hidden_states)
        
        # Save the hidden states
        num_to_hidden[(variant_index, data_index)] = new_hidden_states
        
        # Clean up memory
        del hidden_states
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save the results for this variant
    print(f"Saving activations for variant {variant_index}")
    torch.save(num_to_hidden, f"{output_folder}/activation_{variant_index}")
    print(f"Saved to {output_folder}/activation_{variant_index}")
    
    # Clear memory before processing next variant
    num_to_hidden = {}
    gc.collect()
    torch.cuda.empty_cache()

print("Processing complete!")