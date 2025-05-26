import json
import re
import random
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from huggingface_hub import login
import os
import sys
import csv
from utils import *



start_index = int(sys.argv[1])

model_name = sys.argv[2]

mode = sys.argv[3] # random or wrong data only

folder = f"{model_name.split('/')[-1]}_2_shots_3_digit_sum_output"

result = load_model_result_dic(folder)
correct_prompt = []
wrong_prompt = []
for i in result:
    if int(i[0])+int(i[1])-int(result[i]) == 0:
        correct_prompt.append((int(i[0]), int(i[1])))
    else:
        wrong_prompt.append((int(i[0]), int(i[1])))

all_prompt_below_1000 = []
for i in range(100, 1000):
    for j in range(100, 1000):
        if i + j < 1000:
            all_prompt_below_1000.append((i,j))

def count_steps(sample):
    steps = sample['intermediate_steps']
    equations = re.findall(r'<<(.+?)>>', steps)
    return len(equations)


def is_valid_equation(equation):
    if any(op in equation for op in ['-', '*', '/']):
        return False

    left_side = equation.split('=')[0].strip()

    plus_count = left_side.count('+')

    if plus_count > 1:
        return False

    return True

def has_colon(equations):
    """
    Check if equations contain colons.

    Args:
        equations (list or str): Either a list of equation strings or a single string with equations

    Returns:
        bool: True if any equation contains a colon, False otherwise
    """
    # If input is a single string, convert to list
    if isinstance(equations, str):
        # Check if it contains intermediate steps in the <<>> format
        import re
        eq_matches = re.findall(r'<<(.*?)>>', equations)
        if eq_matches:
            equations = eq_matches
        else:
            # Treat as a single equation
            equations = [equations]

    # Check each equation for colons
    for eq in equations:
        if ':' in eq:
            return True

    return False

def is_valid_sample(sample):
    steps = sample['intermediate_steps']

    equations = re.findall(r'<<(.+?)>>', steps)

    for eq in equations:
        if not is_valid_equation(eq):
            return False

    return True and not has_colon(equations)




# Function to format a single example for few-shot prompting
def format_shot(example):
    question = example.get('question')

    # Extract equations from intermediate_steps using the <<>> format
    # We want to extract the equations AFTER variables have been substituted
    # NOT from original_intermediate_steps which has variables like x1, x2
    intermediate_steps = example['intermediate_steps']  # Use the version with substituted variables
    equations = re.findall(r'<<(.*?)>>', intermediate_steps)

    # Format the answer
    final_answer = str(example['answer'])

    # Format the shot with one equation per line and final answer
    formatted_equations = "<<" + ">>\n<<".join(equations) + ">>"
    formatted_shot = f"{formatted_equations}\nfinal answer: [{final_answer}]"

    return formatted_shot

# Function to filter out examples with time-related calculations
def filter_time_examples(data):
    filtered_data = []
    time_keywords = ['hour', 'minute', 'second', 'day', 'week', 'month', 'year', 'time', 'clock', 'a.m.', 'p.m.', 'AM', 'PM']

    for example in data:
        # Check if any time-related keywords are in the question
        has_time = any(keyword in example['question'].lower() for keyword in time_keywords)

        # Also check if time operations are in the equations
        if not has_time and 'intermediate_steps' in example:
            equations = re.findall(r'<<(.*?)>>', example['intermediate_steps'])
            for eq in equations:
                # Check for obvious time calculations in equations
                if any(keyword in eq.lower() for keyword in time_keywords):
                    has_time = True
                    break

        if not has_time:
            filtered_data.append(example)

    print(f"Filtered out {len(data) - len(filtered_data)} time-related examples out of {len(data)} total examples")
    return filtered_data

# Function to create variants of the dataset with different digit replacements
def create_digit_variants(data):
    # First, filter out time-related examples
    filtered_data = [example for example in data if not has_colon(example.get('intermediate_steps', ''))]

    print(f"Filtered out {len(data) - len(filtered_data)} examples with colons out of {len(data)} total examples")

    variants = {
        "one_digit": [],
        "two_digit": [],
        "three_digit": []
    }

    # Process each example
    for example in filtered_data:
        #print(f"\nProcessing example: {example['question'][:50]}...")

        # === ONE DIGIT VARIANT (1-9) ===
        one_digit = example.copy()
        one_digit['original_question'] = one_digit['question']
        one_digit['original_intermediate_steps'] = one_digit['intermediate_steps']

        # Get x variables and generate random values (1-9)
        x_vars_one = {k: v for k, v in one_digit['variables'].items() if k.startswith('x')}
        new_values_one = {k: random.randint(1, 9) for k in x_vars_one}

        # Deep copy and update variables dictionary
        one_digit_vars = one_digit['variables'].copy()
        one_digit_vars.update(new_values_one)
        one_digit['variables'] = one_digit_vars

        # Replace variables in question text
        one_digit_question = one_digit['question']
        one_digit_interm = one_digit['intermediate_steps']

        for var_name, var_value in new_values_one.items():
            one_digit_question = one_digit_question.replace(var_name, str(var_value))
            one_digit_interm = one_digit_interm.replace(var_name, str(var_value))

        one_digit['question'] = one_digit_question
        one_digit['intermediate_steps'] = one_digit_interm

        # Recalculate y variables
        equations_one = re.findall(r'<<(.*?)>>', one_digit['original_intermediate_steps'])
        namespace_one = {k: v for k, v in one_digit['variables'].items() if k.startswith('x')}

        for eq in equations_one:
            left, right = eq.split('=')
            result = eval(left, {"__builtins__": {}}, namespace_one)
            namespace_one[right] = result
            one_digit['variables'][right] = result

        # Update answer
        for var_name, var_value in one_digit['variables'].items():
            one_digit_interm = one_digit_interm.replace(var_name, str(var_value))

        one_digit['intermediate_steps'] = one_digit_interm

        last_y_one = equations_one[-1].split('=')[1]
        one_digit['answer'] = namespace_one[last_y_one]

        # Print x variables for verification
        one_digit_x_vars = {k: v for k, v in one_digit['variables'].items() if k.startswith('x')}
        #print(f"One-digit variant x variables: {one_digit_x_vars}")

        # Add to variants list
        variants["one_digit"].append(one_digit)

        # === TWO DIGIT VARIANT (10-99) ===
        two_digit = example.copy()
        two_digit['original_question'] = two_digit['question']
        two_digit['original_intermediate_steps'] = two_digit['intermediate_steps']

        # Get x variables and generate random values (10-99)
        x_vars_two = {k: v for k, v in two_digit['variables'].items() if k.startswith('x')}
        new_values_two = {k: random.randint(10, 99) for k in x_vars_two}

        # Deep copy and update variables dictionary
        two_digit_vars = two_digit['variables'].copy()
        two_digit_vars.update(new_values_two)
        two_digit['variables'] = two_digit_vars

        # Replace variables in question text
        two_digit_question = two_digit['question']
        two_digit_interm = two_digit['intermediate_steps']

        for var_name, var_value in new_values_two.items():
            two_digit_question = two_digit_question.replace(var_name, str(var_value))
            two_digit_interm = two_digit_interm.replace(var_name, str(var_value))

        two_digit['question'] = two_digit_question
        two_digit['intermediate_steps'] = two_digit_interm

        # Recalculate y variables
        equations_two = re.findall(r'<<(.*?)>>', two_digit['original_intermediate_steps'])
        namespace_two = {k: v for k, v in two_digit['variables'].items() if k.startswith('x')}

        for eq in equations_two:
            left, right = eq.split('=')
            result = eval(left, {"__builtins__": {}}, namespace_two)
            namespace_two[right] = result
            two_digit['variables'][right] = result

        # Update answer
        for var_name, var_value in two_digit['variables'].items():
            two_digit_interm = two_digit_interm.replace(var_name, str(var_value))

        two_digit['intermediate_steps'] = two_digit_interm

        last_y_two = equations_two[-1].split('=')[1]
        two_digit['answer'] = namespace_two[last_y_two]

        # Print x variables for verification
        two_digit_x_vars = {k: v for k, v in two_digit['variables'].items() if k.startswith('x')}
        #print(f"Two-digit variant x variables: {two_digit_x_vars}")

        # Add to variants list
        variants["two_digit"].append(two_digit)

        # === THREE DIGIT VARIANT (100-999) ===
        three_digit = example.copy()
        three_digit['original_question'] = three_digit['question']
        three_digit['original_intermediate_steps'] = three_digit['intermediate_steps']

        # Get x variables and generate random values (100-999)
        x_vars_three = {k: v for k, v in three_digit['variables'].items() if k.startswith('x')}
        new_values_three = {k: random.randint(100, 999) for k in x_vars_three}

        # Deep copy and update variables dictionary
        three_digit_vars = three_digit['variables'].copy()
        three_digit_vars.update(new_values_three)
        three_digit['variables'] = three_digit_vars

        # Replace variables in question text
        three_digit_question = three_digit['question']
        three_digit_interm = three_digit['intermediate_steps']

        for var_name, var_value in new_values_three.items():
            three_digit_question = three_digit_question.replace(var_name, str(var_value))
            three_digit_interm = three_digit_interm.replace(var_name, str(var_value))

        three_digit['question'] = three_digit_question
        three_digit['intermediate_steps'] = three_digit_interm

        # Recalculate y variables
        equations_three = re.findall(r'<<(.*?)>>', three_digit['original_intermediate_steps'])
        namespace_three = {k: v for k, v in three_digit['variables'].items() if k.startswith('x')}

        for eq in equations_three:
            left, right = eq.split('=')
            result = eval(left, {"__builtins__": {}}, namespace_three)
            namespace_three[right] = result
            three_digit['variables'][right] = result

        # Update answer
        for var_name, var_value in three_digit['variables'].items():
            three_digit_interm = three_digit_interm.replace(var_name, str(var_value))

        three_digit['intermediate_steps'] = three_digit_interm

        last_y_three = equations_three[-1].split('=')[1]
        three_digit['answer'] = namespace_three[last_y_three]


        # Print x variables for verification
        three_digit_x_vars = {k: v for k, v in three_digit['variables'].items() if k.startswith('x')}
        #print(f"Three-digit variant x variables: {three_digit_x_vars}")

        # Add to variants list
        variants["three_digit"].append(three_digit)

    # Print statistics about the variants
    for variant_name, variant_data in variants.items():
        #print(f"\nCreated {variant_name} variant with {len(variant_data)} examples")
        if variant_data:
            sample = variant_data[0]
            x_vars = {k: v for k, v in sample['variables'].items() if k.startswith('x')}
            print(f"Sample {variant_name} x variables: {x_vars}")

    return variants

def create_custom_variants(data, tuple_list):
    """
    Create a custom variant of the dataset where x1 and x2 are randomly selected
    from the provided tuple list, and other x variables are replaced with random
    three-digit numbers.

    Parameters:
    data (list): Original dataset
    tuple_list (list): List of tuples containing two numbers

    Returns:
    list: Processed dataset
    """
    # Filter out time-related examples
    filtered_data = [example for example in data if not has_colon(example.get('intermediate_steps', ''))]

    # Create new variant list
    custom_variant = []
    tuple_index = 0
    # Process each example
    for example in filtered_data:
        # Copy the example
        custom = example.copy()
        custom['original_question'] = custom['question']
        custom['original_intermediate_steps'] = custom['intermediate_steps']

        # Get all x variables
        x_vars = {k: v for k, v in custom['variables'].items() if k.startswith('x')}

        # Randomly select a tuple from the list
        random_tuple = tuple_list[tuple_index]
        tuple_index += 1

        # Create new values dictionary
        new_values = {}


        # Assign random three-digit numbers to other x variables
        for x_var in x_vars:
            if x_var not in new_values:  # Skip already assigned x1 and x2
                new_values[x_var] = random.randint(100, 999)

        # Deep copy and update variables dictionary
        custom_vars = custom['variables'].copy()
        custom_vars.update(new_values)
        custom['variables'] = custom_vars

        # Replace variables in question text
        custom_question = custom['question']
        custom_interm = custom['intermediate_steps']

        for var_name, var_value in new_values.items():
            custom_question = custom_question.replace(var_name, str(var_value))
            custom_interm = custom_interm.replace(var_name, str(var_value))

        custom['question'] = custom_question
        custom['intermediate_steps'] = custom_interm

        # Recalculate y variables
        equations = re.findall(r'<<(.*?)>>', custom['original_intermediate_steps'])
        namespace = {k: v for k, v in custom['variables'].items() if k.startswith('x')}

        for eq in equations:
            left, right = eq.split('=')
            result = eval(left, {"__builtins__": {}}, namespace)
            namespace[right] = result
            custom['variables'][right] = result

        # Update answer
        for var_name, var_value in custom['variables'].items():
            custom_interm = custom_interm.replace(var_name, str(var_value))

        custom['intermediate_steps'] = custom_interm

        last_y = equations[-1].split('=')[1]
        custom['answer'] = namespace[last_y]

        # Add to variants list
        custom_variant.append(custom)

    # Print statistics about the variants
    print(f"\nCreated custom variant with {len(custom_variant)} examples")
    if custom_variant:
        sample = custom_variant[0]
        x_vars = {k: v for k, v in sample['variables'].items() if k.startswith('x')}
        print(f"Sample custom x variables: {x_vars}")

    return custom_variant

# Function to create the prompt with system message and examples
def create_prompt(shots, test_example):
    # For some models, the system message needs to be included in the first user message
    system_message = 'As an expert problem solver, solve step by step the following mathematical questions. Each step should include exactly one arithmetic operation, and should be formatted as "<<a+b=c>>". Do not combine more than two numbers in one step. Any step with more than one arithmetic operator is incorrect and must be avoided.'

    # Initialize messages without system role to avoid compatibility issues
    messages = []

    # Add the 8 shots - for shots we use the original questions with variable names
    # but the concrete calculations in the answers
    for i, shot in enumerate(shots):
        if i == 0:
            # For the first shot, prepend the system message to the user query
            messages.append({"role": "user", "content": f"{system_message}\n\n{shot.get('question')}"})
        else:
            messages.append({"role": "user", "content": shot.get('question')})

        # Format each shot to only include the equations (not the question again)
        # We want to extract the equations with concrete values, not variable names
        #equations = re.findall(r'<<(.*?)>>', shot['intermediate_steps'])
        #formatted_answer = "\n".join(equations) + f"\nfinal answer: [{shot['answer']}]"
        formatted_answer = format_shot(shot)

        messages.append({"role": "assistant", "content": formatted_answer})

    # Add the test example - this should use the concrete question with substituted values
    messages.append({"role": "user", "content": test_example['question']})

    try:
        # Try to apply the chat template with add_generation_prompt
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )+"<<"
    except Exception as e:
        # Fallback method if the standard approach fails
        print(f"Warning: Standard chat template failed ({str(e)}). Using fallback method.")
        prompt = ""

        # Manually construct the prompt
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                if i > 0:
                    prompt += "\n\n"
                prompt += f"User: {msg['content']}\n\n"
            else:
                prompt += f"Assistant: {msg['content']}\n\n"

        prompt += "Assistant:\n<<"

    return prompt

# Function to extract the answer from model output
def extract_answer(output):
    # Try to extract the final answer in the format [X]
    final_answer_match = re.search(r'final answer: \[(\d+)\]', output)
    if final_answer_match:
        return int(final_answer_match.group(1))

    # If no final answer format is found, try to find the last number in the response
    numbers = re.findall(r'\d+', output)
    if numbers:
        return int(numbers[-1])

    return None

# Function to count equations and extract their equal sign positions
def count_and_extract_equations(response):
    """
    Count equations in the response and extract them with equal sign positions.
    
    Args:
        response (str): The model's response
        
    Returns:
        tuple: (count of equations, list of equations, positions of equal signs)
    """
    equations = re.findall(r'<<(.+?)>>', "<<"+response)
    
    # Calculate positions of equal signs
    eq_positions = []
    start_search = 0
    for eq in equations:
        if '=' in eq:
            eq_match = '<<' + eq + '>>'
            match_start = ("<<"+response).find(eq_match, start_search)
            if match_start != -1:
                eq_pos_in_match = eq.find('=')
                eq_pos_in_text = match_start + eq_pos_in_match  
                eq_positions.append(eq_pos_in_text)
                start_search = match_start + len(eq_match)
            else:
                eq_positions.append(-1)  # Equal sign not found
        else:
            eq_positions.append(-1)  # Equation has no equal sign
    
    return len(equations), equations, eq_positions

# Function to evaluate the model on a dataset
def evaluate_model(model, tokenizer, test_data, shots_data, variant_name=None, n_shots = 2):
    results = []
    correct_count = 0
    total_count = len(test_data)

    print(f"\nEvaluating on {variant_name if variant_name else 'dataset'} ({total_count} examples)...")

    # We'll use the original shot examples with variables, not the variant-specific ones
    # This is important because shots should show the general pattern with variables
    # while the test examples should have concrete values

    for i, test_example in enumerate(tqdm(test_data)):
        try:
            # Use 2 examples as shots (skip the test example if it's in the shots_data)
            # For shots, we want to use the original data with variable names, not concrete values
            shots = []
            for shot in shots_data:
                # Don't use the current test example as a shot
                if not is_same_example(shot, test_example):
                    shots.append(shot)
                if len(shots) >= n_shots:
                    break

            # If we don't have enough shots, just repeat some
            while len(shots) < n_shots:
                shots.append(random.choice(shots))

            # Create the prompt
            prompt = create_prompt(shots, test_example)

            # Generate the response
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens = False).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=512,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    do_sample=False,
                    num_beams=1
                )

            # Decode the output
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            # Extract the predicted answer
            predicted_answer = extract_answer(response)

            # Calculate if it's correct
            correct = predicted_answer == test_example['answer'] if predicted_answer is not None else False
            if correct:
                correct_count += 1

            # Count equations and extract them
            equation_count, equations, eq_positions = count_and_extract_equations(response)
            
            # Filter valid equations (no minus, multiply, divide, or non-numeric chars)
            valid_indices = []
            for idx, eq in enumerate(equations):
                # Check if equation only contains numbers, plus signs, equals signs, spaces, and periods
                if not ('-' in eq or '*' in eq or '/' in eq or 
                        any(char not in '0123456789+=. ' for char in eq)):
                    valid_indices.append(idx)
            
            # Check for incorrect equations (left side != right side)
            incorrect_indices = []
            for idx in valid_indices:
                eq = equations[idx]
                if '=' in eq:
                    try:
                        left, right = eq.split('=')
                        left_value = eval(left.strip())
                        right_value = float(right.strip())
                        if abs(left_value - right_value) > 1e-10:  # Small threshold for float comparison
                            incorrect_indices.append(idx)
                    except:
                        # If evaluation fails, consider it incorrect
                        incorrect_indices.append(idx)
            
            # Select an equation index
            if incorrect_indices:
                # If there are incorrect equations, choose one
                selected_index = random.choice(incorrect_indices)
            elif valid_indices:
                # If all valid equations are correct, randomly select one
                selected_index = random.choice(valid_indices)
            else:
                # If no valid equations, set to -1
                selected_index = -1

            # Log progress
            if (i + 1) % 5 == 0 or i == 0 or i == total_count - 1:
                current_accuracy = correct_count / (i + 1)
                print(f"Progress: {i+1}/{total_count}, Current accuracy: {current_accuracy:.2%}")

            # Store the result
            result = {
                "question": test_example['question'],
                "expected_answer": test_example['answer'],
                "predicted_answer": predicted_answer,
                "correct": correct,
                "variables": test_example['variables'],
                "model_response": response,
                "equation_count": equation_count,
                "valid_equation_count": len(valid_indices),
                "incorrect_equation_count": len(incorrect_indices),
                "selected_equation_index": selected_index,
                "selected_equation": equations[selected_index] if equation_count > 0 and selected_index >= 0 else None,
                "selected_equation_equal_pos": eq_positions[selected_index] if equation_count > 0 and selected_index >= 0 else -1,
                "is_incorrect_equation": selected_index in incorrect_indices if equation_count > 0 and selected_index >= 0 else False,
                "prompt": prompt  # Store the full prompt for debugging
            }

            results.append(result)

        except Exception as e:
            print(f"Error processing example {i}: {str(e)}")
            # Store the error
            result = {
                "question": test_example['question'] if 'question' in test_example else "Unknown",
                "expected_answer": test_example['answer'] if 'answer' in test_example else "Unknown",
                "predicted_answer": None,
                "correct": False,
                "error": str(e),
                "variables": test_example.get('variables', {})
            }
            results.append(result)

    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f"\nFinal accuracy on {variant_name if variant_name else 'dataset'}: {accuracy:.2%} ({correct_count}/{total_count})")

    return results, accuracy

# Helper function to check if two examples are the same
def is_same_example(ex1, ex2):
    # Compare by question text
    if ex1['question'] == ex2['question']:
        return True
    return False


# Main execution
def main_custom():

    with open('gsm8k/gsm_symbolic_addition_only.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    filtered_data = []
    for i in data["samples"]:
        if count_steps(i)>1 and is_valid_sample(i):
            filtered_data.append(i)

    import random
    random.seed(42)
    random.shuffle(filtered_data)
    random.shuffle(wrong_prompt)
    random.shuffle(all_prompt_below_1000)

    # Create the variants
    variants = {}
    n_data = len(filtered_data)
    for i in range(start_index, start_index + 10):
        random.seed(42 + i*10)
        if mode == "wrong_data_only":
            variants[i] = create_custom_variants(filtered_data,wrong_prompt[i*n_data: (i+1)*n_data])
        elif mode == "all":
            variants[i] = create_custom_variants(filtered_data,all_prompt_below_1000[i*n_data: (i+1)*n_data])

    # Evaluate on each variant
    all_results = {}

    for variant_name, variant_data in variants.items():
        print(f"Evaluating on {variant_name} variant...")
        results, accuracy = evaluate_model(model, tokenizer, variant_data, variant_data, n_shots = 2)

        all_results[variant_name] = {
            "accuracy": accuracy,
            "results": results
        }

        print(f"{variant_name} accuracy: {accuracy:.2%}")

    # Save the results
    os.makedirs("gsm8k/model_response", exist_ok=True)

    with open(f"gsm8k/model_response/{model_name.split('/')[-1]}_evaluation_results_gen_{mode}_{start_index}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Create a summary DataFrame
    summary = pd.DataFrame({
        "Variant": list(all_results.keys()),
        "Accuracy": [all_results[v]["accuracy"] for v in all_results]
    })

    print("\nEvaluation Summary:")
    print(summary)

    # Save the summary
    summary.to_csv(f"{model_name.split('/')[-1]}_evaluation_summary.csv", index=False)

    # Analyze results in more detail
    analyze_results(all_results)

# Function to analyze the results in more detail
def analyze_results(all_results):
    print("\n=== Detailed Analysis ===\n")

    # Analyze performance by question complexity (number of variables)
    for variant_name, variant_results in all_results.items():
        print(f"\nAnalysis for {variant_name}:")

        results = variant_results["results"]

        # Group by number of x variables
        var_counts = {}
        for r in results:
            if "variables" in r:
                x_vars = len([k for k in r["variables"] if k.startswith('x')])
                if x_vars not in var_counts:
                    var_counts[x_vars] = {"correct": 0, "total": 0}
                var_counts[x_vars]["total"] += 1
                if r.get("correct", False):
                    var_counts[x_vars]["correct"] += 1

        # Print accuracy by variable count
        print("Accuracy by number of x variables:")
        for var_count, stats in sorted(var_counts.items()):
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {var_count} variables: {accuracy:.2%} ({stats['correct']}/{stats['total']})")

        # Analyze common error patterns
        error_examples = [r for r in results if not r.get("correct", False)]
        if error_examples:
            print(f"\nCommon error patterns (from {len(error_examples)} incorrect examples):")

            # Categorize errors
            error_types = {
                "no_answer": 0,
                "calculation_error": 0,
                "parsing_error": 0,
                "other": 0
            }

            for ex in error_examples:
                if ex.get("predicted_answer") is None:
                    error_types["no_answer"] += 1
                elif "expected_answer" in ex and "predicted_answer" in ex:
                    error = abs(ex["expected_answer"] - ex["predicted_answer"])
                    if error > 0 and error <= 5:  # Small calculation errors
                        error_types["calculation_error"] += 1
                    else:
                        error_types["other"] += 1
                else:
                    error_types["parsing_error"] += 1

            # Print error distribution
            for error_type, count in error_types.items():
                percentage = count / len(error_examples) if len(error_examples) > 0 else 0
                print(f"  {error_type}: {percentage:.2%} ({count}/{len(error_examples)})")

            # Sample a few incorrect examples
            if error_examples:
                print("\nSample incorrect examples:")
                for ex in random.sample(error_examples, min(3, len(error_examples))):
                    print(f"  Question: {ex['question']}")
                    print(f"  Expected: {ex['expected_answer']}")
                    print(f"  Predicted: {ex['predicted_answer']}")
                    print(f"  Variables: {ex.get('variables', {})}")
                    print()


login(token=os.environ.get('HUGGINGFACE_HUB_TOKEN'))

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

main_custom()