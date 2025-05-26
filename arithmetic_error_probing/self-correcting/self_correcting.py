'''
python self-correcting.py --detector mlp --layer 25 --model google/gemma-2-2b-it --data_dir gsm8k --results_dir detector_guided_correction_gemma --detector_results_dir error_detector_detailed_results_gemma-2-2b-it
'''
import time
start_time = time.time()

import torch
import os
import json
import re
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from utils import get_digit, all_equations_correct


def load_model_tokenizer(model_name):
    """
    Load model and tokenizer
    
    Args:
        model_name (str): The name or path of the model
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    return model, tokenizer

def extract_equation_result(output):
    """
    Extract the result (number before >>) from model output
    
    Args:
        output (str): Model output text
        
    Returns:
        int or None: The extracted number, or None if not found
    """
    # Look for a number right before >>
    right_before_tokens = re.search(r'(\d+)\s*>>', output)
    if right_before_tokens:
        return int(right_before_tokens.group(1))
    
    # If we can't find the pattern with >>, look for the last number after an equal sign
    equal_match = re.search(r'=\s*(\d+)(?!\d*=)', output)
    if equal_match:
        return int(equal_match.group(1))
    
    # If we still can't find it, try to find the last number in the output
    numbers = re.findall(r'\d+', output)
    if numbers:
        return int(numbers[-1])  # Return the last number found
        
    return None

def create_correction_messages():
    """
    Create different correction messages to test
    
    Returns:
        dict: Dictionary with message types and their content
    """
    return {
        #sligtly increase the preservation rate by emphasising on calculating the exact same equation. but decrease the correction rate
        #"neutral": "That step looks incorrect. Let's re-calculate the exact same equation in the same format:",
        #"specific": "There seems to be an error in that arithmetic step. Let's carefully repeat the same calculation:",
        #"stronger": "That result is incorrect. I will now recompute that exact same step:",
        #"detailed": "I made a mistake in that arithmetic step. Let me go through the same equation again step by step:",
        #"reflective": "I should double-check that specific step. Let me re-do the same calculation carefully:",
        #"focused": "Let me revise that previous equation only, using the same structure as before:"

        #sometimes not calculate the original equation
        "suspecious": "That step looks suspecious. Let's re-do just this step:",
        "neutral": "That step looks incorrect. Let's re-do just this step:",
        "specific": "The calculation in this step is incorrect. Let's recalculate:",
        "stronger": "That's definitely wrong. The correct calculation should be:",
        "detailed": "I made an error in adding these numbers. Let me compute the sum correctly step by step:",

        #the following prmpts do not work (not correcting)
        #"verify_soft": "Let's double-check this step to make sure it's accurate:",
        #"verify_question": "Is this result definitely correct? Let's take a moment to confirm:",
        #"neutral_check": "Just to be safe, let me try this step one more time:",
        #"reflect_and_verify": "Before moving on, I want to make sure this calculation is right:",
        #"show_and_check": "I originally got this result. Let's verify if it's correct by recalculating:",
        #"check_with_reasoning": "Let me walk through the logic of this step again to confirm it adds up:",
        #"mild_flag": "Hmm, this might be slightly off. Let's go through it again carefully:",
        #"double_check_hint": "It's always good to double-check—just to be sure:"'
        
    }

def load_detector_results(detector_type, layer_idx, base_dir="error_detector_detailed_results"):
    """
    Load the results of a specific error detector on a specific layer.
    
    Args:
        detector_type: Type of the error detector (e.g., "mlp", "circular_jointly", "mlp_seperately")
        layer_idx: Index of the layer to analyze
        base_dir: Base directory where results are stored
        
    Returns:
        List of dictionaries with detailed results
    """
    # Construct the path to the results file
    file_path = os.path.join(base_dir, detector_type, f"layer_{layer_idx}_results.json")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file not found at: {file_path}")
    
    # Load the results
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} results from {file_path}")
    return results

def categorize_detector_results(results, detector_type, left_equations, model_answers):
    """
    Categorize detector results into TP, FP, TN, FN based on complete number equality.
    
    Args:
        results: List of result dictionaries
        detector_type: Type of the detector for interpreting results
        left_equations: Dictionary of left equation strings
        model_answers: Dictionary of model answers
        
    Returns:
        Tuple of (tp_indices, fp_indices, tn_indices, fn_indices)
    """
    tp_indices = []  # True Positives: correctly identified errors
    fp_indices = []  # False Positives: correct samples flagged as errors
    tn_indices = []  # True Negatives: correctly identified correct samples
    fn_indices = []  # False Negatives: errors that were missed
    
    for result in results:
        idx = result["idx"]
        
        # Convert list to tuple if needed
        if isinstance(idx, list) and len(idx) == 2:
            idx = tuple(idx)
        
        # Skip if we don't have equation data for this index
        if idx not in left_equations or idx not in model_answers:
            continue
        
        try:
            # Calculate actual correctness based on complete numbers
            left_value = eval(left_equations[idx].strip(), {"__builtins__": None}, {})
            model_value = int(model_answers[idx])
            
            # Set ground truth based on complete number equality (1=correct, 0=wrong)
            actual_correct = (left_value == model_value)
            gt = 1 if actual_correct else 0
            
            # Get the prediction from the detector
            if detector_type in ["mlp", "circular_jointly"]:
                # For binary detectors
                pred = result["prediction"]
            else:
                # For separately trained detectors
                pred = 1 if result["model_prediction"] == result["true_prediction"] else 0
            
            # Categorize the result - CORRECTED LOGIC
            if gt == 0 and pred == 0:
                tp_indices.append(idx)  # Error correctly detected
            elif gt == 1 and pred == 0:
                fp_indices.append(idx)  # Not an error but flagged as one
            elif gt == 1 and pred == 1:
                tn_indices.append(idx)  # Correctly identified as not an error
            elif gt == 0 and pred == 1:
                fn_indices.append(idx)  # Error missed by detector
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            continue
    
    print(f"Categorized results:")
    print(f"  True Positives (correct errors): {len(tp_indices)}")
    print(f"  False Positives (false alarms): {len(fp_indices)}")
    print(f"  True Negatives (correct non-errors): {len(tn_indices)}")
    print(f"  False Negatives (missed errors): {len(fn_indices)}")
    
    return tp_indices, fp_indices, tn_indices, fn_indices

def load_equation_data(gsm8k_folder, model_name, mode="_all"):
    """
    Load model responses and extract equations
    
    Args:
        gsm8k_folder: Folder containing GSM8K data
        model_name: Name of the model
        mode: Mode for file name
        
    Returns:
        tuple: (response_data, left_equations, model_answers, equal_sign_indices)
    """
    response_folder = os.path.join(gsm8k_folder, "model_response")
    response_data = {}
    left_equations = {}
    model_answers = {}
    equal_sign_indices = {}
    
    for dataset_index in range(0, 760, 10):
        results_file = os.path.join(response_folder, f"{model_name.split('/')[-1]}_evaluation_results_gen{mode}_{dataset_index}.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                try:
                    results = json.load(f)
                    for variant_index in results:
                        data = results[str(variant_index)]["results"]
                        for data_index in range(len(data)):
                            idx = (int(variant_index), data_index)
                            response_data[idx] = data[data_index]
                            
                            # Direct extraction of equation from the result data
                            selected_equation = data[data_index].get("selected_equation", "")
                            equal_sign_pos = data[data_index].get("selected_equation_equal_pos", -1)
                            
                            if selected_equation and '=' in selected_equation:
                                parts = selected_equation.split('=', 1)
                                left = parts[0].strip()
                                right = parts[1].strip() if len(parts) > 1 else ""
                                left_equations[idx] = left
                                model_answers[idx] = right
                                equal_sign_indices[idx] = equal_sign_pos
                            else:
                                left_equations[idx] = ""
                                model_answers[idx] = ""
                                equal_sign_indices[idx] = -1
                                
                except Exception as e:
                    print(f"Error processing {results_file}: {e}")
    
    return response_data, left_equations, model_answers, equal_sign_indices

def prepare_correction_prompts(indices, response_data, left_equations, model_answers, equal_sign_indices):
    """
    Prepare prompts for self-correction based on detector-identified indices
    
    Args:
        indices: List of sample indices identified by the detector
        response_data, left_equations, model_answers, equal_sign_indices: Data from load_equation_data
        
    Returns:
        tuple: (prompts, equation_data) for self-correction
    """
    correction_prompts = []
    correction_equations = []
    
    for idx in indices:
        # Convert list to tuple if needed
        if isinstance(idx, list) and len(idx) == 2:
            idx = tuple(idx)
            
        if idx not in left_equations or idx not in model_answers:
            continue
            
        left, right = left_equations[idx], model_answers[idx]
        str_index = equal_sign_indices[idx]
        
        try:
            if left != "":
                int(right)  # Make sure the right side is a valid integer
                left_value = eval(left.strip(), {"__builtins__": None}, {})
                
                if left_value < 1000:
                    # Create the prompt with the equation
                    prompt = response_data[idx]["prompt"] + response_data[idx]["model_response"][:str_index+1] + right + ">>"
                    correction_prompts.append(prompt)
                    correction_equations.append((idx, left, right, left_value))
        except Exception as e:
            print(f"Error preparing correction prompt for {idx}: {e}")
            continue
    
    return correction_prompts, correction_equations

def evaluate_self_correction(model, tokenizer, prompts, equations, message_type, message):
    """
    Evaluate self-correction with a specific correction message
    
    Args:
        model, tokenizer: The language model and tokenizer
        prompts: List of prompts to test
        equations: List of equation data (idx, left, right, left_value)
        message_type: Type of correction message
        message: The correction message text
        
    Returns:
        dict: Statistics about correction success rate
    """
    results = {
        "message_type": message_type,
        "message": message,
        "total": len(prompts),
        "corrected": 0,
        "still_incorrect": 0,
        "no_answer": 0,
        "details": []
    }
    
    print(f"Evaluating self-correction with message: '{message}' ({results['total']} examples)...")
    
    # Print a few complete prompts for verification
    print("\n=== SAMPLE PROMPTS FOR VERIFICATION ===")
    for i in range(min(3, len(prompts))):
        original_prompt = prompts[i]
        if message_type == "user_instruction":
            new_user_message = [
                {"role": "user", "content": message + " <<"},
            ]
                
            user_instruction = tokenizer.apply_chat_template(
                new_user_message,
                tokenize=False,
                add_generation_prompt=True
            )
            # for gemma 
            if "<end_of_turn>" in original_prompt:     
                modified_prompt = original_prompt + "<end_of_turn>\n" + user_instruction
            #for phil
            elif "<|end|>" in original_prompt:     
                modified_prompt = original_prompt + "<|end|>\n" + user_instruction
            else:
                modified_prompt = original_prompt + "\n" + user_instruction
        else:
            modified_prompt = original_prompt + "\n" + message + " <<" + equations[i][1] + "="

        
        print(f"\n--- Sample {i+1} ---")
        print("===== ORIGINAL PROMPT =====")
        print(original_prompt)
        print("\n===== MODIFIED PROMPT WITH CORRECTION MESSAGE =====")
        print(modified_prompt)
        print("\n" + "="*50)

    for i, (prompt, eq_data) in enumerate(zip(prompts, equations)):
        idx, left, right, expected_value = eq_data
        
        # Add correction message
        if message_type == "user_instruction":
            new_user_message = [
                {"role": "user", "content": message + " <<" + eq_data[1]},
            ]
                
            user_instruction = tokenizer.apply_chat_template(
                new_user_message,
                tokenize=False,
                add_generation_prompt=True
            )
            # for gemma 
            if "<end_of_turn>" in prompt:     
                modified_prompt = prompt + "<end_of_turn>\n" + user_instruction
            #for phil
            elif "<|end|>" in prompt:     
                modified_prompt = prompt + "<|end|>\n" + user_instruction
            else:
                modified_prompt = prompt + "\n" + user_instruction
        else:
            modified_prompt = prompt + "\n" + message + " <<" + eq_data[1] + "="

        try:
            # Generate continuation with correction prompt
            inputs = tokenizer(modified_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=512,
                    temperature=0,
                    top_p=None,
                    top_k=None,
                    do_sample=False,
                    num_beams=1
                )
            
            # Decode the output
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Extract the predicted answer
            predicted_answer = extract_equation_result(response)
            
            # Check if the correction worked
            if predicted_answer is not None:
                if predicted_answer == expected_value:
                    results["corrected"] += 1
                    status = "corrected"
                else:
                    results["still_incorrect"] += 1
                    status = "still_incorrect"
            else:
                results["no_answer"] += 1
                status = "no_answer"
            
            # Store detailed results
            results["details"].append({
                "index": idx,
                "original_value": int(right),
                "expected_value": expected_value,
                "corrected_value": predicted_answer,
                "status": status,
                "original_equation": left,
                "correction_response": response[:200]  # First 200 chars for brevity
            })
            
            # Log progress
            if (i + 1) % 5 == 0 or i == 0 or i == len(prompts) - 1:
                corrected_so_far = results["corrected"]
                progress_percent = (i + 1) / len(prompts) * 100
                correction_rate = corrected_so_far / (i + 1) * 100
                print(f"Progress: {i+1}/{len(prompts)} ({progress_percent:.1f}%), "
                      f"Correction rate: {correction_rate:.1f}%")
                
        except Exception as e:
            print(f"Error processing example {i}: {str(e)}")
            results["no_answer"] += 1
            results["details"].append({
                "index": idx,
                "error": str(e),
                "status": "error"
            })
    
    # Calculate correction rate
    if results["total"] > 0:
        results["correction_rate"] = results["corrected"] / results["total"]
    else:
        results["correction_rate"] = 0
        
    print(f"\nFinal correction rate for '{message_type}': {results['correction_rate']:.2%} "
          f"({results['corrected']}/{results['total']})")
    print(f"Still incorrect: {results['still_incorrect']}, "
          f"No answer: {results['no_answer']}")
    
    return results

def evaluate_detector_guided_correction(model, tokenizer, tp_indices, fp_indices, 
                                        response_data, left_equations, model_answers, equal_sign_indices, 
                                        output_dir="detector_guided_correction"):
    """
    Evaluate self-correction guided by detector results
    
    Args:
        model, tokenizer: The model and tokenizer
        tp_indices, fp_indices: Indices from detector categorization
        response_data, left_equations, model_answers, equal_sign_indices: Data from load_equation_data
        output_dir: Directory to save results
        
    Returns:
        dict: Results of the evaluation
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get correction messages
    correction_messages = create_correction_messages()
    
    # Prepare prompts for TP samples (detector-identified errors)
    print("\nPreparing prompts for True Positives (detector-identified errors)...")
    tp_prompts, tp_equations = prepare_correction_prompts(
        tp_indices, response_data, left_equations, model_answers, equal_sign_indices
    )
    
    # Prepare prompts for FP samples (false alarms)
    print("\nPreparing prompts for False Positives (false alarms)...")
    fp_prompts, fp_equations = prepare_correction_prompts(
        fp_indices, response_data, left_equations, model_answers, equal_sign_indices
    )
    
    # Results container
    results = {
        "true_positives": {},
        "false_positives": {}
    }
    
    # Test each message type on TP samples
    print("\n=== Evaluating Self-Correction on True Positives (actual errors) ===")
    for message_type, message in correction_messages.items():
        print(f"\nTesting message type: {message_type}")
        tp_results = evaluate_self_correction(
            model, tokenizer, tp_prompts, tp_equations, message_type, message
        )
        results["true_positives"][message_type] = tp_results
    
    # Test each message type on FP samples
    print("\n=== Evaluating Self-Correction on False Positives (false alarms) ===")
    for message_type, message in correction_messages.items():
        print(f"\nTesting message type: {message_type}")
        fp_results = evaluate_self_correction(
            model, tokenizer, fp_prompts, fp_equations, message_type, message
        )
        results["false_positives"][message_type] = fp_results
    
    # Save results
    results_file = os.path.join(output_dir, "detector_guided_correction_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Generate analysis
    generate_analysis_report(results, output_dir)
    
    return results


def generate_analysis_report(results, output_dir):
    """
    Generate a detailed analysis report of the detector-guided correction results
    
    Args:
        results: Results from evaluation
        output_dir: Directory to save the report
    """
    report = ["# Detector-Guided Self-Correction Analysis Report\n"]
    
    # Add a summary table for all message types on both TP and FP
    report.append("## Summary of Correction Rates\n")
    report.append("| Message Type | TP Correction Rate | FP Preservation Rate | Combined Effectiveness |\n")
    report.append("|--------------|---------------------|----------------------|------------------------|\n")
    
    combined_effectiveness = {}
    
    for message_type in results["true_positives"].keys():
        tp_rate = results["true_positives"][message_type]["correction_rate"]
        
        # For FP, we want to know if it preserved the correct answer
        fp_details = results["false_positives"][message_type]["details"]
        preserved_count = sum(1 for detail in fp_details 
                            if detail["status"] != "error" and 
                            detail["corrected_value"] == detail["expected_value"])
        fp_preservation_rate = preserved_count / len(fp_details) if fp_details else 0
        
        # Combined effectiveness - weighted average of TP correction and FP preservation
        # giving higher weight to TP correction since that's our primary goal
        combined = (0.7 * tp_rate) + (0.3 * fp_preservation_rate)
        combined_effectiveness[message_type] = combined
        
        report.append(f"| {message_type} | {tp_rate:.2%} | {fp_preservation_rate:.2%} | {combined:.2%} |\n")
    
    # Find the best message type
    best_message = max(combined_effectiveness.items(), key=lambda x: x[1])
    report.append(f"\n**Best overall message type: '{best_message[0]}' with combined effectiveness {best_message[1]:.2%}**\n")
    
    # Detailed analysis for TP
    report.append("\n## Analysis of True Positives (Actual Errors)\n")
    
    for message_type, data in results["true_positives"].items():
        report.append(f"\n### {message_type}\n")
        report.append(f"Message: '{data['message']}'\n")
        report.append(f"Correction rate: {data['correction_rate']:.2%} ({data['corrected']}/{data['total']})\n")
        
        # Error analysis
        if data["still_incorrect"] > 0:
            report.append("\nError patterns in failed corrections:\n")
            
            failures = [d for d in data["details"] if d["status"] == "still_incorrect"]
            error_diff = [(d["expected_value"] - d["corrected_value"]) for d in failures if "corrected_value" in d and d["corrected_value"] is not None]
            
            if error_diff:
                avg_error = sum(abs(e) for e in error_diff) / len(error_diff)
                report.append(f"Average absolute error: {avg_error:.2f}\n")
                
                # Check if errors tend to be positive or negative
                bias = sum(e for e in error_diff) / len(error_diff)
                if bias > 1:
                    report.append(f"Error bias: Model tends to undercorrect (average: {bias:.2f})\n")
                elif bias < -1:
                    report.append(f"Error bias: Model tends to overcorrect (average: {bias:.2f})\n")
                else:
                    report.append(f"Error bias: No significant bias in corrections (average: {bias:.2f})\n")
    
    # Detailed analysis for FP
    report.append("\n## Analysis of False Positives (False Alarms)\n")
    
    for message_type, data in results["false_positives"].items():
        report.append(f"\n### {message_type}\n")
        report.append(f"Message: '{data['message']}'\n")
        
        # For FP, we want to see if correction requests preserved the correct answer
        fp_details = data["details"]
        preserved_count = sum(1 for detail in fp_details 
                            if detail["status"] != "error" and 
                            detail["corrected_value"] == detail["expected_value"])
        fp_preservation_rate = preserved_count / len(fp_details) if fp_details else 0
        
        report.append(f"Preservation rate: {fp_preservation_rate:.2%} ({preserved_count}/{len(fp_details)})\n")
        
        # Check if corrections made things worse
        made_wrong_count = sum(1 for detail in fp_details 
                              if detail["status"] != "error" and 
                              detail["corrected_value"] != detail["expected_value"])
        
        if made_wrong_count > 0:
            made_wrong_rate = made_wrong_count / len(fp_details)
            report.append(f"Rate of introducing errors: {made_wrong_rate:.2%} ({made_wrong_count}/{len(fp_details)})\n")
    
    # Overall recommendations
    report.append("\n## Overall Recommendations\n")
    
    # Find best for TP
    best_tp_message = max(results["true_positives"].items(), key=lambda x: x[1]["correction_rate"])
    
    # Find best for FP preservation
    best_fp_preservation = None
    best_fp_rate = 0
    
    for message_type, data in results["false_positives"].items():
        fp_details = data["details"]
        preserved_count = sum(1 for detail in fp_details 
                            if detail["status"] != "error" and 
                            detail["corrected_value"] == detail["expected_value"])
        fp_preservation_rate = preserved_count / len(fp_details) if fp_details else 0
        
        if fp_preservation_rate > best_fp_rate:
            best_fp_rate = fp_preservation_rate
            best_fp_preservation = message_type
    
    report.append(f"1. Best message for correcting actual errors: '{best_tp_message[0]}' with {best_tp_message[1]['correction_rate']:.2%} correction rate\n")
    report.append(f"2. Best message for preserving correct answers: '{best_fp_preservation}' with {best_fp_rate:.2%} preservation rate\n")
    report.append(f"3. Best overall message considering both factors: '{best_message[0]}' with {best_message[1]:.2%} combined effectiveness\n")
    
    # Save the report
    report_path = os.path.join(output_dir, "detector_guided_correction_analysis.md")
    with open(report_path, 'w') as f:
        f.writelines(report)
    
    print(f"Analysis report saved to {report_path}")
def main():
    """
    Main function to run the detector-guided self-correction evaluation
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate detector-guided self-correction")
    parser.add_argument("--detector", type=str, required=True, 
                        help="Detector type (e.g., mlp, circular_jointly, mlp_seperately)")
    parser.add_argument("--layer", type=int, required=True, 
                        help="Layer index to analyze")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it",
                        help="Model name or path")
    parser.add_argument("--data_dir", type=str, default="gsm8k",
                        help="Directory containing GSM8K data")
    parser.add_argument("--results_dir", type=str, default="detector_guided_correction",
                        help="Directory to save results")
    parser.add_argument("--detector_results_dir", type=str, default="error_detector_detailed_results",
                        help="Directory containing detector results")
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_tokenizer(args.model)
    
    # Load equation data first (needed for complete number comparison)
    response_data, left_equations, model_answers, equal_sign_indices = load_equation_data(
        args.data_dir, args.model
    )
    
    # Load detector results
    detector_results = load_detector_results(args.detector, args.layer, args.detector_results_dir)
    
    # Categorize detector results with complete number comparison
    tp_indices, fp_indices, tn_indices, fn_indices = categorize_detector_results(
        detector_results, args.detector, left_equations, model_answers
    )
    
    # Evaluate detector-guided correction
    results = evaluate_detector_guided_correction(
        model, tokenizer, tp_indices, fp_indices,
        response_data, left_equations, model_answers, equal_sign_indices,
        output_dir=args.results_dir
    )
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
    end_time = time.time()
    print(f"total running time: {end_time - start_time:.4f} second")