'''
model_name=google/gemma-2-2b-it
trained_probes_folder=gemma-2-2b-it_probing_results_2_shots
python test_2_shots_probe_on_gsm8k.py $trained_probes_folder $model_name
'''


import torch
import os
import sys
import json
import random
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Any
from transformers import AutoConfig

from model import RidgeRegression, MLP, CircularProbe, MultiClassLogisticRegression, CircularErrorDetector

from utils import get_digit, all_equations_correct


def test_circular_probe(X, Y, probe, device="cuda"):
    """Test circular probe performance"""
    X = X.to(device)
    Y = Y.to(device)
    
    # Print Y distribution
    y_values = Y.detach().cpu().numpy()
    print("Test data Y distribution:")
    for i in range(10):
        count = (y_values == i).sum()
        print(f"Number {i} appears {count} times in test data")
    
    probe = probe.to(device)
    probe.eval()
    with torch.no_grad():
        y_pred = torch.round(probe.forward_digit(X)) % 10
    correct_predictions = (y_pred == Y).float()
    accuracy = correct_predictions.mean().item()
    return accuracy


def test_ridge_probe(X, Y, probe, device="cuda"):
    """Test ridge regression probe performance"""
    X = X.to(device)
    Y = Y.to(device)
    
    # Print Y distribution
    y_values = Y.detach().cpu().numpy()
    print("Test data Y distribution:")
    for i in range(10):
        count = (y_values == i).sum()
        print(f"Number {i} appears {count} times in test data")
    
    probe = probe.to(device)
    probe.eval()
    with torch.no_grad():
        y_pred = probe(X).long()
        y_test_class = Y.long()
    correct_predictions = (y_pred == y_test_class).float()
    accuracy = correct_predictions.mean().item()
    return accuracy


def test_mlp_probe(X, Y, probe, device="cuda"):
    """Test MLP probe performance"""
    X = X.to(device)
    Y = Y.to(device)
    
    # Print Y distribution
    y_values = Y.detach().cpu().numpy()
    print("Test data Y distribution:")
    for i in range(10):
        count = (y_values == i).sum()
        print(f"Number {i} appears {count} times in test data")
    
    probe = probe.to(device)
    probe.eval()
    with torch.no_grad():
        _, y_pred = torch.max(probe(X), 1)
    correct_predictions = (y_pred == Y).float()
    accuracy = correct_predictions.mean().item()
    return accuracy


def test_logistic_probe(X, Y, probe, device="cuda"):
    """Test logistic regression probe performance"""
    X = X.to(device)
    Y = Y.to(device)
    
    # Print Y distribution
    y_values = Y.detach().cpu().numpy()
    print("Test data Y distribution:")
    for i in range(10):
        count = (y_values == i).sum()
        print(f"Number {i} appears {count} times in test data")
    
    probe = probe.to(device)
    probe.eval()
    with torch.no_grad():
        outputs = probe(X)
        _, y_pred = torch.max(outputs, 1)
    correct_predictions = (y_pred == Y).float()
    accuracy = correct_predictions.mean().item()
    return accuracy

def test_mlp_error_detector(X, Y, probe, device="cuda"):
    """Test MLP error detector (jointly trained) with confidence scores"""
    X = X.to(device)
    Y = Y.to(device)
    
    # Print Y distribution
    y_values = Y.detach().cpu().numpy()
    print("Test data Y distribution:")
    for i in range(2):  # Binary classification: 0=wrong, 1=correct
        count = (y_values == i).sum()
        print(f"Class {i} appears {count} times in test data")
    
    probe = probe.to(device)
    probe.eval()
    
    with torch.no_grad():
        outputs = probe(X)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, y_pred = torch.max(outputs, 1)
        
        # Calculate entropy-based confidence
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=1)
        max_entropy = torch.log(torch.tensor(outputs.size(1), dtype=torch.float, device=device))
        confidence = 1.0 - (entropy / max_entropy)
    
    # Original metrics calculation 
    TP = ((y_pred == 0) & (Y == 0)).float().sum().item()
    FP = ((y_pred == 0) & (Y == 1)).float().sum().item()
    TN = ((y_pred == 1) & (Y == 1)).float().sum().item()
    FN = ((y_pred == 1) & (Y == 0)).float().sum().item()
    
    accuracy = (y_pred == Y).float().mean().item()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    
    # Collect detailed results for each sample
    detailed_results = []
    for i in range(len(X)):
        result = {
            "idx": None,  # Will be filled in the main function
            "prediction": y_pred[i].item(),  # Binary prediction (0=wrong, 1=correct)
            "ground_truth": Y[i].item(),     # Binary label (0=wrong, 1=correct)
            "confidence": confidence[i].item(),  # Add confidence score
            "is_correct": (y_pred[i] == Y[i]).item()
        }
        detailed_results.append(result)
    
    return accuracy, precision, recall, f1, detailed_results


# 2. Modify test_mlp_error_detector_seperately function to calculate confidence based on model agreement
def test_mlp_error_detector_seperately(X, Y_model, Y_true, probes, device="cuda"):
    """Test MLP error detector (separately trained) with confidence scores"""
    X = X.to(device)
    Y_model = Y_model.to(device)
    Y_true = Y_true.to(device)
    
    # Print Y distribution
    y_model_values = Y_model.detach().cpu().numpy()
    y_true_values = Y_true.detach().cpu().numpy()
    print("Test data Y_model and Y_true distributions:")
    for i in range(10):
        count_model = (y_model_values == i).sum()
        count_true = (y_true_values == i).sum()
        print(f"Number {i} appears {count_model} times in Y_model and {count_true} times in Y_true")
    
    probe_1, probe_2 = probes
    probe_1 = probe_1.to(device)
    probe_2 = probe_2.to(device)
    
    probe_1.eval()
    probe_2.eval()
    
    with torch.no_grad():
        outputs_1 = probe_1(X)
        outputs_2 = probe_2(X)
        
        # Apply softmax to get probabilities
        probs_1 = torch.nn.functional.softmax(outputs_1, dim=1)
        probs_2 = torch.nn.functional.softmax(outputs_2, dim=1)
        
        # Get maximum probability as model-specific confidence
        max_probs_1, pred_model = torch.max(probs_1, dim=1)
        max_probs_2, pred_true = torch.max(probs_2, dim=1)
        
        # Combined confidence: geometric mean of both confidences
        combined_confidence = torch.sqrt(max_probs_1 * max_probs_2)
    
    # Check if the prediction about whether model matches ground truth is correct
    model_correct = (Y_model == Y_true)
    pred_correct = (pred_model == pred_true)
    
    # Calculate metrics  
    TP = ((pred_correct == 0) & (model_correct == 0)).float().sum().item()
    FP = ((pred_correct == 0) & (model_correct == 1)).float().sum().item()
    TN = ((pred_correct == 1) & (model_correct == 1)).float().sum().item()
    FN = ((pred_correct == 1) & (model_correct == 0)).float().sum().item()
    
    accuracy = (model_correct == pred_correct).float().mean().item()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    
    # Collect detailed results for each sample
    detailed_results = []
    for i in range(len(X)):
        result = {
            "idx": None,  # Will be filled in the main function
            "model_prediction": pred_model[i].item(),
            "true_prediction": pred_true[i].item(),
            "model_digit": Y_model[i].item(),
            "ground_truth": Y_true[i].item(),
            "confidence": combined_confidence[i].item(),  # Add confidence score
            "conf_model": max_probs_1[i].item(),  # Individual model confidence
            "conf_true": max_probs_2[i].item(),   # Individual true confidence
            "is_correct": (model_correct[i] == pred_correct[i]).item()
        }
        detailed_results.append(result)
    
    return accuracy, precision, recall, f1, detailed_results


# 3. Modify test_logistic_error_detector_seperately function with similar confidence calculation
def test_logistic_error_detector_seperately(X, Y_model, Y_true, probes, device="cuda"):
    """Test logistic regression error detector (separately trained) with confidence scores"""
    X = X.to(device)
    Y_model = Y_model.to(device)
    Y_true = Y_true.to(device)
    
    # Print Y distribution
    y_model_values = Y_model.detach().cpu().numpy()
    y_true_values = Y_true.detach().cpu().numpy()
    print("Test data Y_model and Y_true distributions:")
    for i in range(10):
        count_model = (y_model_values == i).sum()
        count_true = (y_true_values == i).sum()
        print(f"Number {i} appears {count_model} times in Y_model and {count_true} times in Y_true")
    
    probe_1, probe_2 = probes
    probe_1 = probe_1.to(device)
    probe_2 = probe_2.to(device)
    
    probe_1.eval()
    probe_2.eval()
    
    with torch.no_grad():
        outputs_1 = probe_1(X)
        outputs_2 = probe_2(X)
        
        # Apply softmax to get probabilities
        probs_1 = torch.nn.functional.softmax(outputs_1, dim=1)
        probs_2 = torch.nn.functional.softmax(outputs_2, dim=1)
        
        # Get maximum probability as model-specific confidence
        max_probs_1, pred_model = torch.max(probs_1, dim=1)
        max_probs_2, pred_true = torch.max(probs_2, dim=1)
        
        # Combined confidence: geometric mean of both confidences
        combined_confidence = torch.sqrt(max_probs_1 * max_probs_2)
    
    # Check if the prediction about whether model matches ground truth is correct
    model_correct = (Y_model == Y_true)
    pred_correct = (pred_model == pred_true)
    
    # Calculate metrics - treating 1 (correct) as positive class
    TP = ((pred_correct == 0) & (model_correct == 0)).float().sum().item()
    FP = ((pred_correct == 0) & (model_correct == 1)).float().sum().item()
    TN = ((pred_correct == 1) & (model_correct == 1)).float().sum().item()
    FN = ((pred_correct == 1) & (model_correct == 0)).float().sum().item()
    
    accuracy = (model_correct == pred_correct).float().mean().item()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    
    # Collect detailed results for each sample
    detailed_results = []
    for i in range(len(X)):
        result = {
            "idx": None,  # Will be filled in the main function
            "model_prediction": pred_model[i].item(),
            "true_prediction": pred_true[i].item(),
            "model_digit": Y_model[i].item(),
            "ground_truth": Y_true[i].item(),
            "confidence": combined_confidence[i].item(),  # Add confidence score
            "conf_model": max_probs_1[i].item(),  # Individual model confidence
            "conf_true": max_probs_2[i].item(),   # Individual true confidence
            "is_correct": (model_correct[i] == pred_correct[i]).item()
        }
        detailed_results.append(result)
    
    return accuracy, precision, recall, f1, detailed_results


# 4. Modify test_circular_error_detector_seperately function with vector length-based confidence
def test_circular_error_detector_seperately(X, Y_model, Y_true, probes, device="cuda"):
    """Test circular error detector (separately trained) with confidence scores"""
    X = X.to(device)
    Y_model = Y_model.to(device)
    Y_true = Y_true.to(device)
    
    # Print Y distribution
    y_model_values = Y_model.detach().cpu().numpy()
    y_true_values = Y_true.detach().cpu().numpy()
    print("Test data Y_model and Y_true distributions:")
    for i in range(10):
        count_model = (y_model_values == i).sum()
        count_true = (y_true_values == i).sum()
        print(f"Number {i} appears {count_model} times in Y_model and {count_true} times in Y_true")
    
    probe_1, probe_2 = probes
    probe_1 = probe_1.to(device)
    probe_2 = probe_2.to(device)
    
    probe_1.eval()
    probe_2.eval()
    
    with torch.no_grad():
        # Get raw projections
        proj_1 = probe_1(X)
        proj_2 = probe_2(X)
        
        # Calculate vector lengths as clarity measures
        clarity_1 = torch.sqrt(torch.sum(proj_1**2, dim=1))
        clarity_2 = torch.sqrt(torch.sum(proj_2**2, dim=1))
        
        # Get digit predictions
        pred_model = torch.round(probe_1.forward_digit(X)) % 10
        pred_true = torch.round(probe_2.forward_digit(X)) % 10
        
        # Calculate combined confidence based on vector clarity
        combined_confidence = (clarity_1 + clarity_2) / 2
        
        # Normalize confidence to 0-1 range (assuming typical range for clarity)
        # This is a heuristic and may need adjustment
        normalized_confidence = torch.tanh(combined_confidence / 5.0)
    
    # Check if the prediction about whether model matches ground truth is correct
    model_correct = (Y_model == Y_true)
    pred_correct = (pred_model == pred_true)
    
    # Calculate metrics - treating 1 (correct) as positive class
    TP = ((pred_correct == 0) & (model_correct == 0)).float().sum().item()
    FP = ((pred_correct == 0) & (model_correct == 1)).float().sum().item()
    TN = ((pred_correct == 1) & (model_correct == 1)).float().sum().item()
    FN = ((pred_correct == 1) & (model_correct == 0)).float().sum().item()
    
    accuracy = (model_correct == pred_correct).float().mean().item()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    
    # Collect detailed results for each sample
    detailed_results = []
    for i in range(len(X)):
        result = {
            "idx": None,  # Will be filled in the main function
            "model_prediction": pred_model[i].item(),
            "true_prediction": pred_true[i].item(),
            "model_digit": Y_model[i].item(),
            "ground_truth": Y_true[i].item(),
            "confidence": normalized_confidence[i].item(),  # Add normalized confidence
            "clarity_model": clarity_1[i].item(),  # Individual clarity measures
            "clarity_true": clarity_2[i].item(),
            "is_correct": (model_correct[i] == pred_correct[i]).item()
        }
        detailed_results.append(result)
    
    return accuracy, precision, recall, f1, detailed_results


# 5. Modify test_circular_error_detector_jointly function using sigmoid distance for confidence
def test_circular_error_detector_jointly(X, Y, probe, device="cuda"):
    """Test circular error detector (jointly trained) with confidence scores"""
    X = X.to(device)
    Y = Y.to(device)
    
    # Print Y distribution
    y_values = Y.detach().cpu().numpy()
    print("Test data Y distribution:")
    for i in range(2):  # Binary classification: 0=wrong, 1=correct
        count = (y_values == i).sum()
        print(f"Class {i} appears {count} times in test data")
    
    probe = probe.to(device)
    probe.eval()
    
    with torch.no_grad():
        predicted_result = probe(X)  # This is the sigmoid output
        predicted_labels = (predicted_result > 0.5).int()
        
        # Calculate confidence: distance from decision boundary (0.5)
        confidence = 2 * torch.abs(predicted_result - 0.5)  # Scale to [0,1]
    
    # Calculate metrics 
    TP = ((predicted_labels == 0) & (Y == 0)).float().sum().item()
    FP = ((predicted_labels == 0) & (Y == 1)).float().sum().item()
    TN = ((predicted_labels == 1) & (Y == 1)).float().sum().item()
    FN = ((predicted_labels == 1) & (Y == 0)).float().sum().item()
    
    accuracy = (predicted_labels == Y).float().mean().item()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    
    # Collect detailed results for each sample
    detailed_results = []
    for i in range(len(X)):
        result = {
            "idx": None,  # Will be filled in the main function
            "prediction": predicted_labels[i].item(),  # Binary prediction (0=wrong, 1=correct)
            "prediction_raw": predicted_result[i].item(),  # Raw prediction probability
            "ground_truth": Y[i].item(),  # Binary label (0=wrong, 1=correct)
            "confidence": confidence[i].item(),  # Add confidence score
            "is_correct": (predicted_labels[i] == Y[i]).item()
        }
        detailed_results.append(result)
    
    return accuracy, precision, recall, f1, detailed_results

def save_detector_results(results: List[Dict[str, Any]], detector_type: str, layer_idx: int, output_dir: str):
    """
    Save detailed results from error detector to a JSON file, including confidence scores.
    
    Args:
        results: List of dictionaries with detailed prediction information
        detector_type: Type of the error detector
        layer_idx: Index of the layer
        output_dir: Base directory for saving results
    """
    # Create output directories if they don't exist
    detector_dir = os.path.join(output_dir, detector_type)
    os.makedirs(detector_dir, exist_ok=True)
    
    # Create output file path
    output_file = os.path.join(detector_dir, f"layer_{layer_idx}_results.json")
    
    # Convert any tensor values to Python types
    serializable_results = []
    for result in results:
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                serializable_result[key] = value.item()
            elif isinstance(value, tuple) and all(isinstance(item, int) for item in value):
                serializable_result[key] = list(value)  # Convert tuple to list for JSON
            else:
                serializable_result[key] = value
        serializable_results.append(serializable_result)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Saved {len(results)} detailed results to {output_file}")







def check_equation_overlap(sample_indices, left_equations, training_data_path="training_samples_probing"):
    """
    Check overlap between test samples and training data
    For equations with form a+b, check if (a,b) is in training data
    """
    try:
        # Load training data
        training_data = torch.load(training_data_path)
        print(f"Loaded training data with {len(training_data)} samples")
        print(f"Example training data: {training_data[:5] if training_data else 'Empty'}")
        
        # Initialize counters
        overlap_count = 0
        valid_equation_count = 0

        example_count = 0
        examples = []
        
        # Process each sample
        for idx in sample_indices:
            if idx in left_equations and left_equations[idx] != -1:
                equation = left_equations[idx].strip()
                
                # Check if it's an addition equation with two operands
                if "+" in equation and not "-" in equation and not "*" in equation and not "/" in equation:
                    parts = equation.split("+")
                    if len(parts) == 2:
                        valid_equation_count += 1
                        try:
                            a, b = int(parts[0].strip()), int(parts[1].strip())

                            if example_count<5:
                                examples.append((a,b))
                                '''
                                print(f"idx: {idx}")
                                print(f"equation: {equation}")'
                                '''
                            
                            # Check if (a,b) or (b,a) is in training data
                            if (a, b) in training_data or (b, a) in training_data:
                                overlap_count += 1
                        except ValueError:
                            # Skip if not integer values
                            pass

        print(f"Example test data: {examples[:5] if examples else 'Empty'}")
        
        # Calculate overlap ratio
        overlap_ratio = overlap_count / valid_equation_count if valid_equation_count > 0 else 0
        
        print(f"Total valid equations: {valid_equation_count}")
        print(f"Equations with overlap: {overlap_count}")
        print(f"Overlap ratio: {overlap_ratio:.4f} ({overlap_count}/{valid_equation_count})")
        
        return overlap_ratio, overlap_count, valid_equation_count
        
    except Exception as e:
        print(f"Error checking overlap: {e}")
        return 0, 0, 0

def update_error_results_saving(error_output_file, error_results):
    """Update how error detector results are saved to include basic confidence information"""
    with open(error_output_file, 'w') as f:
        # Convert tensors to Python types
        serializable_results = {}
        for detector_type in error_results:
            serializable_results[detector_type] = [
                {
                    "layer": layer,
                    "accuracy": float(acc), 
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1)
                }
                for layer, acc, prec, rec, f1 in error_results[detector_type]
            ]
        
        json.dump(serializable_results, f, indent=2)
    print(f"Error detector results saved to {error_output_file}")


def main():
    """Test probes trained in the first code on data from the second code"""
    # Specify device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Parameter processing
    if len(sys.argv) < 3:
        print("Usage: python script.py <probes_folder> <model_name> [training_data_path]")
        sys.exit(1)
        
    probes_folder = sys.argv[1]   
    model_name = sys.argv[2]
    gsm8k_folder = "gsm8k"
    
    # Optional training data path
    training_data_path = sys.argv[3] if len(sys.argv) > 3 else f"{probes_folder}/training_samples"
    
    # Target digit position
    target_digit_index = 3
    
    # Create output directory for detailed results
    detailed_results_dir = f"error_detector_detailed_results_{model_name.split('/')[-1]}"
    os.makedirs(detailed_results_dir, exist_ok=True)
    print(f"Created output directory: {detailed_results_dir}")
    
    # Load data from the second code
    print("Loading activations from GSM8K data...")
    activations = {}
    activation_folder = os.path.join(gsm8k_folder, f"{model_name.split('/')[-1]}_activation")
    
    for variant_index in range(0, 560):
        try:
            activation_path = os.path.join(activation_folder, f"activation_{variant_index}")
            if os.path.exists(activation_path):
                variant_activation = torch.load(activation_path, map_location=torch.device(device))
                for i in variant_activation:
                    activations[i] = variant_activation[i]
        except Exception as e:
            print(f"Error loading activation_{variant_index}: {e}")
    
    if not activations:
        print("No activations loaded. Check the path and file availability.")
        sys.exit(1)
    
    print(f"Loaded activations for {len(activations)} samples")
    
    # Load model responses and extract equations
    print("Loading model responses and extracting equations...")
    response_data = {}
    left_equations = {}
    model_answers = {}
    
    response_folder = os.path.join(gsm8k_folder, "model_response")
    mode = "_all"  # Default value from the second code
    
    for dataset_index in range(0, 560, 10):
        results_file = os.path.join(response_folder, f"{model_name.split('/')[-1]}_evaluation_results_gen{mode}_{dataset_index}.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                try:
                    results = json.load(f)
                    for variant_index in results:
                        data = results[str(variant_index)]["results"]
                        for data_index in range(len(data)):
                            response_data[(int(variant_index), data_index)] = data[data_index]
                            
                            selected_equation = data[data_index].get("selected_equation", "")
                            if selected_equation and '=' in selected_equation:
                                parts = selected_equation.split('=', 1)
                                left = parts[0].strip()
                                right = parts[1].strip() if len(parts) > 1 else ""
                                left_equations[(int(variant_index), data_index)] = left
                                model_answers[(int(variant_index), data_index)] = right
                            else:
                                left_equations[(int(variant_index), data_index)] = ""
                                model_answers[(int(variant_index), data_index)] = ""
                except Exception as e:
                    print(f"Error processing {results_file}: {e}")
    
    if not left_equations:
        print("No equation data extracted. Check the path and file availability.")
        sys.exit(1)
    
    print(f"Extracted equations for {len(left_equations)} samples")
    
    # Prepare indices for testing
    index_starting_by_digit_correct = {}
    index_starting_by_digit_wrong = {}
    
    for i in range(10):
        index_starting_by_digit_correct[i] = []
        index_starting_by_digit_wrong[i] = []

    for i in left_equations:
        if i not in activations:
            continue
            
        left, right = left_equations[i], model_answers[i]
        try:
            if left != "":  
                int(right)  # Make sure the right side is a valid integer
                left_value = eval(left.strip(), {"__builtins__": None}, {})
                if left_value < 1000:
                    if left_value == int(right):
                        index_starting_by_digit_correct[get_digit(left_value, target_digit_index)].append(i)
                    else:
                        index_starting_by_digit_wrong[get_digit(left_value, target_digit_index)].append(i)
        except Exception:
            continue

    # Prepare balanced test samples for regular probes
    random.seed(42)
    test_samples = []
    
    for i in range(10):
        combined_samples = index_starting_by_digit_correct[i] + index_starting_by_digit_wrong[i]
        if combined_samples:
            test_samples.extend(
                random.sample(
                    combined_samples, 
                    min(100, len(combined_samples))
                )
            )
    
    print(f"Created balanced test set with {len(test_samples)} samples for regular probes")
    
    # Check overlap between test samples and training data
    print("\n=== Checking overlap with training data ===")
    overlap_ratio, overlap_count, valid_count = check_equation_overlap(
        test_samples,
        left_equations,
        training_data_path
    )
    
    # Prepare balanced test samples for error detection
    error_test_samples = []
    for i in range(10):
        # Add correct samples
        if index_starting_by_digit_correct[i]:
            error_test_samples.extend(
                random.sample(
                    index_starting_by_digit_correct[i], 
                    min(50, len(index_starting_by_digit_correct[i]))
                )
            )
        # Add wrong samples
        if index_starting_by_digit_wrong[i]:
            error_test_samples.extend(
                random.sample(
                    index_starting_by_digit_wrong[i], 
                    min(50, len(index_starting_by_digit_wrong[i]))
                )
            )
    
    print(f"Created balanced test set with {len(error_test_samples)} samples for error detection")
    
    # Check overlap for error detection samples as well
    print("\n=== Checking overlap for error detection samples ===")
    error_overlap_ratio, error_overlap_count, error_valid_count = check_equation_overlap(
        error_test_samples,
        left_equations,
        training_data_path
    )
    
    # Load probe folders and file structure
    probe_types = ["linear", "mlp", "circular", "logistic"] 
    probe_targets = {
        "output_probe": lambda idx: get_digit(int(model_answers[idx]), target_digit_index),
        "gt_probe": lambda idx: get_digit(eval(left_equations[idx].strip(), {"__builtins__": None}, {}), target_digit_index)
    }
    
    test_functions = {
        "linear": test_ridge_probe,
        "mlp": test_mlp_probe,
        "circular": test_circular_probe,
        "logistic": test_logistic_probe
    }
    
    # Define error detector types and functions
    error_detector_types = ["logistic_seperately", "mlp", "mlp_seperately", "circular_seperately", "circular_jointly"]
    
    test_error_detector_function = {
        "logistic_seperately": test_logistic_error_detector_seperately,
        "mlp": test_mlp_error_detector,
        "mlp_seperately": test_mlp_error_detector_seperately,
        "circular_seperately": test_circular_error_detector_seperately,
        "circular_jointly": test_circular_error_detector_jointly
    }
    
    # Test each probe
    results = {}
    error_results = {}
    

    # Test regular probes
    for probe_type in probe_types:
        results[probe_type] = {}
        probe_type_folder = os.path.join(probes_folder, f"{probe_type}")
        
        if not os.path.exists(probe_type_folder):
            print(f"Probe folder {probe_type_folder} not found, skipping...")
            continue
            
        for probe_target_name, target_function in probe_targets.items():
            probe_path = os.path.join(probe_type_folder, f"{probe_target_name}_digit{target_digit_index}")
            
            if not os.path.exists(probe_path):
                print(f"Probe not found at {probe_path}, skipping...")
                continue
                
            print(f"\n===== Testing {probe_type} {probe_target_name} =====")
            
            try:
                # Load probes (list of tuples: (accuracy, probe))
                trained_probes = torch.load(probe_path, map_location=torch.device(device))
                results[probe_type][probe_target_name] = []
                
                # Get layer info from the data
                sample_key = next(iter(activations))

                # Get model configuration
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                num_layers = getattr(config, "num_hidden_layers")
                print(f"Number of layers: {num_layers}")
                
                # Determine start layer (some models have embeddings as layer 0)
                if len(activations[sample_key]) == num_layers + 1:
                    start_layer = 1
                else:
                    start_layer = 0
                            
                # For each layer probe
                for layer_idx, (_, trained_probe) in enumerate(trained_probes):
                    # Check if the layer exists in our data
                    if layer_idx >= num_layers:
                        print(f"Layer {layer_idx} exceeds available layers in the data, skipping...")
                        continue
                        
                    print(f"\n--- Testing on layer {layer_idx} ---")
                    
                    # Prepare test data for this layer
                    X = []
                    Y = []
                    
                    for idx in test_samples:
                        if idx in activations and layer_idx < len(activations[idx]):
                            try:
                                # Get representation from the layer
                                x = activations[idx][layer_idx+start_layer][0][-1]
                                X.append(x)
                                
                                # Get target value based on probe target type
                                Y.append(torch.tensor(target_function(idx)))
                            except Exception as e:
                                print(f"Error processing sample {idx} for layer {layer_idx}: {e}")
                                continue
                    
                    if not X or not Y:
                        print(f"No valid data for layer {layer_idx}, skipping...")
                        continue
                        
                    # Convert lists to tensors
                    X = torch.stack(X).to(device)
                    Y = torch.stack(Y).to(device)
                    
                    # Test the probe
                    try:
                        accuracy = test_functions[probe_type](X, Y, trained_probe, device)
                        print(f"Accuracy of {probe_type} {probe_target_name} on layer {layer_idx}: {accuracy:.4f}")
                        results[probe_type][probe_target_name].append((layer_idx, accuracy))
                    except Exception as e:
                        print(f"Error testing probe on layer {layer_idx}: {e}")
            except Exception as e:
                print(f"Error loading or testing {probe_path}: {e}")
    
    # Test error detector probes
    for detector_type in error_detector_types:
        error_results[detector_type] = []
        error_detector_path = os.path.join(probes_folder, "error_detectors", f"{detector_type}_digit{target_digit_index}")
        
        if not os.path.exists(error_detector_path):
            print(f"Error detector not found at {error_detector_path}, skipping...")
            continue
            
        print(f"\n===== Testing {detector_type} error detector =====")
        
        try:
            # Load error detectors (list of tuples: (accuracy, detector))
            trained_detectors = torch.load(error_detector_path, map_location=torch.device(device))
            
            # Get layer info from the data
            sample_key = next(iter(activations))

            # Get model configuration
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            num_layers = getattr(config, "num_hidden_layers")
            
            # Determine start layer
            if len(activations[sample_key]) == num_layers + 1:
                start_layer = 1
            else:
                start_layer = 0
            
            # For each layer detector
            for layer_idx, (_, _, _, _, trained_detector) in enumerate(trained_detectors):

                # Check if the layer exists in our data
                if layer_idx >= num_layers:
                    print(f"Layer {layer_idx} exceeds available layers in the data, skipping...")
                    continue
                    
                print(f"\n--- Testing error detector on layer {layer_idx} ---")
                
                # Prepare test data for error detection
                X = []
                Y = []  # Binary: 1 if correct, 0 if wrong
                Y_model = []  # Model's digit prediction
                Y_true = []  # True digit
                sample_indices = []  # Store indices for reference
                
                for idx in error_test_samples:
                    if idx in activations and layer_idx < len(activations[idx]):
                        try:
                            # Get representation from the layer
                            x = activations[idx][layer_idx+start_layer][0][-1]
                            X.append(x)
                            
                            # Get ground truth and model prediction
                            true_value = eval(left_equations[idx].strip(), {"__builtins__": None}, {})
                            model_value = int(model_answers[idx])
                            
                            true_digit = get_digit(true_value, target_digit_index)
                            model_digit = get_digit(model_value, target_digit_index)
                            
                            # Binary label: 1 if correct, 0 if wrong
                            is_correct = true_digit == model_digit
                            Y.append(torch.tensor(1 if is_correct else 0))
                            
                            # Store digits for separate detectors
                            Y_model.append(torch.tensor(model_digit))
                            Y_true.append(torch.tensor(true_digit))
                            
                            # Store index
                            sample_indices.append(idx)
                            
                        except Exception as e:
                            print(f"Error processing sample {idx} for layer {layer_idx}: {e}")
                            continue
                
                if not X:
                    print(f"No valid data for layer {layer_idx}, skipping...")
                    continue
                    
                # Convert lists to tensors
                X = torch.stack(X).to(device)
                
                # Test the error detector based on its type
                try:
                    if detector_type in ["mlp", "circular_jointly"]:
                        # Binary detectors use Y
                        Y = torch.stack(Y).to(device)
                        accuracy, precision, recall, f1, detailed_results = test_error_detector_function[detector_type](X, Y, trained_detector, device)
                    else:
                        # Separate detectors use Y_model and Y_true
                        Y_model = torch.stack(Y_model).to(device)
                        Y_true = torch.stack(Y_true).to(device)
                        accuracy, precision, recall, f1, detailed_results = test_error_detector_function[detector_type](X, Y_model, Y_true, trained_detector, device)
                        
                    # Add sample indices to detailed results
                    for i, result in enumerate(detailed_results):
                        result["idx"] = sample_indices[i]
                    
                    # Save detailed results with confidence scores
                    save_detector_results(detailed_results, detector_type, layer_idx, detailed_results_dir)
                    
                    print(f"Accuracy of {detector_type} error detector on layer {layer_idx}: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
                    error_results[detector_type].append((layer_idx, accuracy, precision, recall, f1))
                except Exception as e:
                    print(f"Error testing detector on layer {layer_idx}: {e}")
        except Exception as e:
            print(f"Error loading or testing {error_detector_path}: {e}")
    

    
    # Print summary of results for error detectors
    print("\n===== Error Detector Results Summary =====")
    for detector_type in error_results:
        print(f"\n{detector_type} error detector:")
        
        if not error_results[detector_type]:
            print("  No results available")
            continue
            
        # Find best performing layer by accuracy
        best_layer, best_acc = max([(layer, acc) for layer, acc, _, _, _ in error_results[detector_type]], key=lambda x: x[1])
        # Get other metrics for best layer
        for layer, acc, prec, rec, f1 in error_results[detector_type]:
            if layer == best_layer:
                best_prec, best_rec, best_f1 = prec, rec, f1
                break
                
        print(f"  Best layer: {best_layer}, Accuracy: {best_acc:.4f}, Precision: {best_prec:.4f}, Recall: {best_rec:.4f}, F1-score: {best_f1:.4f}")
        
        # Print all results
        print("  All layers:")
        for layer, acc, prec, rec, f1 in sorted(error_results[detector_type], key=lambda x: x[0]):
            print(f"    Layer {layer}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    
    # Save results to file
    output_file = f"test_2_shots_probe_on_gsm8k_{model_name.split('/')[-1]}.json"
    error_output_file = f"test_2_shots_error_detector_on_gsm8k_{model_name.split('/')[-1]}.json"
    confidence_output_file = f"error_detector_confidence_analysis_{model_name.split('/')[-1]}.json"
    overlap_file = f"equation_overlap_analysis_{model_name.split('/')[-1]}.json"
    
    # Save regular probe results
    with open(output_file, 'w') as f:
        # Convert tensors to Python types
        serializable_results = {}
        for probe_type in results:
            serializable_results[probe_type] = {}
            for probe_target in results[probe_type]:
                serializable_results[probe_type][probe_target] = [
                    (layer, float(acc)) for layer, acc in results[probe_type][probe_target]
                ]
        
        json.dump(serializable_results, f, indent=2)
    
    # Save error detector results
    with open(error_output_file, 'w') as f:
        # Convert tensors to Python types
        serializable_results = {}
        for detector_type in error_results:
            serializable_results[detector_type] = [
                {
                    "layer": layer,
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1)
                }
                for layer, acc, prec, rec, f1 in error_results[detector_type]
            ]
        
        json.dump(serializable_results, f, indent=2)
    
    # Save basic confidence information for each detector type and layer
    with open(confidence_output_file, 'w') as f:
        confidence_info = {}
        
        for detector_type in error_results:
            confidence_info[detector_type] = {}
            detector_dir = os.path.join(detailed_results_dir, detector_type)
            
            if os.path.exists(detector_dir):
                for layer, _, _, _, _ in error_results[detector_type]:
                    results_file = os.path.join(detector_dir, f"layer_{layer}_results.json")
                    
                    if os.path.exists(results_file):
                        try:
                            with open(results_file, 'r') as rf:
                                layer_results = json.load(rf)
                                
                                # Calculate basic confidence statistics
                                confidences = [r.get("confidence", 0) for r in layer_results]
                                correct_confidences = [r.get("confidence", 0) for r in layer_results if r.get("is_correct", False)]
                                incorrect_confidences = [r.get("confidence", 0) for r in layer_results if not r.get("is_correct", True)]
                                
                                if confidences:
                                    confidence_info[detector_type][f"layer_{layer}"] = {
                                        "avg_confidence": sum(confidences) / len(confidences),
                                        "avg_confidence_correct": sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0,
                                        "avg_confidence_incorrect": sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0,
                                        "num_samples": len(confidences),
                                        "num_correct": len(correct_confidences),
                                        "num_incorrect": len(incorrect_confidences)
                                    }
                        except Exception as e:
                            print(f"Error processing confidence for {detector_type}, layer {layer}: {e}")
        
        json.dump(confidence_info, f, indent=2)
    
    # Save overlap analysis
    with open(overlap_file, 'w') as f:
        overlap_data = {
            "test_samples": {
                "overlap_ratio": overlap_ratio,
                "overlap_count": overlap_count,
                "valid_count": valid_count
            },
            "error_detection_samples": {
                "overlap_ratio": error_overlap_ratio,
                "overlap_count": error_overlap_count,
                "valid_count": error_valid_count
            }
        }
        json.dump(overlap_data, f, indent=2)
    
    print(f"\nRegular probe results saved to {output_file}")
    print(f"Error detector results saved to {error_output_file}")
    print(f"Confidence analysis summary saved to {confidence_output_file}")
    print(f"Overlap analysis saved to {overlap_file}")
    print(f"Detailed error detector results saved to {detailed_results_dir}")


if __name__ == "__main__":
    main()