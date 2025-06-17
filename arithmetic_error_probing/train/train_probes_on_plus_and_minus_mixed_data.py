'''
python train_probes_mixed.py $model_name $sum_data_folder $difference_data_folder $sum_hidden_states_path $difference_hidden_states_path $base_folder
'''

import torch
import sys
import os
import arithmetic_error_probing.model as model
from arithmetic_error_probing.model import RidgeRegression, MultiClassLogisticRegression, MLP, CircularProbe, CircularErrorDetector, LinearBinaryClassifier, RidgeRegressionErrorDetector
from tqdm import tqdm
from transformers import AutoConfig
import random
import json
from arithmetic_error_probing.utils import random_select_tuples, seed_everything, get_digit, load_model_result_dic
from arithmetic_error_probing.model import (
    train_circular_probe, test_probe_circular,
    train_ridge_probe, test_probe_ridge,
    train_mlp_probe, test_probe_mlp,
    train_logistic_probe, test_probe_logistic,
    train_logistic_error_detector_seperately,
    test_logistic_error_detector_seperately,
    train_mlp_error_detector,
    train_mlp_error_detector_seperately,
    test_mlp_error_detector,
    test_mlp_error_detector_seperately,
    train_circular_error_detector_seperately,
    train_circular_error_detector_jointly,
    test_circular_error_detector_seperately,
    test_circular_error_detector_jointly
)

sys.modules['model'] = model

# Global variables
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = ""
sum_data_folder = ""
difference_data_folder = ""
sum_hidden_states_path = ""
difference_hidden_states_path = ""
base_folder = "probing_results"
target_digit_index = 3
train_ratio = 0.7
seed = 42

# Data containers
sum_num_to_hidden = {}
difference_num_to_hidden = {}
samples = []
samples_train = []
samples_test = []
sum_result_dic = {}
difference_result_dic = {}

def load_data():
    """Load both sum and difference data."""
    global sum_num_to_hidden, difference_num_to_hidden, samples, samples_train, samples_test
    global sum_result_dic, difference_result_dic

    # Load result dictionaries
    sum_result_dic = load_model_result_dic(sum_data_folder)
    difference_result_dic = load_model_result_dic(difference_data_folder)
    
    # Load hidden states
    print(f"Loading sum hidden states from {sum_hidden_states_path}")
    sum_num_to_hidden = torch.load(sum_hidden_states_path, map_location=torch.device(device), weights_only=False)
    
    print(f"Loading difference hidden states from {difference_hidden_states_path}")
    difference_num_to_hidden = torch.load(difference_hidden_states_path, map_location=torch.device(device), weights_only=False)
    
    # Process sum samples
    sum_samples = process_samples(sum_num_to_hidden, sum_result_dic, "sum")
    print(f"Sum samples: {len(sum_samples)}")
    
    # Process difference samples
    difference_samples = process_samples(difference_num_to_hidden, difference_result_dic, "difference")
    print(f"Difference samples: {len(difference_samples)}")
    
    # Balance sample counts
    min_count = min(len(sum_samples), len(difference_samples))
    sum_samples = sum_samples[:min_count]
    difference_samples = difference_samples[:min_count]
    
    # Combine samples
    samples = sum_samples + difference_samples
    random.shuffle(samples)
    
    # Split train/test
    samples_train = random_select_tuples(samples, int(train_ratio * len(samples)))
    samples_test = list(set(samples) - set(samples_train))
    
    print(f"Total samples: {len(samples)}, Train: {len(samples_train)}, Test: {len(samples_test)}")

def process_samples(num_to_hidden, result_dic, operation_type):
    """Process samples for one operation type using existing logic."""
    all_samples = list(num_to_hidden.keys())
    
    # Filter valid samples
    if operation_type == "sum":
        all_samples = [pair for pair in all_samples if int(pair[0]) + int(pair[1]) < 1000]
    else:  # difference
        all_samples = [pair for pair in all_samples if int(pair[0]) - int(pair[1]) >= 0]
    
    # Group by target digit
    index_by_digit = {}
    for i in range(10):
        index_by_digit[i] = []
    
    for pair in all_samples:
        output = result_dic[pair]
        digit = get_digit(output, target_digit_index)
        index_by_digit[digit].append(pair)
    
    # Balance samples by digit
    balanced_samples = []
    for i in range(10):
        if len(index_by_digit[i]) > 0:
            count = min(100, len(index_by_digit[i]))
            selected = random.sample(index_by_digit[i], count)
            # Add operation type to each sample
            balanced_samples.extend([(pair[0], pair[1], operation_type) for pair in selected])
    
    return balanced_samples

def get_arithmetic_result(i, j, operation_type):
    """Get ground truth result based on operation type."""
    if operation_type == "sum":
        return i + j
    elif operation_type == "difference":
        return i - j
    else:
        raise ValueError(f"Unknown operation type: {operation_type}")

def get_model_output(i, j, operation_type):
    """Get model output based on operation type."""
    if operation_type == "sum":
        return sum_result_dic[(i, j)]
    elif operation_type == "difference":
        return difference_result_dic[(i, j)]
    else:
        raise ValueError(f"Unknown operation type: {operation_type}")

def get_hidden_states(i, j, operation_type):
    """Get hidden states based on operation type."""
    if operation_type == "sum":
        return sum_num_to_hidden[(i, j)]
    elif operation_type == "difference":
        return difference_num_to_hidden[(i, j)]
    else:
        raise ValueError(f"Unknown operation type: {operation_type}")

def prepare_data_for_layer(layer_index, value_func, samples_list):
    """Prepare input and target tensors for training or testing."""
    X = []
    Y = []
    
    for i, j, op_type in samples_list:
        hidden_states = get_hidden_states(i, j, op_type)
        x = hidden_states[layer_index][0][-1]
        X.append(x)
        
        value = value_func(i, j, op_type)
        if isinstance(value, int) or isinstance(value, float):
            Y.append(torch.tensor(value))
        else:
            digit = get_digit(value, target_digit_index)
            Y.append(torch.tensor(digit))
    
    X = torch.stack(X).to(device)
    Y = torch.stack(Y).to(device)
    
    # Shuffle data
    perm = torch.randperm(X.shape[0])
    X = X[perm]
    Y = Y[perm]
    
    return X, Y

def prepare_error_data_for_layer(layer_index, model_func, gt_func, samples_list):
    """Prepare input and target tensors for error detection probing."""
    X = []
    Y_model = []
    Y_true = []
    Y_binary = []
    
    for i, j, op_type in samples_list:
        hidden_states = get_hidden_states(i, j, op_type)
        x = hidden_states[layer_index][0][-1]
        X.append(x)
        
        model_value = model_func(i, j, op_type)
        true_value = gt_func(i, j, op_type)
        
        model_digit = get_digit(model_value, target_digit_index)
        true_digit = get_digit(true_value, target_digit_index)
        
        Y_model.append(torch.tensor(model_digit))
        Y_true.append(torch.tensor(true_digit))
        
        is_correct = model_digit == true_digit
        Y_binary.append(torch.tensor(1 if is_correct else 0))
    
    X = torch.stack(X).to(device)
    Y_model = torch.stack(Y_model).to(device)
    Y_true = torch.stack(Y_true).to(device)
    Y_binary = torch.stack(Y_binary).to(device)
    
    # Shuffle data consistently
    perm = torch.randperm(X.shape[0])
    X = X[perm]
    Y_model = Y_model[perm]
    Y_true = Y_true[perm]
    Y_binary = Y_binary[perm]
    
    return X, Y_model, Y_true, Y_binary

def get_first_number_tens_digit(i, j, operation_type):
    return (i // 10) % 10

def get_first_number_ones_digit(i, j, operation_type):
    return i % 10

def get_second_number_tens_digit(i, j, operation_type):
    return (j // 10) % 10

def get_second_number_ones_digit(i, j, operation_type):
    return j % 10

def train_all_probes_and_error_detectors():
    """Train probes and error detectors for each layer."""
    # Get model configuration
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = getattr(config, "num_hidden_layers")
    print(f"Number of layers: {num_layers}")
    
    # Determine start layer
    sample_key = samples[0]
    sample_hidden = get_hidden_states(sample_key[0], sample_key[1], sample_key[2])
    if len(sample_hidden) == num_layers + 1:
        start_layer = 1
    else:
        start_layer = 0
    
    # Define probe types and functions
    probes = ["circular", "linear", "mlp", "logistic"]
    probe_types = [
        "gt_probe", 
        "output_probe", 
        "first_num_probe", 
        "second_num_probe",
        "first_num_tens_digit_probe",    
        "first_num_ones_digit_probe",   
        "second_num_tens_digit_probe",   
        "second_num_ones_digit_probe"    
    ]

    train_probe_function = {
        "circular": train_circular_probe,
        "linear": train_ridge_probe,
        "mlp": train_mlp_probe,
        "logistic": train_logistic_probe
    }

    test_probe_function = {
        "circular": test_probe_circular,
        "linear": test_probe_ridge,
        "mlp": test_probe_mlp,
        "logistic": test_probe_logistic
    }
    
    # Define error detector types and functions
    error_detector_types = ["logistic_seperately", "mlp", "mlp_seperately", "circular_seperately", "circular_jointly"]
    
    train_error_detector_function = {
        "logistic_seperately": train_logistic_error_detector_seperately,
        "mlp": train_mlp_error_detector,
        "mlp_seperately": train_mlp_error_detector_seperately,
        "circular_seperately": train_circular_error_detector_seperately,
        "circular_jointly": train_circular_error_detector_jointly
    }
    
    test_error_detector_function = {
        "logistic_seperately": test_logistic_error_detector_seperately,
        "mlp": test_mlp_error_detector,
        "mlp_seperately": test_mlp_error_detector_seperately,
        "circular_seperately": test_circular_error_detector_seperately,
        "circular_jointly": test_circular_error_detector_jointly
    }
    
    # Initialize accuracy dictionaries
    accuracy_and_models = {}
    for probe in probes:
        accuracy_and_models[probe] = {
            "gt_probe": [],
            "output_probe": [],
            "first_num_probe": [],
            "second_num_probe": [],
            "first_num_tens_digit_probe": [],
            "first_num_ones_digit_probe": [],
            "second_num_tens_digit_probe": [],
            "second_num_ones_digit_probe": []
        }
    
    # Initialize error detector accuracy dictionary
    error_detector_accuracy = {}
    for detector_type in error_detector_types:
        error_detector_accuracy[detector_type] = []
    
    # Train probes for each type and layer
    for probe in probes:
        print(f"\n=== Training {probe} probes ===")
        
        for probe_type in probe_types:
            print(f"\n--- Training {probe} on {probe_type} ---")
            
            for layer_index in tqdm(range(start_layer, len(sample_hidden))):
                seed_everything(seed)
                
                # Select the appropriate value function
                if probe_type == "gt_probe":
                    value_func = lambda x,y,op: get_digit(get_arithmetic_result(x,y,op), target_digit_index)
                elif probe_type == "output_probe":
                    value_func = lambda x,y,op: get_digit(get_model_output(x,y,op), target_digit_index)
                elif probe_type == "first_num_probe":
                    value_func = lambda x,y,op: get_digit(x, target_digit_index)
                elif probe_type == "second_num_probe":
                    value_func = lambda x,y,op: get_digit(y, target_digit_index)
                elif probe_type == "first_num_tens_digit_probe":
                    value_func = get_first_number_tens_digit
                elif probe_type == "first_num_ones_digit_probe":
                    value_func = get_first_number_ones_digit
                elif probe_type == "second_num_tens_digit_probe":
                    value_func = get_second_number_tens_digit
                elif probe_type == "second_num_ones_digit_probe":
                    value_func = get_second_number_ones_digit
                
                # Prepare data for this layer
                X_train, Y_train = prepare_data_for_layer(layer_index, value_func, samples_train)
                X_test, Y_test = prepare_data_for_layer(layer_index, value_func, samples_test)
                
                # Train the probe
                trained_probe = train_probe_function[probe](X_train, Y_train)
                
                # Test the probe
                accuracy = test_probe_function[probe](X_test, Y_test, trained_probe)
                
                print(f"Accuracy of {probe} probe on {probe_type} on layer {layer_index} is {accuracy}.")
                accuracy_and_models[probe][probe_type].append((accuracy, trained_probe))
    
    # Train error detectors for each type and layer
    print(f"\n=== Training error detectors ===")
    
    for detector_type in error_detector_types:
        print(f"\n--- Training {detector_type} error detector ---")
        
        for layer_index in tqdm(range(start_layer, len(sample_hidden))):
            seed_everything(seed)
            
            # Prepare data for error detection
            X_train, Y_model_train, Y_true_train, Y_binary_train = prepare_error_data_for_layer(
                layer_index, get_model_output, get_arithmetic_result, samples_train
            )
            X_test, Y_model_test, Y_true_test, Y_binary_test = prepare_error_data_for_layer(
                layer_index, get_model_output, get_arithmetic_result, samples_test
            )
            
            # Train the detector based on its type
            if detector_type in ["mlp", "circular_jointly"]:
                # Binary detectors use Y_binary
                trained_detector = train_error_detector_function[detector_type](X_train, Y_binary_train)
                accuracy, precision, recall, f1 = test_error_detector_function[detector_type](X_test, Y_binary_test, trained_detector)
            else:
                # Separate detectors use Y_model and Y_true
                trained_detector = train_error_detector_function[detector_type](X_train, Y_model_train, Y_true_train)
                accuracy, precision, recall, f1 = test_error_detector_function[detector_type](X_test, Y_model_test, Y_true_test, trained_detector)
            
            print(f"Accuracy of {detector_type} error detector on layer {layer_index}: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
            error_detector_accuracy[detector_type].append((accuracy, precision, recall, f1, trained_detector))
    
    return accuracy_and_models, error_detector_accuracy

def save_results(results, error_results):
    """Save the training results and test samples to files."""
    os.makedirs(base_folder, exist_ok=True)
    
    # Save test samples
    test_samples_path = f"{base_folder}/test_samples_mixed_digit{target_digit_index}"
    torch.save(samples_test, test_samples_path)
    print(f"Saved test samples to {test_samples_path}")
    
    # Print and save results for each probe type
    for probe_type, type_results in results.items():
        print(f"\n{probe_type.capitalize()} probe results:")
        
        probe_dir = f"{base_folder}/{probe_type}"
        os.makedirs(probe_dir, exist_ok=True)
        
        for name, probes in type_results.items():
            if not probes:
                continue
                
            best_idx, (best_acc, _) = max(enumerate(probes), key=lambda x: x[1][0])
            print(f"  {name}: Layer {best_idx}, Accuracy {best_acc:.4f}")
            
            torch.save(probes, f"{probe_dir}/{name}_mixed_digit{target_digit_index}")
    
    # Create directory for error detectors
    error_dir = f"{base_folder}/error_detectors"
    os.makedirs(error_dir, exist_ok=True)
    
    # Print and save results for each error detector type
    print(f"\nError detector results:")
    for detector_type, detectors in error_results.items():
        if not detectors:
            continue
            
        # Find best performing detector and its layer
        best_idx, (best_acc, best_prec, best_rec, best_f1, _) = max(enumerate(detectors), key=lambda x: x[1][0])  # Sort by accuracy
        print(f"  {detector_type}: Layer {best_idx}, Accuracy {best_acc:.4f}, Precision: {best_prec:.4f}, Recall: {best_rec:.4f}, F1-score: {best_f1:.4f}")
        
        # Save results
        torch.save(detectors, f"{error_dir}/{detector_type}_mixed_digit{target_digit_index}")

def main():
    """Main function to run the mixed probe training pipeline."""
    global model_name, sum_data_folder, difference_data_folder
    global sum_hidden_states_path, difference_hidden_states_path, base_folder
    
    # Parse command line arguments
    if len(sys.argv) < 7:
        print("Usage: python script.py <model_name> <sum_data_folder> <difference_data_folder> <sum_hidden_states_path> <difference_hidden_states_path> <base_folder>")
        sys.exit(1)
        
    model_name = sys.argv[1] 
    sum_data_folder = sys.argv[2]
    difference_data_folder = sys.argv[3]
    sum_hidden_states_path = sys.argv[4]
    difference_hidden_states_path = sys.argv[5]
    base_folder = sys.argv[6]
    random.seed(42)

    print(f"Training on mixed sum and difference data")
    
    # Load data
    load_data()
    
    # Save train/test splits
    os.makedirs(base_folder, exist_ok=True)
    torch.save(samples_train, f"{base_folder}/training_samples_mixed")
    torch.save(samples_test, f"{base_folder}/testing_samples_mixed")
    
    # Train all probes and error detectors
    results, error_results = train_all_probes_and_error_detectors()
    
    # Save results
    save_results(results, error_results)

if __name__ == "__main__":
    main()