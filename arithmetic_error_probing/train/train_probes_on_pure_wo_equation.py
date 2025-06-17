'''
python train_probes_on_pure.py $model_name $data_folder $hidden_states_path $base_folder
'''

import torch
import sys
import os
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

# Global variables and constants
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = ""
data_folder = ""
hidden_states_path = ""
base_folder = "probing_results"
target_digit_index = "start"  # Fixed target digit
train_ratio = 0.7
seed = 42

# Global data containers
num_to_hidden = {}
samples = []
samples_train = []
samples_test = []
result_dic = {}

# Load hidden states from file
def load_hidden_states():
    """Load hidden states from the specified path."""
    global num_to_hidden, samples, samples_train, samples_test

    def get_model_output(i, j):
        return result_dic[(i, j)]
    
    if not os.path.exists(hidden_states_path):
        raise FileNotFoundError(f"Hidden states file not found at {hidden_states_path}. Please ensure the file exists.")
    
    print(f"Loading hidden states from {hidden_states_path}")
    num_to_hidden = torch.load(hidden_states_path, map_location=torch.device(device))
    
    # Update samples to match loaded hidden states
    all_samples = list(num_to_hidden.keys())
    print(len(all_samples))
    #all_samples = [pair for pair in all_samples if int(pair[0]) + int(pair[1]) < 1000]

    if len(all_samples) > 0:
        sample = all_samples[0]
        output = get_model_output(sample[0], sample[1])
        digit = get_digit(output, target_digit_index)
        print(f"First sample: {sample}, model output: {output}, digit at position {target_digit_index}: {digit}")

    # Group samples by target digit
    index_starting_by_digit = {}
    for i in range(10):
        index_starting_by_digit[i] = []
    for i in all_samples:
        index_starting_by_digit[get_digit(get_model_output(i[0], i[1]), target_digit_index)].append(i)
    for i in range(10):
        print(f"digit {i}:{len(index_starting_by_digit[i])}")
        
    # Create balanced dataset
    samples = []
    for i in range(10):  
        if len(index_starting_by_digit[i]) > 0:
            samples += random.sample(index_starting_by_digit[i], min(100, len(index_starting_by_digit[i])))

    # Split into train and test sets
    samples_train = random_select_tuples(samples, int(train_ratio * len(samples)))
    samples_test = list(set(samples) - set(samples_train))
    print(f"Loaded hidden states for {len(samples)} samples")

# Prepare data for training and testing
def prepare_data_for_layer(layer_index, value_func, samples_list):
    """Prepare input and target tensors for training or testing."""
    X = []
    Y = []
    
    for i, j in samples_list:
        hidden_states = num_to_hidden[(i, j)]
        x = hidden_states[layer_index][0][-1]
        X.append(x)
        
        # Get the target value and extract the specific digit
        value = value_func(i, j)
        if isinstance(value, int) or isinstance(value, float):
            # The value_func already returns the target digit
            Y.append(torch.tensor(value))
        else:
            # The value_func returns a number which needs digit extraction
            digit = get_digit(value, target_digit_index)
            Y.append(torch.tensor(digit))
        
    X = torch.stack(X).to(device)
    Y = torch.stack(Y).to(device)
    
    # Shuffle data
    perm = torch.randperm(X.shape[0])
    X = X[perm]
    Y = Y[perm]
    
    return X, Y

# Prepare data for error detection probing
def prepare_error_data_for_layer(layer_index, model_func, gt_func, samples_list):
    """Prepare input and target tensors for error detection probing."""
    X = []
    Y_model = []
    Y_true = []
    Y_binary = []
    
    for i, j in samples_list:
        hidden_states = num_to_hidden[(i, j)]
        x = hidden_states[layer_index][0][-1]
        X.append(x)
        
        # Get model's predicted value and ground truth
        model_value = model_func(i, j)
        true_value = gt_func(i, j)
        
        # Extract the specific digit
        model_digit = get_digit(model_value, target_digit_index)
        true_digit = get_digit(true_value, target_digit_index)
        
        Y_model.append(torch.tensor(model_digit))
        Y_true.append(torch.tensor(true_digit))
        
        # Binary label: 1 if correct, 0 if wrong
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

#-----------------------------------------------------------------------------
# New digit extraction functions for tens and ones digits
#-----------------------------------------------------------------------------

# Function to get tens digit of first number
def get_first_number_tens_digit(i, j):
    # Extract tens digit (i // 10) % 10
    return (i // 10) % 10

# Function to get ones digit of first number
def get_first_number_ones_digit(i, j):
    # Extract ones digit i % 10
    return i % 10

# Function to get tens digit of second number
def get_second_number_tens_digit(i, j):
    # Extract tens digit (j // 10) % 10
    return (j // 10) % 10

# Function to get ones digit of second number
def get_second_number_ones_digit(i, j):
    # Extract ones digit j % 10
    return j % 10

#-----------------------------------------------------------------------------
# Main training and evaluation functions
#-----------------------------------------------------------------------------

# Train all probes and error detectors
def train_all_probes_and_error_detectors():
    """Train probes and error detectors for each layer."""
    # Value functions
    def get_sum(i, j):
        return i + j
    
    def get_model_output(i, j):
        return result_dic[(i, j)]
    
    def get_first_number(i, j):
        return i
    
    def get_second_number(i, j):
        return j
    
    # Get model configuration
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = getattr(config, "num_hidden_layers")
    print(f"Number of layers: {num_layers}")
    
    # Determine start layer (some models have embeddings as layer 0)
    if len(num_to_hidden[samples[0]]) == num_layers + 1:
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
            "first_num_tens_digit_probe": [],    # New probe type
            "first_num_ones_digit_probe": [],    # New probe type
            "second_num_tens_digit_probe": [],   # New probe type
            "second_num_ones_digit_probe": []    # New probe type
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
            
            for layer_index in tqdm(range(start_layer, len(num_to_hidden[samples[0]]))):
                # Set seed for reproducibility
                seed_everything(seed)
                
                # Select the appropriate value function
                if probe_type == "gt_probe":
                    value_func = lambda x,y: get_digit(get_sum(x,y),target_digit_index)
                elif probe_type == "output_probe":
                    value_func = lambda x,y: get_digit(get_model_output(x,y),target_digit_index)
                elif probe_type == "first_num_probe":
                    value_func = lambda x,y: get_digit(get_first_number(x,y),target_digit_index)
                elif probe_type == "second_num_probe":
                    value_func = lambda x,y: get_digit(get_second_number(x,y),target_digit_index)
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
        
        for layer_index in tqdm(range(start_layer, len(num_to_hidden[samples[0]]))):
            # Set seed for reproducibility
            seed_everything(seed)
            
            # Prepare data for error detection
            X_train, Y_model_train, Y_true_train, Y_binary_train = prepare_error_data_for_layer(
                layer_index, get_model_output, get_sum, samples_train
            )
            X_test, Y_model_test, Y_true_test, Y_binary_test = prepare_error_data_for_layer(
                layer_index, get_model_output, get_sum, samples_test
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

# Save probe and error detector results to files
def save_results(results, error_results):
    """Save the training results and test samples to files."""
    # Create base directory
    os.makedirs(base_folder, exist_ok=True)
    
    # Save test samples
    test_samples_path = f"{base_folder}/test_samples_digit{target_digit_index}"
    torch.save(samples_test, test_samples_path)
    print(f"Saved test samples to {test_samples_path}")
    
    # Print and save results for each probe type
    for probe_type, type_results in results.items():
        print(f"\n{probe_type.capitalize()} probe results:")
        
        # Create directory for this probe type
        probe_dir = f"{base_folder}/{probe_type}"
        os.makedirs(probe_dir, exist_ok=True)
        
        for name, probes in type_results.items():
            if not probes:
                continue
                
            # Find best performing probe and its layer
            best_idx, (best_acc, _) = max(enumerate(probes), key=lambda x: x[1][0])
            print(f"  {name}: Layer {best_idx}, Accuracy {best_acc:.4f}")
            
            # Save results
            torch.save(probes, f"{probe_dir}/{name}_digit{target_digit_index}")
    
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
        torch.save(detectors, f"{error_dir}/{detector_type}_digit{target_digit_index}")

# Main function
def main():
    """Main function to run the probe training pipeline."""
    global model_name, data_folder, hidden_states_path, base_folder, result_dic
    
    # Parse command line arguments
    if len(sys.argv) < 4:
        print("Usage: python script.py <model_name> <data_folder> <hidden_states_path> <base_folder>")
        sys.exit(1)
        
    model_name = sys.argv[1] 
    data_folder = sys.argv[2]
    hidden_states_path = sys.argv[3]
    base_folder = sys.argv[4]
    random.seed(42)

    # Load result dictionary
    result_dic = load_model_result_dic(data_folder)
    
    # Load hidden states
    load_hidden_states()
    os.makedirs(base_folder, exist_ok=True)
    torch.save(samples_train, f"{base_folder}/training_samples")
    torch.save(samples_test, f"{base_folder}/testing_samples")
    
    # Train all probes and error detectors
    results, error_results = train_all_probes_and_error_detectors()
    
    # Save results
    save_results(results, error_results)

if __name__ == "__main__":
    main()