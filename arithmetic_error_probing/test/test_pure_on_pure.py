'''
python -m arithmetic_error_probing.test.test_pure_on_pure \
google/gemma-2-2b-it \
gemma-2-2b-it_2_shots_3_digit_sum_output \
arithmetic_error_probing/generate_response_and_activation/gemma-2-2b-it_sum_3_num_to_hidden_2_shots \
gemma-2-2b-it_difference_probing_results_2_shots \
gemma-2-2b-it_probing_results_2_shots \
sum \
difference

'''
    

import torch
import sys
import os
from tqdm import tqdm
from transformers import AutoConfig
import random
from arithmetic_error_probing.utils import random_select_tuples, seed_everything, get_digit, load_model_result_dic
from arithmetic_error_probing.model import (
    test_probe_circular, test_probe_ridge, test_probe_mlp, test_probe_logistic,
    test_logistic_error_detector_seperately, test_mlp_error_detector, 
    test_mlp_error_detector_seperately, test_circular_error_detector_seperately, 
    test_circular_error_detector_jointly
)
import arithmetic_error_probing.model as model
sys.modules['model'] = model



# Global variables
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = ""
data_folder = ""
hidden_states_path = ""
model_base_folder = ""
test_data_base_folder = ""
test_arithmetic_mode = ""
model_arithmetic_mode = ""
target_digit_index = 3
train_ratio = 0.7
seed = 42

# Arithmetic functions
test_arithmetic_function = None
model_arithmetic_function = None
num_to_hidden = {}
samples_test = []
result_dic = {}
probe_results = {}
error_detector_results = {}

def setup_arithmetic_functions():
    """Setup arithmetic functions for test and model."""
    global test_arithmetic_function, model_arithmetic_function
    
    def get_function(mode):
        if mode == "sum":
            return lambda a, b: a + b
        elif mode == "difference":
            return lambda a, b: a - b
        elif mode == "product":
            return lambda a, b: a * b
        else:
            raise ValueError(f"Unsupported arithmetic mode: {mode}")
    
    test_arithmetic_function = get_function(test_arithmetic_mode)
    model_arithmetic_function = get_function(model_arithmetic_mode)

def load_test_data():
    """Load pre-saved test data."""
    global num_to_hidden, samples_test, result_dic
    
    print(f"Loading hidden states from {hidden_states_path}")
    num_to_hidden = torch.load(hidden_states_path, map_location=torch.device(device), weights_only=False)
    result_dic = load_model_result_dic(data_folder)
    
    # Load pre-saved test samples
    test_samples_path = f"{test_data_base_folder}/testing_samples"
    if not os.path.exists(test_samples_path):
        raise FileNotFoundError(f"Test samples file not found at {test_samples_path}")
    
    samples_test = torch.load(test_samples_path, map_location=torch.device(device))
    print(f"Loaded {len(samples_test)} test samples")

def prepare_data_for_layer(layer_index, value_func, samples_list):
    """Prepare test data - same as training script."""
    X, Y = [], []
    
    for i, j in samples_list:
        hidden_states = num_to_hidden[(i, j)]
        x = hidden_states[layer_index][0][-1]
        X.append(x)
        
        value = value_func(i, j)
        if isinstance(value, int) or isinstance(value, float):
            Y.append(torch.tensor(value))
        else:
            digit = get_digit(value, target_digit_index)
            Y.append(torch.tensor(digit))
    
    X = torch.stack(X).to(device)
    Y = torch.stack(Y).to(device)
    return X, Y

def prepare_error_data_for_layer(layer_index, model_func, gt_func, samples_list):
    """Prepare error detection test data - same as training script."""
    X, Y_model, Y_true, Y_binary = [], [], [], []
    
    for i, j in samples_list:
        hidden_states = num_to_hidden[(i, j)]
        x = hidden_states[layer_index][0][-1]
        X.append(x)
        
        model_value = model_func(i, j)
        true_value = gt_func(i, j)
        
        model_digit = get_digit(model_value, target_digit_index)
        true_digit = get_digit(true_value, target_digit_index)
        
        Y_model.append(torch.tensor(model_digit))
        Y_true.append(torch.tensor(true_digit))
        Y_binary.append(torch.tensor(1 if model_digit == true_digit else 0))
    
    X = torch.stack(X).to(device)
    Y_model = torch.stack(Y_model).to(device)
    Y_true = torch.stack(Y_true).to(device)
    Y_binary = torch.stack(Y_binary).to(device)
    return X, Y_model, Y_true, Y_binary

def test_cross_task():
    """Test models trained on one arithmetic task using another task's data."""
    global probe_results, error_detector_results
    # Initialize result dictionaries
    probe_results = {}
    for probe in ["circular", "linear", "mlp", "logistic"]:
        probe_results[probe] = {
            "gt_probe": [],
            "output_probe": [],
            "first_num_probe": [],
            "second_num_probe": [],
            "first_num_tens_digit_probe": [],
            "first_num_ones_digit_probe": [],
            "second_num_tens_digit_probe": [],
            "second_num_ones_digit_probe": []
        }
    
    error_detector_results = {}
    for detector_type in ["logistic_seperately", "mlp", "mlp_seperately", "circular_seperately", "circular_jointly"]:
        error_detector_results[detector_type] = []

    # Define test functions using test arithmetic mode
    def get_test_result(i, j):
        return test_arithmetic_function(i, j)
    
    def get_model_output(i, j):
        return result_dic[(i, j)]
    
    def get_first_number_tens_digit(i, j):
        return (i // 10) % 10
    
    def get_first_number_ones_digit(i, j):
        return i % 10
    
    def get_second_number_tens_digit(i, j):
        return (j // 10) % 10
    
    def get_second_number_ones_digit(i, j):
        return j % 10
    
    # Get model configuration
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = getattr(config, "num_hidden_layers")
    start_layer = 1 if len(num_to_hidden[list(num_to_hidden.keys())[0]]) == num_layers + 1 else 0
    
    # Define probe and detector configurations
    probes = ["circular", "linear", "mlp", "logistic"]
    probe_types = ["gt_probe", "output_probe", "first_num_probe", "second_num_probe",
                   "first_num_tens_digit_probe", "first_num_ones_digit_probe", 
                   "second_num_tens_digit_probe", "second_num_ones_digit_probe"]
    
    test_probe_function = {
        "circular": test_probe_circular,
        "linear": test_probe_ridge, 
        "mlp": test_probe_mlp,
        "logistic": test_probe_logistic
    }
    
    error_detector_types = ["logistic_seperately", "mlp", "mlp_seperately", 
                           "circular_seperately", "circular_jointly"]
    
    test_error_detector_function = {
        "logistic_seperately": test_logistic_error_detector_seperately,
        "mlp": test_mlp_error_detector,
        "mlp_seperately": test_mlp_error_detector_seperately,
        "circular_seperately": test_circular_error_detector_seperately,
        "circular_jointly": test_circular_error_detector_jointly
    }
    
    # Test probes
    for probe in probes:
        print(f"\n=== Testing {probe} probes (trained on {model_arithmetic_mode}, tested on {test_arithmetic_mode}) ===")
        
        for probe_type in probe_types:
            print(f"\n--- Testing {probe} on {probe_type} ---")
            
            # Load trained models
            model_path = f"{model_base_folder}/{probe}/{probe_type}_digit{target_digit_index}"
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue
                
            trained_models = torch.load(model_path, map_location=device)
            
            for layer_index in tqdm(range(start_layer, min(len(trained_models), len(num_to_hidden[list(num_to_hidden.keys())[0]])))):
                seed_everything(seed)
                
                # Select value function based on test arithmetic mode
                if probe_type == "gt_probe":
                    value_func = lambda x,y: get_digit(get_test_result(x,y), target_digit_index)
                elif probe_type == "output_probe":
                    value_func = lambda x,y: get_digit(get_model_output(x,y), target_digit_index)
                elif probe_type == "first_num_probe":
                    value_func = lambda x,y: get_digit(x, target_digit_index)
                elif probe_type == "second_num_probe":
                    value_func = lambda x,y: get_digit(y, target_digit_index)
                elif probe_type == "first_num_tens_digit_probe":
                    value_func = get_first_number_tens_digit
                elif probe_type == "first_num_ones_digit_probe":
                    value_func = get_first_number_ones_digit
                elif probe_type == "second_num_tens_digit_probe":
                    value_func = get_second_number_tens_digit
                elif probe_type == "second_num_ones_digit_probe":
                    value_func = get_second_number_ones_digit
                
                # Prepare test data
                X_test, Y_test = prepare_data_for_layer(layer_index, value_func, samples_test)
                
                # Test with trained model
                if layer_index < len(trained_models):
                    _, trained_probe = trained_models[layer_index]
                    accuracy = test_probe_function[probe](X_test, Y_test, trained_probe)
                    print(f"Accuracy of {probe} probe on {probe_type} on layer {layer_index} is {accuracy}.")
                    probe_results[probe][probe_type].append(accuracy)
    
    # Test error detectors
    print(f"\n=== Testing error detectors (trained on {model_arithmetic_mode}, tested on {test_arithmetic_mode}) ===")
    
    for detector_type in error_detector_types:
        print(f"\n--- Testing {detector_type} error detector ---")
        
        # Load trained models
        model_path = f"{model_base_folder}/error_detectors/{detector_type}_digit{target_digit_index}"
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
            
        trained_models = torch.load(model_path, map_location=device)
        
        for layer_index in tqdm(range(start_layer, min(len(trained_models), len(num_to_hidden[list(num_to_hidden.keys())[0]])))):
            seed_everything(seed)
            
            # Prepare error detection test data using test arithmetic mode
            X_test, Y_model_test, Y_true_test, Y_binary_test = prepare_error_data_for_layer(
                layer_index, get_model_output, get_test_result, samples_test
            )
            
            # Test with trained detector
            if layer_index < len(trained_models):
                trained_detector = trained_models[layer_index][-1]  # Last element is the model
                
                if detector_type in ["mlp", "circular_jointly"]:
                    accuracy, precision, recall, f1 = test_error_detector_function[detector_type](X_test, Y_binary_test, trained_detector)
                else:
                    accuracy, precision, recall, f1 = test_error_detector_function[detector_type](X_test, Y_model_test, Y_true_test, trained_detector)
                
                print(f"Accuracy of {detector_type} error detector on layer {layer_index}: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
                error_detector_results[detector_type].append((accuracy, precision, recall, f1)) 

def summarize_results():
    """Summarize cross-task testing results."""
    print(f"\n" + "="*80)
    print(f"CROSS-TASK TESTING SUMMARY")
    print(f"Models trained on: {model_arithmetic_mode}")
    print(f"Tested on: {test_arithmetic_mode}")
    print(f"="*80)
    
    # Summarize probe results
    for probe_type, type_results in probe_results.items():
        print(f"\n{probe_type.capitalize()} probe results:")
        
        for name, accuracies in type_results.items():
            if not accuracies:
                continue
                
            # Find best performing layer
            best_idx = accuracies.index(max(accuracies))
            best_acc = max(accuracies)
            print(f"  {name}: Layer {best_idx}, Accuracy {best_acc:.4f}")
    
    # Summarize error detector results
    print(f"\nError detector results:")
    for detector_type, results in error_detector_results.items():
        if not results:
            continue
            
        # Find best performing layer
        best_idx, (best_acc, best_prec, best_rec, best_f1) = max(enumerate(results), key=lambda x: x[1][0])
        print(f"  {detector_type}: Layer {best_idx}, Accuracy {best_acc:.4f}, Precision: {best_prec:.4f}, Recall: {best_rec:.4f}, F1-score: {best_f1:.4f}")


def main():
    """Main function for cross-task testing."""
    global model_name, data_folder, hidden_states_path, model_base_folder, test_data_base_folder, test_arithmetic_mode, model_arithmetic_mode
    
    if len(sys.argv) < 8:
        print("Usage: python script.py <model_name> <data_folder> <hidden_states_path> <model_base_folder> <test_data_base_folder> <test_arithmetic_mode> <model_arithmetic_mode>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    data_folder = sys.argv[2] 
    hidden_states_path = sys.argv[3]
    model_base_folder = sys.argv[4]
    test_data_base_folder = sys.argv[5]
    test_arithmetic_mode = sys.argv[6]
    model_arithmetic_mode = sys.argv[7]
    random.seed(42)
    
    print(f"Testing {model_arithmetic_mode}-trained models on {test_arithmetic_mode} data")
    
    setup_arithmetic_functions()
    load_test_data()
    test_cross_task()
    summarize_results()

if __name__ == "__main__":
    main()