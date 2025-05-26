'''
python train_probes_on_gsm8k google/gemma-2-2b-it
'''

import torch
import re
import sys
import os
from model import RidgeRegression, MultiClassLogisticRegression, MLP, CircularProbe, RidgeRegressionErrorDetector, LinearBinaryClassifier, CircularErrorDetector
from general_ps_utils import ModelAndTokenizer
from tqdm import tqdm
import json
import random
from utils import get_digit
from tqdm import tqdm

from model import (
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

device = "cuda"
#model_name = "google/gemma-2-2b-it"
model_name = sys.argv[1]
#mode = "_wrong_data_only"
#mode = "_all"
mode = "all"
if len(mode) != 0:
    mode = "_"+mode


#get all activation
activations = {}
activation_folder = f"gsm8k/{model_name.split('/')[-1]}_activation"
for variant_index in range(0,560):
    variant_activation = torch.load(f"{activation_folder}/activation_{variant_index}",map_location = torch.device(device))
    for i in variant_activation:
        activations[i] = variant_activation[i]

#get all model response
response_data = {}
response_folder = "gsm8k/model_response"
left_equations = {}
model_answers = {}
for dataset_index in range(0,560,10):
    results_file = f"{response_folder}/{model_name.split('/')[-1]}_evaluation_results_gen{mode}_{dataset_index}.json"
    with open(results_file, 'r') as f:
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

#prepare the data
num_layers = len(list(activations.values())[0]) #to
start_layer = 1

#Enhanced probe types and functions
probes = ["circular", "linear", "mlp", "logistic"]
probe_types = ["gt_probe", "output_probe"]

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

target_digit_index = 3
index_starting_by_digit_correct = {}
index_starting_by_digit_wrong = {}

for i in range(10):
    index_starting_by_digit_correct[i] = []
    index_starting_by_digit_wrong[i] = []
for i in left_equations:
    left, right = left_equations[i], model_answers[i]
    try:
        if left:  
            int_right = int(right)
            left_value = eval(left.strip(), {"__builtins__": None}, {})
            if left_value < 1000 and left_value == int_right: 
                index_starting_by_digit_correct[get_digit(left_value, target_digit_index)].append(i)
            if left_value < 1000 and left_value != int_right: 
                index_starting_by_digit_wrong[get_digit(left_value, target_digit_index)].append(i)
    except:
        pass

# Initialize accuracy dictionaries for all probe types
accuracy_and_models = {}
for probe in probes:
    accuracy_and_models[probe] = {
        "gt_probe": [],
        "output_probe": []
    }

# Train all probe types
random.seed(42)
samples = []
for i in range(0,10):  # Include all digits 0-9
    if len(index_starting_by_digit_correct[i]) > 0:
        samples += random.sample(index_starting_by_digit_correct[i], min(50, len(index_starting_by_digit_correct[i])))
    if len(index_starting_by_digit_wrong[i]) > 0:
        samples += random.sample(index_starting_by_digit_wrong[i], min(50, len(index_starting_by_digit_wrong[i])))

for probe in probes:
    print(f"\n=== Training {probe} probes ===")
    
    for probe_type in probe_types:
        print(f"\n--- Training {probe} on {probe_type} ---")
        
        for layer_index in tqdm(range(start_layer, num_layers)):
            X = []
            Y = []
            for j in samples:
                X.append(activations[j][layer_index][0][-1])
                if probe_type == "gt_probe":
                    Y.append(torch.tensor(get_digit(eval(left_equations[j].strip(), {"__builtins__": None}, {}), target_digit_index)))
                elif probe_type == "output_probe":
                    Y.append(torch.tensor(get_digit(int(model_answers[j]), target_digit_index)))

            if len(X) == 0 or len(Y) == 0:
                print(f"  Layer {layer_index}: No valid samples.")
                continue
            
            combined = list(zip(X, Y))
            random.shuffle(combined)

            X_shuffled, Y_shuffled = zip(*combined)

            split_idx = int(0.8 * len(X_shuffled))
            X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
            Y_train, Y_test = Y_shuffled[:split_idx], Y_shuffled[split_idx:]
            X_train, X_test = torch.stack(X_train), torch.stack(X_test)
            Y_train, Y_test = torch.stack(Y_train), torch.stack(Y_test)
            
            # Print class distribution for both train and test data
            print("Training data Y distribution:")
            counts_train = torch.bincount(Y_train)
            for i, count in enumerate(counts_train):
                if i < 10:  # Ensure we only print 0-9
                    print(f"Number {i} appears {count.item()} times in Y_train")
            
            print("Test data Y distribution:")
            counts_test = torch.bincount(Y_test)
            for i, count in enumerate(counts_test):
                if i < 10:  # Ensure we only print 0-9
                    print(f"Number {i} appears {count.item()} times in Y_test")
            # Train the probe
            trained_probe = train_probe_function[probe](X_train, Y_train)
            
            # Test the probe
            accuracy = test_probe_function[probe](X_test, Y_test, trained_probe)
            print(f"Accuracy of {probe} probe on {probe_type} on layer {layer_index} is {accuracy}.")
            accuracy_and_models[probe][probe_type].append((accuracy, trained_probe))

# Create folder if it doesn't exist
folder = f"gsm8k/{model_name.split('/')[-1]}_trained_detectors"
os.makedirs(folder, exist_ok=True)




# Save all trained probes
for probe in probes:
    os.makedirs(f"{folder}/{probe}", exist_ok=True)
    for target_type in accuracy_and_models[probe]:
        torch.save(accuracy_and_models[probe][target_type], f"{folder}/{probe}/{target_type}_probe_digit{target_digit_index}")
        print(f"Saved {probe} probe for {target_type} to {folder}/{probe}/{target_type}_probe_digit{target_digit_index}")

# Initialize accuracy dictionaries for all error detector types
error_detector_types = ["logistic_seperately", "mlp", "mlp_seperately", "circular_seperately", "circular_jointly"]
error_detector_accuracy = {}
for detector_type in error_detector_types:
    error_detector_accuracy[detector_type] = []

# Train all error detector types
random.seed(42)
for detector_type in error_detector_types:
    print(f"\n=== Training {detector_type} error detector ===")
    
    for layer_index in tqdm(range(start_layer, num_layers)):
        X = []
        Y = []
        Y_model = []
        Y_true = []
        
            
        for j in samples:
            X.append(activations[j][layer_index][0][-1])
            # For binary detectors (mlp, circular_jointly)
            if detector_type in ["mlp", "circular_jointly"]:
                # 1 if correct, 0 if wrong
                # Get model's answer digit
                model_digit = get_digit(int(model_answers[j]), target_digit_index)
                # Get ground truth digit
                true_digit = get_digit(eval(left_equations[j].strip(), {"__builtins__": None}, {}), target_digit_index)
                Y.append(torch.tensor(1 if model_digit == true_digit else 0))
            # For separate detectors (logistic_seperately, mlp_seperately, circular_seperately)
            else:
                try:
                    # Get model's answer digit
                    model_digit = get_digit(int(model_answers[j]), target_digit_index)
                    # Get ground truth digit
                    true_digit = get_digit(eval(left_equations[j].strip(), {"__builtins__": None}, {}), target_digit_index)
                    Y_model.append(torch.tensor(model_digit))
                    Y_true.append(torch.tensor(true_digit))
                except:
                    continue

        # Skip if no valid samples
        if detector_type in ["mlp", "circular_jointly"]:
            if len(X) == 0 or len(Y) == 0:
                print(f"  Layer {layer_index}: No valid samples.")
                continue
                
            # Prepare data
            combined = list(zip(X, Y))
            random.shuffle(combined)
            X_shuffled, Y_shuffled = zip(*combined)
            
            split_idx = int(0.8 * len(X_shuffled))
            X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
            Y_train, Y_test = Y_shuffled[:split_idx], Y_shuffled[split_idx:]
            X_train, X_test = torch.stack(X_train), torch.stack(X_test)
            Y_train, Y_test = torch.stack(Y_train), torch.stack(Y_test)
            
            # Train the detector
            trained_detector = train_error_detector_function[detector_type](X_train, Y_train)
            
            # Test the detector
            accuracy, precision, recall, f1 = test_error_detector_function[detector_type](X_test, Y_test, trained_detector)
            print(f"Accuracy of {detector_type} error detector on layer {layer_index} is {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
            error_detector_accuracy[detector_type].append((accuracy, precision, recall, f1, trained_detector))

        else:  # For separate detectors
            if len(X) == 0 or len(Y_model) == 0 or len(Y_true) == 0:
                print(f"  Layer {layer_index}: No valid samples.")
                continue
                
            # Prepare data
            combined = list(zip(X, Y_model, Y_true))
            random.shuffle(combined)
            X_shuffled, Y_model_shuffled, Y_true_shuffled = zip(*combined)
            
            split_idx = int(0.8 * len(X_shuffled))
            X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
            Y_model_train, Y_model_test = Y_model_shuffled[:split_idx], Y_model_shuffled[split_idx:]
            Y_true_train, Y_true_test = Y_true_shuffled[:split_idx], Y_true_shuffled[split_idx:]
            
            X_train, X_test = torch.stack(X_train), torch.stack(X_test)
            Y_model_train, Y_model_test = torch.stack(Y_model_train), torch.stack(Y_model_test)
            Y_true_train, Y_true_test = torch.stack(Y_true_train), torch.stack(Y_true_test)
            
            # Train the detector
            trained_detector = train_error_detector_function[detector_type](X_train, Y_model_train, Y_true_train)
            
            # Test the detector
            accuracy, precision, recall, f1 = test_error_detector_function[detector_type](X_test, Y_model_test, Y_true_test, trained_detector)
            print(f"Accuracy of {detector_type} error detector on layer {layer_index} is {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
            error_detector_accuracy[detector_type].append((accuracy, precision, recall, f1, trained_detector))

# Create folder if it doesn't exist
folder = f"gsm8k/{model_name.split('/')[-1]}_trained_detectors"
os.makedirs(folder, exist_ok=True)

torch.save(samples, f"{folder}/samples")

# Save all trained error detectors
for detector_type in error_detector_types:
    os.makedirs(f"{folder}/error_detectors", exist_ok=True)
    torch.save(error_detector_accuracy[detector_type], f"{folder}/error_detectors/{detector_type}_digit{target_digit_index}")
    print(f"Saved {detector_type} error detector to {folder}/error_detectors/{detector_type}_digit{target_digit_index}")

# Print summary of best performing error detectors for each type
print("\n=== Error Detector Performance Summary ===")
for detector_type in error_detector_types:
    if error_detector_accuracy[detector_type]:
        best_entry = max(error_detector_accuracy[detector_type], key=lambda x: x[0])  # Sort by accuracy
        best_layer = error_detector_accuracy[detector_type].index(best_entry) + start_layer
        print(f"Best {detector_type} error detector: Layer {best_layer}")
        print(f"  Accuracy: {best_entry[0]:.4f}, Precision: {best_entry[1]:.4f}, Recall: {best_entry[2]:.4f}, F1-score: {best_entry[3]:.4f}")
    else:
        print(f"No valid {detector_type} error detector trained")