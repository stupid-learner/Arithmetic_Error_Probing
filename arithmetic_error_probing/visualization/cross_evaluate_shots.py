import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from matplotlib.gridspec import GridSpec
from model import CircularProbe, RidgeRegression, MultiClassLogisticRegression, MLP
from utils import get_digit, load_model_result_dic

# Global font settings 
plt.rcParams.update({
    'font.size': 40,             # Base font size
    'axes.titlesize': 60,        # Axes title
    'axes.labelsize': 52,        # Axes labels
    'xtick.labelsize': 40,       # X-axis tick labels
    'ytick.labelsize': 40,       # Y-axis tick labels
    'legend.fontsize': 48,       # Legend
    'figure.titlesize': 64       # Figure title
})

def evaluate_single_combination(fig, plot_position, probe_shots, data_shots, target_type):
    """Process a single probe/data combination for a specific target type"""
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe_types = ["circular", "linear", "mlp", "logistic"]
    error_detector_types = ["logistic_seperately", "mlp", "mlp_seperately", "circular_seperately", "circular_jointly"]
    target_digit_index = 3
    
    # Create subplot
    ax = fig.add_subplot(3, 3, plot_position)
    ax.set_title(f"Probe: {probe_shots}-shot, Data: {data_shots}-shot", fontsize=44, pad=20)
    
    try:

        from model import (
            test_probe_circular, test_probe_ridge, test_probe_mlp, test_probe_logistic,
            test_logistic_error_detector_seperately,
            test_mlp_error_detector,
            test_mlp_error_detector_seperately,
            test_circular_error_detector_seperately,
            test_circular_error_detector_jointly
        )
        
        # Use the imported test functions
        test_functions = {
            "circular": test_probe_circular,
            "linear": test_probe_ridge,
            "mlp": test_probe_mlp,
            "logistic": test_probe_logistic
        }
        
        error_detector_test_functions = {
            "logistic_seperately": test_logistic_error_detector_seperately,
            "mlp": test_mlp_error_detector,
            "mlp_seperately": test_mlp_error_detector_seperately,
            "circular_seperately": test_circular_error_detector_seperately,
            "circular_jointly": test_circular_error_detector_jointly
        }
        
        probe_names = {
            "circular": "Circular Probe",
            "linear": "Ridge (Linear) Probe",
            "mlp": "MLP Probe",
            "logistic": "Logistic Probe"
        }
        
        error_detector_names = {
            "logistic_seperately": "Logistic Error Detector",
            "mlp": "MLP Error Detector (Binary)",
            "mlp_seperately": "MLP Error Detector",
            "circular_seperately": "Circular Error Detector",
            "circular_jointly": "Circular Error Detector (Joint)"
        }
        
        # Load data
        hidden_states_path = f"double_circular_probe/gemma-2-2b-it_num_to_hidden_{data_shots}_shots"
        data_folder = f"gemma-2-2b-it_{data_shots}_shots_3_digit_sum_output"
        probe_base_folder = f"gemma-2-2b-it_probing_results_{probe_shots}_shots"
        data_base_folder = f"gemma-2-2b-it_probing_results_{data_shots}_shots"
        
        # Load test samples from data_shots 
        test_samples_path = f"{data_base_folder}/test_samples_digit{target_digit_index}"
        
        print(f"Loading test samples from {test_samples_path}")
        samples = torch.load(test_samples_path, map_location=device)
        
        print(f"Loading hidden states from {hidden_states_path}")
        num_to_hidden = torch.load(hidden_states_path, map_location=device)
        
        print(f"Loading result dictionary from {data_folder}")
        result_dic = load_model_result_dic(data_folder)
        
        # Check if all test samples are available in the current hidden states
        valid_samples = []
        for pair in samples:
            if pair in num_to_hidden and pair in result_dic:
                valid_samples.append(pair)
        
        if len(valid_samples) < len(samples):
            print(f"Warning: Only {len(valid_samples)} out of {len(samples)} test samples found in current data")
        
        if not valid_samples:
            print(f"Error: No valid samples for (probe: {probe_shots}, data: {data_shots})")
            ax.text(0.5, 0.5, f"No valid samples", ha='center', va='center', transform=ax.transAxes, fontsize=32)
            return ax
            
        samples = valid_samples
        print(f"Using {len(samples)} test samples")
        
        # Store all accuracies for y-axis scaling
        all_accuracies = []
        has_valid_plot = False  # Track if at least one valid plot was created
        
        if target_type == "error_detector":
            # Use the error detector color palette from code 2
            error_detector_colors = sns.color_palette("Set2", 5)
            
            for i, detector_type in enumerate(error_detector_types):
                probe_path = f"{probe_base_folder}/error_detectors/{detector_type}_digit{target_digit_index}"
                
                if not os.path.exists(probe_path):
                    print(f"Warning: {probe_path} does not exist, skipping")
                    continue
                
                try:
                    print(f"Loading error detector from {probe_path}")
                    detectors = torch.load(probe_path, map_location=device)
                    layers = range(len(detectors))
                    accuracies = []

                    start_index = 1
                    
                    # Test each layer
                    for layer_idx in layers:
                        try:
                            # Get detector for this layer
                            saved_data = detectors[layer_idx]
                            
                            if len(saved_data) == 5:  
                                saved_accuracy, _, _, _, detector = saved_data
                            else:   
                                saved_accuracy, detector = saved_data
                            
                            # Move detector to device
                            if isinstance(detector, tuple):
                                detector = tuple(d.to(device) for d in detector)
                            else:
                                detector = detector.to(device)
                            
                            # Prepare data
                            X = []
                            Y_model = []
                            Y_true = []
                            Y_binary = []
                            
                            def get_sum(i, j):
                                return i + j
                            
                            def get_model_output(i, j):
                                return result_dic[(i, j)]
                            
                            for pair in samples:
                                try:
                                    hidden_states = num_to_hidden[pair]
                                    
                                    # Check if layer_idx+start_index is valid
                                    if layer_idx+start_index >= len(hidden_states):
                                        raise IndexError(f"Layer index {layer_idx+start_index} out of bounds for hidden states with length {len(hidden_states)}")
                                        
                                    x = hidden_states[layer_idx+start_index][0][-1]
                                    
                                    # Set target value
                                    i, j = pair
                                    model_output = result_dic[pair]
                                    true_output = i + j
                                    model_digit = get_digit(model_output, target_digit_index)
                                    true_digit = get_digit(true_output, target_digit_index)
                                    
                                    X.append(x)
                                    Y_model.append(torch.tensor(model_digit))
                                    Y_true.append(torch.tensor(true_digit))
                                    Y_binary.append(torch.tensor(1 if model_digit == true_digit else 0))
                                    
                                except Exception as e:
                                    print(f"Error processing sample {pair} for layer {layer_idx}: {e}")
                                    continue
                            
                            if not X:  # Check if X is empty
                                print(f"No valid samples for layer {layer_idx}, skipping")
                                accuracies.append(0)  # Add placeholder accuracy
                                continue
                            
                            # Create Tensors
                            X = torch.stack(X).to(device)
                            Y_model = torch.stack(Y_model).to(device)
                            Y_true = torch.stack(Y_true).to(device)
                            Y_binary = torch.stack(Y_binary).to(device)
                            
                            try:
                                if detector_type in ["mlp", "circular_jointly"]:
                                    result = error_detector_test_functions[detector_type](X, Y_binary, detector)
                                else:
                                    result = error_detector_test_functions[detector_type](X, Y_model, Y_true, detector)
                                
                                if isinstance(result, tuple) and len(result) >= 1:
                                    accuracy = result[0] 
                                else:
                                    accuracy = result
                                
                                # Print original accuracy versus newly calculated accuracy
                                if probe_shots == data_shots:
                                    print(f"Layer {layer_idx}: Original accuracy: {saved_accuracy:.4f}, New accuracy: {accuracy:.4f}")
                                
                                accuracies.append(accuracy)
                                all_accuracies.append(accuracy)
                            except Exception as e:
                                print(f"Error testing detector on layer {layer_idx}: {e}")
                                accuracies.append(0)  # Add placeholder accuracy
                                
                        except Exception as e:
                            print(f"Error processing layer {layer_idx}: {e}")
                            accuracies.append(0)  # Add placeholder accuracy
                    
                    # Only plot if we have valid accuracies
                    if any(acc > 0 for acc in accuracies):
                        # Use color from error_detector_colors matching code 2
                        marker = MARKERS[i % len(MARKERS)]
                        linestyle = LINESTYLES[i % len(LINESTYLES)]
                        color = error_detector_colors[i]
                        
                        ax.plot(list(layers), accuracies, label=error_detector_names[detector_type], 
                               marker=marker, linestyle=linestyle, linewidth=4.0, markersize=14, color=color)
                        has_valid_plot = True
                    else:
                        print(f"No valid accuracies for {detector_type}, skipping plot")
                        
                except Exception as e:
                    print(f"Error processing detector type {detector_type}: {e}")
            
            # Add dashed line at 0.5 for error detector plots
            ax.axhline(y=0.5, color='black', linestyle='--', linewidth=3.0, alpha=0.7)
            
        else:
            # Use the probe color palette from code 2
            probe_colors = sns.color_palette("deep", len(probe_types))
            
            # Test each probe type
            for i, probe_type in enumerate(probe_types):
                # Load probes
                probe_path = f"{probe_base_folder}/{probe_type}/{target_type}_digit{target_digit_index}"
                
                if not os.path.exists(probe_path):
                    print(f"Warning: {probe_path} does not exist, skipping")
                    continue
                
                try:
                    print(f"Loading probes from {probe_path}")
                    probes = torch.load(probe_path, map_location=device)
                    layers = range(len(probes))
                    accuracies = []

                    start_index = 1
                    
                    # Test each layer
                    for layer_idx in layers:
                        try:
                            # Get probe for this layer
                            saved_accuracy, probe = probes[layer_idx]
                            
                            # Move probe to device
                            if isinstance(probe, tuple):
                                probe = tuple(d.to(device) for d in detector)
                            else:
                                probe = probe.to(device)
                            
                            X = []
                            Y = []
                            
                            def get_sum(i, j):
                                return i + j
                            
                            def get_model_output(i, j):
                                return result_dic[(i, j)]
                            
                            def get_first_number(i, j):
                                return i
                            
                            def get_second_number(i, j):
                                return j
                            
                            def get_first_number_tens_digit(i, j):
                                # Extract tens digit (i // 10) % 10
                                return (i // 10) % 10
                            
                            def get_first_number_ones_digit(i, j):
                                # Extract ones digit i % 10
                                return i % 10
                            
                            def get_second_number_tens_digit(i, j):
                                # Extract tens digit (j // 10) % 10
                                return (j // 10) % 10
                            
                            def get_second_number_ones_digit(i, j):
                                # Extract ones digit j % 10
                                return j % 10
                            
                            # Select the correct value function based on target_type
                            if target_type == "gt_probe":
                                value_func = lambda x,y: get_digit(get_sum(x,y), target_digit_index)
                            elif target_type == "output_probe":
                                value_func = lambda x,y: get_digit(get_model_output(x,y), target_digit_index)
                            elif target_type == "first_num_probe":
                                value_func = lambda x,y: get_digit(get_first_number(x,y), target_digit_index)
                            elif target_type == "second_num_probe":
                                value_func = lambda x,y: get_digit(get_second_number(x,y), target_digit_index)
                            elif target_type == "first_num_tens_digit_probe":
                                value_func = get_first_number_tens_digit
                            elif target_type == "first_num_ones_digit_probe":
                                value_func = get_first_number_ones_digit
                            elif target_type == "second_num_tens_digit_probe":
                                value_func = get_second_number_tens_digit
                            elif target_type == "second_num_ones_digit_probe":
                                value_func = get_second_number_ones_digit
                            else:
                                raise ValueError(f"Unknown target type: {target_type}")
                            
                            for pair in samples:
                                try:
                                    hidden_states = num_to_hidden[pair]
                                    
                                    # Check if layer_idx+start_index is valid
                                    if layer_idx+start_index >= len(hidden_states):
                                        raise IndexError(f"Layer index {layer_idx+start_index} out of bounds for hidden states with length {len(hidden_states)}")
                                        
                                    x = hidden_states[layer_idx+start_index][0][-1]
                                    
                                    # Set target value
                                    i, j = pair
                                    target_value = value_func(i, j)
                                    Y.append(torch.tensor(target_value))
                                    X.append(x)
                                except Exception as e:
                                    print(f"Error processing sample {pair} for layer {layer_idx}: {e}")
                                    continue
                            
                            if not X:  # Check if X is empty
                                print(f"No valid samples for layer {layer_idx}, skipping")
                                accuracies.append(0)  # Add placeholder accuracy
                                continue
                            
                            # Create Tensors - check for consistent dimensions
                            X = torch.stack(X).to(device)
                            Y = torch.stack(Y).to(device)
                            
                            try:
                                accuracy = test_functions[probe_type](X, Y, probe)
                                
                                # Print original accuracy versus newly calculated accuracy
                                if probe_shots == data_shots:
                                    print(f"Layer {layer_idx}: Original accuracy: {saved_accuracy:.4f}, New accuracy: {accuracy:.4f}")
                                
                                accuracies.append(accuracy)
                                all_accuracies.append(accuracy)
                            except Exception as e:
                                print(f"Error testing probe on layer {layer_idx}: {e}")
                                accuracies.append(0)  # Add placeholder accuracy
                                
                        except Exception as e:
                            print(f"Error processing layer {layer_idx}: {e}")
                            accuracies.append(0)  # Add placeholder accuracy
                    
                    # Only plot if we have valid accuracies
                    if any(acc > 0 for acc in accuracies):
                        probe_index = probe_types.index(probe_type)
                        marker = MARKERS[probe_index]
                        linestyle = LINESTYLES[probe_index]
                        color = probe_colors[probe_index]
                        
                        ax.plot(list(layers), accuracies, label=probe_names[probe_type], 
                               marker=marker, linestyle=linestyle, linewidth=4.0, markersize=14, color=color)
                        has_valid_plot = True
                    else:
                        print(f"No valid accuracies for {probe_type}, skipping plot")
                        
                except Exception as e:
                    print(f"Error processing probe type {probe_type}: {e}")
        
        # Set consistent y-axis limits for all plots (0 to 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        
        # Update labels and font sizes
        if has_valid_plot:
            ax.set_xlabel("Layer", fontsize=40)
            ax.set_ylabel("Accuracy", fontsize=40)
            ax.tick_params(axis='both', which='major', labelsize=32, length=8, width=2)
            ax.grid(True, linewidth=1.5)
            
            # We'll handle the legend separately to make a shared legend later
        else:
            ax.text(0.5, 0.5, "No valid results", ha='center', va='center', transform=ax.transAxes, fontsize=32)
        
    except Exception as e:
        print(f"Error processing (probe: {probe_shots}, data: {data_shots}): {e}")
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes, fontsize=32)
    
    return ax

def cross_evaluate_probes():
    """Main function to run cross-evaluation"""
    # Set global style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("deep")
    
    # Create output directory
    os.makedirs("plot_cross_shots", exist_ok=True)
    
    # Define probe types to evaluate 
    target_types = [
        "gt_probe", "output_probe", "error_detector",
        "first_num_probe", "second_num_probe",
        "first_num_tens_digit_probe", "first_num_ones_digit_probe",
        "second_num_tens_digit_probe", "second_num_ones_digit_probe"
    ]
    
    # Define shots to evaluate
    n_shots_list = [0, 1, 2]
    
    # Define global markers and linestyles
    global MARKERS, LINESTYLES
    MARKERS = ['o', 's', 'D', '^', 'p', 'h']   
    LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]   
    
    # Define plot positions
    combinations = [
        # row 1
        (1, 0, 0),  # position 1: probe 0-shot, data 0-shot
        (2, 0, 1),  # position 2: probe 0-shot, data 1-shot
        (3, 0, 2),  # position 3: probe 0-shot, data 2-shot
        
        # row 2
        (4, 1, 0),  # position 4: probe 1-shot, data 0-shot
        (5, 1, 1),  # position 5: probe 1-shot, data 1-shot
        (6, 1, 2),  # position 6: probe 1-shot, data 2-shot
        
        # row 3
        (7, 2, 0),  # position 7: probe 2-shot, data 0-shot
        (8, 2, 1),  # position 8: probe 2-shot, data 1-shot
        (9, 2, 2),  # position 9: probe 2-shot, data 2-shot
    ]
    
    target_display_names = {
        "gt_probe": "Ground Truth Probe",
        "output_probe": "Model Output Probe",
        "error_detector": "Error Detector",
        "first_num_probe": "First Number Probe",
        "second_num_probe": "Second Number Probe",
        "first_num_tens_digit_probe": "First Number Tens Digit Probe",
        "first_num_ones_digit_probe": "First Number Ones Digit Probe",
        "second_num_tens_digit_probe": "Second Number Tens Digit Probe",
        "second_num_ones_digit_probe": "Second Number Ones Digit Probe"
    }
    
    # Process each target type
    for target_type in target_types:
        print(f"\n{'='*50}")
        print(f"Processing {target_type}")
        print(f"{'='*50}")
        
        # Check if this target type exists in any of the probe folders
        exists = False
        for probe_shots in n_shots_list:
            probe_base_folder = f"gemma-2-2b-it_probing_results_{probe_shots}_shots"
            
            # For error_detector, check in the error_detectors directory
            if target_type == "error_detector":
                path = f"{probe_base_folder}/error_detectors"
                if os.path.exists(path) and os.listdir(path):
                    exists = True
                    break
            else:
                # For other probe types, check if they exist in any probe type directory
                for probe_type in ["circular", "linear", "mlp", "logistic"]:
                    path = f"{probe_base_folder}/{probe_type}/{target_type}_digit3"
                    if os.path.exists(path):
                        exists = True
                        break
                if exists:
                    break
        
        if not exists:
            print(f"Skipping {target_type} as it doesn't exist in any of the probe folders")
            continue
        
        # Create figure for this target type with improved layout - matching code 2's large figure size
        # Using larger figure size for the 3x3 grid
        fig = plt.figure(figsize=(36, 30))
        
        # Process each combination
        subplot_axes = []
        for position, probe_shots, data_shots in combinations:
            print(f"\nProcessing position {position}: Probe {probe_shots}-shot, Data {data_shots}-shot")
            ax = evaluate_single_combination(fig, position, probe_shots, data_shots, target_type)
            subplot_axes.append(ax)
        
        # Add more descriptive title with improved formatting
        fig.suptitle(f"{target_display_names.get(target_type, target_type)} Cross-Evaluation Performance", 
                    fontsize=60, y=0.98)
        
        # Collect all unique handles and labels for a shared legend
        all_handles = []
        all_labels = []
        
        # Create a dictionary to collect unique legend entries
        legend_dict = {}
        
        for ax in subplot_axes:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in legend_dict:
                    legend_dict[label] = handle
        
        # For error detector, use specific order and colors
        if target_type == "error_detector":
            error_detector_types = ["logistic_seperately", "mlp", "mlp_seperately", "circular_seperately", "circular_jointly"]
            error_detector_names = {
                "logistic_seperately": "Logistic Error Detector",
                "mlp": "MLP Error Detector (Binary)",
                "mlp_seperately": "MLP Error Detector",
                "circular_seperately": "Circular Error Detector",
                "circular_jointly": "Circular Error Detector (Joint)"
            }
            
            # Sort the legend entries to match the order in error_detector_types
            for detector_type in error_detector_types:
                detector_name = error_detector_names[detector_type]
                if detector_name in legend_dict:
                    all_labels.append(detector_name)
                    all_handles.append(legend_dict[detector_name])
        # For probe types, use a different order
        else:
            probe_types = ["circular", "linear", "mlp", "logistic"]
            probe_names = {
                "circular": "Circular Probe",
                "linear": "Ridge (Linear) Probe",
                "mlp": "MLP Probe",
                "logistic": "Logistic Probe"
            }
            
            # Sort the legend entries to match the order in probe_types
            for probe_type in probe_types:
                probe_name = probe_names[probe_type]
                if probe_name in legend_dict:
                    all_labels.append(probe_name)
                    all_handles.append(legend_dict[probe_name])
        
        # Add a single shared legend at the bottom
        if all_handles:
            # Create shared legend at the bottom with larger font size
            fig.legend(all_handles, all_labels, 
                      loc='lower center', bbox_to_anchor=(0.5, 0.01),
                      fontsize=36, ncol=min(len(all_handles), 3),
                      title_fontsize=40)
        
        # Adjust layout to make room for the shared legend at the bottom
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        
        # Save figure with higher DPI for better quality
        output_file = f"plot_cross_shots/{target_type}_cross_evaluation.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved {target_type} cross-evaluation to '{output_file}'")
        
        # Also save as PDF
        output_pdf = f"plot_cross_shots/{target_type}_cross_evaluation.pdf"
        plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
        print(f"Saved {target_type} cross-evaluation to '{output_pdf}'")
        
        plt.close(fig)
    
    print("All cross-evaluations completed successfully.")

if __name__ == "__main__":
    cross_evaluate_probes()