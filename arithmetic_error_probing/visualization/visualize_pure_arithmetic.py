#!/usr/bin/env python3
"""
Extended Neural Probe and Error Detector Visualization Script

This script generates visualization plots for probe and error detector performance results,
including all probe targets (gt_probe, output_probe, first_num_probe, second_num_probe,
and digit-specific probes). Added dashed lines showing maximum class proportion.

Usage:
python visualize_pure_arithmetic.py --input_folder gemma-2-2b-it_probing_results_2_shots --output_folder plots_pure --data_folder gemma-2-2b-it_2_shots_3_digit_sum_output
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import seaborn as sns
import argparse
from matplotlib.gridspec import GridSpec
import glob
import sys
from utils import get_digit, load_model_result_dic

# Global font settings - extremely large fonts for LaTeX visibility
plt.rcParams.update({
    'font.size': 40,             # Base font size
    'axes.titlesize': 60,        # Axes title
    'axes.labelsize': 52,        # Axes labels
    'xtick.labelsize': 40,       # X-axis tick labels
    'ytick.labelsize': 40,       # Y-axis tick labels
    'legend.fontsize': 48,       # Legend
    'figure.titlesize': 64       # Figure title
})


def load_test_samples(base_folder, target_digit_index=3):
    """Load test samples from saved files."""
    # Use the first path format as requested
    test_samples_path = os.path.join(base_folder, f"test_samples_digit{target_digit_index}")
    if os.path.exists(test_samples_path):
        return torch.load(test_samples_path, map_location="cpu")
    
    print(f"Warning: Could not find test samples in {base_folder}")
    return None


def calculate_error_detector_max_class_proportion(test_samples, result_dic, target_digit_index=3):
    """Calculate the proportion of the most common class in error detection (correct vs incorrect predictions)"""
    if not test_samples or not result_dic:
        return 0.5
    
    correct_count = 0
    total_count = 0
    
    for i, j in test_samples:
        if (i, j) not in result_dic:
            continue
            
        # Get model's predicted value and ground truth
        model_value = result_dic[(i, j)]
        true_value = i + j
        
        # Extract the specific digit
        model_digit = get_digit(model_value, target_digit_index)
        true_digit = get_digit(true_value, target_digit_index)
        
        # Count correct predictions
        if model_digit == true_digit:
            correct_count += 1
        
        total_count += 1
    
    if total_count == 0:
        return 0.5
    
    # Calculate proportion of correct predictions
    correct_prop = correct_count / total_count
    
    # Return the maximum proportion (either correct or incorrect)
    return max(correct_prop, 1 - correct_prop)


def calculate_probe_max_class_proportion(test_samples, result_dic, probe_type, target_digit_index=3):
    """Calculate the proportion of the most common digit class in probe targets"""
    if not test_samples or (probe_type == "output" and not result_dic):
        return 0.1
    
    # Dictionary to count occurrences of each digit
    digit_counts = {digit: 0 for digit in range(10)}
    total_count = 0
    
    for i, j in test_samples:
        # Determine the value to extract digit from
        if probe_type == "gt":
            value = i + j
        elif probe_type == "output":
            if (i, j) not in result_dic:
                continue
            value = result_dic[(i, j)]
        else:
            raise ValueError(f"Unknown probe type: {probe_type}")
        
        # Extract digit and count
        digit = get_digit(value, target_digit_index)
        digit_counts[digit] += 1
        total_count += 1
    
    if total_count == 0:
        return 0.1
    
    # Find the most common digit and its proportion
    most_common_digit = max(digit_counts, key=digit_counts.get)
    most_common_prop = digit_counts[most_common_digit] / total_count
    
    return most_common_prop


def load_probe_results(base_folder):
    """
    Load probe training results for all targets with robust file path handling
    """
    # Define all probe types
    probe_types = ["circular", "linear", "mlp", "logistic"]
    
    # Define all probe targets
    all_targets = [
        "gt_probe", 
        "output_probe", 
        "first_num_probe", 
        "second_num_probe",
        "first_num_tens_digit_probe",
        "first_num_ones_digit_probe",
        "second_num_tens_digit_probe",
        "second_num_ones_digit_probe"
    ]
    
    # Initialize results dictionary
    results = {}
    for probe_type in probe_types:
        results[probe_type] = {}
        for target in all_targets:
            results[probe_type][target] = []
    
    # Check if directory exists
    if not os.path.exists(base_folder):
        print(f"Warning: {base_folder} not found")
        return results
    
    for probe_type in probe_types:
        probe_dir = os.path.join(base_folder, probe_type)
        if not os.path.exists(probe_dir):
            print(f"Warning: {probe_dir} not found")
            continue
        
        # List all files in directory to check what's available
        available_files = os.listdir(probe_dir)
        print(f"Available files in {probe_dir}: {available_files}")
            
        for target in all_targets:
            # Search for files with glob pattern
            pattern = os.path.join(probe_dir, f"{target}*digit3*")
            matching_files = glob.glob(pattern)
            
            if not matching_files:
                # Try without 'digit3' in case it's part of another naming scheme
                pattern = os.path.join(probe_dir, f"{target}*")
                matching_files = glob.glob(pattern)
            
            loaded = False
            for probe_path in matching_files:
                try:
                    probes_data = torch.load(probe_path, map_location="cpu")
                    
                    # Handle different data formats
                    if isinstance(probes_data, list):
                        if len(probes_data) > 0:
                            # Check if it's a list of tuples (accuracy, model)
                            if isinstance(probes_data[0], tuple) and len(probes_data[0]) >= 2:
                                accuracies = [acc for acc, _ in probes_data]
                                results[probe_type][target] = accuracies
                                print(f"Loaded {len(accuracies)} accuracy values from {probe_path}")
                                loaded = True
                                break
                            # Check if it's just a list of accuracies
                            elif isinstance(probes_data[0], (int, float)):
                                results[probe_type][target] = probes_data
                                print(f"Loaded {len(probes_data)} accuracy values from {probe_path}")
                                loaded = True
                                break
                    elif isinstance(probes_data, dict):
                        # Handle dictionary format if present
                        if target in probes_data:
                            results[probe_type][target] = probes_data[target]
                            print(f"Loaded {len(probes_data[target])} accuracy values from {probe_path}")
                            loaded = True
                            break
                except Exception as e:
                    print(f"Error loading {probe_path}: {e}")
            
            if not loaded:
                print(f"No data found for {probe_type}/{target}")
    
    return results


def load_error_detector_results(base_folder):
    """
    Load error detector training results with robust file path handling
    """
    results = {
        "logistic_seperately": [],
        "mlp": [],
        "mlp_seperately": [],
        "circular_seperately": [],
        "circular_jointly": []
    }
    
    error_dir = os.path.join(base_folder, "error_detectors")
    if not os.path.exists(error_dir):
        print(f"Warning: {error_dir} not found")
        return results
        
    for detector_type in results.keys():
        # Search for files with glob pattern
        pattern = os.path.join(error_dir, f"{detector_type}*digit3*")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            # Try without 'digit3' in case it's part of another naming scheme
            pattern = os.path.join(error_dir, f"{detector_type}*")
            matching_files = glob.glob(pattern)
        
        loaded = False
        for detector_path in matching_files:
            try:
                # Load saved detectors
                detectors_data = torch.load(detector_path, map_location="cpu")
                
                # Handle different data formats
                if isinstance(detectors_data, list):
                    if len(detectors_data) > 0:
                        # Check if format is (accuracy, precision, recall, f1, detector)
                        if isinstance(detectors_data[0], tuple) and len(detectors_data[0]) >= 4:
                            results[detector_type] = detectors_data
                            print(f"Loaded {len(detectors_data)} items from {detector_path}")
                            loaded = True
                            break
                        # Check if format is (accuracy, detector)
                        elif isinstance(detectors_data[0], tuple) and len(detectors_data[0]) == 2:
                            # Convert to the format with precision, recall, f1
                            accuracies = [acc for acc, _ in detectors_data]
                            results[detector_type] = [(acc, 0.0, 0.0, 0.0, None) for acc in accuracies]
                            print(f"Loaded {len(accuracies)} accuracy values (legacy format) from {detector_path}")
                            loaded = True
                            break
                        # Check if it's just a list of accuracies
                        elif isinstance(detectors_data[0], (int, float)):
                            results[detector_type] = [(acc, 0.0, 0.0, 0.0, None) for acc in detectors_data]
                            print(f"Loaded {len(detectors_data)} accuracy values from {detector_path}")
                            loaded = True
                            break
                elif isinstance(detectors_data, dict):
                    # Handle dictionary format if present
                    if "accuracy" in detectors_data:
                        accuracies = detectors_data["accuracy"]
                        if "precision" in detectors_data and "recall" in detectors_data and "f1" in detectors_data:
                            # Create tuples from the dict values
                            results[detector_type] = list(zip(
                                detectors_data["accuracy"],
                                detectors_data["precision"],
                                detectors_data["recall"],
                                detectors_data["f1"],
                                [None] * len(detectors_data["accuracy"])
                            ))
                        else:
                            results[detector_type] = [(acc, 0.0, 0.0, 0.0, None) for acc in accuracies]
                        print(f"Loaded {len(accuracies)} values from {detector_path}")
                        loaded = True
                        break
            except Exception as e:
                print(f"Error loading {detector_path}: {e}")
        
        if not loaded:
            print(f"Could not load data for {detector_type}")
    
    return results


def visualize_probe_accuracy(probe_results, output_dir, test_samples=None, result_dic=None, target_digit_index=3):
    """
    Visualize probe accuracy across layers for different probe types and all targets
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("deep")
    
    # Standard figure size matching the three-subplot comparison subplots
    standard_figsize = (12, 10)
    
    # Define all target groups with friendly names
    target_groups = {
        "Basic": {
            "gt_probe": "Probe Accuracy on Ground Truth",
            "output_probe": "Probe Accuracy on Model Output"
        },
        "Numbers": {
            "first_num_probe": "Operand #1 - Hundreds",
            "second_num_probe": "Operand #2 - Hundreds"
        },
        "First Number Digits": {
            "first_num_tens_digit_probe": "Operand #1 - Tens",
            "first_num_ones_digit_probe": "Operand #1 - Ones"
        },
        "Second Number Digits": {
            "second_num_tens_digit_probe": "Operand #2 - Tens",
            "second_num_ones_digit_probe": "Operand #2 - Ones"
        }
    }
    
    # Define markers and linestyles for consistency
    markers = ['o', 's', 'D', '^']
    linestyles = ['-', '--', '-.', ':']
    
    # Define friendly names for each probe type
    probe_names = {
        "linear": "Linear",
        "mlp": "MLP",
        "circular": "Circular",
        "logistic": "Logistic"
    }
    
    # Create plots for all targets grouped by category
    for group_name, targets in target_groups.items():
        for target, target_friendly_name in targets.items():
            # Check if we have any data for this target
            has_data = False
            for probe_type in probe_results:
                if target in probe_results[probe_type] and probe_results[probe_type][target]:
                    has_data = True
                    break
            
            if not has_data:
                print(f"No data for {target}, skipping visualization")
                continue
            
            # Create figure with standard size
            fig, ax = plt.subplots(figsize=standard_figsize)
            
            # Plot each probe type
            for i, probe_type in enumerate(["circular", "linear", "mlp", "logistic"]):
                if probe_type in probe_results and target in probe_results[probe_type]:
                    accs = probe_results[probe_type][target]
                    if accs:  # Only plot if we have data
                        layers = np.arange(len(accs))
                        ax.plot(layers, accs, label=probe_names[probe_type], 
                               marker=markers[i % len(markers)], 
                               markersize=12,  # Reduced marker size
                               linestyle=linestyles[i % len(linestyles)], 
                               linewidth=3.0)  # Reduced line width
            
            # Add dashed line for maximum class proportion (without text)
            if test_samples and result_dic:
                if target == "gt_probe":
                    max_prop = calculate_probe_max_class_proportion(test_samples, result_dic, "gt", target_digit_index)
                    ax.axhline(y=max_prop, color='black', linestyle='--', linewidth=2.0, alpha=0.7)
                elif target == "output_probe":
                    max_prop = calculate_probe_max_class_proportion(test_samples, result_dic, "output", target_digit_index)
                    ax.axhline(y=max_prop, color='black', linestyle='--', linewidth=2.0, alpha=0.7)
            
            # Adjusted title and labels with smaller font sizes
            ax.set_title(f"{target_friendly_name}", fontsize=36, pad=20)
            ax.set_xlabel("Layer Index", fontsize=32)
            ax.set_ylabel("Accuracy", fontsize=32)
            
            # Set y-axis limits
            ax.set_ylim(0, 1.0)
            ax.set_yticks(np.arange(0, 1.1, 0.1))
            
            # Consistent tick label size (reduced)
            ax.tick_params(axis='both', which='major', labelsize=28, length=8, width=2)
            
            # Consistent legend (reduced size)
            ax.legend(fontsize=30, handlelength=3)
            ax.grid(True, linewidth=1.5)
            
            # Save figure
            plt.tight_layout()
            output_file = os.path.join(output_dir, f"probe_accuracy_{target}(pure).pdf")
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Saved plot to {output_file}")
            
            plt.close(fig)


def visualize_error_detector_metrics(error_results, output_dir, test_samples=None, result_dic=None, target_digit_index=3):
    """
    Visualize error detector metrics (accuracy, precision, recall, f1) across layers
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("deep")
    
    # Standard figure size matching the three-subplot comparison subplots
    standard_figsize = (12, 10)
    
    # Define markers and linestyles for consistency
    markers = ['o', 's', 'D', '^', 'X']
    linestyles = ['-', '--', '-.', ':', '-']
    
    # Define friendly names for each detector type
    detector_names = {
        "logistic_seperately": "Logistic (Separate)",
        "mlp": "MLP (Joint)",
        "mlp_seperately": "MLP (Separate)",
        "circular_seperately": "Circular (Separate)",
        "circular_jointly": "Circular (Joint)"
    }
    
    # Create plots for each metric
    metrics = [
        ("accuracy", 0, "Error Detector Accuracy"), 
        ("precision", 1, "Error Detector Precision"),
        ("recall", 2, "Error Detector Recall"),
        ("f1", 3, "Error Detector F1-Score")
    ]
    
    for metric_name, metric_idx, title_text in metrics:
        # Create figure with standard size
        fig, ax = plt.subplots(figsize=standard_figsize)
        
        # Plot each detector type
        for i, detector_type in enumerate(detector_names.keys()):
            if detector_type in error_results and error_results[detector_type]:
                # Extract the specific metric values
                if isinstance(error_results[detector_type][0], tuple) and len(error_results[detector_type][0]) > metric_idx:
                    # Extract the metric at the specified index
                    metric_values = [item[metric_idx] for item in error_results[detector_type]]
                    layers = np.arange(len(metric_values))
                    ax.plot(layers, metric_values, label=detector_names[detector_type], 
                           marker=markers[i % len(markers)],
                           markersize=12,  # Reduced marker size
                           linestyle=linestyles[i % len(linestyles)], 
                           linewidth=3.0)  # Reduced line width
        
        # Add dashed line for maximum class proportion for accuracy metric (without text)
        if metric_name == "accuracy" and test_samples and result_dic:
            error_max_prop = calculate_error_detector_max_class_proportion(
                test_samples, result_dic, target_digit_index)
            ax.axhline(y=error_max_prop, color='black', linestyle='--', linewidth=2.0, alpha=0.7)
        
        # Consistent title and labels with reduced size
        ax.set_title(f"{title_text}", fontsize=36, pad=20)
        ax.set_xlabel("Layer Index", fontsize=32)
        ax.set_ylabel(title_text.split(" ")[-1], fontsize=32)  # Just "Accuracy", "Precision", etc.
        
        # Set y-axis limits
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        
        # Consistent tick labels (reduced size)
        ax.tick_params(axis='both', which='major', labelsize=28, length=8, width=2)
        
        # Consistent legend (reduced size)
        ax.legend(fontsize=30, handlelength=3)
        ax.grid(True, linewidth=1.5)
        
        # Save figure
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"error_detector_{metric_name}(pure).pdf")
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
        
        plt.close(fig)


def visualize_last_layer_comparison(probe_results, output_dir):
    """
    Create a bar chart comparing the last layer performance of each probe type
    on both ground truth and model output
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("deep")
    
    # Adjusted figure size to match subplot proportions
    fig, ax = plt.subplots(figsize=(14, 10))  # Slightly wider for bar chart
    
    # Define friendly names for each probe type
    probe_names = {
        "linear": "Linear",
        "mlp": "MLP",
        "circular": "Circular",
        "logistic": "Logistic"
    }
    
    # Extract last layer accuracy for each probe type
    probe_types = ["circular", "linear", "mlp", "logistic"]
    gt_accs = []
    output_accs = []
    labels = []
    
    for probe_type in probe_types:
        if probe_type in probe_results:
            # Extract last layer accuracy (if available)
            if "gt_probe" in probe_results[probe_type] and probe_results[probe_type]["gt_probe"]:
                gt_accs.append(probe_results[probe_type]["gt_probe"][-1])
            else:
                gt_accs.append(0)  # Default value if data is missing
                
            if "output_probe" in probe_results[probe_type] and probe_results[probe_type]["output_probe"]:
                output_accs.append(probe_results[probe_type]["output_probe"][-1])
            else:
                output_accs.append(0)  # Default value if data is missing
                
            labels.append(probe_names[probe_type])
    
    # Set width of bars - adjusted for consistent proportions
    barWidth = 0.35
    
    # Consistent spacing
    r1 = np.arange(len(labels)) * 1.2
    r2 = [x + barWidth for x in r1]
    
    # Create grouped bars with consistent styling
    ax.bar(r1, gt_accs, width=barWidth, label='Ground Truth', color='#3274A1', edgecolor='black', linewidth=1.5)
    ax.bar(r2, output_accs, width=barWidth, label='Model Output', color='#E1812C', edgecolor='black', linewidth=1.5)
    
    # Consistent labels and title with reduced font size
    ax.set_title('Last Layer Probe Accuracy Comparison', fontsize=36, pad=20)
    ax.set_xlabel('Probe Type', fontsize=32)
    ax.set_ylabel('Accuracy', fontsize=32)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    # Add xticks with consistent formatting
    ax.set_xticks([r + barWidth/2 for r in r1])
    ax.set_xticklabels(labels)
    
    # Consistent tick labels (reduced size)
    ax.tick_params(axis='both', which='major', labelsize=28, length=8, width=2)
    
    # Consistent legend (reduced size)
    ax.legend(fontsize=30, handlelength=3, loc='upper right')
    ax.grid(True, linewidth=1.5, axis='y')
    
    # Save figure
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"last_layer_comparison(pure).pdf")
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    
    plt.close(fig)


def create_three_subplot_comparison(probe_results, error_results, output_dir,
                                   output_max_prop=None, gt_max_prop=None, error_max_prop=None):
    """
    Create a horizontal arrangement of three subplots showing:
    1. Probe Accuracy on Model Prediction
    2. Probe Accuracy on Ground-Truth Result
    3. Error Detector Accuracy with dashed lines showing maximum class proportions
    
    Parameters:
    -----------
    probe_results : dict
        Dictionary of probe results loaded from load_probe_results()
    error_results : dict
        Dictionary of error detector results loaded from load_error_detector_results()
    output_dir : str
        Directory to save the output figure
    output_max_prop : float
        Maximum class proportion for model output probe
    gt_max_prop : float
        Maximum class proportion for ground truth probe
    error_max_prop : float
        Maximum class proportion for error detector
    """
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a figure with 3 subplots arranged horizontally
    fig = plt.figure(figsize=(36, 10))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Define probe types and their properties
    probe_types = ["circular", "linear", "mlp", "logistic"]
    probe_names = {
        "linear": "Linear",
        "mlp": "MLP",
        "circular": "Circular",
        "logistic": "Logistic"
    }
    
    # Define markers and linestyles for consistency
    markers = ['o', 's', 'D', '^']
    linestyles = ['-', '--', '-.', ':']
    
    # Color palette for probes - using a consistent set of colors
    probe_colors = sns.color_palette("deep", len(probe_types))
    
    # Different color palette for error detectors
    error_detector_colors = sns.color_palette("Set2", 5)
    
    # Plot Output probes (left subplot)
    for i, probe_type in enumerate(probe_types):
        if probe_type in probe_results and "output_probe" in probe_results[probe_type]:
            accs = probe_results[probe_type]["output_probe"]
            if accs:  # Only plot if we have data
                layers = np.arange(len(accs))
                ax1.plot(layers, accs, 
                       label=probe_names[probe_type],
                       marker=markers[i % len(markers)], 
                       markersize=14,
                       linestyle=linestyles[i % len(linestyles)], 
                       linewidth=4.0,
                       color=probe_colors[i])

    # Plot Ground Truth probes (middle subplot)
    for i, probe_type in enumerate(probe_types):
        if probe_type in probe_results and "gt_probe" in probe_results[probe_type]:
            accs = probe_results[probe_type]["gt_probe"]
            if accs:  # Only plot if we have data
                layers = np.arange(len(accs))
                ax2.plot(layers, accs, 
                       label=probe_names[probe_type],
                       marker=markers[i % len(markers)], 
                       markersize=14, 
                       linestyle=linestyles[i % len(linestyles)], 
                       linewidth=4.0,
                       color=probe_colors[i])
    
    # Plot Error Detector accuracy (Right subplot)
    detector_names = {
        "logistic_seperately": "Logistic (Separate)",
        "mlp": "MLP (Joint)",
        "mlp_seperately": "MLP (Separate)",
        "circular_seperately": "Circular (Separate)",
        "circular_jointly": "Circular (Joint)"
    }
    
    for i, detector_type in enumerate(detector_names.keys()):
        if detector_type in error_results and error_results[detector_type]:
            # Extract accuracy values (first element in each tuple)
            if isinstance(error_results[detector_type][0], tuple):
                accuracy_values = [item[0] for item in error_results[detector_type]]
                layers = np.arange(len(accuracy_values))
                ax3.plot(layers, accuracy_values, 
                       label=detector_names[detector_type],
                       marker=markers[i % len(markers)], 
                       markersize=14,
                       linestyle=linestyles[i % len(linestyles)], 
                       linewidth=4.0,
                       color=error_detector_colors[i])
    
    # Add dashed lines for maximum class proportions (without text)
    if output_max_prop is not None:
        ax1.axhline(y=output_max_prop, color='black', linestyle='--', linewidth=3.0, alpha=0.7)
    
    if gt_max_prop is not None:
        ax2.axhline(y=gt_max_prop, color='black', linestyle='--', linewidth=3.0, alpha=0.7)
    
    if error_max_prop is not None:
        ax3.axhline(y=error_max_prop, color='black', linestyle='--', linewidth=3.0, alpha=0.7)
    
    # Set titles and labels with original font sizes
    ax1.set_title("Probe Accuracy on Model Prediction", fontsize=44, pad=20)   
    ax2.set_title("Probe Accuracy on Ground-Truth", fontsize=44, pad=20)   
    ax3.set_title("Error Detector Accuracy", fontsize=44, pad=20)   
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Layer Index", fontsize=40)
        ax.set_ylabel("Accuracy", fontsize=40)
        
        # Set y-axis limits
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        
        # Consistent tick label size
        ax.tick_params(axis='both', which='major', labelsize=32, length=8, width=2)
        
        # Add grid
        ax.grid(True, linewidth=1.5)

    # Add subplot labels (a, b, c)
    label_fontsize = 36
    label_fontweight = 'bold'
    label_color = 'black'
    
    ax1.text(0.5, -0.2, '(a)', transform=ax1.transAxes, fontsize=label_fontsize, 
            fontweight=label_fontweight, color=label_color, va='top', ha='center')
    ax2.text(0.5, -0.2, '(b)', transform=ax2.transAxes, fontsize=label_fontsize, 
            fontweight=label_fontweight, color=label_color, va='top', ha='center')
    ax3.text(0.5, -0.2, '(c)', transform=ax3.transAxes, fontsize=label_fontsize, 
            fontweight=label_fontweight, color=label_color, va='top', ha='center')
    
    # Get handles and labels for legends
    probe_handles, probe_labels = ax1.get_legend_handles_labels()
    detector_handles, detector_labels = ax3.get_legend_handles_labels()
        
    # Create shared legends
    plt.subplots_adjust(bottom=0.5)  
    
    probe_legend = fig.legend(probe_handles, probe_labels, 
                           loc='lower center', bbox_to_anchor=(0.3, -0.3),
                           fontsize=36, ncol=2, title="Probes", title_fontsize=40)
    
    detector_legend = fig.legend(detector_handles, detector_labels, 
                              loc='lower center', bbox_to_anchor=(0.7, -0.3),
                              fontsize=36, ncol=2, title="Error Detectors", title_fontsize=40)
    
    fig.add_artist(probe_legend)
    fig.add_artist(detector_legend)
    
    # Adjust layout
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.95, wspace=0.25)

    for ax in [ax1, ax2, ax3]:
        pos = ax.get_position()
        new_height = pos.height * 0.95  
        new_bottom = pos.y0 + pos.height - new_height  
        ax.set_position([pos.x0, new_bottom, pos.width, new_height])
    
    # Save figure
    output_file = os.path.join(output_dir, "three_subplot_comparison(pure).pdf")
    plt.savefig(output_file, bbox_inches='tight')
    
    output_file = os.path.join(output_dir, "three_subplot_comparison(pure).png")
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved three-subplot comparison to {output_file}")
    
    plt.close(fig)
    
    return output_file


def main():
    """Main function to run visualizations including the three-subplot comparison with max class proportions"""
    parser = argparse.ArgumentParser(description="Visualize probe and error detector performance results")
    parser.add_argument("--input_folder", required=True, help="Folder containing trained detectors")
    parser.add_argument("--output_folder", default="plots", help="Directory to save plots")
    parser.add_argument("--data_folder", required=True, help="Folder containing model result dictionary")
    parser.add_argument("--target_digit", type=int, default=3, help="Target digit position (default: 3)")
    
    args = parser.parse_args()
    
    print(f"Loading results from {args.input_folder}...")
    
    # Load probe results
    probe_results = load_probe_results(args.input_folder)
    
    # Load error detector results
    error_results = load_error_detector_results(args.input_folder)
    
    # Load test samples and result dictionary
    test_samples = load_test_samples(args.input_folder, args.target_digit)
    result_dic = load_model_result_dic(args.data_folder)
    
    # Calculate maximum class proportions
    if test_samples and result_dic:
        # Calculate max class proportions
        error_max_prop = calculate_error_detector_max_class_proportion(
            test_samples, result_dic, args.target_digit)
        gt_max_prop = calculate_probe_max_class_proportion(
            test_samples, result_dic, "gt", args.target_digit)
        output_max_prop = calculate_probe_max_class_proportion(
            test_samples, result_dic, "output", args.target_digit)
        
        print(f"Maximum class proportions: Error detector: {error_max_prop:.4f}, "
              f"GT probe: {gt_max_prop:.4f}, Output probe: {output_max_prop:.4f}")
    else:
        # Use default values
        error_max_prop = 0.5
        gt_max_prop = 0.1
        output_max_prop = 0.1
        print("Warning: Using default max class proportion values")
    
    print("\nCreating visualizations...")
    
    # Create standard visualizations with test samples and result_dic for dashed lines
    visualize_probe_accuracy(probe_results, args.output_folder, test_samples, result_dic, args.target_digit)
    visualize_error_detector_metrics(error_results, args.output_folder, test_samples, result_dic, args.target_digit)
    visualize_last_layer_comparison(probe_results, args.output_folder)
    
    # Create the three-subplot comparison with maximum class proportions
    create_three_subplot_comparison(
        probe_results, 
        error_results, 
        args.output_folder,
        output_max_prop,
        gt_max_prop,
        error_max_prop
    )
    
    print(f"\nAll visualizations saved to {args.output_folder}")


if __name__ == "__main__":
    main()