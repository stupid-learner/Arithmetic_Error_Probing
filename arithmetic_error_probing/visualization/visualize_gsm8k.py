#!/usr/bin/env python3
"""
Probe and Error Detector Performance Visualization Script

This script generates visualization plots for probe and error detector performance results.
It's compatible with both training approaches (with or without templates separated).

Usage:
python visualize_gsm8k.py --input_folder gsm8k/gemma-2-2b-it_trained_detectors --output_folder plots_gsm8k
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import seaborn as sns
import argparse
from matplotlib.gridspec import GridSpec


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


def load_probe_results(base_folder):
    """
    Load probe training results
    """
    results = {
        "circular": {"gt_probe": [], "output_probe": []},
        "linear": {"gt_probe": [], "output_probe": []},
        "mlp": {"gt_probe": [], "output_probe": []},
        "logistic": {"gt_probe": [], "output_probe": []}
    }
    
    for probe_type in results.keys():
        probe_dir = f"{base_folder}/{probe_type}"
        if not os.path.exists(probe_dir):
            print(f"Warning: {probe_dir} not found")
            continue
            
        for target in ["gt_probe", "output_probe"]:
            # Try both naming conventions
            possible_paths = [
                f"{probe_dir}/{target}_probe_digit3",
                f"{probe_dir}/{target}_digit3"
            ]
            
            loaded = False
            for probe_path in possible_paths:
                if os.path.exists(probe_path):
                    try:
                        probes_data = torch.load(probe_path, map_location="cpu")
                        
                        # Extract accuracies
                        accuracies = [acc for acc, _ in probes_data]
                        results[probe_type][target] = accuracies
                        print(f"Loaded {len(accuracies)} accuracy values from {probe_path}")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"Error loading {probe_path}: {e}")
            
            if not loaded:
                print(f"Could not load data for {probe_type}/{target}")
    
    return results


def load_error_detector_results(base_folder):
    """
    Load error detector training results
    """
    results = {
        "logistic_seperately": [],
        "mlp": [],
        "mlp_seperately": [],
        "circular_seperately": [],
        "circular_jointly": []
    }
    
    error_dir = f"{base_folder}/error_detectors"
    if not os.path.exists(error_dir):
        print(f"Warning: {error_dir} not found")
        return results
        
    for detector_type in results.keys():
        detector_path = f"{error_dir}/{detector_type}_digit3"
        if not os.path.exists(detector_path):
            print(f"Warning: {detector_path} not found")
            continue
            
        try:
            # Load saved detectors
            detectors_data = torch.load(detector_path, map_location="cpu")
            
            # Extract metrics - each item should be (accuracy, precision, recall, f1, trained_detector)
            if len(detectors_data) > 0 and len(detectors_data[0]) >= 4:
                # It has the expected format with the metrics
                results[detector_type] = detectors_data
                print(f"Loaded {len(detectors_data)} items from {detector_path}")
            else:
                # Handle older format where it might just be (accuracy, detector)
                accuracies = [acc for acc, _ in detectors_data]
                # Fill with placeholder values for missing metrics
                results[detector_type] = [(acc, 0.0, 0.0, 0.0, None) for acc in accuracies]
                print(f"Loaded {len(accuracies)} accuracy values (legacy format) from {detector_path}")
        except Exception as e:
            print(f"Error loading {detector_path}: {e}")
    
    return results


def visualize_probe_accuracy(probe_results, output_dir):
    """
    Visualize probe accuracy across layers for different probe types
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("deep")
    
    # Create plots for gt_probe and output_probe
    for target in ["gt_probe", "output_probe"]:
        # Create larger figure with more padding for title
        fig, ax = plt.subplots(figsize=(24, 22))  # Increased height for title space
        
        # Define markers and linestyles for consistency
        markers = ['o', 's', 'D', '^']
        linestyles = ['-', '--', '-.', ':']
        
        # Define friendly names for each probe type
        probe_names = {
            "linear": "Ridge (Linear) Probe",
            "mlp": "MLP Probe",
            "circular": "Circular Probe",
            "logistic": "Logistic Probe"
        }
        
        # Plot each probe type
        for i, probe_type in enumerate(["circular", "linear", "mlp", "logistic"]):
            if probe_type in probe_results and target in probe_results[probe_type]:
                accs = probe_results[probe_type][target]
                if accs:  # Only plot if we have data
                    layers = np.arange(len(accs))
                    ax.plot(layers, accs, label=probe_names[probe_type], 
                           marker=markers[i % len(markers)],
                           markersize=12,  # Increased marker size
                           linestyle=linestyles[i % len(linestyles)], 
                           linewidth=5.0)  # Increased line width
        
        target_name = "Ground Truth" if target == "gt_probe" else "Model Output"
        
        # Extremely large font sizes with adjusted title positioning
        ax.set_title(f"Probe Accuracy on {target_name} (Digit 3) Across Layers", 
                    fontsize=64, pad=30)  # Added padding between title and plot
        ax.set_xlabel("Layer Index", fontsize=56)
        ax.set_ylabel("Accuracy", fontsize=56)
        
        # Set y-axis limits
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        
        # Extremely large tick label size
        ax.tick_params(axis='both', which='major', labelsize=44, length=10, width=2)
        
        # Extremely large legend font
        ax.legend(fontsize=48, handlelength=3)
        ax.grid(True, linewidth=1.5)
        
        # Save figure with higher DPI and adjusted layout
        plt.tight_layout(pad=4.0)  # Increased padding around the plot
        output_file = os.path.join(output_dir, f"probe_accuracy_{target}(gsm8k).pdf")
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
        
        plt.close(fig)


def visualize_error_detector_metrics(error_results, output_dir):
    """
    Visualize error detector metrics (accuracy, precision, recall, f1) across layers
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("deep")
    
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
        ("accuracy", 0, "Accuracy"), 
        ("precision", 1, "Precision"),
        ("recall", 2, "Recall"),
        ("f1", 3, "F1-Score")
    ]
    
    for metric_name, metric_idx, title_text in metrics:
        # Create extremely large figure with more padding for title
        fig, ax = plt.subplots(figsize=(24, 22))  # Increased height for title space
        
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
                           markersize=12,  # Increased marker size
                           linestyle=linestyles[i % len(linestyles)], 
                           linewidth=5.0)  # Increased line width
        
        # Extremely large font sizes with adjusted title positioning
        ax.set_title(f"Error Detector {title_text} (Digit 3) Across Layers", 
                   fontsize=64, pad=30)  # Added padding between title and plot
        ax.set_xlabel("Layer Index", fontsize=56)
        ax.set_ylabel(title_text, fontsize=56)
        
        # Set y-axis limits
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        
        # Extremely large tick labels
        ax.tick_params(axis='both', which='major', labelsize=44, length=10, width=2)
        
        # Extremely large legend font
        ax.legend(fontsize=48, handlelength=3)
        ax.grid(True, linewidth=1.5)
        
        # Save figure with higher DPI and adjusted layout
        plt.tight_layout(pad=4.0)  # Increased padding around the plot
        output_file = os.path.join(output_dir, f"error_detector_{metric_name}(gsm8k).pdf")
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
    
    # Create larger figure with extra space for legend
    fig, ax = plt.subplots(figsize=(30, 22))  # Further increased width
    
    # Define friendly names for each probe type
    probe_names = {
        "linear": "Ridge (Linear)",
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
    
    # Set width of bars - increase bar width ratio
    barWidth = 0.38  # Slightly wider bars
    
    # Increase bar group spacing by adjusting the positions
    r1 = np.arange(len(labels)) * 1.2  # Multiply by 1.2 to increase spacing between groups
    r2 = [x + barWidth for x in r1]
    
    # Create grouped bars
    ax.bar(r1, gt_accs, width=barWidth, label='Ground Truth', color='#3274A1', edgecolor='black', linewidth=1.5)
    ax.bar(r2, output_accs, width=barWidth, label='Model Output', color='#E1812C', edgecolor='black', linewidth=1.5)
    
    # Add labels and title
    ax.set_title('Last Layer Probe Accuracy Comparison (Digit 3)', fontsize=64, pad=30)
    ax.set_xlabel('Probe Type', fontsize=56)
    ax.set_ylabel('Accuracy', fontsize=56)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    # Add xticks on the middle of the group bars with increased spacing
    ax.set_xticks([r + barWidth/2 for r in r1])
    ax.set_xticklabels(labels)
    
    # Extremely large tick labels
    ax.tick_params(axis='both', which='major', labelsize=44, length=10, width=2)
    
    # Move legend to the right side of the chart
    ax.legend(fontsize=48, handlelength=3, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, linewidth=1.5, axis='y')
    
    # Save figure with higher DPI and adjusted layout
    plt.tight_layout(pad=4.0)
    output_file = os.path.join(output_dir, f"last_layer_comparison(gsm8k).pdf")
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    
    plt.close(fig)

def print_last_layer_metrics(error_results):
    """
    Print the metrics for the last layer of each error detector
    """
    print("\n=== Error Detector Performance at Last Layer (Digit 3) ===")
    print("{:<20} {:<10} {:<10} {:<10} {:<10}".format(
        "Detector Type", "Accuracy", "Precision", "Recall", "F1-Score"))
    print("-" * 65)
    
    # Define friendly names for each detector type
    detector_names = {
        "logistic_seperately": "Logistic (Separate)",
        "mlp": "MLP (Joint)",
        "mlp_seperately": "MLP (Separate)",
        "circular_seperately": "Circular (Separate)",
        "circular_jointly": "Circular (Joint)"
    }
    
    # Print metrics for each detector type
    for detector_type, friendly_name in detector_names.items():
        if detector_type in error_results and error_results[detector_type]:
            # Get the last layer metrics
            last_layer_metrics = error_results[detector_type][-1]
            
            # Check if we have all metrics (the tuple should have at least 4 elements)
            if len(last_layer_metrics) >= 4:
                accuracy = last_layer_metrics[0]
                precision = last_layer_metrics[1]
                recall = last_layer_metrics[2]
                f1 = last_layer_metrics[3]
                
                print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                    friendly_name, accuracy, precision, recall, f1))
            else:
                # Handle the case where we only have accuracy
                accuracy = last_layer_metrics[0]
                print("{:<20} {:<10.4f} {:<10} {:<10} {:<10}".format(
                    friendly_name, accuracy, "N/A", "N/A", "N/A"))
        else:
            print("{:<20} {:<10} {:<10} {:<10} {:<10}".format(
                friendly_name, "N/A", "N/A", "N/A", "N/A"))
    
    print("=" * 65)

def create_three_subplot_comparison(probe_results, error_results, output_dir):
    """
    Create a horizontal arrangement of three subplots showing:
    1. Probe Accuracy on Ground-Truth Result 
    2. Probe Accuracy on Model Prediction
    3. Error Detector Accuracy with a dashed line at y=0.5
    
    Parameters:
    -----------
    probe_results : dict
        Dictionary of probe results loaded from load_probe_results()
    error_results : dict
        Dictionary of error detector results loaded from load_error_detector_results()
    output_dir : str
        Directory to save the output figure
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a figure with 3 subplots arranged horizontally
    # Increase figure height to accommodate the shared legend at bottom
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
    error_detector_colors = sns.color_palette("Set2", 5)  # "Set2" palette for a different color scheme
    
    
    # Plot Output probes (Middle subplot)
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
                
    # Plot Ground Truth probes (Left subplot)
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
                       color=error_detector_colors[i])  # Different color palette
    
    # Add a dashed line at y=0.5 for error detector plot
    ax3.axhline(y=0.5, color='black', linestyle='--', linewidth=3.0, alpha=0.7)
    
    # Set titles and labels with increased font sizes and more padding
    ax1.set_title("Probe Accuracy on Model Prediction", fontsize=44, pad=40)
    ax2.set_title("Probe Accuracy on Ground-Truth", fontsize=44, pad=40)
    ax3.set_title("Error Detector Accuracy", fontsize=44, pad=40)
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Layer Index", fontsize=40)
        ax.set_ylabel("Accuracy", fontsize=40)
        
        # Set y-axis limits
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        
        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=32, length=8, width=2)
        
        # Add grid
        ax.grid(True, linewidth=1.5)
    
    # Add subplot labels (a, b, c) at the bottom of each subplot
    label_fontsize = 36  
    label_fontweight = 'bold'  # Bold font weight for visibility
    label_color = 'black'
    
    # Positioning for the labels - centered below each subplot

    ax1.text(0.5, -0.2, '(a)', transform=ax1.transAxes, fontsize=label_fontsize, 
            fontweight=label_fontweight, color=label_color, va='top', ha='center')
    ax2.text(0.5, -0.2, '(b)', transform=ax2.transAxes, fontsize=label_fontsize, 
            fontweight=label_fontweight, color=label_color, va='top', ha='center')
    ax3.text(0.5, -0.2, '(c)', transform=ax3.transAxes, fontsize=label_fontsize, 
            fontweight=label_fontweight, color=label_color, va='top', ha='center')
    
    
    # Get the probe handles and labels from first subplot (they're the same across the first two subplots)
    handles, labels = ax1.get_legend_handles_labels()
    probe_handles = handles
    probe_labels = labels
    
    # Get the detector handles and labels from the third subplot
    handles, labels = ax3.get_legend_handles_labels()
    detector_handles = handles
    detector_labels = labels
    
    # Create a single shared legend below all subplots
    # First adjust the subplot positions to make room for the legend
    plt.subplots_adjust(bottom=0.5)  
    
    # Create the legend in two parts side by side - moved lower by adjusting y value in bbox_to_anchor
    probe_legend = fig.legend(probe_handles, probe_labels, 
                           loc='lower center', bbox_to_anchor=(0.3, -0.3),  
                           fontsize=36, ncol=2, title="Probes", title_fontsize=40)
    
    detector_legend = fig.legend(detector_handles, detector_labels, 
                              loc='lower center', bbox_to_anchor=(0.7, -0.3),   
                              fontsize=36, ncol=2, title="Error Detectors", title_fontsize=40)
    
    # Add the legends to the figure
    fig.add_artist(probe_legend)
    fig.add_artist(detector_legend)
    
    # Adjust layout with more space for the legend at the bottom
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.95, wspace=0.25)   
    
    for ax in [ax1, ax2, ax3]:
        pos = ax.get_position()
        new_height = pos.height * 0.95  
        new_bottom = pos.y0 + pos.height - new_height  
        ax.set_position([pos.x0, new_bottom, pos.width, new_height])
    
    # Save figure in both PDF and PNG formats
    output_file = os.path.join(output_dir, "three_subplot_comparison(gsm8k).pdf")
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved three-subplot comparison to {output_file}")
    
    output_file = os.path.join(output_dir, "three_subplot_comparison(gsm8k).png")
    plt.savefig(output_file, bbox_inches='tight')
    
    plt.close(fig)
    
    return output_file

def main():
    """Main function to run visualizations"""
    parser = argparse.ArgumentParser(description="Visualize probe and error detector performance results")
    parser.add_argument("--input_folder", required=True, help="Folder containing trained detectors")
    parser.add_argument("--output_folder", default="plots", help="Directory to save plots")
    
    args = parser.parse_args()
    
    print(f"Loading results from {args.input_folder}...")
    
    # Load probe results
    probe_results = load_probe_results(args.input_folder)
    
    # Load error detector results
    error_results = load_error_detector_results(args.input_folder)
    
    # Print last layer metrics for error detectors
    print_last_layer_metrics(error_results)
    
    print("\nCreating visualizations...")
    
    # Create probe accuracy visualization
    visualize_probe_accuracy(probe_results, args.output_folder)
    
    # Create error detector metric visualizations
    visualize_error_detector_metrics(error_results, args.output_folder)
    
    # Create last layer comparison visualization
    visualize_last_layer_comparison(probe_results, args.output_folder)
    
    # Create the new three-subplot comparison visualization
    create_three_subplot_comparison(probe_results, error_results, args.output_folder)
    
    print(f"\nAll visualizations saved to {args.output_folder}")


if __name__ == "__main__":
    main()