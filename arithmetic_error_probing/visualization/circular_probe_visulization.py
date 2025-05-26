import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import get_digit, load_model_result_dic, seed_everything

# Parameters 
n_shots = 2
model_name = "google/gemma-2-2b-it"
data_folder = f"gemma-2-2b-it_{n_shots}_shots_3_digit_sum_output"
hidden_states_path = f"double_circular_probe/gemma-2-2b-it_num_to_hidden_{n_shots}_shots"
base_folder = f"gemma-2-2b-it_probing_results_{n_shots}_shots"

# Set the target digit 
target_digit_index = 3

# Create a directory to save visualizations
visualization_dir = f"plots_circular_probe_visulization"
os.makedirs(visualization_dir, exist_ok=True)

# Load necessary data
test_samples_path = f"{base_folder}/test_samples_digit{target_digit_index}"
test_samples = torch.load(test_samples_path)
num_to_hidden = torch.load(hidden_states_path)
result_dic = load_model_result_dic(data_folder)

# Function to prepare data for a specific layer
def prepare_data_for_layer(layer_index, test_samples):
    X = []
    Y_model = []
    Y_ground_truth = []
    
    for i, j in test_samples:
        # Extract hidden states
        hidden_states = num_to_hidden[(i, j)]
        x = hidden_states[layer_index][0][-1]
        X.append(x)
        
        # Get model output and ground truth
        model_output = result_dic[(i, j)]
        ground_truth = i + j
        
        # Extract the specific digit
        model_digit = get_digit(model_output, target_digit_index)
        true_digit = get_digit(ground_truth, target_digit_index)
        
        Y_model.append(model_digit)
        Y_ground_truth.append(true_digit)
    
    X = torch.stack(X)
    Y_model = torch.tensor(Y_model)
    Y_ground_truth = torch.tensor(Y_ground_truth)
    
    # Create a binary label indicating if the model prediction is correct
    Y_correct = (Y_model == Y_ground_truth).long()
    
    return X, Y_model, Y_ground_truth, Y_correct

# Load circular probes
circular_probes = {}
probe_types = ["gt_probe", "output_probe"]
for probe_type in probe_types:
    probe_path = f"{base_folder}/circular/{probe_type}_digit{target_digit_index}"
    if os.path.exists(probe_path):
        circular_probes[probe_type] = torch.load(probe_path)
    else:
        print(f"Warning: {probe_path} does not exist")

# Load circular error detectors
error_detector_sep_path = f"{base_folder}/error_detectors/circular_seperately_digit{target_digit_index}"
if os.path.exists(error_detector_sep_path):
    circular_error_detector_sep = torch.load(error_detector_sep_path)
else:
    print(f"Warning: {error_detector_sep_path} does not exist")
    circular_error_detector_sep = None

error_detector_joint_path = f"{base_folder}/error_detectors/circular_jointly_digit{target_digit_index}"
if os.path.exists(error_detector_joint_path):
    circular_error_detector_joint = torch.load(error_detector_joint_path)
else:
    print(f"Warning: {error_detector_joint_path} does not exist")
    circular_error_detector_joint = None

# Create 10 distinct colors for digits 0-9
colors = plt.cm.tab10.colors

# Function to visualize a circular probe
def visualize_circular_probe(probe, X, Y, title, filename):
    # Get 2D projections
    with torch.no_grad():
        projections = probe.weights(X).cpu().numpy()
    
    # Convert labels to numpy array
    labels = Y.cpu().numpy()
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    # Plot points with colors based on labels
    for i in range(10):  # Digits 0-9
        mask = (labels == i)
        plt.scatter(projections[mask, 0], projections[mask, 1], 
                    color=colors[i], label=f'Digit {i}', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig(f"{visualization_dir}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Function to visualize circular error detector separately
def visualize_circular_error_detector_sep(detector_tuple, X, Y_model, Y_gt, Y_correct):
    # Unpack the detector tuple
    detector_1, detector_2 = detector_tuple
    
    # Get 2D projections for both detectors
    with torch.no_grad():
        projections_1 = detector_1.weights(X).cpu().numpy()
        projections_2 = detector_2.weights(X).cpu().numpy()
    
    # Convert labels to numpy arrays
    y_model = Y_model.cpu().numpy()
    y_gt = Y_gt.cpu().numpy()
    y_correct = Y_correct.cpu().numpy()
    
    # Create visualizations for detector 1 (model output)
    plt.figure(figsize=(10, 8))
    for i in range(10):
        mask = (y_model == i)
        plt.scatter(projections_1[mask, 0], projections_1[mask, 1], 
                    color=colors[i], label=f'Digit {i}', alpha=0.7)
    plt.title("Separate Detector 1 (Model Output)")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{visualization_dir}/separate_detector1.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create visualizations for detector 2 (ground truth)
    plt.figure(figsize=(10, 8))
    for i in range(10):
        mask = (y_gt == i)
        plt.scatter(projections_2[mask, 0], projections_2[mask, 1], 
                    color=colors[i], label=f'Digit {i}', alpha=0.7)
    plt.title("Separate Detector 2 (Ground Truth)")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{visualization_dir}/separate_detector2.png", dpi=300, bbox_inches='tight')
    plt.close()

# Function to visualize circular error detector jointly
def visualize_circular_error_detector_joint(detector, X, Y_model, Y_gt, Y_correct):
    # Get 2D projections
    with torch.no_grad():
        projections_1 = detector.projection_1(X).cpu().numpy()
        projections_2 = detector.projection_2(X).cpu().numpy()
    
    # Convert labels to numpy arrays
    y_model = Y_model.cpu().numpy()
    y_gt = Y_gt.cpu().numpy()
    y_correct = Y_correct.cpu().numpy()
    
    # Visualize projection_1 with different colorings
    # 1. Color by ground truth
    plt.figure(figsize=(10, 8))
    for i in range(10):
        mask = (y_gt == i)
        plt.scatter(projections_1[mask, 0], projections_1[mask, 1], 
                    color=colors[i], label=f'Digit {i}', alpha=0.7)
    plt.title("Joint Detector Projection 1 - Colored by Ground Truth")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{visualization_dir}/joint_detector1_gt.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Color by model output
    plt.figure(figsize=(10, 8))
    for i in range(10):
        mask = (y_model == i)
        plt.scatter(projections_1[mask, 0], projections_1[mask, 1], 
                    color=colors[i], label=f'Digit {i}', alpha=0.7)
    plt.title("Joint Detector Projection 1 - Colored by Model Output")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{visualization_dir}/joint_detector1_model.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Color by correctness
    plt.figure(figsize=(10, 8))
    plt.scatter(projections_1[y_correct == 0, 0], projections_1[y_correct == 0, 1], 
                color='red', label='Incorrect', alpha=0.7)
    plt.scatter(projections_1[y_correct == 1, 0], projections_1[y_correct == 1, 1], 
                color='green', label='Correct', alpha=0.7)
    plt.title("Joint Detector Projection 1 - Colored by Correctness")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{visualization_dir}/joint_detector1_correct.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualize projection_2 with different colorings
    # 1. Color by ground truth
    plt.figure(figsize=(10, 8))
    for i in range(10):
        mask = (y_gt == i)
        plt.scatter(projections_2[mask, 0], projections_2[mask, 1], 
                    color=colors[i], label=f'Digit {i}', alpha=0.7)
    plt.title("Joint Detector Projection 2 - Colored by Ground Truth")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{visualization_dir}/joint_detector2_gt.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Color by model output
    plt.figure(figsize=(10, 8))
    for i in range(10):
        mask = (y_model == i)
        plt.scatter(projections_2[mask, 0], projections_2[mask, 1], 
                    color=colors[i], label=f'Digit {i}', alpha=0.7)
    plt.title("Joint Detector Projection 2 - Colored by Model Output")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{visualization_dir}/joint_detector2_model.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Color by correctness
    plt.figure(figsize=(10, 8))
    plt.scatter(projections_2[y_correct == 0, 0], projections_2[y_correct == 0, 1], 
                color='red', label='Incorrect', alpha=0.7)
    plt.scatter(projections_2[y_correct == 1, 0], projections_2[y_correct == 1, 1], 
                color='green', label='Correct', alpha=0.7)
    plt.title("Joint Detector Projection 2 - Colored by Correctness")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{visualization_dir}/joint_detector2_correct.png", dpi=300, bbox_inches='tight')
    plt.close()

# Set seed for reproducibility
seed_everything(42)

# Get a sample to determine the number of layers
sample_key = list(num_to_hidden.keys())[0]
num_layers = len(num_to_hidden[sample_key])
last_layer_idx = num_layers - 2

print(f"Model has {num_layers} layers, visualizing layer {last_layer_idx}")

# Get probes for the last layer
gt_probe = None
output_probe = None

if 'gt_probe' in circular_probes and len(circular_probes['gt_probe']) > last_layer_idx:
    gt_probe_acc, gt_probe = circular_probes['gt_probe'][last_layer_idx]
    print(f"Ground Truth probe at layer {last_layer_idx} has accuracy: {gt_probe_acc:.4f}")
else:
    print(f"Ground Truth probe not found for layer {last_layer_idx}")

if 'output_probe' in circular_probes and len(circular_probes['output_probe']) > last_layer_idx:
    output_probe_acc, output_probe = circular_probes['output_probe'][last_layer_idx]
    print(f"Model Output probe at layer {last_layer_idx} has accuracy: {output_probe_acc:.4f}")
else:
    print(f"Model Output probe not found for layer {last_layer_idx}")

# Get error detectors for the last layer
sep_detector = None
joint_detector = None

if circular_error_detector_sep is not None and len(circular_error_detector_sep) > last_layer_idx:
    sep_detector_acc, sep_detector_prec, sep_detector_recall, sep_detector_f1, sep_detector = circular_error_detector_sep[last_layer_idx]
    print(f"Separate error detector at layer {last_layer_idx} has accuracy: {sep_detector_acc:.4f}")
else:
    print(f"Separate error detector not found for layer {last_layer_idx}")

if circular_error_detector_joint is not None and len(circular_error_detector_joint) > last_layer_idx:
    joint_detector_acc, joint_detector_prec, joint_detector_recall, joint_detector_f1, joint_detector = circular_error_detector_joint[last_layer_idx]
    print(f"Joint error detector at layer {last_layer_idx} has accuracy: {joint_detector_acc:.4f}")
else:
    print(f"Joint error detector not found for layer {last_layer_idx}")

# Prepare data for the last layer
X, Y_model, Y_gt, Y_correct = prepare_data_for_layer(last_layer_idx, test_samples)

# Visualize the probes from the last layer
if gt_probe is not None:
    visualize_circular_probe(gt_probe, X, Y_gt, 
                           f"Ground Truth Probe (Layer {last_layer_idx}, Acc: {gt_probe_acc:.4f})",
                           "gt_probe")

if output_probe is not None:
    visualize_circular_probe(output_probe, X, Y_model, 
                           f"Model Output Probe (Layer {last_layer_idx}, Acc: {output_probe_acc:.4f})",
                           "output_probe")

# Visualize the error detectors from the last layer
if sep_detector is not None:
    # Check if it's a tuple of two probes as expected
    if isinstance(sep_detector, tuple) and len(sep_detector) == 2:
        visualize_circular_error_detector_sep(sep_detector, X, Y_model, Y_gt, Y_correct)
    else:
        print(f"Warning: Separate detector has unexpected structure: {type(sep_detector)}")

if joint_detector is not None:
    # Check if it has the expected attributes
    if hasattr(joint_detector, 'projection_1') and hasattr(joint_detector, 'projection_2'):
        visualize_circular_error_detector_joint(joint_detector, X, Y_model, Y_gt, Y_correct)
    else:
        print(f"Warning: Joint detector has unexpected structure: {dir(joint_detector)}")

print("Visualization complete. All figures saved to:", visualization_dir)