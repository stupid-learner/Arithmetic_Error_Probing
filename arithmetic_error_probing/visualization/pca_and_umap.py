#!/usr/bin/env python3
"""
<model_name> <data_folder> <hidden_states_path> <base_folder>
Usage:
    python pca_and_umap.py google/gemma-2-2b-it gemma-2-2b-it_2_shots_3_digit_sum_output double_circular_probe/gemma-2-2b-it_num_to_hidden_2_shots pca_and_umap_gemma-2
"""

import torch
import sys
import os
from utils import random_select_tuples, seed_everything, get_digit, load_model_result_dic
from tqdm import tqdm
from transformers import AutoConfig
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap
from huggingface_hub import login

# Set global font settings - extremely large fonts for LaTeX visibility
plt.rcParams.update({
    'font.size': 88,             # Base font size, doubled
    'axes.titlesize': 128,       # Axes title, doubled
    'axes.labelsize': 112,       # Axes labels, doubled
    'xtick.labelsize': 88,       # X-axis tick labels, doubled
    'ytick.labelsize': 88,       # Y-axis tick labels, doubled
    'legend.fontsize': 104,      # Legend, doubled
    'figure.titlesize': 136      # Figure title, doubled
})

login(token=os.environ.get('HUGGINGFACE_HUB_TOKEN'))
# Global variables and constants
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = ""
data_folder = ""
hidden_states_path = ""
base_folder = "visualization_results"
target_digit_index = 3  # Fixed target digit
seed = 42

# Global data containers
num_to_hidden = {}
samples = []
result_dic = {}

# Load hidden states from file
def load_hidden_states():
    """Load hidden states from the specified path."""
    global num_to_hidden, samples

    def get_model_output(i, j):
        return result_dic[(i, j)]
    
    if not os.path.exists(hidden_states_path):
        raise FileNotFoundError(f"Hidden states file not found at {hidden_states_path}. Please ensure the file exists.")
    
    print(f"Loading hidden states from {hidden_states_path}")
    num_to_hidden = torch.load(hidden_states_path, map_location=torch.device(device))
    
    # Update samples to match loaded hidden states
    all_samples = list(num_to_hidden.keys())
    print(len(all_samples))
    all_samples = [pair for pair in all_samples if int(pair[0]) + int(pair[1]) < 1000]

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

    print(f"Loaded hidden states for {len(samples)} samples")

# Function to visualize hidden states using PCA and UMAP
def visualize_hidden_states():
    """Visualize hidden states using PCA and UMAP."""
    # Get model configuration
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = getattr(config, "num_hidden_layers")
    print(f"Number of layers: {num_layers}")
    
    # Determine start layer
    if len(num_to_hidden[samples[0]]) == num_layers + 1:
        start_layer = 1
    else:
        start_layer = 0
    
    # Create output directory
    os.makedirs(base_folder, exist_ok=True)
    
    # Set style for plots
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("deep")
    
    # Create visualizations for each layer
    for layer_index in tqdm(range(start_layer, len(num_to_hidden[samples[0]]))):
        print(f"Processing layer {layer_index-start_layer}")
        
        # Extract hidden states and digits for this layer
        hidden_states = []
        digits = []
        
        for i, j in samples:
            hidden = num_to_hidden[(i, j)]
            x = hidden[layer_index][0][-1]  # Last token's hidden state
            hidden_states.append(x.cpu().numpy())
            digits.append(get_digit(result_dic[(i, j)], target_digit_index))
        
        hidden_states = np.array(hidden_states)
        digits = np.array(digits)
        
        # Create PCA visualization with larger figure size and fonts
        fig_pca = plt.figure(figsize=(30, 18))
        ax_pca = fig_pca.add_subplot(111)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(hidden_states)
        scatter_pca = ax_pca.scatter(pca_result[:, 0], pca_result[:, 1], c=digits, cmap='tab10', alpha=0.7, s=100)  # Increased marker size
        
        # Set title and labels with large fonts
        ax_pca.set_title(f'PCA Visualization - Layer {layer_index-start_layer}', fontsize=64, pad=30)
        ax_pca.set_xlabel('Principal Component 1', fontsize=56)
        ax_pca.set_ylabel('Principal Component 2', fontsize=56)
        
        # Set tick parameters
        ax_pca.tick_params(axis='both', which='major', labelsize=44, length=10, width=2)
        
        # Add colorbar with large font
        cbar_pca = plt.colorbar(scatter_pca, ax=ax_pca)
        cbar_pca.set_label('Digit', fontsize=52)
        cbar_pca.ax.tick_params(labelsize=44)
        
        plt.grid(True, linewidth=1.5)
        plt.tight_layout()
        plt.savefig(f"{base_folder}/layer_{layer_index-start_layer}_pca(pure).pdf", bbox_inches='tight')
        plt.close()
        
        # Create UMAP visualization with larger figure size and fonts
        fig_umap = plt.figure(figsize=(24, 22))
        ax_umap = fig_umap.add_subplot(111)
        umap_reducer = umap.UMAP()
        umap_result = umap_reducer.fit_transform(hidden_states)
        scatter_umap = ax_umap.scatter(umap_result[:, 0], umap_result[:, 1], c=digits, cmap='tab10', alpha=0.7, s=100)  # Increased marker size
        
        # Set title and labels with large fonts
        ax_umap.set_title(f'UMAP Visualization - Layer {layer_index-start_layer}', fontsize=64, pad=30)
        ax_umap.set_xlabel('UMAP Dimension 1', fontsize=56)
        ax_umap.set_ylabel('UMAP Dimension 2', fontsize=56)
        
        # Set tick parameters
        ax_umap.tick_params(axis='both', which='major', labelsize=44, length=10, width=2)
        
        # Add colorbar with large font
        cbar_umap = plt.colorbar(scatter_umap, ax=ax_umap)
        cbar_umap.set_label('Digit', fontsize=52)
        cbar_umap.ax.tick_params(labelsize=44)
        
        plt.grid(True, linewidth=1.5)
        plt.tight_layout()
        plt.savefig(f"{base_folder}/layer_{layer_index-start_layer}_umap.pdf", bbox_inches='tight')
        plt.close()

# Main function
def main():
    """Main function to run the visualization pipeline."""
    global model_name, data_folder, hidden_states_path, base_folder, result_dic
    
    # Parse command line arguments
    if len(sys.argv) < 4:
        print("Usage: python pca_and_umap.py <model_name> <data_folder> <hidden_states_path> <base_folder>")
        sys.exit(1)
        
    model_name = sys.argv[1] 
    data_folder = sys.argv[2]
    hidden_states_path = sys.argv[3]
    base_folder = sys.argv[4] if len(sys.argv) > 4 else "visualization_results"
    random.seed(seed)
    seed_everything(seed)
    
    # Load result dictionary
    result_dic = load_model_result_dic(data_folder)
    
    # Load hidden states
    load_hidden_states()
    
    # Visualize hidden states
    visualize_hidden_states()

if __name__ == "__main__":
    main()