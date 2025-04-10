import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_result_file(file_path):
    """Parse a result file and extract dataset names and latencies."""
    results = {}
    in_data_section = False
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Check if we're at the start of the data section
                if line.strip() == "==================================================":
                    in_data_section = True
                    continue
                
                # Skip header and separator lines
                if "Dataset" in line or "--------------------------------------------------" in line:
                    continue
                
                # Check if we're at the end of the data section
                if in_data_section and line.strip() == "==================================================":
                    break
                
                # Process data lines
                if in_data_section and "|" in line:
                    parts = line.strip().split("|")
                    if len(parts) == 2:
                        dataset = parts[0].strip()
                        latency_str = parts[1].strip()
                        
                        # Skip entries with 'X' as latency
                        if latency_str != "X":
                            try:
                                latency = int(latency_str)
                                results[dataset] = latency
                            except ValueError:
                                # Skip entries with non-integer latency
                                pass
        
        return results
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return {}

def find_result_files(directory="."):
    """Find all .result files in the given directory."""
    result_files = []
    for file in os.listdir(directory):
        if file.endswith(".result"):
            result_files.append(os.path.join(directory, file))
    return result_files

def generate_latency_plot(all_results):
    """Generate a bar chart for latency comparison."""
    # Get unique datasets across all methods
    all_datasets = set()
    for method in all_results:
        all_datasets.update(all_results[method].keys())
    
    # Sort datasets by category and name
    def sort_key(dataset):
        if dataset.startswith('small/'):
            return (0, dataset)
        elif dataset.startswith('medium/'):
            return (1, dataset)
        elif dataset.startswith('large/'):
            return (2, dataset)
        else:
            return (3, dataset)
    
    all_datasets = sorted(list(all_datasets), key=sort_key)
    
    # Calculate the best (lowest) latency for each dataset
    best_latencies = {}
    for dataset in all_datasets:
        best_latency = float('inf')
        for method in all_results:
            if dataset in all_results[method] and all_results[method][dataset] < best_latency:
                best_latency = all_results[method][dataset]
        best_latencies[dataset] = best_latency
    
    # Calculate the average gap between each method and the best method
    avg_gaps = {}
    for method in all_results:
        total_gap = 0
        count = 0
        for dataset in all_datasets:
            if dataset in all_results[method]:
                gap = all_results[method][dataset] - best_latencies[dataset]
                total_gap += gap
                count += 1
        
        if count > 0:
            avg_gaps[method] = total_gap / count
        else:
            avg_gaps[method] = 0
    
    # Prepare data for plotting
    methods = list(all_results.keys())
    
    # Group datasets by category (small, medium, large)
    dataset_categories = {"small": [], "medium": [], "large": []}
    for dataset in all_datasets:
        if "small/" in dataset:
            dataset_categories["small"].append(dataset)
        elif "medium/" in dataset:
            dataset_categories["medium"].append(dataset)
        elif "large/" in dataset:
            dataset_categories["large"].append(dataset)
            
    # Add category dividers and labels to the plot
    category_dividers = []
    last_idx = 0
    category_order = ["small", "medium", "large"]
    
    for category in category_order:
        if dataset_categories[category]:
            last_idx += len(dataset_categories[category])
            category_dividers.append(last_idx - 0.5)
    
    # Create figure and axes - wider and shorter
    fig, ax = plt.subplots(figsize=(20, 6))
    
    # Set width of bars
    bar_width = 0.8 / len(methods)
    
    # Set positions of bars on X axis
    indices = np.arange(len(all_datasets))
    
    # Plot bars for each method with a colormap
    cmap = plt.cm.get_cmap('tab10', len(methods))
    
    for i, method in enumerate(methods):
        latencies = [all_results[method].get(dataset, 0) for dataset in all_datasets]
        positions = indices + i * bar_width - (len(methods) - 1) * bar_width / 2
        bars = ax.bar(positions, latencies, bar_width, 
                     label=f"{method} (avg gap: {avg_gaps[method]:.2f})",
                     color=cmap(i))
    
    # Add dataset names to x-axis with larger font
    ax.set_xticks(indices)
    ax.set_xticklabels(all_datasets, rotation=45, ha='right', fontsize=12)
    
    # Add labels and title with larger fonts
    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_ylabel('Latency', fontsize=14)
    ax.set_title('Latency Comparison of Different Methods', fontsize=16)
    
    # Increase font size of tick labels on y-axis
    ax.tick_params(axis='y', labelsize=12)
    
    # Add legend with larger font at the top of the plot
    ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.5), 
             ncol=len(methods), frameon=True, fancybox=True, shadow=True)
    
    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add category dividers if there are multiple categories
    if len(category_dividers) > 0:
        for divider in category_dividers[:-1]:  # Don't add a line after the last category
            ax.axvline(x=divider, color='black', linestyle='-', alpha=0.3)
        
        # Add category labels
        current_pos = 0
        for category in category_order:
            if dataset_categories[category]:
                category_size = len(dataset_categories[category])
                middle_pos = current_pos + category_size / 2
                ax.text(middle_pos, ax.get_ylim()[1] * 1.05, category.upper(), 
                        ha='center', va='bottom', fontsize=14, fontweight='bold')
                current_pos += category_size
    
    # Add more space at the top for the legend and category labels
    plt.subplots_adjust(top=0.85, bottom=0.2)
    
    # Highlight the best method for each dataset
    for i, dataset in enumerate(all_datasets):
        best_latency = float('inf')
        best_method_idx = -1
        
        for j, method in enumerate(methods):
            if dataset in all_results[method]:
                if all_results[method][dataset] < best_latency:
                    best_latency = all_results[method][dataset]
                    best_method_idx = j
        
        if best_method_idx >= 0:
            x_pos = i + best_method_idx * bar_width - (len(methods) - 1) * bar_width / 2
            ax.plot(x_pos, best_latency, marker='*', markersize=12, color='gold', 
                   markeredgecolor='black', zorder=10)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('latency_comparison.png', dpi=300)
    print(f"Plot saved as 'latency_comparison.png'")
    
    # Show the plot
    plt.show()
    
    # Print average gaps
    print("\nAverage gaps between each method and the best method:")
    for method, gap in sorted(avg_gaps.items(), key=lambda x: x[1]):
        print(f"{method}: {gap:.2f}")

def main():
    # Find all result files
    result_files = find_result_files()
    
    if not result_files:
        print("No result files found. Using example data for demonstration.")
        # Use the example data provided
        example_data = """Results Summary:
==================================================
Dataset                        | Latency   
--------------------------------------------------
large/dag_1000                 | X       
large/dag_1500                 | X       
large/dag_500                  | X       
medium/idctcol                 | 20       
medium/interpolate             | 12       
medium/invert                  | 22       
medium/jpeg_fdct               | 21       
medium/jpeg_idct               | 19       
medium/matmul                  | 13       
medium/smooth                  | 21       
medium/write                   | 13       
small/arf                      | 17       
small/collapse                 | 12       
small/cosine1                  | 15       
small/cosine2                  | 13       
small/demo                     | 4       
small/ewf                      | 22       
small/feedback                 | 14       
small/fir1                     | 17       
small/fir2                     | 15       
small/h2v2                     | 23       
small/hal                      | 9       
small/horner                   | 13       
small/motion                   | 13       
=================================================="""
        
        # Save example data to a temporary file
        with open("ilp.result", "w") as f:
            f.write(example_data)
        
        # Create some additional example files with different latencies
        methods = ["greedy", "opt"]
        for method in methods:
            with open(f"{method}.result", "w") as f:
                lines = example_data.split('\n')
                new_lines = []
                for line in lines:
                    if "|" in line and not "Dataset" in line and not "---------" in line:
                        parts = line.split("|")
                        dataset = parts[0].strip()
                        latency = parts[1].strip()
                        
                        # Modify latency for demonstration
                        if latency != "X":
                            if method == "greedy":
                                # Make greedy slightly worse
                                new_latency = int(latency) + np.random.randint(1, 5)
                            elif method == "opt":
                                # Make opt slightly better
                                new_latency = max(1, int(latency) - np.random.randint(1, 3))
                            
                            new_lines.append(f"{dataset} | {new_latency}")
                        else:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                
                f.write('\n'.join(new_lines))
        
        result_files = ["ilp.result", "greedy.result", "opt.result"]
    
    # Parse all result files
    all_results = {}
    for file_path in result_files:
        method_name = os.path.basename(file_path).replace('.result', '')
        results = parse_result_file(file_path)
        if results:
            all_results[method_name] = results
    
    # Generate the latency plot
    if all_results:
        generate_latency_plot(all_results)
    else:
        print("No valid data found in the result files.")

if __name__ == "__main__":
    main()