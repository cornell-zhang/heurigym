import os
import json
import re
from pathlib import Path

def parse_dot_file(dot_path):
    """Parse a DOT file and convert it to our JSON format."""
    with open(dot_path, 'r') as f:
        content = f.read()
    
    # Extract graph name
    name_match = re.search(r'digraph\s+(\w+)\s*{', content)
    graph_name = name_match.group(1) if name_match else "unnamed"
    
    # Extract nodes
    nodes = []
    # Handle arbitrary spaces and optional quotes around label values
    node_pattern = r'"?([^"\s\[\]]+)"?\s*\[\s*label\s*=\s*([^;[\]]+?)\s*\]\s*;?'
    for match in re.finditer(node_pattern, content):
        node_id = match.group(1)
        # Remove quotes and extra spaces from label
        label = match.group(2).strip().strip('"')
        nodes.append([node_id, label])
    
    # Extract edges
    edges = []
    # Handle arbitrary spaces and optional quotes around name values
    edge_pattern = r'"?([^"\s\[\]]+)"?\s*->\s*"?([^"\s\[\]]+)"?\s*\[\s*name\s*=\s*([^;[\]]+?)\s*\]\s*;?'
    for match in re.finditer(edge_pattern, content):
        from_node = match.group(1)
        to_node = match.group(2)
        # Remove quotes and extra spaces from name
        name = match.group(3).strip().strip('"')
        edges.append([from_node, to_node, name])
    
    # Find all node and edge declarations in the file
    all_nodes = set()
    all_edges = set()
    
    # Process each line to find nodes and edges, skipping node style definitions
    for line in content.split('\n'):
        # Skip node style definitions
        if 'node [' in line:
            continue
        
        # Find node declarations
        node_matches = re.findall(r'"?([^"\s\[\]]+)"?\s*\[', line)
        all_nodes.update(node_matches)
        
        # Find edge declarations
        edge_matches = re.findall(r'"?([^"\s\[\]]+)"?\s*->\s*"?([^"\s\[\]]+)"?', line)
        all_edges.update(edge_matches)
    
    # Check for unprocessed nodes
    processed_nodes = {node[0] for node in nodes}
    unprocessed_nodes = all_nodes - processed_nodes
    if unprocessed_nodes:
        # Print the problematic lines for debugging
        print(f"\nContent of {dot_path}:")
        for line in content.split('\n'):
            if any(node in line for node in unprocessed_nodes):
                print(f"Unprocessed line: {line}")
        raise ValueError(f"Failed to process some nodes in {dot_path}: {unprocessed_nodes}")
    
    # Check for unprocessed edges
    processed_edges = {(edge[0], edge[1]) for edge in edges}
    unprocessed_edges = all_edges - processed_edges
    if unprocessed_edges:
        # Print the problematic lines for debugging
        print(f"\nContent of {dot_path}:")
        for line in content.split('\n'):
            if any(f"{edge[0]} -> {edge[1]}" in line for edge in unprocessed_edges):
                print(f"Unprocessed line: {line}")
        raise ValueError(f"Failed to process some edges in {dot_path}: {unprocessed_edges}")
    
    return {
        "name": graph_name,
        "nodes": nodes,
        "edges": edges
    }

def is_already_processed(data):
    """Check if the JSON data is already in the succinct format."""
    if "nodes" not in data or "edges" not in data:
        return False
    
    # Check if nodes are in succinct format
    if not data["nodes"] or not isinstance(data["nodes"][0], list):
        return False
    
    # Check if edges are in succinct format
    if not data["edges"] or not isinstance(data["edges"][0], list):
        return False
    
    return True

def process_file(file_path):
    """Process a single file and convert it to our JSON format."""
    print(f"Processing {file_path}")
    
    if file_path.suffix == '.dot':
        try:
            # Parse the DOT file
            dot_data = parse_dot_file(file_path)
            
            # Look for corresponding JSON file
            json_path = file_path.with_suffix('.json')
            if not json_path.exists():
                print(f"Warning: No corresponding JSON file found for {file_path}")
                return
            
            # Read the JSON file for delay and resource info
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # Combine the data
            result = {
                "name": dot_data["name"],
                "delay": json_data.get("delay", {}),
                "resource": json_data.get("resource", {}),
                "nodes": dot_data["nodes"],
                "edges": dot_data["edges"]
            }
            
            # Save as .input file with compact formatting
            output_path = file_path.with_suffix('.input')
            with open(output_path, 'w') as f:
                # Write the header
                f.write('{\n')
                f.write(f'  "name": "{result["name"]}",\n')
                
                # Write delay dictionary, one item per line
                f.write('  "delay": {\n')
                delay_items = list(result["delay"].items())
                for i, (key, value) in enumerate(delay_items):
                    f.write(f'    "{key}": {value}')
                    if i < len(delay_items) - 1:
                        f.write(',\n')
                    else:
                        f.write('\n')
                f.write('  },\n')
                
                # Write resource dictionary, one item per line
                f.write('  "resource": {\n')
                resource_items = list(result["resource"].items())
                for i, (key, value) in enumerate(resource_items):
                    f.write(f'    "{key}": {value}')
                    if i < len(resource_items) - 1:
                        f.write(',\n')
                    else:
                        f.write('\n')
                f.write('  },\n')
                
                # Write nodes, one per line
                f.write('  "nodes": [\n')
                for i, node in enumerate(result["nodes"]):
                    f.write(f'    {json.dumps(node, separators=(", ", ":"))}')
                    if i < len(result["nodes"]) - 1:
                        f.write(',\n')
                    else:
                        f.write('\n')
                f.write('  ],\n')
                
                # Write edges, one per line
                f.write('  "edges": [\n')
                for i, edge in enumerate(result["edges"]):
                    f.write(f'    {json.dumps(edge, separators=(", ", ":"))}')
                    if i < len(result["edges"]) - 1:
                        f.write(',\n')
                    else:
                        f.write('\n')
                f.write('  ]\n')
                f.write('}\n')
            
            print(f"Saved to {output_path}")
        except ValueError as e:
            print(f"Error processing {file_path}: {str(e)}")
            return
    
    elif file_path.suffix == '.json':
        # Skip JSON files that don't have a corresponding DOT file
        dot_path = file_path.with_suffix('.dot')
        if not dot_path.exists():
            print(f"Skipping {file_path} - no corresponding DOT file")
            return
        
        # Skip if the DOT file hasn't been processed yet
        input_path = dot_path.with_suffix('.input')
        if not input_path.exists():
            print(f"Skipping {file_path} - DOT file not yet processed")
            return

def main():
    dataset_dir = Path(__file__).parent
    
    # Find all .dot and .json files recursively
    for file_path in dataset_dir.rglob('*'):
        if file_path.suffix in ['.dot', '.json']:
            process_file(file_path)

if __name__ == '__main__':
    main() 