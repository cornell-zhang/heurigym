import json
import os
from pathlib import Path

class Node:
    def __init__(self, num: int, type: str, delay: int):
        self.num = num
        self.type = type
        self.delay = delay
        self.asap = 0
        self.alap = None
        self.predecessors = []
        self.successors = []

    def set_alap(self, value: int):
        self.alap = value

class Graph:
    def __init__(self, vertex_count=None, constrained_latency=None, max_resources=None):
        self.vertex_count = vertex_count
        self.adjacency_list = []  # List of nodes
        self.resource_types = {}  # Resource types
        self.resource_usage = {}  # Resource usage per time step
        self.constrained_latency = constrained_latency
        self.max_resources = max_resources or {}
        self.visited_nodes = set()  # Track visited nodes
        self.topological_order = []  # Store topological order
        self.node_map = {}  # Map node names to indices

    @classmethod
    def from_json(cls, json_file: str, constrained_latency: int = None) -> 'Graph':
        """Create a Graph instance from a JSON file.
        
        Args:
            json_file: Path to the JSON file
            constrained_latency: Optional latency constraint
            
        Returns:
            Graph instance initialized from the JSON data
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Create graph with number of nodes from JSON
        graph = cls(len(data['nodes']), constrained_latency)
        
        # Set delays and resource constraints
        graph.max_resources = data['resource']
        
        # Create nodes
        for i, (node_name, node_type) in enumerate(data['nodes']):
            node = Node(i, node_type, data['delay'][node_type])
            graph.adjacency_list.append(node)
            graph.node_map[node_name] = i
        
        # Create edges
        for src_name, dst_name, _ in data['edges']:
            src_idx = graph.node_map[src_name]
            dst_idx = graph.node_map[dst_name]
            graph.adjacency_list[src_idx].successors.append(graph.adjacency_list[dst_idx])
            graph.adjacency_list[dst_idx].predecessors.append(graph.adjacency_list[src_idx])
        
        return graph

    def topological_sort_dfs(self):
        """Perform topological sorting using DFS and set ASAP/ALAP times"""
        self.visited_nodes = set()
        self.topological_order = []
        
        # First pass: Perform DFS to get topological order
        for node in self.adjacency_list:
            if node not in self.visited_nodes:
                self._depth_first_search(node)
        
        # Second pass: Calculate ASAP times
        self._calculate_asap_times()
        
        # Third pass: Calculate ALAP times
        self._calculate_alap_times()
    
    def _depth_first_search(self, node):
        """Helper function for DFS traversal"""
        self.visited_nodes.add(node)
        
        # Visit all successors first
        for successor in node.successors:
            if successor not in self.visited_nodes:
                self._depth_first_search(successor)
        
        # Add node to topological order after visiting all successors
        self.topological_order.insert(0, node)
    
    def _calculate_asap_times(self):
        """Calculate As Soon As Possible (ASAP) times"""
        # Reset ASAP times
        for node in self.adjacency_list:
            node.asap = 0
        
        # Process nodes in topological order
        for node in self.topological_order:
            # For each predecessor, update ASAP time
            for predecessor in node.predecessors:
                node.asap = max(node.asap, predecessor.asap + predecessor.delay)
    
    def _calculate_alap_times(self):
        """Calculate As Late As Possible (ALAP) times"""
        # Reset ALAP times
        for node in self.adjacency_list:
            node.alap = (self.constrained_latency - 1) if self.constrained_latency else (self.vertex_count - 1)
        
        # Process nodes in reverse topological order
        for node in reversed(self.topological_order):
            # For each successor, update ALAP time
            for successor in node.successors:
                node.alap = min(node.alap, successor.alap - node.delay)

    def map_resource_type(self, resource_type):
        """Map resource type to internal representation"""
        return resource_type

    def generate_time_constrained_ilp(self, outfile):
        """Generate ILP for time-constrained problems"""
        self.topological_sort_dfs()
        print("\nStart generating ILP formulas for latency-constrained problems...")

        outfile.write("Minimize\n")
        outfile.write("M1 + M2\n\n")
        outfile.write("Subject To\n")

        # Time frame constraints
        for i, node in enumerate(self.adjacency_list):
            time_frame = [f"x{i},{j}" for j in range(node.asap, node.alap + 1)]
            outfile.write(" + ".join(time_frame) + " = 1\n")
        print("Time frame constraints generated.")

        # Resource constraints
        for i in range(self.vertex_count):
            for j in range(self.adjacency_list[i].asap, self.adjacency_list[i].alap + self.adjacency_list[i].delay):
                resource_type = self.map_resource_type(self.adjacency_list[i].type)
                if j not in self.resource_usage:
                    self.resource_usage[j] = {}
                if resource_type not in self.resource_usage[j]:
                    self.resource_usage[j][resource_type] = []
                self.resource_usage[j][resource_type].append(i)

        # Initialize resource usage for all time steps
        for i in range(0, self.vertex_count):
            if i not in self.resource_usage:
                self.resource_usage[i] = {}
            for resource_type in self.max_resources.keys():
                if resource_type not in self.resource_usage[i]:
                    self.resource_usage[i][resource_type] = []

        for i in range(0, self.vertex_count):
            for resource_type in self.max_resources.keys():
                if len(self.resource_usage[i][resource_type]) < 2:
                    continue
                for j, node_idx in enumerate(self.resource_usage[i][resource_type]):
                    node = self.adjacency_list[node_idx]
                    for d in range(node.delay):
                        if i - d >= 0:
                            outfile.write(f"x{node_idx},{i-d}")
                            if not (j == len(self.resource_usage[i][resource_type]) - 1 and (d == node.delay - 1 or i - d == 0)):
                                outfile.write(" + ")
                        else:
                            break
                outfile.write(f" <= {self.max_resources.get(resource_type, 1)}\n")
        print("Resource constraints generated.")

        # Precedence constraints
        for node in self.adjacency_list:
            for successor in node.successors:
                node_terms = [f"{i} x{node.num},{i}" for i in range(node.asap, node.alap + 1)]
                successor_terms = [f"{i} x{successor.num},{i}" for i in range(successor.asap, successor.alap + 1)]
                outfile.write(" + ".join(node_terms) + " - " + " - ".join(successor_terms) + 
                            f" <= -{node.delay}\n")
        print("Precedence constraints generated.")

        # Bounds
        outfile.write("\nBounds\n")
        for i in range(self.vertex_count):
            for j in range(self.adjacency_list[i].asap, self.adjacency_list[i].alap + 1):
                outfile.write(f"0 <= x{i},{j} <= 1\n")
        outfile.write("M1 >= 1\nM2 >= 1\n")
        print("Bounds generated.")

        # Generals
        outfile.write("\nGenerals\n")
        for i in range(self.vertex_count):
            for j in range(self.adjacency_list[i].asap, self.adjacency_list[i].alap + 1):
                outfile.write(f"x{i},{j}\n")
        outfile.write("M1\nM2\n")
        print("Generals generated.")
        outfile.write("End\n")
        print("Finished ILP generation!")

    def generate_resource_constrained_ilp(self, outfile):
        """Generate ILP for resource-constrained problems"""
        self.topological_sort_dfs()
        print("Time frame:")
        for i, node in enumerate(self.adjacency_list):
            node.set_alap(self.vertex_count)
            print(f"{i+1}: [ {node.asap} , {node.alap} ]")
        print("\nStart generating ILP formulas for resource-constrained problems...")

        outfile.write("Minimize\n")
        outfile.write("L\n\n")
        outfile.write("Subject To\n")

        # Time frame and upper latency constraints
        for i, node in enumerate(self.adjacency_list):
            time_frame = [f"x{i},{j}" for j in range(node.asap, node.alap + 1)]
            outfile.write(" + ".join(time_frame) + " = 1\n")
            for j in range(node.asap, node.alap + 1):
                outfile.write(f"{j + node.delay} x{i},{j} - L <= 0\n")
        print("Time frame and upper latency constraints generated.")

        # Resource constraints
        for i in range(self.vertex_count):
            for j in range(self.adjacency_list[i].asap, self.adjacency_list[i].alap + self.adjacency_list[i].delay):
                resource_type = self.map_resource_type(self.adjacency_list[i].type)
                if j not in self.resource_usage:
                    self.resource_usage[j] = {}
                if resource_type not in self.resource_usage[j]:
                    self.resource_usage[j][resource_type] = []
                self.resource_usage[j][resource_type].append(i)

        # Initialize resource usage for all time steps
        for i in range(0, self.vertex_count):
            if i not in self.resource_usage:
                self.resource_usage[i] = {}
            for resource_type in self.max_resources.keys():
                if resource_type not in self.resource_usage[i]:
                    self.resource_usage[i][resource_type] = []

        for i in range(0, self.vertex_count):
            for resource_type in self.max_resources.keys():
                if len(self.resource_usage[i][resource_type]) < 2:
                    continue
                for j, node_idx in enumerate(self.resource_usage[i][resource_type]):
                    node = self.adjacency_list[node_idx]
                    for d in range(node.delay):
                        if i - d >= 0:
                            outfile.write(f"x{node_idx},{i-d}")
                            if not (j == len(self.resource_usage[i][resource_type]) - 1 and (d == node.delay - 1 or i - d == 0)):
                                outfile.write(" + ")
                        else:
                            break
                outfile.write(f" <= {self.max_resources.get(resource_type, 1)}\n")
        print("Resource constraints generated.")

        # Precedence constraints
        for node in self.adjacency_list:
            for successor in node.successors:
                node_terms = [f"{i} x{node.num},{i}" for i in range(node.asap, node.alap + 1)]
                successor_terms = [f"{i} x{successor.num},{i}" for i in range(successor.asap, successor.alap + 1)]
                outfile.write(" + ".join(node_terms) + " - " + " - ".join(successor_terms) + 
                            f" <= -{node.delay}\n")
        print("Precedence constraints generated.")

        # Bounds
        outfile.write("\nBounds\n")
        for i in range(self.vertex_count):
            for j in range(self.adjacency_list[i].asap, self.adjacency_list[i].alap + 1):
                outfile.write(f"0 <= x{i},{j} <= 1\n")
        outfile.write("L >= 1\n")
        print("Bounds generated.")

        # Generals
        outfile.write("\nGenerals\n")
        for i in range(self.vertex_count):
            for j in range(self.adjacency_list[i].asap, self.adjacency_list[i].alap + 1):
                outfile.write(f"x{i},{j}\n")
        outfile.write("L\n")
        print("Generals generated.")
        outfile.write("End\n")
        print("Finished ILP generation!")

def main():
    # Create ilp directory if it doesn't exist
    ilp_dir = Path("ilp")
    ilp_dir.mkdir(exist_ok=True)
    
    # Process all JSON files in dataset directories
    dataset_dirs = [
        "../dataset/demo",
        "../dataset/small",
        "../dataset/medium",
        "../dataset/large"
    ]
    
    for dataset_dir in dataset_dirs:
        if not os.path.exists(dataset_dir):
            print(f"Directory {dataset_dir} does not exist, skipping...")
            continue
            
        # Process each JSON file in the directory
        for json_file in Path(dataset_dir).glob("*.json"):
            print(f"Processing {json_file}...")
            
            # Create graph from JSON file
            graph = Graph.from_json(str(json_file))
            
            # Generate resource-constrained ILP
            output_file = ilp_dir / f"{json_file.stem}_rc.lp"
            with open(output_file, "w") as f:
                graph.generate_resource_constrained_ilp(f)
            print(f"Generated {output_file}")

if __name__ == "__main__":
    main() 