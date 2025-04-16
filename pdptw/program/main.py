import argparse
import random
from solver import solve
from structure import Node, Instance

def read_instance(file_path: str) -> Instance:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    name = ""
    type_ = ""
    dimension = 0
    vehicles = 0
    capacity = 0.0
    edge_weight_type = ""
    nodes = {}
    depot_node = []

    section = None
    for line in lines:
        line = line.strip()
        if line.startswith("NAME"):
            name = line.split(":")[1].strip()
        elif line.startswith("TYPE"):
            type_ = line.split(":")[1].strip()
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())
        elif line.startswith("VEHICLES"):
            vehicles = int(line.split(":")[1].strip())
        elif line.startswith("CAPACITY"):
            capacity = float(line.split(":")[1].strip())
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":")[1].strip()
        elif line.startswith("NODE_COORD_SECTION"):
            section = "NODE_COORD_SECTION"
        elif line.startswith("PICKUP_AND_DELIVERY_SECTION"):
            section = "PICKUP_AND_DELIVERY_SECTION"
        elif line.startswith("DEPOT_SECTION"):
            section = "DEPOT_SECTION"
        elif line.startswith("EOF"):
            break
        elif section == "NODE_COORD_SECTION":
            parts = line.split()
            index = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            nodes[index] = Node(
                index=index,
                x=x,
                y=y,
                demand=0.0,
                earliest_time=0.0,
                latest_time=0.0,
                service_time=0.0,
                pickup_sibling=0,
                delivery_sibling=0,
                is_pickup=False  # Default value
            )
        elif section == "PICKUP_AND_DELIVERY_SECTION":
            parts = line.split()
            index = int(parts[0])
            if index in nodes:
                nodes[index].demand = float(parts[1])
                nodes[index].earliest_time = float(parts[2])
                nodes[index].latest_time = float(parts[3])
                nodes[index].service_time = float(parts[4])
                nodes[index].pickup_sibling = int(parts[5])
                nodes[index].delivery_sibling = int(parts[6])
                nodes[index].is_pickup = nodes[index].pickup_sibling == 0  # Set indicator
        elif section == "DEPOT_SECTION":
            if line != "-1":
                depot_node.append(int(line))

    return Instance(
        name=name,
        type=type_,
        dimension=dimension,
        vehicles=vehicles,
        capacity=capacity,
        edge_weight_type=edge_weight_type,
        nodes=nodes,
        depot_node=depot_node
    )

def test_instance(instance: Instance):
    print(f"Instance Name: {instance.name}")
    print(f"Type: {instance.type}")
    print(f"Dimension: {instance.dimension}")
    print(f"Vehicles: {instance.vehicles}")
    print(f"Capacity: {instance.capacity}")
    print(f"Edge Weight Type: {instance.edge_weight_type}")
    print(f"Number of Nodes: {len(instance.nodes)}")
    print(f"Depot Node: {instance.depot_node}")
    sample_nodes = random.sample(instance.nodes.values(), min(5, len(instance.nodes)))  # Randomly sample up to 5 nodes
    print("Sample Nodes:")
    for node in sample_nodes:
        print(node)

def main():
    parser = argparse.ArgumentParser(description="Solve a PDPTW instance.")
    parser.add_argument("input_file_path", type=str, help="Path to the PDPTW instance file")
    parser.add_argument("output_file_path", type=str, help="Path to save the solution output")
    args = parser.parse_args()

    instance = read_instance(args.input_file_path)
    test_instance(instance)
    cost, routes = solve(instance)
    with open(args.output_file_path, 'w') as output_file:
        output_file.write(f"{instance.name}, cost: {cost}\n")
        for route in routes:
            output_file.write(" ".join(map(str, route)) + "\n")

if __name__ == "__main__":
    main()
