import argparse
from io_utils import read_blif
from solver_baseline import solve # baseline solver
# from solver import solve # placeholder solver
import os

def write_blif_solution(out_path: str, area: float, mapping, netlist):
    """
    Write the mapping solution in BLIF format
    
    Args:
        out_path: Path to write the output file
        area: Total area of the mapping
        mapping: List of (node, lut) pairs
        netlist: The original logic network
    """
    # Get the filename without extension
    basename = os.path.basename(out_path)
    model_name = os.path.splitext(basename)[0]
    
    with open(out_path, "w") as f:
        # Write header with statistics as comments
        f.write(f"# Mapped netlist with {len(mapping)} LUTs, total area = {area}\n")
        f.write(f".model {model_name}\n")
        
        # Write primary inputs
        pi_str = " ".join(sorted(netlist.PI))
        f.write(f".inputs {pi_str}\n")
        
        # Write primary outputs
        po_str = " ".join(sorted(netlist.PO))
        f.write(f".outputs {po_str}\n")
        
        # Write each LUT
        for node, lut in mapping:
            # Get the BLIF representation for this LUT
            lut_blif = lut.get_blif_representation(node)
            for line in lut_blif:
                f.write(f"{line}\n")
        
        # End the model
        f.write(".end\n")

def write_mapping_summary(out_path: str, area: float, mapping):
    """
    Write a simple mapping summary that shows which node maps to which LUT
    
    Args:
        out_path: Path to write the output file
        area: Total area of the mapping
        mapping: List of (node, lut) pairs
    """
    with open(f"{out_path}.summary", "w") as f:
        f.write(f"# total area: {area}\n")
        for node, lut in mapping:
            f.write(f"{node} -> {lut.name}\n")

def main():
    parser = argparse.ArgumentParser(description="Technology-mapping driver")
    parser.add_argument("--net",  required=True, 
                        help="Path to logic netlist (.blif)")
    parser.add_argument("--out",  required=True, 
                        help="Output mapped netlist file path (.blif)")
    parser.add_argument("--delay_budget", type=float, default=None,
                        help="Optional timing constraint (ps)")
    parser.add_argument("--k", required=True, type=int,
                        help="K value for k-LUT mapping")
    parser.add_argument("--optimize", choices=["size", "depth"], default="size",
                        help="Optimization target: 'size' (number of LUTs) or 'depth' (number of levels)")
    args = parser.parse_args()
    

    netlist = read_blif(args.net)
    
    # print out something about the netlist
    print(f"Netlist has {len(netlist.graph.nodes)} nodes and {len(netlist.graph.edges)} edges")
    print(f"Netlist has {len(netlist.PI)} PIs and {len(netlist.PO)} POs")
    # print PI and PO
    print(f"PIs: {netlist.PI}")
    print(f"POs: {netlist.PO}")
    # # For each node, print its successors, print out the successors dict
    # for node in netlist.graph.nodes:
    #     for successor in netlist.graph.successors(node):
    #         print(f"Node {node} has successor {successor}")
    #     # print out the truth table if it exists
    #     if 'truth_table' in netlist.graph.nodes[node]:
    #         print(f"Node {node} has truth table {netlist.graph.nodes[node]['truth_table']}")

    
    # k-LUT mapping mode
    area, mapping = solve(netlist, args.k, args.delay_budget, optimize_for=args.optimize)


    # mkdir -p $(dirname $args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Write the mapping in BLIF format
    write_blif_solution(args.out, area, mapping, netlist)
    
    # Also write a simple summary file
    write_mapping_summary(args.out, area, mapping)
    
    print(f"Mapping finished, total area = {area}")
    print(f"BLIF output written to {args.out}")
    print(f"Mapping summary written to {args.out}.summary")

if __name__ == "__main__":
    main()
