from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import networkx as nx

class KLut:
    """Simplified class for k-LUT representation"""
    def __init__(self, k: int, inputs: List[str], truth_table: List[Tuple[str, str]] = None):
        self.k = k
        self.inputs = inputs
        self.name = f"LUT{len(inputs)}"
        if inputs:
            self.name += f"({','.join(inputs)})"
        
        # Truth table in the form of [(input_pattern, output_value), ...] 
        # e.g. [('11', '1'), ('00', '0')] for an AND gate
        if truth_table is None:
            # Initialize empty truth table with all outputs as '0'
            n_inputs = len(inputs)
            self.truth_table = []
            for i in range(2**n_inputs):
                # Convert i to binary string of length n_inputs
                pattern = format(i, f'0{n_inputs}b')
                self.truth_table.append((pattern, '0'))
        else:
            self.truth_table = truth_table
    
    def get_blif_representation(self, node_name: str) -> List[str]:
        """
        Generate the BLIF representation for this LUT
        
        Args:
            node_name: The name of the node this LUT belongs to
            
        Returns:
            List of strings representing the BLIF format for this LUT
        """
        blif_lines = []
        
        # Add .names line with inputs and output
        inputs_str = " ".join(self.inputs)
        blif_lines.append(f".names {inputs_str} {node_name}")
        
        # Add truth table entries - only for cases where output is '1'
        # BLIF format uses care/don't care notation:
        # For example: 1-0 1 means if first input is 1, third is 0, and second is don't care, output is 1
        one_outputs = [(pattern, output) for pattern, output in self.truth_table if output == '1']
        
        if len(one_outputs) == 0:
            # If no 1 outputs, use the BLIF format constant 0 representation
            # (an empty truth table means the output is always 0)
            pass
        elif len(one_outputs) == 2**len(self.inputs):
            # If all outputs are 1, use the BLIF format constant 1 representation
            blif_lines.append("1")
        else:
            # Normal case: list all patterns that result in output 1
            for pattern, output in one_outputs:
                blif_lines.append(f"{pattern} {output}")
        
        return blif_lines



@dataclass
class LogicNetwork:
    """Original Logic Network (technology-independent)"""
    graph: nx.DiGraph
    PI: Set[str] = field(default_factory=set)
    PO: Set[str] = field(default_factory=set)

    def topological_order(self) -> List[str]:
        return list(nx.topological_sort(self.graph))
