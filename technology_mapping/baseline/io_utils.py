import json
import networkx as nx
import itertools
from structure import LogicNetwork

def create_complete_truth_table(explicit_truth_table, num_inputs):
    """
    Create a complete truth table with all 2^num_inputs rows.
    
    In BLIF format:
    - For rows with output '1': Only these input patterns produce '1', all others produce '0'
    - For rows with output '0': Only these input patterns produce '0', all others produce '1'
    
    Args:
        explicit_truth_table: List of (input_pattern, output) tuples from BLIF
        num_inputs: Number of inputs to the logic gate
        
    Returns:
        List of (input_pattern, output) tuples representing the complete truth table
    """
    # Generate all possible input combinations
    all_input_patterns = [''.join(p) for p in itertools.product('01', repeat=num_inputs)]
    
    # Check if we have any entries with output 0
    has_output_0 = any(output == '0' for _, output in explicit_truth_table)
    has_output_1 = any(output == '1' for _, output in explicit_truth_table)
    
    # if has both, raise error
    if has_output_0 and has_output_1:
        raise ValueError("BLIF file has both output 0 and 1 entries within the same node")

    # Determine the default output value (what to use for unlisted patterns)
    if has_output_0:
        # If only output '0' entries exist, default is '1'
        default_output = '1'
    else:
        default_output = '0'
    
    # Create a mapping of existing patterns to their outputs
    explicit_outputs = {}
    for pattern, output in explicit_truth_table:
        # Handle don't-care terms in the pattern
        if '-' in pattern:
            # Expand patterns with don't-cares into multiple patterns
            concrete_patterns = expand_dont_care_pattern(pattern)
            for concrete in concrete_patterns:
                explicit_outputs[concrete] = output
        else:
            explicit_outputs[pattern] = output
    
    # Create the complete truth table
    complete_table = []
    for pattern in all_input_patterns:
        # If this pattern is in the explicit table, use that output
        # Otherwise, use the default output
        output = explicit_outputs.get(pattern, default_output)
        complete_table.append((pattern, output))
    
    return complete_table

def expand_dont_care_pattern(pattern):
    """
    Expand a pattern with don't-care terms into all concrete patterns.
    
    Args:
        pattern: Input pattern with possible '-' (don't-care) terms
        
    Returns:
        List of concrete patterns without don't-care terms
    """
    if '-' not in pattern:
        return [pattern]
    
    # Find the position of the first don't-care
    pos = pattern.find('-')
    
    # Replace it with both '0' and '1' and recursively expand
    pattern_with_0 = pattern[:pos] + '0' + pattern[pos+1:]
    pattern_with_1 = pattern[:pos] + '1' + pattern[pos+1:]
    
    return expand_dont_care_pattern(pattern_with_0) + expand_dont_care_pattern(pattern_with_1)


def _collect_tokens(first_parts: list[str], lines: list[str], idx: int) -> tuple[list[str], int]:
    """
    Collect tokens that may be continued on the following lines with a trailing back-slash.
    `first_parts`     – result of .split() on the current line
    `lines`           – full file split by lines
    `idx`             – index *after* the current line (so the next unread line)
    returns tokens, new_idx
    """
    tokens = first_parts[1:]              # skip the directive itself
    while tokens and tokens[-1] == '\\':  # line continues
        tokens.pop()                      # drop the back-slash token
        if idx >= len(lines):
            break                         # just in case the file ends unexpectedly
        next_parts = lines[idx].strip().split()
        idx += 1
        tokens.extend(next_parts)
    return tokens, idx


# ---------- read BLIF format ----------
def read_blif(blif_path: str) -> LogicNetwork:
    G = nx.DiGraph()
    PI, PO = set(), set()

    with open(blif_path) as f:
        lines = f.read().splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        cmd = parts[0]

        # ---------- model name ----------
        if cmd == '.model':
            continue

        # ---------- primary inputs ----------
        elif cmd == '.inputs':
            pins, i = _collect_tokens(parts, lines, i)
            for sig in pins:
                PI.add(sig)
                G.add_node(sig, type='PI')

        # ---------- primary outputs ----------
        elif cmd == '.outputs':
            pins, i = _collect_tokens(parts, lines, i)
            for sig in pins:
                PO.add(sig)
                G.add_node(sig, type='PO')

        # ---------- combinational node (.names) ----------
        elif cmd == '.names':
            inputs = parts[1:-1]
            out = parts[-1]
            G.add_node(out, type='names')
            truth = []
            # read cube rows
            while i < len(lines) and lines[i] and not lines[i].lstrip().startswith('.'):
                tok = lines[i].strip().split()
                if len(tok) == 2:
                    truth.append((tok[0], tok[1]))
                i += 1
            # connect edges
            for inp in inputs:
                G.add_node(inp, type=G.nodes.get(inp, {}).get('type', 'wire'))
                G.add_edge(inp, out)
            # store truth table
            G.nodes[out]['truth_table'] = create_complete_truth_table(truth, len(inputs))
            G.nodes[out]['explicit_truth_table'] = truth
            continue

        elif cmd == '.end':
            break

    return LogicNetwork(graph=G, PI=PI, PO=PO)

