from utils import parse_json, parse_schedule


def evaluate(input_file: str, output_file: str) -> int:
    """Cost calculation function: calculates the final latency.

    Args:
        input_file: Path to the input file containing graph and constraints
        output_file: Path to the schedule file containing node start times

    Returns:
        int: The final latency of the schedule

    Final latency is defined as the maximum over operations of (start cycle + delay).
    """
    nodes, delay, _ = parse_json(input_file)
    schedule = parse_schedule(output_file)

    latency = 0
    for node_id, node in nodes.items():
        finish_time = schedule[node_id] + delay[node.resource]
        latency = max(latency, finish_time)
    return latency
