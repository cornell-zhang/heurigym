#!/usr/bin/env python3
import sys
from utils import parse_json, parse_schedule


def verify(input_file: str, output_file: str) -> bool:
    """Verification function: checks dependency and resource constraints.

    Args:
        input_file: Path to the input JSON file containing graph and constraints
        output_file: Path to the schedule file containing node start times

    Returns:
        bool: True if schedule is valid, False otherwise

    Dependency: For each edge, finish time of predecessor (start + delay)
    must be less than or equal to the start time of the successor.

    Resource: At each cycle, the active operations for a resource type must
    not exceed the available functional units.
    """
    # Parse input files
    nodes, delay, resource_constraints = parse_json(input_file)
    schedule = parse_schedule(output_file)

    valid = True

    # Check data dependency constraints
    for node_id, node in nodes.items():
        node_delay = delay[node.resource]
        for succ_id in node.succs:
            if schedule[node_id] + node_delay > schedule[succ_id]:
                print(
                    f"Dependency constraint violated: {node_id} "
                    f"finishes at {schedule[node_id] + node_delay} "
                    f"but {succ_id} starts at {schedule[succ_id]}",
                    file=sys.stderr,
                )
                valid = False

    # Determine overall latency (final cycle when operations end)
    final_cycle = 0
    for node_id, node in nodes.items():
        finish_time = schedule[node_id] + delay[node.resource]
        final_cycle = max(final_cycle, finish_time)

    # Check resource constraints at each cycle from 0 to finalCycle
    for t in range(final_cycle + 1):
        # Count active operations per resource type
        resource_usage = {resource: 0 for resource in resource_constraints.keys()}

        # For each node, if its active time covers cycle t, increment usage
        for node_id, node in nodes.items():
            start = schedule[node_id]
            finish = schedule[node_id] + delay[node.resource]
            if start <= t < finish:
                resource_usage[node.resource] += 1

        # Verify that usage does not exceed available units
        for resource, available in resource_constraints.items():
            if resource_usage[resource] > available:
                print(
                    f"Resource constraint violated for resource {resource} "
                    f"at time {t}: used {resource_usage[resource]}, "
                    f"available {available}",
                    file=sys.stderr,
                )
                valid = False

    return valid
