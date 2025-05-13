import verifier


def evaluate(input_file: str, output_file: str) -> int:
    """
    Evaluate the solution for a given input file and output file.
    
    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file.
    
    Returns:
        int: the cost of the solution.
    """
    # Load the input and output files
    problem = verifier.load_problem(input_file)
    solution = verifier.load_solution(output_file)

    max_time = max(node.interval[1] for node in problem.nodes)
    total_cost = 0.0

    # Evaluate node strategies
    for node_idx, strategy_idx in enumerate(solution.strategies):
        node = problem.nodes[node_idx]
        strategy = node.strategies[strategy_idx]
        total_cost += strategy.cost

    # Evaluate edge strategies
    for edge in problem.edges:
        strategy_idx = 0
        for node_idx in edge.nodes:
            strategy_idx *= len(problem.nodes[node_idx].strategies)
            strategy_idx += solution.strategies[node_idx]

        edge_strategy = edge.strategies[strategy_idx]
        total_cost += edge_strategy.cost

    return total_cost


if __name__ == "__main__":
    print(evaluate("dataset/demo/example.json", "dataset/demo/example.sol"))
