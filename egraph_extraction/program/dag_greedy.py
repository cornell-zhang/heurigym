import argparse
import numpy as np
from collections import defaultdict
from collections import deque
import time
from egraph_data import EGraphData
import os
import pickle


class CostSet:

    def __init__(self, costs=None, total=0, choice=None):
        self.costs = costs or {}
        self.total = total
        self.choice = choice


class FasterGreedyDagExtractor:

    def calculate_cost_set(self, node, costs, cost_of_node, egraph_enodes,
                           best_cost):
        if not egraph_enodes[node].eclass_id:
            return CostSet(
                {egraph_enodes[node].belong_eclass_id: cost_of_node[node]},
                cost_of_node[node], node)

        children_classes = list(
            set(child for child in egraph_enodes[node].eclass_id))

        cid = egraph_enodes[node].belong_eclass_id
        if cid in children_classes:
            return CostSet({}, float("inf"), node)

        first_cost = costs[children_classes[0]]
        if (
                # len(children_classes) == 1
                cost_of_node[node] + first_cost.total > best_cost):
            return CostSet({}, float("inf"), node)

        result = costs[children_classes[0]].costs.copy()
        for child_cid in children_classes[1:]:
            result.update(costs[child_cid].costs)

        contain = cid in result
        result[cid] = cost_of_node[node]
        result_cost = float("inf") if contain else sum(result.values())

        return CostSet(result, result_cost, node)

    # @profile
    def extract(self, cost_of_node, egraph_enodes, egraph_eclasses=None):
        "int index for elcass id and enode id"
        parents = defaultdict(list)
        analysis_pending = UniqueQueue()

        for node in egraph_enodes:
            if egraph_enodes[node].eclass_id == [] or egraph_enodes[
                    node].eclass_id == set():
                analysis_pending.insert(node)  #leaf node
            else:
                for child_class in egraph_enodes[node].eclass_id:
                    parents[child_class].append(node)

        costs = {}
        while analysis_pending:
            node = analysis_pending.pop()
            class_id = egraph_enodes[node].belong_eclass_id

            if all(child_class in costs
                   for child_class in egraph_enodes[node].eclass_id):
                if class_id in costs:
                    prev_cost = costs.get(class_id).total
                    # prev_choice = costs.get(class_id).choice
                else:
                    prev_cost = float("inf")

                cost_set = self.calculate_cost_set(node, costs, cost_of_node,
                                                   egraph_enodes, prev_cost)
                if cost_set.total < prev_cost:
                    costs[class_id] = cost_set
                    analysis_pending.extend(parents[class_id])
                # elif cost_set.total == prev_cost and prev_cost != float("inf"):
                #     if cost_set.choice < prev_choice:  # we remove the randomness
                #         costs[class_id] = cost_set
                #         analysis_pending.extend(parents[class_id])

        result = ExtractionResult()
        for cid, cost_set in costs.items():
            result.choose(cid, cost_set.choice)

        return result, costs


class UniqueQueue:

    def __init__(self):
        self.set = set()
        self.queue = deque()

    def insert(self, item):
        if item not in self.set:
            self.set.add(item)
            self.queue.append(item)

    def extend(self, items):
        for item in items:
            self.insert(item)

    def pop(self):
        if not self.queue:
            return None
        item = self.queue.popleft()
        self.set.remove(item)
        return item

    def __bool__(self):
        return bool(self.queue)


class ExtractionResult:

    def __init__(self):
        self.choices = {}
        self.final_dag = []

    def choose(self, class_id, node_id):
        self.choices[class_id] = node_id

    def find_cycles(self, egraph_enodes, roots):
        status = defaultdict(lambda: "Todo")
        cycles = []
        for root in roots:
            self._cycle_dfs(egraph_enodes, root, status, cycles)
        return cycles

    def _cycle_dfs(self, egraph_enodes, class_id, status, cycles):
        if status[class_id] == "Done":
            return
        elif status[class_id] == "Doing":
            cycle_start = False
            cycle_path = []
            for class_idx, class_status in status.items():
                if class_idx == class_id:  #the start of the cycle
                    cycle_start = True
                    assert (class_status == 'Doing')
                if not cycle_start:
                    continue
                else:
                    if class_status == 'Doing':  #Doing means in cycle
                        cycle_path.append([class_idx])
                cycles.append(cycle_path)
            return

        status[class_id] = "Doing"
        node = self.choices[class_id]
        for child in egraph_enodes[node].eclass_id:
            self._cycle_dfs(egraph_enodes, child, status, cycles)
        status[class_id] = "Done"

    def dag_cost(self, egraph_enodes, roots, cost, quad_cost=None):
        choose_enodes = []
        costs = {}
        todo = list(roots)
        while todo:
            cid = todo.pop()
            node = self.choices[cid]
            if cid in costs:
                continue
            costs[cid] = cost[node]
            choose_enodes.append(node)
            for child in egraph_enodes[node].eclass_id:
                todo.append(child)
        linear_cost = sum(costs.values())
        return linear_cost, choose_enodes


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        type=str,
                        default="examples/cunxi_test_egraph2.dot")
    return parser.parse_args()


def greedy(egraph):  #egraph's type is EGraphData or BaseEGraph
    "use int index, needs some preprocess for EGraphData/BaseEGraph"
    cost = egraph.cost_per_node.tolist()  #calculate on cpu
    start_time = time.time()
    extractor = FasterGreedyDagExtractor()
    result, cost_history = extractor.extract(cost, egraph.enodes,
                                             egraph.eclasses)
    end_time = time.time()
    root_classes = egraph.root
    root_classes = [egraph.class_mapping[cid] for cid in root_classes]
    assert result.find_cycles(egraph.enodes, root_classes) == []
    dag_cost, choose_enodes = result.dag_cost(egraph.enodes, root_classes,
                                              cost)
    time_consume = end_time - start_time
    return egraph.node_to_id(choose_enodes)


def main():
    args = get_args()
    # we do greedy on cpu
    egraph = EGraphData(args.input_file,
                        load_cost=False,
                        drop_self_loops=False,
                        device="cpu")
    choose_enodes = greedy(egraph)
    input_file = args.input_file.split("/")[-1]
    input_file = input_file.split(".")[0]
    saved_file_path = os.path.join("output_log", f'greedy_{input_file}.pkl')
    os.makedirs(os.path.dirname(saved_file_path), exist_ok=True)
    with open(saved_file_path, "wb") as f:
        pickle.dump(choose_enodes, f)


if __name__ == "__main__":
    main()
