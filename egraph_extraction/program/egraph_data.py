import re
import os
import json
import pickle
import logging
import numpy as np
from collections import defaultdict


class ENode:

    def __init__(self, eclass_id, belong_eclass_id, label):
        self.eclass_id = eclass_id
        self.belong_eclass_id = belong_eclass_id
        self.normalized_prob = 0
        self.depth = 0
        self.label = label

    def __repr__(self) -> str:
        return f'ENode : {self.eclass_id}, belong_eclass_id: {self.belong_eclass_id}'


class EClass:

    def __init__(self, enode_id, hidden_dim):
        self.enode_id = enode_id
        self.in_nodes = defaultdict(int)
        self.visited_in_nodes = defaultdict(int)
        self.hidden_dim = hidden_dim
        # included = True if at least one in_node is sampled
        # or self is a source eclass
        self.included = False
        self.predecessor_embeddings = []
        self.predecessor_enodes = []
        self.normalized_prob = 1

    def add_in_node(self, in_node):
        self.in_nodes[in_node] += 1

    def add_visited_in_node(self, in_node):
        self.visited_in_nodes[in_node] += 1

    def __repr__(self) -> str:
        return f'EClass: {self.enode_id} in_nodes: {self.in_nodes} visited_in_nodes: {self.visited_in_nodes}'


class EGraphData:

    def __init__(self,
                 input_file,
                 hidden_dim=32,
                 load_cost=False,
                 compress=False,
                 drop_self_loops=False,
                 device='cpu'):
        self.eclasses = {}
        self.enodes = {}
        self.hidden_dim = hidden_dim
        self.load_cost = load_cost
        self.label_cost = False
        self.drop_self_loops = drop_self_loops
        self.compress = compress
        self.device = device

        if input_file.endswith('.json'):
            self.from_json_file(input_file)
        else:
            raise NotImplementedError
        self.set_cost_per_node()

    def __repr__(self) -> str:
        return f'EGraph: EClass {self.eclasses} ENode {self.enodes}'

    def from_dict(self, input_dict):
        self.input_dict = input_dict
        enode_map = {}
        node_to_class_id = {}  #tmp reverse dict
        self.class_mapping = {}
        enode_count = 0
        eclass_count = 0

        enode_cost = {}
        for enode_id, eclass_id in input_dict['nodes'].items():
            input_dict['nodes'][enode_id] = set(eclass_id)
            label = input_dict['labels'][enode_id]
            if label in self.label_cost:
                enode_cost[enode_id] = self.label_cost[label]
            else:
                enode_cost[enode_id] = self.label_cost['default']

        # preprocessing
        self.raw_num_enodes = len(input_dict['nodes'])
        self.raw_num_eclasses = len(input_dict['classes'])
        self.raw_nodes_mapping = {k: [k] for k in input_dict['nodes']}
        if self.drop_self_loops and len(input_dict['classes']) > 10:
            assert self.compress
            total_self_loops = 0
            total_merged = 0
            while True:
                l1 = drop_self_loops()
                l2 = compress()
                total_self_loops += l1
                total_merged += l2
                if l1 + l2 == 0:
                    break
            logging.info(f'Deleted {total_self_loops} self-loops nodes')
            logging.info(f'Merged {total_merged} singleton classes')

        self.enode_cost = [1] * self.raw_num_enodes

        for enode, v in self.raw_nodes_mapping.items():
            if v:
                enode_map[enode] = enode_count
                enode_count += 1
        preprocessed_num_nodes = enode_count
        for enode, v in self.raw_nodes_mapping.items():
            if not v:
                enode_map[enode] = enode_count
                enode_count += 1

        for eclass_id, enode_id in input_dict['classes'].items():
            # map enode_id (str) to enode_num_id (int)
            enode_num_id = []
            for node in enode_id:
                enode_num_id.append(enode_map[node])
                node_to_class_id[enode_map[node]] = eclass_count
            self.eclasses[eclass_count] = EClass(enode_num_id, self.hidden_dim)
            self.class_mapping[
                eclass_id] = eclass_count  # map eclass_id(str) to eclass_id(int)
            eclass_count += 1

        for (enode_id,
             eclass_id), (_, label) in zip(input_dict['nodes'].items(),
                                           input_dict['labels'].items()):
            self.enodes[enode_map[enode_id]] = ENode(
                eclass_id={self.class_mapping[i]
                           for i in eclass_id},
                belong_eclass_id=node_to_class_id[enode_map[enode_id]],
                label=label)
            # if self.load_cost or self.label_cost:
            #     self.enode_cost[enode_map[enode_id]] = enode_cost[enode_id]
            for eclass in eclass_id:
                self.eclasses[self.class_mapping[eclass]].add_in_node(
                    enode_map[enode_id])

        for enode_id in self.raw_nodes_mapping.keys():
            self.enode_cost[enode_map[enode_id]] = enode_cost[enode_id]

        self.enode_map = enode_map

        return self

    def from_json_file(self, json_file):
        with open(json_file, 'r') as f:
            input_dict = json.load(f)
        # the format from the extraction gym repo
        # (https://github.com/egraphs-good/extraction-gym)

        if 'classes' in input_dict:
            # format 1)
            if isinstance(input_dict['classes'], list):
                input_dict['classes'] = {
                    str(k): [str(vi) for vi in v]
                    for k, v in enumerate(input_dict['classes'])
                }
            if isinstance(input_dict['nodes'], list):
                input_dict['nodes'] = {
                    str(k): [str(vi) for vi in v]
                    for k, v in enumerate(input_dict['nodes'])
                }
            self.from_dict(input_dict)
        else:
            class_out_list = defaultdict(list)
            label_cost = defaultdict(int)

            new_dict = {'nodes': {}, 'classes': {}, 'labels': {}}
            for i, node in enumerate(input_dict['nodes']):
                cur_enode = input_dict['nodes'][node]
                pattern1 = r'(\d+)__\d+'
                pattern2 = r'(\d+).\d+'
                eclass_list = []
                for child in cur_enode['children']:
                    p1_result = re.findall(pattern1, child)
                    p2_result = re.findall(pattern2, child)
                    if len(p1_result) > 0:
                        eclass_list.append(p1_result[0])
                    else:
                        eclass_list.append(p2_result[0])
                new_dict['nodes'][node] = eclass_list
                new_dict['labels'][node] = node

                class_out_list[cur_enode['eclass']].append(node)
                label_cost[node] = cur_enode['cost']
            new_dict['classes'] = class_out_list
            self.label_cost = label_cost

            self.root = input_dict['root_eclasses']
            self.from_dict(new_dict)

        return self

    def class_to_id(self, classes):
        class_ids = []
        for eclass in bool_to_index(classes):
            for k, v in self.class_mapping.items():
                if v == eclass:
                    class_ids.append(k)
                    break
        return class_ids

    def node_to_id(self, nodes):
        node_ids = []
        for enode in nodes:
            for k, v in self.enode_map.items():
                if v == enode:
                    node_ids.append(k)
                    break
        return node_ids

    def set_cost_per_node(self):
        cost_per_node = []
        if hasattr(self, 'enode_cost'):
            cost_per_node = np.array(self.enode_cost)
        else:
            cost_per_node = np.array(len(self.enodes))
            cost_per_node.zero_()
            cost_per_node += 1
        self.cost_per_node = cost_per_node

    def linear_cost(self, enodes):
        linear_loss = (self.cost_per_node * enodes).sum(dim=1)
        return linear_loss
