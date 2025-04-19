#include "solver.h"

#include <queue>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <set>

std::map<std::string, int> solve(
    std::map<std::string, Node> &nodes,
    const std::map<std::string, int> &delay,
    const std::map<std::string, int> &resourceConstraints) {
    
    std::map<std::string, int> schedule;
    std::queue<std::string> readyQueue;
    std::unordered_map<std::string, int> remaining_in_degree;
    std::map<std::string, std::vector<std::pair<int, int>>> resourceAllocations;

    for (const auto &pair : nodes) {
        remaining_in_degree[pair.first] = pair.second.in_degree;
        if (pair.second.in_degree == 0) {
            readyQueue.push(pair.first);
        }
    }

    while (!readyQueue.empty()) {
        std::string node_id = readyQueue.front();
        readyQueue.pop();
        Node &node = nodes[node_id];
        std::string resource = node.resource;
        int node_delay = delay.at(resource);
        int earliest_start = 0;

        for (const auto &pred : node.preds) {
            int pred_finish = schedule[pred] + delay.at(nodes[pred].resource);
            earliest_start = std::max(earliest_start, pred_finish);
        }

        int start_time = earliest_start;
        while (true) {
            int end_time = start_time + node_delay;
            bool resource_available = true;

            int concurrent_usage = 0;
            for (const auto &alloc : resourceAllocations[resource]) {
                if (!(alloc.second <= start_time || alloc.first >= end_time)) {
                    concurrent_usage++;
                    if (concurrent_usage >= resourceConstraints.at(resource)) {
                        resource_available = false;
                        break;
                    }
                }
            }
            if (resource_available) {
                break;
            }
            start_time++;
        }

        schedule[node_id] = start_time;
        resourceAllocations[resource].emplace_back(start_time, start_time + node_delay);

        for (const auto &succ : node.succs) {
            remaining_in_degree[succ]--;
            if (remaining_in_degree[succ] == 0) {
                readyQueue.push(succ);
            }
        }
    }

    return schedule;
}