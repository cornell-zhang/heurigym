#include "solve.h"

// The solve function takes the graph (nodes) and the delay constraints, and returns the schedule
// as a mapping from node id to its start cycle. It performs an ASAP (as soon as possible) schedule.
std::map<std::string, int> solve(std::map<std::string, Node> &nodes, const std::map<std::string, int> &delay) {
    std::map<std::string, int> schedule;
    std::queue<std::string> q;
    
    // Push nodes with no incoming edges (in_degree == 0) into the queue.
    for (auto &pair : nodes) {
        if (pair.second.in_degree == 0) {
            q.push(pair.first);
        }
    }
    
    // Process nodes in topological order.
    while (!q.empty()) {
        std::string curId = q.front();
        q.pop();
        Node &curNode = nodes[curId];
        // Calculate finish time for the current node.
        int finish_time = curNode.start_time + delay.at(curNode.resource);
        // For every successor, update its start time and reduce its in-degree.
        for (const auto &succ : curNode.succs) {
            Node &succNode = nodes[succ];
            succNode.start_time = std::max(succNode.start_time, finish_time);
            succNode.in_degree--;
            if (succNode.in_degree == 0) {
                q.push(succ);
            }
        }
        schedule[curId] = curNode.start_time;
    }
    return schedule;
}