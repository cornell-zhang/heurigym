#ifndef SOLVE_H
#define SOLVE_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <regex>
#include <algorithm>

// Data structure to represent an operation in the graph.
struct Node {
    std::string id;              // e.g., "n1"
    std::string resource;        // e.g., "mul" or "sub"
    std::vector<std::string> preds;  // Predecessor nodes
    std::vector<std::string> succs;  // Successor nodes
    int start_time = 0;          // Scheduled start cycle (to be computed)
    int in_degree = 0;           // Number of incoming edges (for topological ordering)
};

std::map<std::string, int> solve(
    std::map<std::string, Node> &nodes,
    const std::map<std::string, int> &delay,
    const std::map<std::string, int> &resourceConstraints);

#endif // SOLVE_H