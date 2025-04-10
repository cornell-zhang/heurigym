// main.cpp
#include "solver.h"

// We use the nlohmann/json library for JSON parsing.
// You can get it from https://github.com/nlohmann/json
#include "nlohmann/json.hpp"
using json = nlohmann::json;

// Function to parse the DOT file and build the graph.
std::map<std::string, Node> parseDot(const std::string &filename) {
    std::map<std::string, Node> nodes;
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening DOT file: " << filename << std::endl;
        exit(1);
    }
    std::string line;
    // Regex to match node definitions: e.g., "n1 [label = mul];"
    std::regex node_regex(R"(^\s*(\w+)\s*\[\s*label\s*=\s*(\w+)\s*\]\s*;)");
    // Regex to match edge definitions: e.g., "n1 -> n3 [name = lhs];"
    std::regex edge_regex(R"(^\s*(\w+)\s*->\s*(\w+)\s*\[\s*name\s*=\s*(\w+)\s*\]\s*;)");
    
    while (std::getline(file, line)) {
        std::smatch match;
        if (std::regex_search(line, match, node_regex)) {
            std::string id = match[1];
            std::string resource = match[2];
            Node node;
            node.id = id;
            node.resource = resource;
            node.start_time = 0;
            node.in_degree = 0;
            nodes[id] = node;
        } else if (std::regex_search(line, match, edge_regex)) {
            std::string src = match[1];
            std::string dst = match[2];
            // Add the edge src -> dst
            nodes[src].succs.push_back(dst);
            nodes[dst].preds.push_back(src);
        }
    }
    // Initialize in-degrees for each node based on its predecessors.
    for (auto &pair : nodes) {
        pair.second.in_degree = pair.second.preds.size();
    }
    return nodes;
}

// Function to parse the JSON file.
// It returns a pair of maps:
// - first: mapping resource types (e.g., "mul", "sub") to their delays,
// - second: mapping resource types to the number of available functional units.
std::pair<std::map<std::string, int>, std::map<std::string, int>> parseJSON(const std::string &filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening JSON file: " << filename << std::endl;
        exit(1);
    }
    json j;
    file >> j;
    std::map<std::string, int> delay;
    std::map<std::string, int> resource;
    for (auto& element : j["delay"].items()) {
        delay[element.key()] = element.value();
    }
    for (auto& element : j["resource"].items()) {
        resource[element.key()] = element.value();
    }
    return {delay, resource};
}

// Verification function: checks dependency and resource constraints.
// Dependency: For each edge, finish time of predecessor (start + delay)
// must be less than or equal to the start time of the successor.
// Resource: At each cycle, the active operations for a resource type must
// not exceed the available functional units.
bool verify(const std::map<std::string, Node>& nodes,
                    const std::map<std::string, int>& schedule,
                    const std::map<std::string, int>& delay,
                    const std::map<std::string, int>& resourceConstraints) {
    bool valid = true;
    // Check data dependency constraints.
    for (const auto& pair : nodes) {
        const std::string &nodeID = pair.first;
        const Node &node = pair.second;
        int nodeDelay = delay.at(node.resource);
        for (const auto &succID : node.succs) {
            if (schedule.at(nodeID) + nodeDelay > schedule.at(succID)) {
                std::cerr << "Dependency constraint violated: " << nodeID 
                          << " finishes at " << (schedule.at(nodeID) + nodeDelay)
                          << " but " << succID << " starts at " 
                          << schedule.at(succID) << std::endl;
                valid = false;
            }
        }
    }
    
    // Determine overall latency (final cycle when operations end).
    int finalCycle = 0;
    for (const auto& pair : nodes) {
        int finishTime = schedule.at(pair.first) + delay.at(pair.second.resource);
        finalCycle = std::max(finalCycle, finishTime);
    }
    
    // Check resource constraints at each cycle from 0 to finalCycle.
    for (int t = 0; t <= finalCycle; t++) {
        // Count active operations per resource type.
        std::map<std::string, int> resourceUsage;
        // Initialize counts to zero.
        for (const auto &rc : resourceConstraints) {
            resourceUsage[rc.first] = 0;
        }
        // For each node, if its active time covers cycle t, increment usage.
        for (const auto& pair : nodes) {
            const std::string &nodeID = pair.first;
            const Node &node = pair.second;
            int start = schedule.at(nodeID);
            int finish = schedule.at(nodeID) + delay.at(node.resource);
            if (t >= start && t <= finish) {
                resourceUsage[node.resource]++;
            }
        }
        // Verify that usage does not exceed available units.
        for (const auto &rc : resourceConstraints) {
            if (resourceUsage[rc.first] > rc.second) {
                std::cerr << "Resource constraint violated for resource " 
                          << rc.first << " at time " << t 
                          << ": used " << resourceUsage[rc.first] 
                          << ", available " << rc.second << std::endl;
                valid = false;
            }
        }
    }
    return valid;
}

// Cost calculation function: calculates the final latency.
// Final latency is defined as the maximum over operations of (start cycle + delay).
int calculateCost(const std::map<std::string, Node>& nodes,
                  const std::map<std::string, int>& schedule,
                  const std::map<std::string, int>& delay) {
    int latency = 0;
    for (const auto &pair : nodes) {
        int finishTime = schedule.at(pair.first) + delay.at(pair.second.resource);
        latency = std::max(latency, finishTime);
    }
    return latency;
}

int main(int argc, char* argv[]) {
    // Expect the DOT file and JSON file names as command line arguments.
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset>" << std::endl;
        return 1;
    }
    std::string dotFile = std::string(argv[1]) + ".dot";
    std::string jsonFile = std::string(argv[1]) + ".json";

    // Parse the input graph and the JSON configuration.
    auto nodes = parseDot(dotFile);
    auto jsonData = parseJSON(jsonFile);
    std::map<std::string, int> delay = jsonData.first;
    std::map<std::string, int> resourceConstraints = jsonData.second;

    auto schedule = solve(nodes, delay, resourceConstraints);

    // Verify that the schedule satisfies all constraints.
    bool valid = verify(nodes, schedule, delay, resourceConstraints);
    if (!valid) {
        std::cerr << "Failed verification" << std::endl;
        // Output the schedule in the format: node:cycle
        std::cout << "Schedule:" << std::endl;
        for (const auto &entry : schedule) {
            std::cout << entry.first << ":" << entry.second << std::endl;
        }
        return 1;
    }

    // Calculate and report the final latency of the schedule.
    int finalLatency = calculateCost(nodes, schedule, delay);
    std::cout << "Final latency: " << finalLatency << std::endl;

    return 0;
}
