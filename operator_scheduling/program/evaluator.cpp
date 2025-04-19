#include "solver.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

// Function to extract filename from path
std::string getFilename(const std::string& path) {
    size_t lastSlash = path.find_last_of("/");
    if (lastSlash == std::string::npos) {
        return path;
    }
    return path.substr(lastSlash + 1);
}

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

// Verification function: checks dependency and resource constraints.
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
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <dataset> <schedule_file>" << std::endl;
        return 1;
    }

    std::string datasetPath = argv[1];
    std::string scheduleFile = argv[2];
    std::string dotFile = datasetPath + ".dot";
    std::string jsonFile = datasetPath + ".json";

    // Parse the input graph and the JSON configuration
    auto nodes = parseDot(dotFile);

    // Parse the JSON configuration
    std::ifstream file(jsonFile);
    if (!file) {
        std::cerr << "Error opening JSON file: " << jsonFile << std::endl;
        return 1;
    }
    json j;
    file >> j;
    std::map<std::string, int> delay;
    std::map<std::string, int> resourceConstraints;
    for (auto& element : j["delay"].items()) {
        delay[element.key()] = element.value();
    }
    for (auto& element : j["resource"].items()) {
        resourceConstraints[element.key()] = element.value();
    }

    // Read the schedule from file
    std::ifstream scheduleStream(scheduleFile);
    if (!scheduleStream) {
        std::cerr << "Error opening schedule file: " << scheduleFile << std::endl;
        return 1;
    }

    std::map<std::string, int> schedule;
    std::string line;
    while (std::getline(scheduleStream, line)) {
        size_t colonPos = line.find(':');
        if (colonPos != std::string::npos) {
            std::string nodeId = line.substr(0, colonPos);
            int cycle = std::stoi(line.substr(colonPos + 1));
            schedule[nodeId] = cycle;
        }
    }

    // Verify the schedule
    bool valid = verify(nodes, schedule, delay, resourceConstraints);
    if (!valid) {
        std::cerr << "Schedule verification failed" << std::endl;
        return 1;
    }

    // Calculate and report the final latency
    int finalLatency = calculateCost(nodes, schedule, delay);
    std::cout << "Cost: " << finalLatency << std::endl;

    return 0;
} 