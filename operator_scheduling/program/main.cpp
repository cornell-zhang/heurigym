// solver header
#include "solve.h"

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

int main(int argc, char* argv[]) {
    // Expect the DOT file and JSON file names as command line arguments.
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <dot file> <json file>" << std::endl;
        return 1;
    }
    std::string dotFile = argv[1];
    std::string jsonFile = argv[2];

    // Parse the input graph and the JSON configuration.
    auto nodes = parseDot(dotFile);
    auto jsonData = parseJSON(jsonFile);
    std::map<std::string, int> delay = jsonData.first;
    // The resource availability (jsonData.second) is parsed but not used in this ASAP demo.

    // Compute the schedule using ASAP scheduling.
    auto schedule = solve(nodes, delay);

    // Output the schedule in the format: node:cycle
    for (const auto &entry : schedule) {
        std::cout << entry.first << ":" << entry.second << std::endl;
    }
    return 0;
}
