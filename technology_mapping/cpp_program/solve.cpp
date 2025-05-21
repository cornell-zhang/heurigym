#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <set>
#include <algorithm>
#include <queue>
#include <limits>
#include <cstdint> // For uint64_t

// K_VAL is the maximum number of inputs for a LUT
const int K_VAL = 6;

// Structure to store original gate information from BLIF
struct OriginalGate {
    std::vector<std::string> input_names; // Ordered list of input signal names
    std::string output_name;
    // SOP lines: pair of (cube string, output character '0' or '1')
    std::vector<std::pair<std::string, char>> sop_lines;
    bool is_const_0 = false; // Flag for constant 0 gate (e.g., .names Z)
    bool is_const_1 = false; // Flag for constant 1 gate (e.g., .names VDD \n 1)
};

// Main structure for a node in the logic network
struct Node {
    int id;                 // Unique integer ID
    std::string name;       // Original name from BLIF
    bool is_pi = false;     // True if primary input
    bool is_po = false;     // True if primary output

    std::vector<int> fanin_ids;  // IDs of fanin nodes
    std::vector<int> fanout_ids; // IDs of fanout nodes

    OriginalGate original_logic; // Stores original gate data if not a PI

    // For dynamic programming
    float dp_cost = std::numeric_limits<float>::infinity(); // Min cost to implement this node as LUT output
    std::vector<int> best_cut_inputs; // Node IDs of inputs for the best LUT cut

    // For final mapping
    bool is_lut_root = false;    // True if this node is selected as an LUT root
    uint64_t lut_truth_table = 0; // Truth table if this node is an LUT root (for K<=6, 2^K <= 64 bits)
    
    int topo_level = -1; // Topological level (distance from PIs)
};

// Global variables to store the network
std::vector<Node> G_NODES;
std::map<std::string, int> G_NODE_NAME_TO_ID;
std::vector<int> G_PI_IDS;
std::vector<int> G_PO_IDS;
std::string G_MODEL_NAME;

// Gets existing node ID or creates a new node if name not found
int get_node_id(const std::string& name) {
    if (G_NODE_NAME_TO_ID.find(name) == G_NODE_NAME_TO_ID.end()) {
        int id = G_NODES.size();
        G_NODES.emplace_back();
        G_NODES[id].id = id;
        G_NODES[id].name = name;
        G_NODE_NAME_TO_ID[name] = id;
    }
    return G_NODE_NAME_TO_ID[name];
}

// Splits a string by whitespace into tokens
std::vector<std::string> split_string(const std::string& s) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (tokenStream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

// Parses the input BLIF file and builds the network graph
void parse_blif(const std::string& input_file) {
    std::ifstream ifs(input_file);
    std::string line;
    std::string accumulated_line; // For handling line continuations with '\'
    int ongoing_names_gate_id = -1; // Tracks current gate for SOP line parsing

    while(std::getline(ifs, line)) {
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line.erase(comment_pos); // Remove comments
        }

        // Trim leading/trailing whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

        if (line.empty()) continue;

        accumulated_line += line;

        if (line.back() == '\\') { // Line continuation
            accumulated_line.pop_back(); 
            accumulated_line += " "; 
            continue; 
        }
        
        std::vector<std::string> tokens = split_string(accumulated_line);
        accumulated_line.clear();

        if (tokens.empty()) continue;

        if (tokens[0] == ".model") {
            if (tokens.size() > 1) G_MODEL_NAME = tokens[1];
            ongoing_names_gate_id = -1;
        } else if (tokens[0] == ".inputs") {
            for (size_t i = 1; i < tokens.size(); ++i) {
                int id = get_node_id(tokens[i]);
                G_NODES[id].is_pi = true;
                G_PI_IDS.push_back(id);
            }
            ongoing_names_gate_id = -1;
        } else if (tokens[0] == ".outputs") {
            for (size_t i = 1; i < tokens.size(); ++i) {
                int id = get_node_id(tokens[i]);
                G_NODES[id].is_po = true;
                G_PO_IDS.push_back(id);
            }
            ongoing_names_gate_id = -1;
        } else if (tokens[0] == ".end") {
            ongoing_names_gate_id = -1;
            break; 
        } else if (tokens[0] == ".names") {
            OriginalGate gate_logic;
            for (size_t i = 1; i < tokens.size() - 1; ++i) {
                gate_logic.input_names.push_back(tokens[i]);
            }
            gate_logic.output_name = tokens.back();
            
            int output_node_id = get_node_id(gate_logic.output_name);
            ongoing_names_gate_id = output_node_id;

            for (const auto& input_name : gate_logic.input_names) {
                int input_node_id = get_node_id(input_name);
                G_NODES[output_node_id].fanin_ids.push_back(input_node_id);
                G_NODES[input_node_id].fanout_ids.push_back(output_node_id);
            }
            G_NODES[output_node_id].original_logic = gate_logic;
        } else { // SOP line or constant value for the previous .names gate
            if (ongoing_names_gate_id != -1) {
                Node& current_gate_node = G_NODES[ongoing_names_gate_id];
                if (tokens.size() == 1 && current_gate_node.original_logic.input_names.empty()) {
                    if (tokens[0] == "1") current_gate_node.original_logic.is_const_1 = true;
                    // is_const_0 is determined later if not is_const_1 and no SOP lines
                } else if (tokens.size() == 2) { // cube output_char
                    current_gate_node.original_logic.sop_lines.push_back({tokens[0], tokens[1][0]});
                } else if (tokens.size() == 1 && !current_gate_node.original_logic.input_names.empty()) { 
                    current_gate_node.original_logic.sop_lines.push_back({tokens[0], '1'}); // Implicit '1' output
                }
            }
        }
    }
    // Post-parsing: identify const 0 gates (.names Z)
    for (Node& n : G_NODES) {
        if (!n.is_pi && n.original_logic.input_names.empty() && 
            !n.original_logic.is_const_1 && n.original_logic.sop_lines.empty()) {
            n.original_logic.is_const_0 = true;
        }
    }
}

// Performs topological sort and sets topo_level for nodes
std::vector<int> topological_sort() {
    std::vector<int> sorted_nodes_result;
    std::map<int, int> in_degree;
    std::queue<int> q;

    for (const auto& node : G_NODES) {
        in_degree[node.id] = node.fanin_ids.size();
        if (in_degree[node.id] == 0) {
            q.push(node.id);
        }
    }

    int current_level = 0;
    while (!q.empty()) {
        int level_size = q.size();
        for (int i = 0; i < level_size; ++i) {
            int u_id = q.front();
            q.pop();
            sorted_nodes_result.push_back(u_id);
            G_NODES[u_id].topo_level = current_level;

            for (int v_id : G_NODES[u_id].fanout_ids) {
                in_degree[v_id]--;
                if (in_degree[v_id] == 0) {
                    q.push(v_id);
                }
            }
        }
        current_level++;
    }
    return sorted_nodes_result;
}

// Helper to unite two sorted vectors of ints (cut members)
std::vector<int> unite_cuts(const std::vector<int>& c1, const std::vector<int>& c2) {
    std::vector<int> result;
    result.reserve(c1.size() + c2.size());
    std::set_union(c1.begin(), c1.end(), c2.begin(), c2.end(), std::back_inserter(result));
    // std::merge + std::unique is an alternative if inputs are not necessarily unique sets
    // but set_union assumes inputs are sorted and produces sorted unique output.
    return result;
}

// Dynamic programming to compute minimum area flow and best cut for each node
void compute_dp(const std::vector<int>& sorted_node_ids) {
    for (int node_id : sorted_node_ids) {
        Node& current_node = G_NODES[node_id];
        if (current_node.is_pi) {
            current_node.dp_cost = 0;
            current_node.best_cut_inputs = {node_id}; // PI is its own "cut"
            continue;
        }

        std::set<std::vector<int>> candidate_cuts_set;

        if (current_node.original_logic.is_const_0 || current_node.original_logic.is_const_1) {
            candidate_cuts_set.insert({}); // Empty cut for constants
        } else {
            // Trivial cut: immediate fanins
            std::vector<int> trivial_cut_members = current_node.fanin_ids;
            std::sort(trivial_cut_members.begin(), trivial_cut_members.end());
            if (trivial_cut_members.size() <= K_VAL) {
                candidate_cuts_set.insert(trivial_cut_members);
            }

            // Cuts by expanding one fanin at a time
            for (int fanin_id_to_expand : current_node.fanin_ids) {
                if (G_NODES[fanin_id_to_expand].is_pi) continue;

                std::vector<int> base_cut_members;
                for (int f_id : current_node.fanin_ids) {
                    if (f_id != fanin_id_to_expand) {
                        base_cut_members.push_back(f_id);
                    }
                }
                std::sort(base_cut_members.begin(), base_cut_members.end());
                
                const std::vector<int>& expansion_part = G_NODES[fanin_id_to_expand].best_cut_inputs;
                // expansion_part is already sorted
                
                std::vector<int> new_cut_members = unite_cuts(base_cut_members, expansion_part);
                
                if (new_cut_members.size() <= K_VAL) {
                    candidate_cuts_set.insert(new_cut_members);
                }
            }
        }
        
        current_node.dp_cost = std::numeric_limits<float>::infinity();

        for (const auto& cut_members : candidate_cuts_set) {
            float current_sum_cost = 0;
            for (int member_id : cut_members) {
                current_sum_cost += G_NODES[member_id].dp_cost;
            }
            float total_cost = 1.0f + current_sum_cost;

            if (total_cost < current_node.dp_cost) {
                current_node.dp_cost = total_cost;
                current_node.best_cut_inputs = cut_members; // cut_members is sorted from set or unite_cuts
            } else if (total_cost == current_node.dp_cost) { // Tie-breaking: prefer smaller cuts
                if (cut_members.size() < current_node.best_cut_inputs.size()) {
                     current_node.best_cut_inputs = cut_members;
                }
            }
        }
    }
}

// Selects LUT roots by traversing backwards from POs
void select_lut_roots() {
    std::queue<int> q;
    for (int po_id : G_PO_IDS) {
        q.push(po_id);
    }

    while(!q.empty()) {
        int current_id = q.front();
        q.pop();

        if (G_NODES[current_id].is_pi || G_NODES[current_id].is_lut_root) {
            continue;
        }
        
        G_NODES[current_id].is_lut_root = true;

        for (int input_node_id : G_NODES[current_id].best_cut_inputs) {
             q.push(input_node_id); // No visited check needed due to DAG and is_lut_root/is_pi checks
        }
    }
}

// Evaluates gate's SOP to get truth table w.r.t LUT inputs
uint64_t evaluate_SOP(const OriginalGate& og, const std::vector<uint64_t>& arg_tts, int num_lut_inputs) {
    if (og.is_const_0) return 0ULL;
    if (og.is_const_1) return ~0ULL; // Full mask, will be implicitly correct for any num_lut_inputs

    uint64_t active_bits_mask; // Mask for relevant bits if num_lut_inputs < 6
    if (num_lut_inputs == 0) active_bits_mask = 1ULL; // Only 0-th bit matters
    else if (num_lut_inputs == 6) active_bits_mask = ~0ULL; // All 64 bits matter
    else active_bits_mask = (1ULL << (1 << num_lut_inputs)) - 1;

    bool first_line_is_on_set = true; 
    if (!og.sop_lines.empty()) {
        if (og.sop_lines[0].second == '0') {
            first_line_is_on_set = false;
        }
    } else { // No SOP lines for a non-constant gate means output is 0
        return 0ULL;
    }

    uint64_t result_tt = first_line_is_on_set ? 0ULL : active_bits_mask;

    for (const auto& sop_line : og.sop_lines) {
        const std::string& cube = sop_line.first;
        uint64_t term_tt = active_bits_mask;

        for (size_t k = 0; k < cube.length(); ++k) {
            if (cube[k] == '1') {
                term_tt &= arg_tts[k];
            } else if (cube[k] == '0') {
                term_tt &= (~arg_tts[k] & active_bits_mask);
            }
        }

        if (first_line_is_on_set) {
            result_tt |= term_tt;
        } else {
            result_tt &= (~term_tt & active_bits_mask);
        }
    }
    return result_tt;
}

// Computes truth tables for all selected LUTs
void compute_lut_truth_tables() {
    for (size_t i = 0; i < G_NODES.size(); ++i) {
        if (!G_NODES[i].is_lut_root) continue;

        Node& lut_root_node = G_NODES[i];
        // best_cut_inputs are already sorted by ID from DP stage
        const auto& lut_inputs = lut_root_node.best_cut_inputs; 
        int num_lut_inputs = lut_inputs.size();

        std::map<int, uint64_t> cone_node_tts; 

        for (int j = 0; j < num_lut_inputs; ++j) {
            uint64_t tt = 0;
            // For input x_j, its truth table has k-th bit = (k >> j) & 1
            // (k is minterm index, j is input index)
            for (int minterm_idx = 0; minterm_idx < (1 << num_lut_inputs); ++minterm_idx) {
                if ((minterm_idx >> j) & 1) {
                    tt |= (1ULL << minterm_idx);
                }
            }
            cone_node_tts[lut_inputs[j]] = tt;
        }
        
        std::vector<int> cone_topo_order;
        std::map<int, int> cone_in_degree;
        std::queue<int> work_q; // For local topological sort of cone

        std::set<int> cone_nodes_set; // Nodes part of the current LUT's cone
        std::queue<int> bfs_q; // For finding cone nodes
        bfs_q.push(lut_root_node.id);
        cone_nodes_set.insert(lut_root_node.id);

        int head = 0;
        std::vector<int> temp_bfs_vec; temp_bfs_vec.push_back(lut_root_node.id);

        while(head < temp_bfs_vec.size()){
            int curr_id = temp_bfs_vec[head++];
            
            bool is_an_lut_input = false; // Check if curr_id is one of lut_inputs
            for(int inp_id : lut_inputs) if(curr_id == inp_id) {is_an_lut_input = true; break;}
            if(is_an_lut_input) continue;

            for(int fanin_id : G_NODES[curr_id].fanin_ids){
                if(cone_nodes_set.find(fanin_id) == cone_nodes_set.end()){
                    cone_nodes_set.insert(fanin_id);
                    temp_bfs_vec.push_back(fanin_id);
                }
            }
        }
        for(int inp_id : lut_inputs) cone_nodes_set.insert(inp_id); // Ensure inputs are in set

        for (int node_id : cone_nodes_set) {
            bool is_an_lut_input = false;
            for(int inp_id : lut_inputs) if(node_id == inp_id) {is_an_lut_input = true; break;}

            if (is_an_lut_input) { // LUT inputs are roots for cone's topo sort
                 work_q.push(node_id);
                 cone_in_degree[node_id] = 0; // Explicitly set, though not strictly needed if handled first
                 continue;
            }
            
            int current_in_degree = 0;
            for (int fanin_id : G_NODES[node_id].fanin_ids) {
                if (cone_nodes_set.count(fanin_id)) { // Count only fanins within the cone
                    current_in_degree++;
                }
            }
            cone_in_degree[node_id] = current_in_degree;
            if (current_in_degree == 0) { // Should only be LUT inputs or consts within cone
                work_q.push(node_id);
            }
        }
        
        while(!work_q.empty()){
            int u_id = work_q.front();
            work_q.pop();
            cone_topo_order.push_back(u_id);

            for(int v_id : G_NODES[u_id].fanout_ids){
                if(cone_nodes_set.count(v_id)){ 
                    cone_in_degree[v_id]--;
                    if(cone_in_degree[v_id] == 0){
                        work_q.push(v_id);
                    }
                }
            }
        }

        for (int cone_node_id : cone_topo_order) {
            bool is_an_lut_input = false;
            for(int inp_id : lut_inputs) if(cone_node_id == inp_id) {is_an_lut_input = true; break;}
            if(is_an_lut_input) continue; // TT already set for LUT inputs

            Node& current_cone_node = G_NODES[cone_node_id];
            std::vector<uint64_t> arg_tts;
            for (const std::string& fanin_name : current_cone_node.original_logic.input_names) {
                int fanin_id = G_NODE_NAME_TO_ID[fanin_name]; // Name to ID
                arg_tts.push_back(cone_node_tts[fanin_id]);   // Get TT of this fanin
            }
            cone_node_tts[cone_node_id] = evaluate_SOP(current_cone_node.original_logic, arg_tts, num_lut_inputs);
        }
        lut_root_node.lut_truth_table = cone_node_tts[lut_root_node.id];
    }
}

// Writes the mapped network in BLIF format
void write_output_blif(const std::string& solution_file) {
    std::ofstream ofs(solution_file);
    ofs << ".model " << (G_MODEL_NAME.empty() ? "mapped_circuit" : G_MODEL_NAME) << std::endl;
    
    ofs << ".inputs";
    for (int pi_id : G_PI_IDS) ofs << " " << G_NODES[pi_id].name;
    ofs << std::endl;

    ofs << ".outputs";
    for (int po_id : G_PO_IDS) ofs << " " << G_NODES[po_id].name;
    ofs << std::endl;

    std::vector<int> lut_root_ids_sorted;
    for(const auto& node : G_NODES) if(node.is_lut_root) lut_root_ids_sorted.push_back(node.id);
    std::sort(lut_root_ids_sorted.begin(), lut_root_ids_sorted.end(), [&](int a, int b){
        return G_NODES[a].topo_level < G_NODES[b].topo_level; // Output LUTs in topological order
    });

    for (int root_id : lut_root_ids_sorted) {
        const Node& lut_node = G_NODES[root_id];
        const auto& lut_inputs = lut_node.best_cut_inputs; // Sorted by ID
        
        ofs << ".names";
        for (int input_id : lut_inputs) ofs << " " << G_NODES[input_id].name;
        ofs << " " << lut_node.name << std::endl;

        int num_lut_inputs = lut_inputs.size();
        if (num_lut_inputs == 0) { // Constant LUT
            if ((lut_node.lut_truth_table & 1ULL)) ofs << "1" << std::endl;
            // Else const 0: print nothing after .names line
        } else {
            int ones_count = 0;
            if (num_lut_inputs <= 5) { // Max 32 entries for TT
                 for (int i = 0; i < (1 << num_lut_inputs); ++i) {
                    if ((lut_node.lut_truth_table >> i) & 1ULL) ones_count++;
                 }
            } else { // num_lut_inputs == 6, 64 entries
                for (int i = 0; i < 32; ++i) { // popcount first half
                    if ((lut_node.lut_truth_table >> i) & 1ULL) ones_count++;
                }
                 for (int i = 32; i < 64; ++i) { // popcount second half
                    if ((lut_node.lut_truth_table >> i) & 1ULL) ones_count++;
                }
            }


            bool print_on_set = (ones_count <= (1 << (num_lut_inputs - 1)));
            // Handle edge cases: all ones or all zeros for non-empty inputs
            if (ones_count == (1 << num_lut_inputs) && num_lut_inputs > 0) print_on_set = true; // All ones
            if (ones_count == 0 && num_lut_inputs > 0) print_on_set = false; // All zeros

            for (int i = 0; i < (1 << num_lut_inputs); ++i) {
                bool bit_is_set = ((lut_node.lut_truth_table >> i) & 1ULL);
                if ((print_on_set && bit_is_set) || (!print_on_set && !bit_is_set)) {
                    for (int j = 0; j < num_lut_inputs; ++j) {
                        ofs << (((i >> j) & 1) ? '1' : '0');
                    }
                    ofs << " " << (print_on_set ? '1' : '0') << std::endl;
                }
            }
        }
    }
    ofs << ".end" << std::endl;
}

// Main solver function
void solve(const std::string& input_file, const std::string& solution_file) {
    // Clear global structures for potential multiple calls in a test harness
    G_NODES.clear();
    G_NODE_NAME_TO_ID.clear();
    G_PI_IDS.clear();
    G_PO_IDS.clear();
    G_MODEL_NAME.clear();

    parse_blif(input_file);
    
    std::vector<int> sorted_node_ids = topological_sort();

    compute_dp(sorted_node_ids);
    
    select_lut_roots();
    
    compute_lut_truth_tables();
    
    write_output_blif(solution_file);
}