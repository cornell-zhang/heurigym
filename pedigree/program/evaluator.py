import math
from utils import get_allele_maps_0_indexed, get_n_alleles_from_sources

def evaluate(input_file: str, output_file: str) -> int:
    individuals_data = {}
    present_ind_ids = []
    max_allele_val_from_obs = 0 

    with open(input_file, 'r') as f_pre:
        for line_num, line_content in enumerate(f_pre, 1):
            line_content = line_content.strip()
            if not line_content:
                continue
            parts = line_content.split()
            # Expecting at least 7 fields for ObsAllele1, ObsAllele2
            if len(parts) < 7:
                raise ValueError(f"Line {line_num}: Invalid format. Expected at least 7 fields, got {len(parts)}.")

            ind_id = int(parts[1])
            father_id = int(parts[2])
            mother_id = int(parts[3])
            # parts[4] is Sex, ignored for now
            obs_allele1 = int(parts[5])
            obs_allele2 = int(parts[6])

            if ind_id in individuals_data:
                raise ValueError(f"Line {line_num}: Duplicate individual ID {ind_id} found.")
            
            observed_genotype_set = None
            is_genotyped = False
            if obs_allele1 != 0 and obs_allele2 != 0: # Both alleles known
                observed_genotype_set = frozenset({obs_allele1, obs_allele2})
                max_allele_val_from_obs = max(max_allele_val_from_obs, obs_allele1, obs_allele2)
                is_genotyped = True

            individuals_data[ind_id] = {
                'fid': father_id, 
                'mid': mother_id,
                'is_genotyped': is_genotyped, # Flag if fully genotyped
                'observed_genotype_set': observed_genotype_set, # frozenset or None
                'assigned_gid_0idx': None 
            }
            present_ind_ids.append(ind_id)
    
    present_ind_ids.sort()

    if not present_ind_ids and not individuals_data:
        with open(output_file, 'r') as f_out:
            if f_out.read().strip() == "": return 0
            else: raise ValueError("Output file is not empty but pedigree file is.")


    max_0_idx_domain_id_from_assign = -1
    with open(output_file, 'r') as f_out:
        line = f_out.readline().strip()
        assigned_gids_0idx = []
        if line:
            assigned_gids_str = line.split()
            assigned_gids_0idx = [int(g) for g in assigned_gids_str]
        
        if len(assigned_gids_0idx) != len(present_ind_ids):
            raise ValueError(f"Number of assignments ({len(assigned_gids_0idx)}) in output file "
                                f"does not match number of individuals ({len(present_ind_ids)}) in .pre file.")

        for i, ind_id_from_sorted_list in enumerate(present_ind_ids):
            gid_0idx = assigned_gids_0idx[i]
            if gid_0idx < 0:
                raise ValueError(f"Invalid assigned 0-indexed genotype ID {gid_0idx} for individual "
                                    f"{ind_id_from_sorted_list}. Must be >= 0.")
            individuals_data[ind_id_from_sorted_list]['assigned_gid_0idx'] = gid_0idx
            max_0_idx_domain_id_from_assign = max(max_0_idx_domain_id_from_assign, gid_0idx)

    # 3. Determine n_alleles and build maps
    n_alleles = get_n_alleles_from_sources(max_0_idx_domain_id_from_assign, max_allele_val_from_obs)
    
    id_to_genotype_map, genotype_to_id_map = get_allele_maps_0_indexed(n_alleles)

    if max_0_idx_domain_id_from_assign >= 0 and (max_0_idx_domain_id_from_assign not in id_to_genotype_map):
        raise RuntimeError(
            f"Internal error: Max assigned 0-indexed ID {max_0_idx_domain_id_from_assign} "
            f"not in generated ID->genotype map for n_alleles={n_alleles}. "
            f"Map size: {len(id_to_genotype_map)}."
        )

    # 4. Calculate costs
    total_cost = 0

    # Unary Costs
    for ind_id in present_ind_ids:
        data = individuals_data[ind_id]
        if data['is_genotyped']: # Individual has a full observed genotype {A,B}
            assigned_gid_0idx = data['assigned_gid_0idx']
            if assigned_gid_0idx is None: 
                raise RuntimeError(f"Ind {ind_id} genotyped but not assigned an ID.")

            assigned_genotype_set = id_to_genotype_map.get(assigned_gid_0idx)
            if assigned_genotype_set is None:
                raise ValueError(f"Assigned GID {assigned_gid_0idx} for Ind {ind_id} not in domain map.")

            if data['observed_genotype_set'] != assigned_genotype_set:
                total_cost += 1
    
    # Binary and Ternary Costs
    for ind_id_i in present_ind_ids:
        ind_i_data = individuals_data[ind_id_i]
        
        assigned_gid_0idx_i = ind_i_data['assigned_gid_0idx']
        if assigned_gid_0idx_i is None: raise RuntimeError(f"Missing assignment for Ind {ind_id_i}") 
        
        genotype_i = id_to_genotype_map.get(assigned_gid_0idx_i)
        if genotype_i is None:
             raise ValueError(f"Assigned 0-indexed GID {assigned_gid_0idx_i} for ind {ind_id_i} is invalid.")

        fid = ind_i_data['fid']
        mid = ind_i_data['mid']

        father_is_in_data = (fid != 0 and fid in individuals_data)
        mother_is_in_data = (mid != 0 and mid in individuals_data)
        
        genotype_f, genotype_m = None, None
        if father_is_in_data:
            parent_f_assigned_gid_0idx = individuals_data[fid]['assigned_gid_0idx']
            if parent_f_assigned_gid_0idx is None: return math.inf 
            genotype_f = id_to_genotype_map.get(parent_f_assigned_gid_0idx)
            if genotype_f is None: raise ValueError(f"Assigned GID {parent_f_assigned_gid_0idx} for father {fid} invalid.")
        
        if mother_is_in_data:
            parent_m_assigned_gid_0idx = individuals_data[mid]['assigned_gid_0idx']
            if parent_m_assigned_gid_0idx is None: return math.inf
            genotype_m = id_to_genotype_map.get(parent_m_assigned_gid_0idx)
            if genotype_m is None: raise ValueError(f"Assigned GID {parent_m_assigned_gid_0idx} for mother {mid} invalid.")

        if father_is_in_data and mother_is_in_data:
            child_allele_list = list(genotype_i) # frozenset({a}) -> [a]; frozenset({a,b}) -> [a,b] or [b,a]
            c_allele1 = child_allele_list[0]
            c_allele2 = child_allele_list[0] if len(child_allele_list) == 1 else child_allele_list[1]

            mendelian_ok = False
            if (c_allele1 in genotype_f and c_allele2 in genotype_m) or \
               (c_allele1 in genotype_m and c_allele2 in genotype_f):
                mendelian_ok = True
            
            if not mendelian_ok: return math.inf

        elif father_is_in_data:
            if genotype_i.isdisjoint(genotype_f): return math.inf
                
        elif mother_is_in_data:
            if genotype_i.isdisjoint(genotype_m): return math.inf
    
    return total_cost