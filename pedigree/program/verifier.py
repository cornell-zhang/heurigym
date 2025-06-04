from typing import Tuple
from utils import get_allele_maps_0_indexed, get_n_alleles_from_sources

def verify(input_file: str, output_file: str) -> Tuple[bool, str]:
    individuals_data = {}
    present_ind_ids = []
    max_allele_val_from_obs = 0

    with open(input_file, 'r') as f_pre:
        for line_num, line_content in enumerate(f_pre, 1):
            line_content = line_content.strip()
            if not line_content:
                continue
            parts = line_content.split()
            if len(parts) < 7:
                # Infeasible: Invalid input format
                return False, "Invalid input format. Expected at least 7 fields."

            ind_id = int(parts[1])
            father_id = int(parts[2])
            mother_id = int(parts[3])
            obs_allele1 = int(parts[5])
            obs_allele2 = int(parts[6])

            if ind_id in individuals_data:
                # Infeasible: Duplicate individual ID
                return False, "Duplicate individual ID found."
            
            observed_genotype_set = None
            is_genotyped = False
            if obs_allele1 != 0 and obs_allele2 != 0:
                if obs_allele1 < 0 or obs_allele2 < 0: # Assuming alleles are positive integers
                    return False, "Invalid allele observation" # Invalid allele observation
                observed_genotype_set = frozenset({obs_allele1, obs_allele2})
                max_allele_val_from_obs = max(max_allele_val_from_obs, obs_allele1, obs_allele2)
                is_genotyped = True
            elif obs_allele1 != 0 or obs_allele2 != 0: # Partially genotyped, which original code treats as ungenotyped
                if (obs_allele1 !=0 and obs_allele1 <0) or \
                    (obs_allele2 !=0 and obs_allele2 <0) :
                        return False, "Invalid allele observation" # Invalid allele observation
                # This case (one allele 0, other not 0) is treated as not genotyped by original logic
                # for 'observed_genotype_set' and 'is_genotyped' flag.
                pass


            individuals_data[ind_id] = {
                'fid': father_id,
                'mid': mother_id,
                'is_genotyped': is_genotyped,
                'observed_genotype_set': observed_genotype_set,
                'assigned_gid_0idx': None
            }
            present_ind_ids.append(ind_id)
    
    present_ind_ids.sort()

    if not present_ind_ids and not individuals_data:
        with open(output_file, 'r') as f_out:
            if f_out.read().strip() == "":
                return True, "Empty input, empty output is feasible" # Empty input, empty output is feasible (cost 0)
            else:
                return False, "Output file not empty but pedigree file is" # Output file not empty but pedigree file is

    max_0_idx_domain_id_from_assign = -1
    with open(output_file, 'r') as f_out:
        line = f_out.readline().strip()
        assigned_gids_0idx = []
        if line:
            assigned_gids_str = line.split()
            try:
                assigned_gids_0idx = [int(g) for g in assigned_gids_str]
            except ValueError:
                return False, "Non-integer assignment" # Non-integer assignment

        if len(assigned_gids_0idx) != len(present_ind_ids):
            # Infeasible: Mismatch in number of assignments
            return False, "Infeasible: Mismatch in number of assignments (each provided data here should have matched number of individuals in input and output files)."

        for i, ind_id_from_sorted_list in enumerate(present_ind_ids):
            gid_0idx = assigned_gids_0idx[i]
            if gid_0idx < 0:
                # Infeasible: Invalid assigned GID (negative)
                return False, "Infeasible: Invalid assigned GID (negative)"
            individuals_data[ind_id_from_sorted_list]['assigned_gid_0idx'] = gid_0idx
            max_0_idx_domain_id_from_assign = max(max_0_idx_domain_id_from_assign, gid_0idx)

    if not present_ind_ids and max_0_idx_domain_id_from_assign != -1 : # output has assignments but input was empty
        return False, "Infeasible: Output has assignments but input was empty"


    n_alleles = get_n_alleles_from_sources(max_0_idx_domain_id_from_assign, max_allele_val_from_obs)
    if n_alleles < 0 : # Should be caught by get_n_alleles_from_sources if it raises ValueError
        return False, "Infeasible: Negative number of alleles"

    id_to_genotype_map, _ = get_allele_maps_0_indexed(n_alleles)

    if max_0_idx_domain_id_from_assign >= 0 and (max_0_idx_domain_id_from_assign not in id_to_genotype_map):
        # Infeasible: Max assigned GID not in generated map (internal error / n_alleles issue)
        # This implies an inconsistency, perhaps n_alleles was too small for the assigned GIDs.
        return False, "Infeasible: Max assigned GID not in generated map (internal error / n_alleles issue)"
    
    # Check for missing assignments for any individual before Mendelian checks
    for ind_id in present_ind_ids:
        if individuals_data[ind_id]['assigned_gid_0idx'] is None:
            # This case should be caught by len(assigned_gids_0idx) != len(present_ind_ids)
            # but as a safeguard:
            return False, "Infeasible: Missing assignment for individual"

    # Unary feasibility: check if assigned genotypes are valid domain members
    # (The cost part of unary is ignored, only hard constraint violations)
    for ind_id in present_ind_ids:
        data = individuals_data[ind_id]
        # Every individual must have an assignment at this point due to earlier checks
        assigned_gid_0idx = data['assigned_gid_0idx']
        # assigned_gid_0idx is already checked to be >= 0.

        # This check is effectively covered by the (max_0_idx_domain_id_from_assign not in id_to_genotype_map)
        # if assigned_gid_0idx is the max. If it's some other ID, it also needs to be in the map.
        if assigned_gid_0idx not in id_to_genotype_map:
                # This means an assigned GID is outside the range covered by n_alleles.
                return False, " Infeasible: Assigned GID not in domain map (out of range)"


    # Binary and Ternary (Mendelian) Feasibility Checks
    for ind_id_i in present_ind_ids:
        ind_i_data = individuals_data[ind_id_i]
        
        assigned_gid_0idx_i = ind_i_data['assigned_gid_0idx']
        # This being None should have been caught earlier or led to assigned_gid_0idx not in id_to_genotype_map
        if assigned_gid_0idx_i is None: return False # Should be caught by earlier checks

        genotype_i = id_to_genotype_map.get(assigned_gid_0idx_i)
        if genotype_i is None:
            # Infeasible: Assigned GID for child not in domain map
            return False, "Infeasible: Assigned GID for child not in domain map"

        fid = ind_i_data['fid']
        mid = ind_i_data['mid']

        father_is_in_data = (fid != 0 and fid in individuals_data)
        mother_is_in_data = (mid != 0 and mid in individuals_data)
        
        genotype_f, genotype_m = None, None
        if father_is_in_data:
            parent_f_data = individuals_data[fid]
            if parent_f_data['assigned_gid_0idx'] is None:
                return False, "Infeasible: Father in data but not assigned a genotype" # Infeasible: Father in data but not assigned a genotype
            parent_f_assigned_gid_0idx = parent_f_data['assigned_gid_0idx']
            genotype_f = id_to_genotype_map.get(parent_f_assigned_gid_0idx)
            if genotype_f is None:
                # Infeasible: Assigned GID for father not in domain map
                return False, "Infeasible: Assigned GID for father not in domain map"
        
        if mother_is_in_data:
            parent_m_data = individuals_data[mid]
            if parent_m_data['assigned_gid_0idx'] is None:
                return False, "Infeasible: Mother in data but not assigned a genotype" # Infeasible: Mother in data but not assigned a genotype
            parent_m_assigned_gid_0idx = parent_m_data['assigned_gid_0idx']
            genotype_m = id_to_genotype_map.get(parent_m_assigned_gid_0idx)
            if genotype_m is None:
                # Infeasible: Assigned GID for mother not in domain map
                return False, "Infeasible: Assigned GID for mother not in domain map"

        if father_is_in_data and mother_is_in_data:
            # Both parents are present and have valid assigned genotypes
            child_allele_list = list(genotype_i)
            c_allele1 = child_allele_list[0]
            c_allele2 = child_allele_list[0] if len(child_allele_list) == 1 else child_allele_list[1]

            mendelian_ok = False
            if (c_allele1 in genotype_f and c_allele2 in genotype_m) or \
                (c_allele1 in genotype_m and c_allele2 in genotype_f):
                mendelian_ok = True
            
            if not mendelian_ok:
                return False, "Infeasible: Mendelian violation (ternary)" # Infeasible: Mendelian violation (ternary)

        elif father_is_in_data: # Only father is present
            if genotype_i.isdisjoint(genotype_f):
                return False, "Infeasible: Child shares no alleles with known father (binary)" # Infeasible: Child shares no alleles with known father (binary)
                
        elif mother_is_in_data: # Only mother is present
            if genotype_i.isdisjoint(genotype_m):
                return False, "Infeasible: Child shares no alleles with known mother (binary)" # Infeasible: Child shares no alleles with known mother (binary)
        
    return True, "All checks passed, solution is feasible" # All checks passed, solution is feasible