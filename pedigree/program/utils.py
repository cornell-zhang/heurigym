def get_n_alleles_from_sources(max_0_idx_domain_id_from_assign, max_allele_val_from_obs_alleles):
    """
    Determines the number of unique alleles (n_alleles).
    It considers the maximum allele value explicitly observed (e.g., if allele '4' is seen, n_alleles >= 4)
    and the number of alleles needed to support the highest 0-indexed domain ID from assignments.
    """
    n_needed_for_assign_ids = 1 # Default minimum
    if max_0_idx_domain_id_from_assign >= 0:
        num_domain_items = max_0_idx_domain_id_from_assign + 1
        n = 0
        current_max_items_for_n = 0
        while current_max_items_for_n < num_domain_items:
            n += 1
            current_max_items_for_n = n * (n + 1) // 2
        n_needed_for_assign_ids = n
    
    n_needed_for_obs_alleles = 1 # Default minimum
    if max_allele_val_from_obs_alleles > 0:
        n_needed_for_obs_alleles = max_allele_val_from_obs_alleles
        
    return max(n_needed_for_assign_ids, n_needed_for_obs_alleles, 1)


def get_allele_maps_0_indexed(n_alleles_total):
    """
    Builds two maps for 0-indexed domain IDs:
    1. id_to_genotype: 0-idx ID -> frozenset({allele1, allele2})
    2. genotype_to_id: frozenset({allele1, allele2}) -> 0-idx ID
    """
    id_to_genotype_map = {}
    genotype_to_id_map = {}
    current_0_idx_id = 0
    for allele1 in range(1, n_alleles_total + 1):
        for allele2 in range(allele1, n_alleles_total + 1):
            genotype_alleles = frozenset({allele1, allele2})
            id_to_genotype_map[current_0_idx_id] = genotype_alleles
            genotype_to_id_map[genotype_alleles] = current_0_idx_id
            current_0_idx_id += 1
    return id_to_genotype_map, genotype_to_id_map
