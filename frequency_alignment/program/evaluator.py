from typing import Union, List, Dict


def evaluate(input_file: str, output_file: str) -> Union[int, float]:
    """
    Cost calculation function: calculates the total number of violations
    (mandatory + CEM) for a given solution.

    Args:
        input_file:    Path to the input file describing domains and constraints
        solution_file: Path to the solver's output file with AL lines

    Returns:
        Union[int, float]: The sum of mandatory constraint violations and
                           CEM violations across all levels.
    """
    # --- parse the solution file (AL lines) ---
    # trajets[path] = {'freq': f, 'pol': p, 'dom_freq': None, 'dom_pol': None}
    trajets: Dict[int, Dict[str, int]] = {}
    with open(output_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != 'AL':
                continue
            path_id = int(parts[1])
            freq    = int(parts[2])
            pol     = int(parts[3])
            trajets[path_id] = {
                'freq':     freq,
                'pol':      pol,
                'dom_freq': None,
                'dom_pol':  None
            }

    # --- initialize data structures for evaluation ---
    domains: Dict[int, List[int]] = {}     # domain_index -> list of allowed freqs
    VIOL_CI = 0                            # count of mandatory (Class 1) violations
    TAB_VIOL = [0] * 11                    # CEM violations per level 0..10
    kmax    = 0                            # highest level at which a violation occurred
    THETA   = 0                            # total number of CEM constraints checked

    # --- parse the input file ---
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            rec = parts[0]

            # 1) Domain definition: DM <dom> <freq>
            if rec == 'DM':
                dom = int(parts[1])
                frq = int(parts[2])
                domains.setdefault(dom, []).append(frq)

            # 2) Path resource domain assignment: TR <path> <dom_freq> <dom_pol>
            elif rec == 'TR':
                path     = int(parts[1])
                dom_freq = int(parts[2])
                dom_pol  = int(parts[3])
                # ensure the path was in the solution
                if path not in trajets:
                    trajets[path] = {'freq': None, 'pol': None, 'dom_freq': None, 'dom_pol': None}
                trajets[path]['dom_freq'] = dom_freq
                trajets[path]['dom_pol']  = dom_pol

            # 3) Mandatory constraints (Class 1): CI <i> <j> <F/P> <E/I> <value>
            elif rec == 'CI':
                i1, i2      = int(parts[1]), int(parts[2])
                X1, X2      = parts[3], parts[4]
                val         = int(parts[5])
                t1, t2      = trajets[i1], trajets[i2]
                f1, f2      = t1['freq'], t2['freq']
                p1, p2      = t1['pol'],  t2['pol']

                if X1 == 'F':  # frequency constraint
                    if X2 == 'E':  # equality
                        if abs(f1 - f2) != val:
                            VIOL_CI += 1
                    else:          # inequality
                        if abs(f1 - f2) == val:
                            VIOL_CI += 1
                else:           # polarization constraint
                    if X2 == 'E':  # same polarization
                        if p1 != p2:
                            VIOL_CI += 1
                    else:          # opposite polarization
                        if p1 == p2:
                            VIOL_CI += 1

            # 4) CEM constraints, same‐polarization: CE <i> <j> t0 t1 … t10
            elif rec == 'CE':
                i1, i2        = int(parts[1]), int(parts[2])
                thresholds    = list(map(int, parts[3:]))
                t1, t2        = trajets[i1], trajets[i2]
                f1, f2        = t1['freq'], t2['freq']
                p1, p2        = t1['pol'],  t2['pol']

                # only check if polarizations match
                if p1 == p2:
                    THETA += 1
                    for lvl, th in enumerate(thresholds):
                        if abs(f1 - f2) < th:
                            TAB_VIOL[lvl] += 1
                            if lvl > kmax:
                                kmax = lvl

            # 5) CEM constraints, different‐polarization: CD <i> <j> t0 t1 … t10
            elif rec == 'CD':
                i1, i2        = int(parts[1]), int(parts[2])
                thresholds    = list(map(int, parts[3:]))
                t1, t2        = trajets[i1], trajets[i2]
                f1, f2        = t1['freq'], t2['freq']
                p1, p2        = t1['pol'],  t2['pol']

                # only check if polarizations differ
                if p1 != p2:
                    THETA += 1
                    for lvl, th in enumerate(thresholds):
                        if abs(f1 - f2) < th:
                            TAB_VIOL[lvl] += 1
                            if lvl > kmax:
                                kmax = lvl

            # any other record types are ignored

    # --- compute total violations ---
    total_cem_viol = sum(TAB_VIOL)
    total_violations = VIOL_CI + total_cem_viol

    return total_violations
