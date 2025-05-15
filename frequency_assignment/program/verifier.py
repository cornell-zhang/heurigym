from typing import Dict, List


def verify(input_file: str, output_file: str) -> Tuple[bool, str]:
    """
    Verifier function: checks whether a proposed solution is feasible.
    It ensures:
      - Each path’s assigned frequency and polarization lie within its declared domains.
      - All mandatory (Class 1) constraints are satisfied.

    Args:
        input_file:    Path to the input file describing domains and constraints
        solution_file: Path to the solver’s output file with AL lines

    Returns:
        bool: True if the solution is feasible, False otherwise.
    """
    # --- parse solution: AL <path_id> <freq> <pol> ---
    trajets: Dict[int, Dict[str, int]] = {}
    with open(output_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4 or parts[0] != "AL":
                continue
            path_id = int(parts[1])
            freq    = int(parts[2])
            pol     = int(parts[3])
            trajets[path_id] = {"freq": freq, "pol": pol}

    # if no assignments found -> infeasible
    if not trajets:
        return False, "no AL lines found in solution file"

    # --- prepare to read input ---
    domains: Dict[int, List[int]] = {}   # dom_index -> allowed frequencies
    # will fill trajets[path]["dom_freq"] and ["dom_pol"]
    # mandatory constraints list
    ci_constraints = []  # each is (i1, i2, X1, X2, val)

    # --- parse input file ---
    with open(input_file) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            rec = parts[0]

            if rec == "DM":
                dom = int(parts[1]); fr = int(parts[2])
                domains.setdefault(dom, []).append(fr)

            elif rec == "TR":
                path    = int(parts[1])
                dom_freq= int(parts[2])
                dom_pol = int(parts[3])
                if path not in trajets:
                    # solution must assign every path declared in TR
                    return False, "path {} not found in solution".format(path)
                trajets[path]["dom_freq"] = dom_freq
                trajets[path]["dom_pol"]  = dom_pol

            elif rec == "CI":
                # mandatory constraint
                i1, i2 = int(parts[1]), int(parts[2])
                X1, X2 = parts[3], parts[4]
                val     = int(parts[5])
                ci_constraints.append((i1, i2, X1, X2, val))

            # ignore CE / CD for feasibility

    # --- check domain membership ---
    for path_id, info in trajets.items():
        dom_f = info.get("dom_freq")
        dom_p = info.get("dom_pol")
        f, p  = info["freq"], info["pol"]
        if dom_f not in domains:
            return False, "domain {} not found in input".format(dom_f)
        if f not in domains[dom_f]:
            return False, "frequency {} not in domain {}".format(f, dom_f)
        # polarization domain check
        if not (dom_p == 0 or p == dom_p):
            # dom_p == 0 means both {–1,1} allowed
            return False, "polarization {} not in domain {}".format(p, dom_p)

    # --- check all mandatory constraints ---
    for i1, i2, X1, X2, val in ci_constraints:
        if i1 not in trajets or i2 not in trajets:
            return False, "path {} or {} not found in solution".format(i1, i2)
        t1, t2 = trajets[i1], trajets[i2]
        f1, f2 = t1["freq"], t2["freq"]
        p1, p2 = t1["pol"],  t2["pol"]

        if X1 == "F":  # frequency constraint
            diff = abs(f1 - f2)
            if X2 == "E":  # equality
                if diff != val:
                    return False, "frequency difference {} != {}".format(diff, val)
            else:          # inequality
                if diff == val:
                    return False, "frequency difference {} == {}".format(diff, val)
        else:          # polarization constraint
            if X2 == "E":  # same
                if p1 != p2:
                    return False, "polarization {} != {}".format(p1, p2)
            else:          # opposite
                if p1 == p2:
                    return False, "polarization {} == {}".format(p1, p2)

    # all checks passed
    return True, "all constraints satisfied"
