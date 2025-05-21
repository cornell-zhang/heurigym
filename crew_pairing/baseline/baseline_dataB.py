"""
baseline that derive near-optimal solution, which improves on greedy solution by genetic algorithm

"""
import argparse
import random
import time
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Constants
BASE_STATION = "NKX"
POSITIONING_FEE = 10_000
REST_THRESHOLD_HOURS = 9.0

class Leg:
    __slots__ = (
        "idx",
        "token",
        "dptr_dt",
        "arrv_dt",
        "dptr_stn",
        "arrv_stn",
        "block",
        "duty_cost",
        "pair_cost",
    )

    def __init__(self, row, idx):
        self.idx = idx
        self.token = f"{row.FltNum}_{row.DptrDateTime.strftime('%Y-%m-%d')}"
        self.dptr_dt = row.DptrDateTime
        self.arrv_dt = row.ArrvDateTime
        self.dptr_stn = row.DptrStn
        self.arrv_stn = row.ArrvStn
        self.block = row.BlockHours
        self.duty_cost = row.DutyCostPerHour
        self.pair_cost = row.ParingCostPerHour

def original_greedy_solution(input_file, output_file=None):
    """Generate the original greedy solution and return both pairings and permutation"""
    # Read and preprocess the input data
    df = pd.read_csv(input_file)
    
    # Forward fill the cost columns
    df['DutyCostPerHour'] = df['DutyCostPerHour'].ffill()
    df['ParingCostPerHour'] = df['ParingCostPerHour'].ffill()
    
    # Convert dates and times to datetime objects
    df['DptrDateTime'] = pd.to_datetime(df['DptrDate'] + ' ' + df['DptrTime'], format='%m/%d/%Y %H:%M')
    df['ArrvDateTime'] = pd.to_datetime(df['ArrvDate'] + ' ' + df['ArrvTime'], format='%m/%d/%Y %H:%M')
    
    # Calculate block hours for each leg
    df['BlockHours'] = (df['ArrvDateTime'] - df['DptrDateTime']).dt.total_seconds() / 3600
    
    # Sort legs by departure time
    df = df.sort_values('DptrDateTime').reset_index(drop=True)
    
    # Generate unique leg tokens
    df['LegToken'] = df['FltNum'] + '_' + df['DptrDateTime'].dt.strftime('%Y-%m-%d')
    
    # Create a mapping from token to index
    token_to_idx = {df.loc[i, 'LegToken']: i for i in range(len(df))}
    
    # Initialize pairings
    pairings = []
    used_legs = set()
    
    # To track the order in which legs are used (for permutation)
    perm_order = []
    
    # Greedy algorithm: assign legs to pairings one by one in chronological order
    for i in range(len(df)):
        if df.loc[i, 'LegToken'] in used_legs:
            continue
        
        current_pairing = [df.loc[i, 'LegToken']]
        used_legs.add(df.loc[i, 'LegToken'])
        perm_order.append(i)  # Add to permutation order
        
        last_arrival = df.loc[i, 'ArrvDateTime']
        last_station = df.loc[i, 'ArrvStn']
        current_duty_start = df.loc[i, 'DptrDateTime']
        current_duty_legs = 1
        current_duty_block_hours = df.loc[i, 'BlockHours']
        current_duty_hours = (df.loc[i, 'ArrvDateTime'] - current_duty_start).total_seconds() / 3600
        
        # Try to extend the pairing with subsequent legs
        for j in range(i + 1, len(df)):
            if df.loc[j, 'LegToken'] in used_legs:
                continue
                
            # Check chronological order
            if df.loc[j, 'DptrDateTime'] < last_arrival:
                continue  # Skip if not chronological
                
            time_gap = (df.loc[j, 'DptrDateTime'] - last_arrival).total_seconds() / 3600
            
            # Check if can be added to current duty
            new_duty_hours = (df.loc[j, 'ArrvDateTime'] - current_duty_start).total_seconds() / 3600
            new_duty_block_hours = current_duty_block_hours + df.loc[j, 'BlockHours']
            
            if (time_gap <= 1.0 and  # Short connection within same duty
                current_duty_legs < 6 and
                new_duty_hours <= 14 and
                new_duty_block_hours <= 10):
                
                current_pairing.append(df.loc[j, 'LegToken'])
                used_legs.add(df.loc[j, 'LegToken'])
                perm_order.append(j)  # Add to permutation order
                
                last_arrival = df.loc[j, 'ArrvDateTime']
                last_station = df.loc[j, 'ArrvStn']
                current_duty_legs += 1
                current_duty_block_hours = new_duty_block_hours
                current_duty_hours = new_duty_hours
            
            # Check if can be added after minimum rest (new duty)
            elif time_gap >= REST_THRESHOLD_HOURS:
                current_pairing.append(df.loc[j, 'LegToken'])
                used_legs.add(df.loc[j, 'LegToken'])
                perm_order.append(j)  # Add to permutation order
                
                last_arrival = df.loc[j, 'ArrvDateTime']
                last_station = df.loc[j, 'ArrvStn']
                current_duty_start = df.loc[j, 'DptrDateTime']
                current_duty_legs = 1
                current_duty_block_hours = df.loc[j, 'BlockHours']
                current_duty_hours = (df.loc[j, 'ArrvDateTime'] - current_duty_start).total_seconds() / 3600
        
        pairings.append(current_pairing)
    
    # Add any remaining legs to the permutation that weren't used
    for i in range(len(df)):
        if i not in perm_order:
            perm_order.append(i)
    
    # Write the solution to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            for pairing in pairings:
                f.write(' '.join(pairing) + '\n')
    
    # Return both the pairings and the permutation
    return pairings, perm_order

def build_pairings(legs, perm):
    """Return list[list[Leg]] built greedily in *perm* order."""
    used = set()
    pairings = []

    for idx in perm:
        if idx in used:
            continue
        leg = legs[idx]
        current_pairing = [leg]
        used.add(idx)

        # duty state trackers
        duty_start = leg.dptr_dt
        duty_block = leg.block
        duty_legs = 1
        last_arrival = leg.arrv_dt

        # attempt to extend with following legs in the *same* chromosome order
        for jdx in perm:
            if jdx in used:
                continue
            next_leg = legs[jdx]
            if next_leg.dptr_dt < last_arrival:
                continue  # not chronological
            gap = (next_leg.dptr_dt - last_arrival).total_seconds() / 3600

            # same duty?
            same_duty = (
                gap <= 1
                and duty_legs < 6
                and (next_leg.arrv_dt - duty_start).total_seconds() / 3600 <= 14
                and duty_block + next_leg.block <= 10
            )
            if same_duty:
                current_pairing.append(next_leg)
                used.add(jdx)
                duty_legs += 1
                duty_block += next_leg.block
                last_arrival = next_leg.arrv_dt
                continue

            # new duty if ≥ REST_THRESHOLD_HOURS rest
            if gap >= REST_THRESHOLD_HOURS:
                current_pairing.append(next_leg)
                used.add(jdx)
                duty_start = next_leg.dptr_dt
                duty_block = next_leg.block
                duty_legs = 1
                last_arrival = next_leg.arrv_dt
        pairings.append(current_pairing)
    return pairings

def compute_cost(pairings):
    """Compute the cost exactly as in evaluator.py"""
    total_cost = 0.0
    
    for pairing in pairings:
        # Sort legs chronologically (to match evaluator logic)
        leg_objs = sorted(pairing, key=lambda l: l.dptr_dt)
        
        # --- block hours: sum of airborne time ---
        block_hours = sum(leg.block for leg in leg_objs)
        
        # --- duty hours: partition by rest >= REST_THRESHOLD_HOURS ---
        duty_hours = 0.0
        duty_start = leg_objs[0].dptr_dt
        prev_arr = leg_objs[0].arrv_dt
        
        for leg in leg_objs[1:]:
            rest = (leg.dptr_dt - prev_arr).total_seconds() / 3600
            if rest >= REST_THRESHOLD_HOURS:
                # close previous duty segment
                duty_hours += (prev_arr - duty_start).total_seconds() / 3600
                duty_start = leg.dptr_dt
            prev_arr = leg.arrv_dt
        
        # close final duty
        duty_hours += (prev_arr - duty_start).total_seconds() / 3600
        
        # Positioning fee if pairing doesn't start at base
        pos_fee = POSITIONING_FEE if leg_objs[0].dptr_stn != BASE_STATION else 0.0
        
        # Add costs using the last leg's cost rates (should be the same for all legs in a pairing)
        total_cost += duty_hours * leg_objs[-1].duty_cost + block_hours * leg_objs[-1].pair_cost + pos_fee
    
    return total_cost

def convert_pairings_to_leg_objects(pairings, df):
    """Convert pairings of tokens to pairings of Leg objects"""
    # Create mapping from token to row
    token_to_row = {}
    for i in range(len(df)):
        token = df.loc[i, 'FltNum'] + '_' + df.loc[i, 'DptrDateTime'].strftime('%Y-%m-%d')
        token_to_row[token] = i
    
    # Convert each pairing
    leg_pairings = []
    for pairing in pairings:
        leg_pairing = []
        for token in pairing:
            idx = token_to_row[token]
            leg = Leg(df.iloc[idx], idx)
            leg_pairing.append(leg)
        leg_pairings.append(leg_pairing)
    
    return leg_pairings

# ENHANCED GA WITH 5 ISLANDS
def enhanced_ga_search(legs, greedy_perm, gens=500, pop_size=200, cx_pb=0.9, mut_pb=0.5, 
                      elite=5, tournament_size=5, seed=42, time_limit=None, num_islands=5,
                      base_name="default"):
    """
    Enhanced genetic algorithm with 5 islands
    """
    if seed is not None:
        random.seed(seed)
    else:
        random.seed(int(time.time()))
        
    n = len(legs)
    base_perm = list(range(n))
    
    # Start time for potential time-based termination
    start_time = time.time()
    
    # Create directory for saving iteration results with base name
    iteration_dir = Path(f"crew_pairing/baseline/iterations_{base_name}")
    iteration_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize iteration history
    iteration_history = {
        'generation': [],
        'best_cost': [],
        'avg_cost': [],
        'std_cost': [],
        'improvement': [],
        'best_permutation': []
    }
    
    # --- INITIALIZATION STRATEGIES ----------------------------------------------
    def new_random_ind():
        """Generate a random permutation"""
        p = base_perm.copy()
        random.shuffle(p)
        return p
    
    def new_base_biased_ind():
        """Generate a permutation biased toward legs starting from base"""
        legs_copy = sorted([(i, legs[i].dptr_stn == BASE_STATION) for i in range(n)])
        # Sort by whether leg starts from base (true first)
        legs_copy.sort(key=lambda x: (not x[1], random.random()))
        return [item[0] for item in legs_copy]
    
    def new_time_clustered_ind():
        """Generate a permutation clustering legs in time windows"""
        window_size = 24 * 3600  # 24-hour window in seconds
        
        # Sort legs by departure time
        time_sorted = sorted(range(n), key=lambda i: legs[i].dptr_dt.timestamp())
        
        # Create time windows
        windows = []
        current_window = []
        base_time = legs[time_sorted[0]].dptr_dt.timestamp()
        
        for idx in time_sorted:
            leg_time = legs[idx].dptr_dt.timestamp()
            if leg_time - base_time <= window_size:
                current_window.append(idx)
            else:
                windows.append(current_window)
                current_window = [idx]
                base_time = leg_time
        
        if current_window:
            windows.append(current_window)
        
        # Shuffle within each window
        for w in windows:
            random.shuffle(w)
        
        # Flatten the result
        return [idx for window in windows for idx in window]
    
    def new_station_clustered_ind():
        """Generate a permutation that clusters legs by arrival/departure stations"""
        # Group legs by station pairs (departure -> arrival)
        station_pairs = {}
        for i in range(n):
            pair = (legs[i].dptr_stn, legs[i].arrv_stn)
            if pair not in station_pairs:
                station_pairs[pair] = []
            station_pairs[pair].append(i)
        
        # Shuffle within each station pair group
        for pair in station_pairs:
            random.shuffle(station_pairs[pair])
        
        # Flatten, prioritizing pairs starting from base
        result = []
        # First, add pairs starting from BASE
        for pair in station_pairs:
            if pair[0] == BASE_STATION:
                result.extend(station_pairs[pair])
        
        # Then add the rest
        for pair in station_pairs:
            if pair[0] != BASE_STATION:
                result.extend(station_pairs[pair])
        
        return result
    
    def new_cost_biased_ind():
        """Generate a permutation that prioritizes legs with lower costs"""
        # Sort legs by a combination of block hours and positioning costs
        leg_costs = []
        for i in range(n):
            # Higher cost for legs not starting from base
            pos_cost = 0 if legs[i].dptr_stn == BASE_STATION else 1
            # Block hours impact cost directly
            block_cost = legs[i].block
            # Combine factors with some randomness
            combined_cost = pos_cost * 10 + block_cost + random.random()
            leg_costs.append((i, combined_cost))
        
        # Sort by combined cost (lower is better)
        leg_costs.sort(key=lambda x: x[1])
        
        return [item[0] for item in leg_costs]
    
    # --- CROSSOVER OPERATORS ---------------------------------------------------
    def order_crossover(p1, p2):
        """Order Crossover (OX)"""
        if random.random() > cx_pb:
            return p1.copy(), p2.copy()
            
        a, b = sorted(random.sample(range(n), 2))
        
        # Create children with middle segment from parent
        child1 = [-1] * n
        child2 = [-1] * n
        
        # Copy middle segment
        for i in range(a, b):
            child1[i] = p1[i]
            child2[i] = p2[i]
        
        # Fill remaining positions
        fill1, fill2 = b, b
        for i in range(n):
            idx = (b + i) % n  # Start from position b
            
            # Fill child1
            if p2[idx] not in child1:
                child1[fill1 % n] = p2[idx]
                fill1 += 1
                
            # Fill child2
            if p1[idx] not in child2:
                child2[fill2 % n] = p1[idx]
                fill2 += 1
        
        return child1, child2
    
    def partially_mapped_crossover(p1, p2):
        """Partially Mapped Crossover (PMX)"""
        if random.random() > cx_pb:
            return p1.copy(), p2.copy()
            
        a, b = sorted(random.sample(range(n), 2))
        
        # Initialize the children with copies of the parents
        child1 = p1.copy()
        child2 = p2.copy()
        
        # Create mapping between segments
        for i in range(a, b):
            # Find positions of swapped values in the other parent
            val1, val2 = child1[i], child2[i]
            pos1 = child1.index(val2)
            pos2 = child2.index(val1)
            
            # Swap values
            child1[i], child1[pos1] = child1[pos1], child1[i]
            child2[i], child2[pos2] = child2[pos2], child2[i]
        
        return child1, child2
    
    def edge_recombination_crossover(p1, p2):
        """Edge Recombination Crossover (ERX)"""
        if random.random() > cx_pb:
            return p1.copy(), p2.copy()
            
        # Build adjacency list
        edges = {i: set() for i in range(n)}
        
        for parent in [p1, p2]:
            for i in range(n):
                # Add edges to neighbors (treating permutation as a cycle)
                prev_idx = (i - 1) % n
                next_idx = (i + 1) % n
                edges[parent[i]].add(parent[prev_idx])
                edges[parent[i]].add(parent[next_idx])
        
        # Choose starting point randomly from either parent
        current = random.choice([p1[0], p2[0]])
        child1 = [current]
        available = set(range(n))
        available.remove(current)
        
        # Build child by selecting edges
        while len(child1) < n:
            neighbors = edges[current] & available
            
            if neighbors:
                # Choose the neighbor with fewest remaining neighbors
                next_val = min(neighbors, key=lambda x: len(edges[x] & available))
            else:
                # If no neighbors, choose random available value
                next_val = random.choice(list(available))
            
            child1.append(next_val)
            available.remove(next_val)
            current = next_val
            
        # Create second child using the same process but starting from a different point
        current = random.choice([p1[-1], p2[-1]])
        child2 = [current]
        available = set(range(n))
        available.remove(current)
        
        while len(child2) < n:
            neighbors = edges[current] & available
            
            if neighbors:
                next_val = min(neighbors, key=lambda x: len(edges[x] & available))
            else:
                next_val = random.choice(list(available))
            
            child2.append(next_val)
            available.remove(next_val)
            current = next_val
            
        return child1, child2
    
    def cycle_crossover(p1, p2):
        """Cycle Crossover (CX)"""
        if random.random() > cx_pb:
            return p1.copy(), p2.copy()
            
        # Initialize offspring
        child1 = [-1] * n
        child2 = [-1] * n
        
        # Start with a random position
        i = random.randrange(n)
        cycle_start = i
        
        # Fill one cycle
        while True:
            child1[i] = p1[i]
            child2[i] = p2[i]
            
            i = p1.index(p2[i])
            if i == cycle_start:
                break
        
        # Fill remaining positions from the other parent
        for i in range(n):
            if child1[i] == -1:
                child1[i] = p2[i]
                child2[i] = p1[i]
        
        return child1, child2
    
    def position_based_crossover(p1, p2):
        """Position-Based Crossover"""
        if random.random() > cx_pb:
            return p1.copy(), p2.copy()
            
        # Select random positions to keep
        positions = random.sample(range(n), n // 3)
        
        # Initialize children
        child1 = [-1] * n
        child2 = [-1] * n
        
        # Copy selected positions
        for pos in positions:
            child1[pos] = p1[pos]
            child2[pos] = p2[pos]
        
        # Fill remaining positions
        remaining1 = [item for item in p2 if item not in child1]
        remaining2 = [item for item in p1 if item not in child2]
        
        for i in range(n):
            if child1[i] == -1:
                child1[i] = remaining1.pop(0)
            if child2[i] == -1:
                child2[i] = remaining2.pop(0)
        
        return child1, child2
    
    # --- MUTATION OPERATORS ----------------------------------------------------
    def swap_mutation(p):
        """Simple swap mutation"""
        if random.random() < mut_pb:
            i, j = random.sample(range(n), 2)
            p[i], p[j] = p[j], p[i]
        return p
    
    def insertion_mutation(p):
        """Remove a value and insert it at a new position"""
        if random.random() < mut_pb:
            i, j = random.sample(range(n), 2)
            if i < j:
                # Remove value at i and insert before j
                value = p[i]
                p = p[:i] + p[i+1:j] + [value] + p[j:]
            else:
                # Remove value at i and insert before j
                value = p[i]
                p = p[:j] + [value] + p[j:i] + p[i+1:]
        return p
    
    def scramble_mutation(p):
        """Scramble a subsequence"""
        if random.random() < mut_pb:
            i, j = sorted(random.sample(range(n), 2))
            segment = p[i:j+1]
            random.shuffle(segment)
            p = p[:i] + segment + p[j+1:]
        return p
    
    def inversion_mutation(p):
        """Invert a subsequence"""
        if random.random() < mut_pb:
            i, j = sorted(random.sample(range(n), 2))
            p[i:j+1] = reversed(p[i:j+1])
        return p
    
    def displacement_mutation(p):
        """Move a subsequence to a different position"""
        if random.random() < mut_pb:
            # Select a subsequence
            i, j = sorted(random.sample(range(n), 2))
            # Select a destination point (not in the subsequence)
            dest = random.choice([k for k in range(n+1) if k <= i or k > j])
            
            # Extract the subsequence
            subseq = p[i:j+1]
            
            # Remove it from the original
            p_new = p[:i] + p[j+1:]
            
            # Insert at destination
            if dest <= i:
                p = p_new[:dest] + subseq + p_new[dest:]
            else:
                dest -= (j - i + 1)  # Adjust for removal
                p = p_new[:dest] + subseq + p_new[dest:]
        return p
    
    # --- Initialize enhanced diverse population -------------------------------
    # Choose different initialization methods
    population = []
    
    # Add the greedy solution
    population.append(greedy_perm.copy())
    
    # Add a balanced mix of initialization strategies
    init_methods = [
        new_random_ind,
        new_base_biased_ind,
        new_time_clustered_ind,
        new_station_clustered_ind,
        new_cost_biased_ind
    ]
    
    individuals_per_method = (pop_size - 1) // len(init_methods)
    remaining = (pop_size - 1) % len(init_methods)
    
    for method in init_methods:
        count = individuals_per_method + (1 if remaining > 0 else 0)
        if remaining > 0:
            remaining -= 1
        population.extend([method() for _ in range(count)])
    
    # --- Evaluation function --------------------------------------------------
    def evaluate(pop):
        costs = []
        for perm in pop:
            pairings = build_pairings(legs, perm)
            costs.append(compute_cost(pairings))
        return costs
    
    # --- Island model with 5 islands -----------------------------------------
    # Divide population into 5 sub-populations (islands)
    island_size = pop_size // num_islands
    islands = []
    for i in range(num_islands):
        start_idx = i * island_size
        end_idx = start_idx + island_size if i < num_islands - 1 else pop_size
        islands.append(population[start_idx:end_idx])
    
    # Define crossover and mutation operators for each island
    island_cx_ops = [
        order_crossover,
        partially_mapped_crossover,
        edge_recombination_crossover,
        cycle_crossover,
        position_based_crossover
    ]
    
    island_mut_ops = [
        [swap_mutation, scramble_mutation],
        [insertion_mutation, inversion_mutation],
        [swap_mutation, inversion_mutation],
        [scramble_mutation, displacement_mutation],
        [insertion_mutation, displacement_mutation]
    ]
    
    # Evaluate the initial greedy solution
    greedy_pairings = build_pairings(legs, greedy_perm)
    greedy_cost = compute_cost(greedy_pairings)
    print(f"Initial greedy solution cost: {greedy_cost:,.0f}")
    
    best_cost = greedy_cost  # Start with the greedy solution as the best
    best_perm = greedy_perm.copy()
    no_improvement_count = 0  # Track generations without improvement
    
    # Save initial greedy solution
    initial_solution = {
        'generation': 0,
        'cost': greedy_cost,
        'permutation': greedy_perm,
        'improvement': 0.0
    }
    with open(iteration_dir / 'initial_solution.json', 'w') as f:
        json.dump(initial_solution, f)
    
    # --- Main GA loop with island model --------------------------------------
    header = f"{'Gen':>4} | {'Best Cost':>12} | {'Avg Cost':>12} | {'Improvement':>10} | {'New Best':>8} | {'Island Bests'}"
    print(header)
    print("-" * len(header))
    
    for g in range(1, gens + 1):
        # Check time limit if specified
        if time_limit and (time.time() - start_time > time_limit):
            print(f"Time limit of {time_limit} seconds reached after {g} generations.")
            break
            
        # Process each island
        all_costs = []
        island_best_costs = []
        generation_improved = False
        
        for island_idx, island in enumerate(islands):
            # Evaluate current island
            costs = evaluate(island)
            all_costs.extend(costs)
            
            # Get island's best cost
            min_cost = min(costs)
            island_best_costs.append(min_cost)
            
            # Check for new global best
            if min_cost < best_cost:
                best_cost = min_cost
                best_perm = island[costs.index(min_cost)].copy()
                generation_improved = True
                no_improvement_count = 0
                
                # Save new best solution
                solution = {
                    'generation': g,
                    'cost': best_cost,
                    'permutation': best_perm,
                    'improvement': (greedy_cost - best_cost) / greedy_cost * 100
                }
                with open(iteration_dir / f'best_solution_gen_{g:04d}.json', 'w') as f:
                    json.dump(solution, f)
            
            # Selection (tournament)
            def tournament_select():
                candidates = random.sample(list(enumerate(island)), tournament_size)
                c_costs = [costs[idx] for idx, _ in candidates]
                return candidates[int(np.argmin(c_costs))][1]
            
            # Get crossover and mutation operators for this island
            crossover_op = island_cx_ops[island_idx % len(island_cx_ops)]
            mutation_ops = island_mut_ops[island_idx % len(island_mut_ops)]
            
            # Create next generation with elitism
            ranked = sorted(zip(costs, island), key=lambda x: x[0])
            next_island = [ranked[i][1].copy() for i in range(elite)]
            
            # Fill the rest with offspring
            while len(next_island) < len(island):
                p1 = tournament_select()
                p2 = tournament_select()
                c1, c2 = crossover_op(p1, p2)
                
                # Apply random mutation from the island's set
                for c in [c1, c2]:
                    # Choose a random mutation operator
                    mut_op = random.choice(mutation_ops)
                    next_island.append(mut_op(c))
                    if len(next_island) >= len(island):
                        break
            
            # Update island
            islands[island_idx] = next_island
        
        # Update no improvement counter if needed
        if not generation_improved:
            no_improvement_count += 1
        
        # Calculate statistics
        avg_cost = np.mean(all_costs)
        std_cost = np.std(all_costs)
        improvement = (greedy_cost - best_cost) / greedy_cost * 100
        
        # Store history
        iteration_history['generation'].append(g)
        iteration_history['best_cost'].append(best_cost)
        iteration_history['avg_cost'].append(avg_cost)
        iteration_history['std_cost'].append(std_cost)
        iteration_history['improvement'].append(improvement)
        iteration_history['best_permutation'].append(best_perm)
        
        # Save iteration history
        with open(iteration_dir / 'iteration_history.json', 'w') as f:
            json.dump(iteration_history, f)
        
        # Print detailed report for each generation
        island_bests_str = ", ".join(f"{cost:,.0f}" for cost in island_best_costs)
        print(f"{g:4d} | {best_cost:12,.0f} | {avg_cost:12,.0f} | {improvement:10.2f}% | {'YES' if generation_improved else 'NO':8} | {island_bests_str}")
        
        # Migration between islands every 20 generations
        if g % 20 == 0:
            print(f"Gen {g}: Migration between islands")
            # For each island, select best individuals to migrate
            migrants = []
            for i, island in enumerate(islands):
                costs = evaluate(island)
                ranked = sorted(zip(costs, island), key=lambda x: x[0])
                migrants.append([ind.copy() for _, ind in ranked[:2]])  # 2 best from each island
            
            # Migrate (send to next island in a ring)
            for i in range(num_islands):
                dest = (i + 1) % num_islands
                # Replace worst individuals in destination with migrants
                dest_costs = evaluate(islands[dest])
                ranked = sorted(zip(range(len(islands[dest])), dest_costs), key=lambda x: -x[1])  # Sort by worst cost
                for j, migrant in enumerate(migrants[i]):
                    if j < len(ranked):
                        islands[dest][ranked[j][0]] = migrant.copy()
        
        # Improvement stagnation check - restart a portion of the population if no improvement
        if no_improvement_count >= 50:
            print(f"Gen {g}: Stagnation detected ({no_improvement_count} gens without improvement) - injecting diversity")
            # Randomly select 30% of each island to be reinitialized
            for i in range(num_islands):
                num_to_replace = len(islands[i]) // 3
                indices_to_replace = random.sample(range(elite, len(islands[i])), num_to_replace)
                for idx in indices_to_replace:
                    # Replace with a new diverse individual
                    strategy = random.choice(init_methods)
                    islands[i][idx] = strategy()
            no_improvement_count = 0  # Reset the counter after diversity injection
    
    # Flatten the islands for final evaluation
    population = [ind for island in islands for ind in island]
    
    # Get the best overall solution
    costs = evaluate(population)
    ranked = sorted(zip(costs, population), key=lambda x: x[0])
    if ranked[0][0] < best_cost:
        best_cost = ranked[0][0]
        best_perm = ranked[0][1].copy()
    
    # Compare final solution with initial greedy
    improvement = (greedy_cost - best_cost) / greedy_cost * 100
    print("\n" + "=" * 50)
    print(f"FINAL RESULTS")
    print("=" * 50)
    print(f"Initial greedy cost: {greedy_cost:,.0f}")
    print(f"Final GA cost:       {best_cost:,.0f}")
    print(f"Improvement:         {improvement:.2f}%")
    print(f"Total generations:   {g}")
    print(f"Number of islands:   {num_islands}")
    
    # Count how many pairings in the final solution
    final_pairings = build_pairings(legs, best_perm)
    print(f"Number of pairings:  {len(final_pairings)} (from original: {len(greedy_pairings)})")
    print("=" * 50)
    
    # Save final solution
    final_solution = {
        'generation': g,
        'cost': best_cost,
        'permutation': best_perm,
        'improvement': (greedy_cost - best_cost) / greedy_cost * 100,
        'total_generations': g,
        'total_time': time.time() - start_time
    }
    with open(iteration_dir / 'final_solution.json', 'w') as f:
        json.dump(final_solution, f)
    
    return best_perm, best_cost

def write_solution(file_path: Path, legs, perm):
    pairings = build_pairings(legs, perm)
    with open(file_path, "w") as f:
        for pairing in pairings:
            f.write(" ".join(l.token for l in pairing) + "\n")
    print(f"Wrote {len(pairings)} pairings → {file_path}")

def preprocess(csv_path: Path):
    df = pd.read_csv(csv_path)
    df["DutyCostPerHour"].ffill(inplace=True)
    df["ParingCostPerHour"].ffill(inplace=True)
    df["DptrDateTime"] = pd.to_datetime(df["DptrDate"] + " " + df["DptrTime"], format="%m/%d/%Y %H:%M")
    df["ArrvDateTime"] = pd.to_datetime(df["ArrvDate"] + " " + df["ArrvTime"], format="%m/%d/%Y %H:%M")
    df["BlockHours"] = (
        df["ArrvDateTime"] - df["DptrDateTime"]
    ).dt.total_seconds() / 3600
    df.sort_values("DptrDateTime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def main():
    parser = argparse.ArgumentParser(description="Enhanced GA solver for crew pairing with multiple improvements")
    parser.add_argument("input_csv")
    parser.add_argument("output_txt")
    parser.add_argument("--gens", type=int, default=200, help="# generations")
    parser.add_argument("--pop", type=int, default=200, help="population size")
    parser.add_argument("--time", type=int, default=None, help="time limit in seconds (optional)")
    parser.add_argument("--islands", type=int, default=3, help="number of islands")
    parser.add_argument("--validate", action="store_true", help="Validate using evaluator.py")
    args = parser.parse_args()

    input_file = args.input_csv
    
    # Extract base name from input file (remove extension and path)
    base_name = Path(input_file).stem
    
    # Step 1: Generate the exact same greedy solution as the original code
    print("Generating original greedy solution...")
    greedy_pairings, greedy_perm = original_greedy_solution(input_file, "greedy_solution.txt")
    
    # Step 2: Preprocess and create legs for the GA
    print("Preprocessing data...")
    df = preprocess(Path(input_file))
    legs = [Leg(row, i) for i, row in df.iterrows()]
    
    # Step 3: Check the cost of the greedy solution to verify it matches
    print("Verifying greedy solution cost...")
    leg_pairings = convert_pairings_to_leg_objects(greedy_pairings, df)
    greedy_cost = compute_cost(leg_pairings)
    print(f"Verified greedy solution cost: {greedy_cost:,.0f}")
    
    # Step 4: Run the enhanced GA with specified parameters
    print("\nStarting enhanced genetic algorithm with the following parameters:")
    print(f"  Generations:      {args.gens}")
    print(f"  Population size:  {args.pop}")
    print(f"  Crossover rate:   0.9")
    print(f"  Mutation rate:    0.5")
    print(f"  Elite count:      5")
    print(f"  Tournament size:  5")
    print(f"  Number of islands: {args.islands}")
    print(f"  Time limit:       {args.time if args.time else 'None'}")
    print(f"  Base name:        {base_name}")
    print("\n" + "=" * 50)
    
    best_perm, best_cost = enhanced_ga_search(
        legs, greedy_perm, 
        gens=args.gens, 
        pop_size=args.pop, 
        cx_pb=0.9,
        mut_pb=0.5,
        elite=5,
        tournament_size=5,
        time_limit=args.time,
        seed=42,
        num_islands=args.islands,
        base_name=base_name
    )
    
    print("\nWriting final solution...")
    write_solution(Path(args.output_txt), legs, best_perm)
    
    # Optionally validate with evaluator.py if installed
    if args.validate:
        try:
            from evaluator import evaluate
            print("\nValidating solution using evaluator.py...")
            eval_cost = evaluate(args.input_csv, args.output_txt)
            print(f"Cost from evaluator.py: {eval_cost:,.0f}")
            print(f"Difference: {abs(best_cost - eval_cost):,.2f}")
            if abs(best_cost - eval_cost) < 1.0:
                print("✓ Costs match! The solution is valid.")
            else:
                print("⚠ Warning: Costs don't match exactly. Check implementation.")
        except ImportError:
            print("\nevaluator.py not found, skipping validation")

if __name__ == "__main__":
    main()