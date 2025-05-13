import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
import random
from typing import List, Dict, Set, Tuple
import time
import math
from collections import defaultdict

def hours_between(t1: datetime, t2: datetime) -> float:
    """Calculate hours between two datetime objects"""
    return (t2 - t1).total_seconds() / 3600

def solve_with_multistart(input_file: str, solution_file: str, num_attempts: int = 5):
    """Solve with multiple attempts, varying parameters each time"""
    print(f"Running multi-start optimization with {num_attempts} attempts")
    
    best_pairings = None
    best_cost = float('inf')
    
    # Read data once for all attempts
    df = preprocess_data(input_file)
    
    # Create different parameter sets for diversity
    parameter_sets = []
    
    # Standard approach
    parameter_sets.append({
        'max_connection_hours': 1.0,  
        'prioritize_base': True,
        'prioritize_station': True,
        'max_pairing_length': None,
        'random_restart': False,
        'favor_long_pairings': True,
        'favor_base_return': True,
        'description': 'Standard approach'
    })
    
    # Longer connections
    parameter_sets.append({
        'max_connection_hours': 3.0,  
        'prioritize_base': True,
        'prioritize_station': True,
        'max_pairing_length': None,
        'random_restart': False,
        'favor_long_pairings': True,
        'favor_base_return': True,
        'description': 'Longer connections'
    })
    
    # Random restarts
    parameter_sets.append({
        'max_connection_hours': 2.0,  
        'prioritize_base': True,
        'prioritize_station': True,
        'max_pairing_length': None,
        'random_restart': True,
        'random_restart_prob': 0.15,
        'favor_long_pairings': True,
        'favor_base_return': True,
        'description': 'Random restarts'
    })
    
    # Shorter pairings
    parameter_sets.append({
        'max_connection_hours': 2.0,  
        'prioritize_base': True,
        'prioritize_station': True,
        'max_pairing_length': 20,
        'random_restart': False,
        'favor_long_pairings': False,
        'favor_base_return': True,
        'description': 'Shorter pairings'
    })
    
    # Balanced approach
    parameter_sets.append({
        'max_connection_hours': 2.0,  
        'prioritize_base': True,
        'prioritize_station': True,
        'max_pairing_length': 50,
        'random_restart': True,
        'random_restart_prob': 0.1,
        'favor_long_pairings': True,
        'favor_base_return': True,
        'favor_duty_utilization': True,
        'description': 'Balanced approach'
    })
    
    # Run additional randomized attempts if requested
    while len(parameter_sets) < num_attempts:
        parameter_sets.append({
            'max_connection_hours': random.uniform(0.5, 3.0),
            'prioritize_base': random.random() > 0.2,
            'prioritize_station': random.random() > 0.1,
            'max_pairing_length': random.choice([None, 30, 50, 70]),
            'random_restart': random.random() > 0.5,
            'random_restart_prob': random.uniform(0.05, 0.2),
            'favor_long_pairings': random.random() > 0.3,
            'favor_base_return': random.random() > 0.2,
            'favor_duty_utilization': random.random() > 0.5,
            'description': f'Random params {len(parameter_sets) - 4}'
        })
    
    for i, params in enumerate(parameter_sets):
        print(f"\n=== Attempt {i+1}/{num_attempts}: {params['description']} ===")
        print(f"Parameters: {', '.join([f'{k}={v}' for k, v in params.items() if k != 'description'])}")
        
        # Run the solver with these parameters
        start_time = time.time()
        pairings = solve_single_attempt(df, params)
        solve_time = time.time() - start_time
        
        # Basic stats
        total_legs = sum(len(p) for p in pairings)
        avg_pairing_length = total_legs / len(pairings) if pairings else 0
        
        print(f"Time taken: {solve_time:.2f}s")
        print(f"Created {len(pairings)} pairings with avg length {avg_pairing_length:.1f}")
        
        # Length distribution
        lengths = [len(p) for p in pairings]
        if lengths:
            print(f"Pairing lengths: min={min(lengths)}, max={max(lengths)}, median={sorted(lengths)[len(lengths)//2]}")
        
        # Write temporary solution file
        temp_file = f"temp_solution_{i}.txt"
        with open(temp_file, 'w') as f:
            for pairing in pairings:
                f.write(' '.join(pairing) + '\n')
        
        # Evaluate
        try:
            from evaluator import evaluate
            cost = evaluate(input_file, temp_file)
            print(f"Solution cost: ${cost:,.0f}")
            
            if cost < best_cost:
                best_cost = cost
                best_pairings = pairings
                print(f"*** NEW BEST SOLUTION FOUND! ***")
                
            # Clean up temp file
            Path(temp_file).unlink(missing_ok=True)
        except Exception as e:
            print(f"Evaluation error: {e}")
            # If no evaluator, use number of pairings as proxy
            cost = len(pairings)
            if best_pairings is None or cost < best_cost:
                best_cost = cost
                best_pairings = pairings
    
    # Write the best solution
    if best_pairings:
        with open(solution_file, 'w') as f:
            for pairing in best_pairings:
                f.write(' '.join(pairing) + '\n')
        
        print(f"\nBest solution written with {len(best_pairings)} pairings")
        print(f"Best cost: ${best_cost:,.0f}" if best_cost != len(best_pairings) else "")
    else:
        print("No valid solution found.")
        
    return best_pairings

def preprocess_data(input_file: str) -> pd.DataFrame:
    """Preprocess the input data"""
    print(f"Loading and preprocessing {input_file}...")
    
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
    
    # Add some useful derived columns
    df['DayOfWeek'] = df['DptrDateTime'].dt.dayofweek
    df['HourOfDay'] = df['DptrDateTime'].dt.hour
    
    # Add station transition frequencies
    station_transitions = defaultdict(int)
    for i in range(len(df) - 1):
        if df.iloc[i]['ArrvStn'] == df.iloc[i+1]['DptrStn']:
            station_transitions[df.iloc[i]['ArrvStn']] += 1
    
    # Normalize to get frequency
    total = sum(station_transitions.values()) or 1
    station_freq = {station: count/total for station, count in station_transitions.items()}
    
    # Add as a column (high value = common connection point)
    df['StationConnectivity'] = df['ArrvStn'].map(lambda x: station_freq.get(x, 0))
    
    print(f"Loaded {len(df)} legs spanning {(df['ArrvDateTime'].max() - df['DptrDateTime'].min()).days + 1} days")
    return df

def solve_single_attempt(df: pd.DataFrame, params: Dict) -> List[List[str]]:
    """Run a single solution attempt with the given parameters"""
    # Constants
    BASE = "NKX"  # Assuming NKX is the base
    MAX_DUTY_LEGS = 6
    MAX_DUTY_HOURS = 14
    MAX_BLOCK_HOURS = 10
    MIN_REST_HOURS = 9
    MAX_CONNECTION_HOURS = params.get('max_connection_hours', 3)
    MAX_PAIRING_LENGTH = params.get('max_pairing_length', None)
    
    # Create station index for faster lookup
    station_index = {}
    for i, row in df.iterrows():
        station_index.setdefault(row['DptrStn'], []).append(i)
    
    # Initialize pairings and tracking
    pairings = []
    used_legs = set()
    start_time = time.time()
    
    # Set random seed for reproducibility but different for each run
    seed = hash(str(params)) % 10000
    random.seed(seed)
    
    while len(used_legs) < len(df):
        # Get unassigned legs
        unassigned_indices = [i for i in range(len(df)) if df.loc[i, 'LegToken'] not in used_legs]
        
        if not unassigned_indices:
            break
        
        # Choose starting leg based on strategy
        start_idx = select_starting_leg(df, unassigned_indices, params, BASE)
        
        # Start a new pairing
        current_pairing = [df.loc[start_idx, 'LegToken']]
        used_legs.add(df.loc[start_idx, 'LegToken'])
        
        # Current state
        last_arrival = df.loc[start_idx, 'ArrvDateTime']
        last_station = df.loc[start_idx, 'ArrvStn']
        current_duty_start = df.loc[start_idx, 'DptrDateTime']
        current_duty_legs = 1
        current_duty_block_hours = df.loc[start_idx, 'BlockHours']
        
        # Track duties in this pairing
        duties_in_pairing = 1
        
        # Try to extend the pairing
        pairing_length = 1
        while True:
            if MAX_PAIRING_LENGTH and pairing_length >= MAX_PAIRING_LENGTH:
                break
                
            # Find next eligible legs
            next_leg_idx = find_next_leg(
                df, 
                unassigned_indices, 
                used_legs,
                last_arrival, 
                last_station, 
                current_duty_start,
                current_duty_legs, 
                current_duty_block_hours,
                MAX_DUTY_LEGS, 
                MAX_DUTY_HOURS, 
                MAX_BLOCK_HOURS,
                MIN_REST_HOURS, 
                MAX_CONNECTION_HOURS,
                BASE,
                prioritize_station=params.get('prioritize_station', True),
                favor_base_return=params.get('favor_base_return', True),
                favor_duty_utilization=params.get('favor_duty_utilization', False)
            )
            
            if next_leg_idx is None:
                break  # No more legs can be added
            
            # Add the leg to the pairing
            current_pairing.append(df.loc[next_leg_idx, 'LegToken'])
            used_legs.add(df.loc[next_leg_idx, 'LegToken'])
            pairing_length += 1
            
            # Check if this is a new duty
            time_gap = hours_between(last_arrival, df.loc[next_leg_idx, 'DptrDateTime'])
            
            if time_gap >= MIN_REST_HOURS:
                # Start a new duty
                current_duty_start = df.loc[next_leg_idx, 'DptrDateTime']
                current_duty_legs = 1
                current_duty_block_hours = df.loc[next_leg_idx, 'BlockHours']
                duties_in_pairing += 1
            else:
                # Continue current duty
                current_duty_legs += 1
                current_duty_block_hours += df.loc[next_leg_idx, 'BlockHours']
            
            # Update state
            last_arrival = df.loc[next_leg_idx, 'ArrvDateTime']
            last_station = df.loc[next_leg_idx, 'ArrvStn']
            
            # Update unassigned indices
            unassigned_indices = [i for i in unassigned_indices if df.loc[i, 'LegToken'] not in used_legs]
        
        # Add completed pairing to the list
        pairings.append(current_pairing)
        
        # Print progress - but not too often
        if len(pairings) % 10 == 0 or len(pairings) < 5:
            elapsed = time.time() - start_time
            print(f"Built pairing {len(pairings)} with {len(current_pairing)} legs, {duties_in_pairing} duties. " +
                  f"Total: {len(used_legs)}/{len(df)} legs ({elapsed:.1f}s)")
    
    return pairings

def select_starting_leg(df: pd.DataFrame, unassigned_indices: List[int], params: Dict, base: str) -> int:
    """Select a starting leg for a new pairing based on strategy"""
    # Random restart if enabled
    if (params.get('random_restart', False) and 
        random.random() < params.get('random_restart_prob', 0.1) and 
        len(unassigned_indices) > 1):
        start_idx = random.choice(unassigned_indices)
        return start_idx
    
    # Start from base if possible and prioritized
    if params.get('prioritize_base', True):
        base_indices = [i for i in unassigned_indices if df.loc[i, 'DptrStn'] == base]
        if base_indices:
            return min(base_indices, key=lambda i: df.loc[i, 'DptrDateTime'])
    
    # Otherwise take earliest unassigned leg
    return min(unassigned_indices, key=lambda i: df.loc[i, 'DptrDateTime'])

def find_next_leg(
    df: pd.DataFrame, 
    unassigned_indices: List[int],
    used_legs: Set[str],
    last_arrival: datetime, 
    last_station: str, 
    duty_start: datetime,
    duty_legs: int, 
    duty_block_hours: float,
    max_duty_legs: int, 
    max_duty_hours: float, 
    max_block_hours: float,
    min_rest_hours: float, 
    max_connection_hours: float,
    base_station: str,
    prioritize_station: bool = True,
    favor_base_return: bool = True,
    favor_duty_utilization: bool = False
) -> int:
    """Find the best next leg to add to a pairing with enhanced scoring"""
    
    # Track candidates with their scores
    candidates = []
    
    for j in unassigned_indices:
        if df.loc[j, 'LegToken'] in used_legs or df.loc[j, 'DptrDateTime'] <= last_arrival:
            continue
        
        time_gap = hours_between(last_arrival, df.loc[j, 'DptrDateTime'])
        same_station = df.loc[j, 'DptrStn'] == last_station
        
        # Base scoring factors
        score_factors = {
            'chronological': df.loc[j, 'DptrDateTime'].timestamp(),  # Lower is better
            'station_match': 0 if same_station else 10,  # 0 if same station (better)
            'connection_time': time_gap,                 # Lower is better
            'to_base': 0 if df.loc[j, 'ArrvStn'] == base_station else 5  # 0 if to base (better)
        }
        
        # Check for same duty extension
        if 0 <= time_gap <= max_connection_hours:
            new_duty_hours = hours_between(duty_start, df.loc[j, 'ArrvDateTime'])
            new_duty_block = duty_block_hours + df.loc[j, 'BlockHours']
            
            if (duty_legs < max_duty_legs and 
                new_duty_hours <= max_duty_hours and 
                new_duty_block <= max_block_hours):
                
                # Calculate duty utilization
                utilization = new_duty_block / new_duty_hours if new_duty_hours > 0 else 0
                score_factors['duty_utilization'] = -utilization * 5 if favor_duty_utilization else 0
                
                # Calculate final score (lower is better)
                score = calculate_score(score_factors, same_duty=True)
                
                candidates.append((j, score, 'same_duty'))
        
        # Check for new duty
        elif time_gap >= min_rest_hours:
            # Calculate rest period quality (closer to min_rest_hours is better)
            rest_quality = abs(time_gap - min_rest_hours)
            score_factors['rest_quality'] = rest_quality
            
            # Calculate final score
            score = calculate_score(score_factors, same_duty=False)
            
            candidates.append((j, score, 'new_duty'))
    
    # If we have candidates, choose the best one
    if candidates:
        # Sort by score (lower is better)
        candidates.sort(key=lambda x: x[1])
        
        # Return the best candidate index
        return candidates[0][0]
    
    # No eligible legs found
    return None

def calculate_score(factors: Dict[str, float], same_duty: bool) -> float:
    """Calculate a score for a leg based on various factors (lower is better)"""
    # Base score from timestamp to maintain chronological order
    score = factors.get('chronological', 0) / 1e9  # Normalize timestamp to a reasonable value
    
    # Station match bonus (big advantage for same station)
    score += factors.get('station_match', 0) * 10
    
    # Connection time factor
    score += factors.get('connection_time', 0) * 5
    
    # Base return bonus
    score += factors.get('to_base', 0) * 3
    
    # Duty utilization factor (negative because higher utilization is better)
    score += factors.get('duty_utilization', 0)
    
    # Rest quality for new duties
    if not same_duty:
        score += factors.get('rest_quality', 0) * 2
        score += 50  # Penalty for new duty vs extending current duty
    
    return score

# Main entry point
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python enhanced_simple_solver.py <input_csv> <output_txt>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    start_time = datetime.now()
    solve_with_multistart(input_file, output_file, num_attempts=5)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\nTotal runtime: {elapsed:.2f} seconds")
    
    # Final evaluation
    try:
        from evaluator import evaluate
        cost = evaluate(input_file, output_file)
        print(f"Final cost: ${cost:,.0f}")
    except Exception as e:
        print(f"Final evaluation error: {e}")
        print("Final evaluation not available.")