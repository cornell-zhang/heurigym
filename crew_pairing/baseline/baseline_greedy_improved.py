import pandas as pd
import numpy as np
import random
import sys
import time
from datetime import datetime, timedelta
from typing import List, Dict, Set, Tuple

class Leg:
    def __init__(self, index, row):
        self.index = index
        self.flt_num = row['FltNum']
        self.dptr_stn = row['DptrStn']
        self.arrv_stn = row['ArrvStn']
        self.dptr_date_time = row['DptrDateTime']
        self.arrv_date_time = row['ArrvDateTime']
        self.block_hours = row['BlockHours']
        self.duty_cost_per_hour = row['DutyCostPerHour']
        self.paring_cost_per_hour = row['ParingCostPerHour']
        self.leg_token = row['LegToken']
    
    def __str__(self):
        return f"{self.leg_token}"

class Duty:
    def __init__(self, legs: List[Leg]):
        self.legs = legs
        self.start_time = legs[0].dptr_date_time
        self.end_time = legs[-1].arrv_date_time
        self.block_hours = sum(leg.block_hours for leg in legs)
        self.duty_hours = (self.end_time - self.start_time).total_seconds() / 3600
        self.cost = self.calculate_cost()
    
    def calculate_cost(self):
        # Use the duty cost per hour from the first leg
        return self.duty_hours * self.legs[0].duty_cost_per_hour
    
    def is_valid(self):
        # Check duty constraints
        if len(self.legs) > 6:  # Maximum 6 legs per duty
            return False
        if self.duty_hours > 14:  # Maximum 14 hours duty time
            return False
        if self.block_hours > 10:  # Maximum 10 block hours
            return False
        
        # Check connectivity between legs
        for i in range(len(self.legs) - 1):
            curr_leg = self.legs[i]
            next_leg = self.legs[i + 1]
            
            # Check stations match
            if curr_leg.arrv_stn != next_leg.dptr_stn:
                return False
            
            # Check time gap is valid (< 1 hour for connection)
            time_gap = (next_leg.dptr_date_time - curr_leg.arrv_date_time).total_seconds() / 3600
            if time_gap > 1.0:
                return False
        
        return True
    
    def __str__(self):
        return " → ".join([str(leg) for leg in self.legs])

class Pairing:
    def __init__(self, duties: List[Duty]):
        self.duties = duties
        self.legs = [leg for duty in duties for leg in duty.legs]
        self.start_time = duties[0].start_time if duties else None
        self.end_time = duties[-1].end_time if duties else None
        self.block_hours = sum(duty.block_hours for duty in duties)
        self.pairing_hours = (self.end_time - self.start_time).total_seconds() / 3600 if duties else 0
        self.cost = self.calculate_cost()
    
    def calculate_cost(self):
        if not self.duties:
            return 0
        
        # Duty costs + pairing overhead cost
        duty_costs = sum(duty.cost for duty in self.duties)
        pairing_cost = self.pairing_hours * self.duties[0].legs[0].paring_cost_per_hour
        
        return duty_costs + pairing_cost
    
    def is_valid(self):
        if not self.duties:
            return False
        
        # Each duty should be valid
        for duty in self.duties:
            if not duty.is_valid():
                return False
        
        # Check rest time between duties
        for i in range(len(self.duties) - 1):
            curr_duty = self.duties[i]
            next_duty = self.duties[i + 1]
            
            # Check rest time (minimum 9 hours)
            rest_time = (next_duty.start_time - curr_duty.end_time).total_seconds() / 3600
            if rest_time < 9:
                return False
            
            # Check station continuity
            if curr_duty.legs[-1].arrv_stn != next_duty.legs[0].dptr_stn:
                return False
        
        return True
    
    def get_leg_tokens(self):
        return [leg.leg_token for leg in self.legs]
    
    def __str__(self):
        return " | ".join([str(duty) for duty in self.duties])

class PairingProblem:
    def __init__(self, input_file: str):
        # Read and preprocess the input data
        self.df = pd.read_csv(input_file)
        
        # Forward fill the cost columns
        self.df['DutyCostPerHour'] = self.df['DutyCostPerHour'].ffill()
        self.df['ParingCostPerHour'] = self.df['ParingCostPerHour'].ffill()
        
        # Convert dates and times to datetime objects
        self.df['DptrDateTime'] = pd.to_datetime(self.df['DptrDate'] + ' ' + self.df['DptrTime'], format='%m/%d/%Y %H:%M')
        self.df['ArrvDateTime'] = pd.to_datetime(self.df['ArrvDate'] + ' ' + self.df['ArrvTime'], format='%m/%d/%Y %H:%M')
        
        # Calculate block hours for each leg
        self.df['BlockHours'] = (self.df['ArrvDateTime'] - self.df['DptrDateTime']).dt.total_seconds() / 3600
        
        # Sort legs by departure time
        self.df = self.df.sort_values('DptrDateTime').reset_index(drop=True)
        
        # Generate unique leg tokens
        self.df['LegToken'] = self.df['FltNum'] + '_' + self.df['DptrDateTime'].dt.strftime('%Y-%m-%d')
        
        # Create leg objects
        self.legs = [Leg(i, row) for i, row in self.df.iterrows()]
        
        # Create a lookup dictionary for legs by token
        self.leg_by_token = {leg.leg_token: leg for leg in self.legs}
        
        # Station connectivity graph
        self.station_legs = {}
        for leg in self.legs:
            if leg.dptr_stn not in self.station_legs:
                self.station_legs[leg.dptr_stn] = []
            self.station_legs[leg.dptr_stn].append(leg)
    
    def generate_initial_solution(self) -> List[Pairing]:
        """Generate an initial solution using the greedy approach from the baseline."""
        pairings = []
        used_legs = set()
        
        # Sort legs by departure time
        sorted_legs = sorted(self.legs, key=lambda x: x.dptr_date_time)
        
        for leg in sorted_legs:
            if leg.leg_token in used_legs:
                continue
            
            # Start a new duty with this leg
            current_duty_legs = [leg]
            used_legs.add(leg.leg_token)
            
            # Try to extend the duty with connected legs
            last_leg = leg
            for next_leg in sorted_legs:
                if next_leg.leg_token in used_legs:
                    continue
                    
                # Check if this leg can be added to the current duty
                if (next_leg.dptr_stn == last_leg.arrv_stn and 
                    next_leg.dptr_date_time > last_leg.arrv_date_time):
                    
                    time_gap = (next_leg.dptr_date_time - last_leg.arrv_date_time).total_seconds() / 3600
                    
                    if time_gap <= 1.0:  # Connection within same duty
                        # Create a temporary duty to check if it's valid
                        temp_duty = Duty(current_duty_legs + [next_leg])
                        if temp_duty.is_valid():
                            current_duty_legs.append(next_leg)
                            used_legs.add(next_leg.leg_token)
                            last_leg = next_leg
            
            # Create a duty from these legs
            duty = Duty(current_duty_legs)
            
            # Create a pairing with just this duty for now
            pairing = Pairing([duty])
            pairings.append(pairing)
        
        # Try to merge pairings with valid rest periods
        merged = True
        while merged:
            merged = False
            for i in range(len(pairings)):
                if i >= len(pairings):
                    break
                    
                for j in range(i + 1, len(pairings)):
                    if j >= len(pairings):
                        break
                        
                    p1, p2 = pairings[i], pairings[j]
                    
                    # Check if p2 can follow p1
                    if (p1.duties[-1].legs[-1].arrv_stn == p2.duties[0].legs[0].dptr_stn and
                        p2.duties[0].start_time > p1.duties[-1].end_time):
                        
                        rest_time = (p2.duties[0].start_time - p1.duties[-1].end_time).total_seconds() / 3600
                        
                        if rest_time >= 9:  # Minimum rest time
                            # Merge the two pairings
                            merged_pairing = Pairing(p1.duties + p2.duties)
                            if merged_pairing.is_valid():
                                pairings[i] = merged_pairing
                                pairings.pop(j)
                                merged = True
                                break
            
        return pairings
    
    def destroy_solution(self, solution: List[Pairing], destroy_ratio: float) -> Tuple[List[Pairing], Set[str]]:
        """Destroy part of the solution by removing some pairings."""
        num_to_destroy = max(1, int(len(solution) * destroy_ratio))
        
        # Select pairings to destroy (random, or can be more strategic)
        to_destroy = random.sample(range(len(solution)), num_to_destroy)
        
        # Remove selected pairings and collect their legs
        remaining_pairings = [p for i, p in enumerate(solution) if i not in to_destroy]
        destroyed_legs = set()
        
        for i in to_destroy:
            for leg in solution[i].legs:
                destroyed_legs.add(leg.leg_token)
        
        return remaining_pairings, destroyed_legs
    
    def repair_solution(self, partial_solution: List[Pairing], free_legs: Set[str]) -> List[Pairing]:
        """Repair the solution by reintegrating the free legs."""
        # Convert set of leg tokens to actual leg objects
        free_leg_objects = [self.leg_by_token[token] for token in free_legs]
        
        # Sort free legs by departure time
        free_leg_objects.sort(key=lambda x: x.dptr_date_time)
        
        # Used legs (those already in partial solution)
        used_legs = set()
        for pairing in partial_solution:
            for leg in pairing.legs:
                used_legs.add(leg.leg_token)
        
        # First try to add legs to existing pairings
        for leg in free_leg_objects[:]:
            if leg.leg_token in used_legs:
                continue
                
            added = False
            for p_idx, pairing in enumerate(partial_solution):
                # Try to add at the beginning of the pairing
                if (leg.arrv_stn == pairing.duties[0].legs[0].dptr_stn and
                    leg.arrv_date_time < pairing.duties[0].start_time):
                    
                    rest_time = (pairing.duties[0].start_time - leg.arrv_date_time).total_seconds() / 3600
                    
                    if rest_time >= 9:
                        new_duty = Duty([leg])
                        new_pairing = Pairing([new_duty] + pairing.duties)
                        if new_pairing.is_valid():
                            partial_solution[p_idx] = new_pairing
                            used_legs.add(leg.leg_token)
                            added = True
                            break
                
                # Try to add at the end of the pairing
                if (leg.dptr_stn == pairing.duties[-1].legs[-1].arrv_stn and
                    leg.dptr_date_time > pairing.duties[-1].end_time):
                    
                    rest_time = (leg.dptr_date_time - pairing.duties[-1].end_time).total_seconds() / 3600
                    
                    if rest_time >= 9:
                        new_duty = Duty([leg])
                        new_pairing = Pairing(pairing.duties + [new_duty])
                        if new_pairing.is_valid():
                            partial_solution[p_idx] = new_pairing
                            used_legs.add(leg.leg_token)
                            added = True
                            break
                
                # Try to add to an existing duty (start)
                if (leg.arrv_stn == pairing.duties[0].legs[0].dptr_stn and
                    leg.arrv_date_time < pairing.duties[0].start_time):
                    
                    time_gap = (pairing.duties[0].start_time - leg.arrv_date_time).total_seconds() / 3600
                    
                    if time_gap <= 1.0:
                        new_legs = [leg] + pairing.duties[0].legs
                        new_duty = Duty(new_legs)
                        if new_duty.is_valid():
                            new_duties = [new_duty] + pairing.duties[1:]
                            new_pairing = Pairing(new_duties)
                            if new_pairing.is_valid():
                                partial_solution[p_idx] = new_pairing
                                used_legs.add(leg.leg_token)
                                added = True
                                break
                
                # Try to add to an existing duty (end)
                for d_idx, duty in enumerate(pairing.duties):
                    if (leg.dptr_stn == duty.legs[-1].arrv_stn and
                        leg.dptr_date_time > duty.legs[-1].arrv_date_time):
                        
                        time_gap = (leg.dptr_date_time - duty.legs[-1].arrv_date_time).total_seconds() / 3600
                        
                        if time_gap <= 1.0:
                            new_legs = duty.legs + [leg]
                            new_duty = Duty(new_legs)
                            if new_duty.is_valid():
                                new_duties = pairing.duties[:d_idx] + [new_duty] + pairing.duties[d_idx+1:]
                                new_pairing = Pairing(new_duties)
                                if new_pairing.is_valid():
                                    partial_solution[p_idx] = new_pairing
                                    used_legs.add(leg.leg_token)
                                    added = True
                                    break
                    
                    if added:
                        break
                
                if added:
                    break
        
        # Create new pairings for remaining free legs
        new_pairings = []
        remaining_legs = [leg for leg in free_leg_objects if leg.leg_token not in used_legs]
        
        # Group remaining legs into duties and pairings
        while remaining_legs:
            leg = remaining_legs[0]
            remaining_legs = remaining_legs[1:]
            
            # Start a new duty with this leg
            current_duty_legs = [leg]
            used_legs.add(leg.leg_token)
            
            # Try to extend the duty with connected legs
            last_leg = leg
            extended = True
            while extended:
                extended = False
                for i, next_leg in enumerate(remaining_legs):
                    # Check if this leg can be added to the current duty
                    if (next_leg.dptr_stn == last_leg.arrv_stn and 
                        next_leg.dptr_date_time > last_leg.arrv_date_time):
                        
                        time_gap = (next_leg.dptr_date_time - last_leg.arrv_date_time).total_seconds() / 3600
                        
                        if time_gap <= 1.0:  # Connection within same duty
                            # Create a temporary duty to check if it's valid
                            temp_duty = Duty(current_duty_legs + [next_leg])
                            if temp_duty.is_valid():
                                current_duty_legs.append(next_leg)
                                last_leg = next_leg
                                used_legs.add(next_leg.leg_token)
                                remaining_legs.pop(i)
                                extended = True
                                break
            
            # Create a duty from these legs
            duty = Duty(current_duty_legs)
            
            # Create a pairing with just this duty for now
            pairing = Pairing([duty])
            new_pairings.append(pairing)
        
        # Try to merge newly created pairings
        merged = True
        while merged:
            merged = False
            for i in range(len(new_pairings)):
                if i >= len(new_pairings):
                    break
                    
                for j in range(i + 1, len(new_pairings)):
                    if j >= len(new_pairings):
                        break
                        
                    p1, p2 = new_pairings[i], new_pairings[j]
                    
                    # Check if p2 can follow p1
                    if (p1.duties[-1].legs[-1].arrv_stn == p2.duties[0].legs[0].dptr_stn and
                        p2.duties[0].start_time > p1.duties[-1].end_time):
                        
                        rest_time = (p2.duties[0].start_time - p1.duties[-1].end_time).total_seconds() / 3600
                        
                        if rest_time >= 9:  # Minimum rest time
                            # Merge the two pairings
                            merged_pairing = Pairing(p1.duties + p2.duties)
                            if merged_pairing.is_valid():
                                new_pairings[i] = merged_pairing
                                new_pairings.pop(j)
                                merged = True
                                break
        
        # Combine partial solution with new pairings
        return partial_solution + new_pairings
    
    def local_search(self, solution: List[Pairing], max_iterations: int) -> List[Pairing]:
        """Apply local search to improve the solution."""
        best_solution = solution
        best_cost = sum(p.cost for p in solution)
        
        for _ in range(max_iterations):
            # Randomly select a leg to move
            source_pairing_idx = random.randint(0, len(solution) - 1)
            source_pairing = solution[source_pairing_idx]
            
            if not source_pairing.legs:
                continue
                
            leg_idx = random.randint(0, len(source_pairing.legs) - 1)
            leg = source_pairing.legs[leg_idx]
            
            # Try to remove this leg from its current pairing
            # First, identify which duty contains this leg
            duty_idx = None
            leg_in_duty_idx = None
            for i, duty in enumerate(source_pairing.duties):
                try:
                    j = duty.legs.index(leg)
                    duty_idx = i
                    leg_in_duty_idx = j
                    break
                except ValueError:
                    continue
            
            if duty_idx is None:
                continue  # Leg not found in any duty
            
            # Check if we can remove this leg without breaking connectivity
            can_remove = False
            
            # If it's the only leg in the duty
            if len(source_pairing.duties[duty_idx].legs) == 1:
                # We can remove the entire duty
                new_duties = source_pairing.duties[:duty_idx] + source_pairing.duties[duty_idx+1:]
                if not new_duties:
                    # If this was the only duty, we're removing the entire pairing
                    new_source_pairing = None
                else:
                    # Check if the pairing is still valid after removing the duty
                    new_source_pairing = Pairing(new_duties)
                    can_remove = new_source_pairing.is_valid()
            else:
                # It's part of a multi-leg duty
                # If it's the first or last leg, we can potentially remove it
                if leg_in_duty_idx == 0 or leg_in_duty_idx == len(source_pairing.duties[duty_idx].legs) - 1:
                    new_legs = (source_pairing.duties[duty_idx].legs[:leg_in_duty_idx] + 
                               source_pairing.duties[duty_idx].legs[leg_in_duty_idx+1:])
                    
                    new_duty = Duty(new_legs)
                    if new_duty.is_valid():
                        new_duties = (source_pairing.duties[:duty_idx] + 
                                     [new_duty] + 
                                     source_pairing.duties[duty_idx+1:])
                        
                        new_source_pairing = Pairing(new_duties)
                        can_remove = new_source_pairing.is_valid()
            
            if not can_remove:
                continue
            
            # Try to insert this leg into another pairing
            best_insert = None
            best_insert_cost = float('inf')
            
            for target_idx, target_pairing in enumerate(solution):
                if target_idx == source_pairing_idx:
                    continue
                
                # Try to add to beginning of first duty
                if (leg.arrv_stn == target_pairing.duties[0].legs[0].dptr_stn and
                    leg.arrv_date_time < target_pairing.duties[0].start_time):
                    
                    time_gap = (target_pairing.duties[0].start_time - leg.arrv_date_time).total_seconds() / 3600
                    
                    if time_gap <= 1.0:  # Can connect within same duty
                        new_legs = [leg] + target_pairing.duties[0].legs
                        new_duty = Duty(new_legs)
                        if new_duty.is_valid():
                            new_duties = [new_duty] + target_pairing.duties[1:]
                            new_target_pairing = Pairing(new_duties)
                            if new_target_pairing.is_valid():
                                total_cost = (new_target_pairing.cost + 
                                             (new_source_pairing.cost if new_source_pairing else 0))
                                
                                if total_cost < best_insert_cost:
                                    best_insert = (target_idx, new_target_pairing, new_source_pairing)
                                    best_insert_cost = total_cost
                    
                    elif time_gap >= 9:  # Can add as new duty
                        new_duty = Duty([leg])
                        new_duties = [new_duty] + target_pairing.duties
                        new_target_pairing = Pairing(new_duties)
                        if new_target_pairing.is_valid():
                            total_cost = (new_target_pairing.cost + 
                                         (new_source_pairing.cost if new_source_pairing else 0))
                            
                            if total_cost < best_insert_cost:
                                best_insert = (target_idx, new_target_pairing, new_source_pairing)
                                best_insert_cost = total_cost
                
                # Try to add to end of last duty
                if (leg.dptr_stn == target_pairing.duties[-1].legs[-1].arrv_stn and
                    leg.dptr_date_time > target_pairing.duties[-1].end_time):
                    
                    time_gap = (leg.dptr_date_time - target_pairing.duties[-1].end_time).total_seconds() / 3600
                    
                    if time_gap <= 1.0:  # Can connect within same duty
                        new_legs = target_pairing.duties[-1].legs + [leg]
                        new_duty = Duty(new_legs)
                        if new_duty.is_valid():
                            new_duties = target_pairing.duties[:-1] + [new_duty]
                            new_target_pairing = Pairing(new_duties)
                            if new_target_pairing.is_valid():
                                total_cost = (new_target_pairing.cost + 
                                             (new_source_pairing.cost if new_source_pairing else 0))
                                
                                if total_cost < best_insert_cost:
                                    best_insert = (target_idx, new_target_pairing, new_source_pairing)
                                    best_insert_cost = total_cost
                    
                    elif time_gap >= 9:  # Can add as new duty
                        new_duty = Duty([leg])
                        new_duties = target_pairing.duties + [new_duty]
                        new_target_pairing = Pairing(new_duties)
                        if new_target_pairing.is_valid():
                            total_cost = (new_target_pairing.cost + 
                                         (new_source_pairing.cost if new_source_pairing else 0))
                            
                            if total_cost < best_insert_cost:
                                best_insert = (target_idx, new_target_pairing, new_source_pairing)
                                best_insert_cost = total_cost
                
                # Try to insert between existing duties
                for d_idx in range(len(target_pairing.duties) - 1):
                    if (leg.dptr_stn == target_pairing.duties[d_idx].legs[-1].arrv_stn and
                        leg.arrv_stn == target_pairing.duties[d_idx+1].legs[0].dptr_stn and
                        leg.dptr_date_time > target_pairing.duties[d_idx].end_time and
                        leg.arrv_date_time < target_pairing.duties[d_idx+1].start_time):
                        
                        rest_time1 = (leg.dptr_date_time - target_pairing.duties[d_idx].end_time).total_seconds() / 3600
                        rest_time2 = (target_pairing.duties[d_idx+1].start_time - leg.arrv_date_time).total_seconds() / 3600
                        
                        if rest_time1 >= 9 and rest_time2 >= 9:  # Valid rest times
                            new_duty = Duty([leg])
                            new_duties = (target_pairing.duties[:d_idx+1] + 
                                         [new_duty] + 
                                         target_pairing.duties[d_idx+1:])
                            
                            new_target_pairing = Pairing(new_duties)
                            if new_target_pairing.is_valid():
                                total_cost = (new_target_pairing.cost + 
                                             (new_source_pairing.cost if new_source_pairing else 0))
                                
                                if total_cost < best_insert_cost:
                                    best_insert = (target_idx, new_target_pairing, new_source_pairing)
                                    best_insert_cost = total_cost
            
            # Apply the best move if it improves the solution
            current_cost = (solution[source_pairing_idx].cost + 
                           solution[best_insert[0]].cost if best_insert else float('inf'))
            
            if best_insert and best_insert_cost < current_cost:
                target_idx, new_target_pairing, new_source_pairing = best_insert
                
                # Update target pairing
                solution[target_idx] = new_target_pairing
                
                # Update or remove source pairing
                if new_source_pairing:
                    solution[source_pairing_idx] = new_source_pairing
                else:
                    solution.pop(source_pairing_idx)
                
                # Update best solution if improved
                new_total_cost = sum(p.cost for p in solution)
                if new_total_cost < best_cost:
                    best_solution = solution.copy()
                    best_cost = new_total_cost
        
        return best_solution
    
    def solve_lns(self, max_iterations: int, destroy_ratio: float = 0.3, local_search_iterations: int = 10) -> List[Pairing]:
        """Solve using Large Neighborhood Search."""
        # Generate initial solution
        print("Generating initial solution...")
        solution = self.generate_initial_solution()
        best_solution = solution
        best_cost = sum(p.cost for p in solution)
        
        print(f"Initial solution: {len(solution)} pairings, total cost: {best_cost:.2f}")
        
        # LNS iterations
        for i in range(max_iterations):
            # Destroy part of the solution
            partial_solution, free_legs = self.destroy_solution(solution, destroy_ratio)
            
            # Repair the solution
            new_solution = self.repair_solution(partial_solution, free_legs)
            
            # Apply local search to improve
            new_solution = self.local_search(new_solution, local_search_iterations)
            
            # Calculate new cost
            new_cost = sum(p.cost for p in new_solution)
            
            # Accept if better
            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost
                solution = new_solution
                print(f"Iteration {i+1}: Found better solution with {len(best_solution)} pairings, cost: {best_cost:.2f}")
            else:
                # Simple acceptance criterion (can be replaced with simulated annealing)
                if random.random() < 0.1:  # Accept worse solution with 10% probability
                    solution = new_solution
                    print(f"Iteration {i+1}: Accepted worse solution with {len(solution)} pairings, cost: {new_cost:.2f}")
                else:
                    print(f"Iteration {i+1}: Keeping current solution with {len(solution)} pairings")
        
        print(f"Final solution: {len(best_solution)} pairings, total cost: {best_cost:.2f}")
        return best_solution

def solve(input_file: str, solution_file: str, max_iterations: int = 50):
    """Solve the pairing problem and write solution to file."""
    start_time = time.time()
    
    # Initialize problem
    problem = PairingProblem(input_file)
    
    # Solve using LNS
    solution = problem.solve_lns(max_iterations)
    
    # Calculate statistics
    total_legs = len(problem.legs)
    covered_legs = set()
    for pairing in solution:
        for leg in pairing.legs:
            covered_legs.add(leg.leg_token)
    
    coverage = len(covered_legs) / total_legs * 100
    total_cost = sum(p.cost for p in solution)
    
    print(f"Solution found in {time.time() - start_time:.2f} seconds")
    print(f"Covered {len(covered_legs)}/{total_legs} legs ({coverage:.2f}%)")
    print(f"Total cost: {total_cost:.2f}")
    
    # Write solution to file
    with open(solution_file, 'w') as f:
        for pairing in solution:
            f.write(' '.join(pairing.get_leg_tokens()) + '\n')

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python baseline_greedy_improved.py <input-csv> <output-txt>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    solution_file = sys.argv[2]
    #max_iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    max_iterations = 50
    
    solve(input_file, solution_file, max_iterations)