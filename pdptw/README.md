# Pick-up and Delivery Problem with Time Windows (PDPTW)

## Background
The Pick-up and Delivery Problem with Time Windows (PDPTW) is a well-known combinatorial optimization problem that extends the classic Capacitated Vehicle Routing Problem (CVRP). In PDPTW, a fleet of vehicles is tasked with transporting goods or people from a set of pick-up locations to corresponding delivery locations, while adhering to specific time windows for both pick-ups and deliveries. The objective is to minimize the total distance traveled by the vehicles while ensuring that all constraints are satisfied.

The problem is often seen in many real-world applications, such as logistics, transportation, and supply chain management. One example is the day-to-day operation of Amazon's last-mile delivery, where massive amount of packages are loaded to the trucks in a distribution center and delivered to various locations within a specified time frame. Meanwhile, to improve the efficiency a truck is usually allowed to pick up items (e.g., returned packages) along its route and carrt them back to the distribution center.

## Formalization
Even within the PDPTW domain, there are many variations of the problem regarding vehicle heterogeneity, soft (hard) time windows constraints, and more. A typical version is described as follows:

**Input**:
- A set of vehicles $V$ with a maximum capacity $Q \in \mathbb{R}_+$.
- A depot $o$ where all vehicles are stationed.
- A set of pick-up locations $N_p$ and delivery locations $N_d$ in the 2D Euclidean space. Each pick-up loc $i \in N_p$ corresponds to a unique delivery loc $j \in N_d$, and vice versus.
- Each pick-up $i \in N_p$ has a demand $d_i > 0$ consuming vehicle capacity, and its corresponding delivery $j \in N_d$ has a demand $d_j < 0$ releasing vehicle capacity.
- Each location $i \in N_p \cup N_d$ has a visit time window $[e_i, l_i]$, where $e_i$ is the earliest visit time and $l_i$ is the latest visit time.
- Each location $i \in N_p \cup N_d$ has a service time $\Delta_i \in \mathbb{R}_+$.

**Objective**:
- Find a set of vehicle routes to minimize the total travel time, while satisfying the following constraints:
  - Each vehicle starts and ends at the depot.
  - Each pick-up location must be visited before its corresponding delivery location.
  - The total demand on each vehicle's route must not exceed its capacity.
  - Each location must be visited within its time window (a vehicle can wait if it arrives earlier).

## Input Format
An instance is stored in a `.txt` file with the format shown below. It consists of some meta data (e.g., instance name, number of vehicles, capacity), followed by three sections: `NODE_COORD_SECTION`, `PICKUP_AND_DELIVERY_SECTION`, and `DEPOT_SECTION`. Every file ends with a line `EOF`. 

 1. `NODE_COORD_SECTION` contains the coordinates of each location.
 2. `PICKUP_AND_DELIVERY_SECTION` contains the pick-up and delivery information, with each line of the form desribed below.
    - The first number is the node index.
    - The second number is the demand.
    - The third and fourth numbers are the earliest and latest time for the node.
    - The fifth number specifies the service time for the node.
    - The sixth number is pickup sibling index if the node itself is a delivery node, and is 0 otherwise.
    - The seventh number is the delivery sibling index if the node itself is a pickup node, and is 0 otherwise.
3. `DEPOT_SECTION` specifies node indices for the departing and returning depot. 

The following file is an example for instance `lc101`:
```
NAME : lc101
TYPE : PDPTW
DIMENSION : 107
VEHICLES : 10
CAPACITY : 200
EDGE_WEIGHT_TYPE : EXACT_2D
NODE_COORD_SECTION
1 40 50
2 45 68
3 45 70
...

PICKUP_AND_DELIVERY_SECTION
1 0 0 1236 0 0 0
2 -10 912 967 90 12 0
3 -20 825 870 90 7 0
...

DEPOT_SECTION
1 
-1
EOF
```


## Output Format
The first line of the output should provide the instance name. From the second line onwards, each line should represent a vehicle's route starting and ending at the depot node (i.e., 1). The following output shows an optimal solution to instance `lc101`. 
```
lc101
1 33 34 32 36 38 39 40 37 106 35 1
1 82 79 105 77 72 71 74 78 80 81 1
1 14 18 19 20 16 17 15 13 1
1 6 4 8 9 11 12 10 7 5 3 2 76 1
1 91 88 87 84 83 85 86 89 90 92 1
1 68 66 64 63 75 73 62 65 103 69 67 70 1
1 99 97 96 95 93 94 98 107 101 100 1
1 58 56 55 54 57 59 61 60 1
1 44 43 42 41 45 47 46 49 52 102 51 53 50 48 1
1 21 25 26 28 30 31 29 27 24 104 23 22 1
```

## References
Li, Haibing, and Andrew Lim. "A metaheuristic for the pickup and delivery problem with time windows." Proceedings 13th IEEE international conference on tools with artificial intelligence. ICTAI 2001. IEEE, 2001.