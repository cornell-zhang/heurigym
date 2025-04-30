# Gloabl Routing

## Background

Global routing (GR) techniques, which establish coarse-grain routing paths for signal nets throughout a Very Large Scale Integration (VLSI) circuit, have a wide range of applications spanning various stages of the modern VLSI design flow. GR significantly impacts circuit timing, power consumption, and overall routability. As an NP-hard problem, GR has been addressed through various heuristics over the past decades, often trading off some degree of optimality for practical efficiency.

## Problem Description

In global routing, a 3D routing space is defined using global routing cells (GCells), created by a regular grid of horizontal and vertical lines. This configuration results in the formation of a grid graph  $\( \mathcal{G}(\mathcal{V}, \mathcal{E}) \)$ where each GCell is treated as a vertex $\( (v \in \mathcal{V}) \)$ and edges $\( (e \in \mathcal{E}) \)$ connect adjacent GCells within the same layer (GCell edges) or between GCells in neighboring layers (via edges). It's important to note that each layer has a preferred routing direction, which means GCell edges can be either horizontal or vertical. The essence of the global routing challenge is to establish concrete path for each net within the grid graph. This process ensures the interconnection of all pins without overflow while minimizing total wire length and the number of vias.

### Inputs
There are two input files for each circuit: a routing resource file (with a .cap extension) and a net information file (with a .net extension). The routing resource file offers a detailed representation of the GCell grid graph and the routing resources it encompasses. Meanwhile, the net information file shows the access points for all the pins within each net.

The routing resource file follows this format:

    # Dimensions of GCell graph 
    nLayers xSize ySize      
    # Weights of performance metrics  
    UnitLengthWireCost UnitViaCost OFWeight[0] OFWeight[1] OFWeight[2] ...   
    # Lengths of horizontal GCell edges (edge count = xSize - 1)  
    HorizontalGCellEdgeLengths[0] HorizontalGCellEdgeLengths[1] HorizontalGCellEdgeLengths[2] ...   
    # Lengths of vertical GCell edges (edge count = ySize - 1)  
    VerticalGCellEdgeLengths[0] VerticalGCellEdgeLengths[1] VerticalGCellEdgeLengths[2] ...   
    # Information for the 0-th layer  
    ## Layer name, preferred direction (Direction: 0 = horizontal, 1 = vertical) and minimum length of a wire at this metal layer (Not useful here)
    layerName layerDirection layerMinLength   
    ## Routing capacities of GCell edges at the 0-th layer    
    ### Capacities of GCell at [x(0), y(0)], [x(1), y(0)], ...  
    10 10 10 ...     
    ### Capacities of GCell at [x(0), y(1)], [x(1), y(1)], ...  
    10 10 10 ...    
    ...      
    ## Information for the 1-th layer 
    ...

The net information file follows this format:

        # Net name  
        Net0  
        (  
        # Access point locations (layer, x, y) for pin 0  
        [(location of access point 0), (location of access point 1), ...]      
        # Access point locations for pin 1  
        [(location of access point 0), (location of access point 1), ...]        
        ...  
        )       
        Net1  
        (  
        [(location of access point 0), (location of access point 1), ...]  
        [(location of access point 0), (location of access point 1), ...]                
        ... 
        )       
        ... 

### Output
The global routing solution is described in the GCell coordinate system. To enhance routability and ensure pin accessibility during the subsequent detailed routing process, we operate under the following assumptions:
1. Metal 1 (the 0-th layer) is not employed for net routing. To reach pins on Metal 1, vias must be utilized to establish connections from Metal 2.
2. Each pin must be connected through at least one of its access points.

Here is an example of a global routing solution for a net:

        # Net name       
        Net0
        (
        # $x_l$ $y_l$ $z_l$ $x_h$ $y_h$ $z_h$          
        0 0 0 0 0 1
        0 0 1 0 2 1
        0 2 1 0 2 2
        0 2 2 3 2 2
        3 2 1 3 2 2
        3 2 1 3 3 1
        3 3 0 3 3 1
        )

where each row ($x_l$ $y_l$ $z_l$ $x_h$ $y_h$ $z_h$) describes a line/rectangle in the 3D GCell graph, spanning from $(x_l, y_l, z_l)$ to $(x_h, y_h, z_h)$.


## Evaluation
### Evaluation Metrics
The global routing solution is measured by the weighted sum of the following metrics: total wire length, via utilization and routing overflow:

    Score = UnitLengthWireCost*TotalWL + UnitViaCost*ViaCount + OverflowScore

TotalWL and ViaCount denote the sum of the wire length for all signal nets and the total number of vias, respectively. UnitLengthWireCost and UnitViaCost are defined in the .cap file. In our evaluation, UnitLengthWireCost and UnitViaCost are set to be $0.00131579$ and $4$, respectively.

The OverflowScore is computed as the total overflow cost across all GCell edges. For a GCell edge at layer $l$ with routing capacity $c$ and demand $d$, the overflow cost is defined as:
$OverflowCost(c,d,l) = OFWeight[l] * (\exp^{0.5(d-c)})$. $OFWeight[l]$ is overflow weight for GCell edges at the $l$-th layer, which is defined in the .cap file.
The smaller the weighted score, the better.
### Benchmarks
We use the benchmark suite from the ISPD 2024 Global Routing Contest [1], which comprises seven RTL designs, each paired with two placed netlists. For the same RTL design, the two netlists differ slightly in netlist structure and implementation settings, such as core density and macro placement. The benchmarks are derived from the open-source TILOS macro placement suite [2] and are synthesized using the NanGate 45nm technology node. The largest design contains approximately 50 million cells. Some testcases feature macros that restrict access to certain routing resources. To simplify the setup, power grid and clock tree routing are excluded.

Below table details the statistics of the test cases. 

|Design | #std cells | #macros | #nets | #pins | density (\%) | GCell grid dimensions |
|  ----  | ----  | ----  | ----  | ---- | ---- | ---- | 
Ariane_sample | 122K | 133 | 129K | 420K | 51 | 844*1144|
MemPool-Tile_sample | 129K | 20 | 136K | 500K | 51 | 475*644|
NVDLA_sample | 166K | 128 | 177K | 630K | 51 | 1240*1682|
BlackParrot_sample | 715K | 220 | 770K | 2.9M | 68 | 1532*2077|
MemPool-Group_sample | 3.1M | 320 | 3.3M | 10.9M | 68 | 1782*2417|
MemPool-Cluster_sample | 9.9M | 1296 | 10.6M | 40.2M | 68 | 3511*4764|
TeraPool-Cluster_sample | 49.7M | 4192 | 59.3M | 213M | 68 | 7891*10708|
|  ----  | ----  | ----  | ----  | ---- | ---- | ---- | 
Ariane_rank | 121K | 133 | 128K | 435K | 68 | 716*971|
MemPool-Tile_rank | 128K | 20 | 136K | 483K | 68 | 429*581|
NVDLA_rank | 164K | 128 | 174K | 610K | 68 | 908*1682|
BlackParrot_rank | 780K | 220 | 825K | 2.9M | 68 | 1532*2077|
MemPool-Group_rank | 3.0M | 320 | 3.2M | 10.9M | 68 | 1782*2417|
MemPool-Cluster_rank | 9.9M | 1296 | 10.6M | 40.2M | 51 | 4113*5580|
TeraPool-Cluster_rank | 49.7M | 4192 | 59.3M | 213M | 51 | 9245*12544|

## Downloads
[Testcases](https://drive.google.com/drive/u/2/folders/1bon65UEAx8cjSvVhYJ-lgC8QMDX0fvUm)

[Evaluation Scripts](https://drive.google.com/drive/u/2/folders/1Ckqd9Fq-CpqVwAlaSObMmv0Uvqbx3IVf)

[Example Outputs](https://drive.google.com/drive/u/2/folders/1FKbYnYVHoroDp9kulaTBWkfKkdly1rdn)

## References
[1] Liang, Rongjian, et al. "Gpu/ml-enhanced large scale global routing contest." International Symposium on Physical Design. 2024.

[2] Cheng, Chung-Kuan, et al. "Assessment of reinforcement learning for macro placement." International Symposium on Physical Design. 2023.
