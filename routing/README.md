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
    ## Layer name, preferred direction and minimum length of a wire at this metal layer  (Direction: 0 = horizontal, 1 = vertical)
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
The global routing solution is measured by the weighted sum of the following metrics: total wire length, via utilization and routing overflow.

$origina\_socre = w_1*TotalWL + w_2*ViaCount + OverflowScore$

TotalWL and ViaCount denote the sum of the wire length for all signal nets and the total number of vias, respectively. $w_1$ and $w_2$ correspond to UnitLengthWireCost and UnitViaCost, respectively, which are defined in the .cap file. In our evaluation, $w_1$ and $w_2$ are set to be $0.5/\texttt{M2 pitch}$ and $2$, respectively.

The overflow cost for a GCell edge with routing capacity $c$ and routing demand $d$ at the $l$-th layer is calculated as follows:
$OverflowCost(c,d,l) = OFWeight[l] * (\exp^{0.5(d-c)})$
### Benchmarks
