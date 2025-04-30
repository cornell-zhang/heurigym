# Gloabl Routing

## Background

Global routing (GR) techniques, which establish coarse-grain routing paths for signal nets throughout a Very Large Scale Integration (VLSI) circuit, have a wide range of applications spanning various stages of the modern VLSI design flow. GR significantly impacts circuit timing, power consumption, and overall routability. As an NP-hard problem, GR has been addressed through various heuristics over the past decades, often trading off some degree of optimality for practical efficiency.

## Problem Description

In global routing, a 3D routing space is defined using global routing cells (GCells), created by a regular grid of horizontal and vertical lines. This configuration results in the formation of a grid graph  $\( \mathcal{G}(\mathcal{V}, \mathcal{E}) \)$ where each GCell is treated as a vertex $\( (v \in \mathcal{V}) \)$ and edges $\( (e \in \mathcal{E}) \)$ connect adjacent GCells within the same layer (GCell edges) or between GCells in neighboring layers (via edges). It's important to note that each layer has a preferred routing direction, which means GCell edges can be either horizontal or vertical. The essence of the global routing challenge is to establish concrete path for each net within the grid graph. This process ensures the interconnection of all pins without overflow while minimizing total wire length and the number of vias.

### Inputs
There are two input files for each circuit: a routing resource file (with a .cap extension) and a net information file (with a .net extension). The routing resource file offers a detailed representation of the GCell grid graph and the routing resources it encompasses. Meanwhile, the net information file shows the access points for all the pins within each net.

The routing resource file follows this format:

\hspace{0.2cm} \textcolor{blue}{\# Dimensions of GCell graph}

\hspace{0.2cm} nLayers xSize ySize 

\hspace{0.2cm} \textcolor{blue}{\# Weights of performance metrics}

\hspace{0.2cm} UnitLengthWireCost UnitViaCost OFWeight[0] OFWeight[1] 
OFWeight[2] \cdots 

\hspace{0.2cm} \textcolor{blue}{\# Lengths of horizontal GCell edges (edge count = xSize - 1)}

\hspace{0.2cm} HorizontalGCellEdgeLengths[0] HorizontalGCellEdgeLengths[1] HorizontalGCellEdgeLengths[2] \cdots  

\hspace{0.2cm} \textcolor{blue}{\# Lengths of vertical GCell edges (edge count = ySize - 1)}

\hspace{0.2cm} VerticalGCellEdgeLengths[0] VerticalGCellEdgeLengths[1] VerticalGCellEdgeLengths[2] \cdots 

\hspace{0.2cm} \textcolor{blue}{\# Information for the $0$-th layer}

\hspace{0.2cm} \textcolor{blue}{\#\# Layer name, prefered direction and minimum length of a wire at this metal layer. For direction, 0 represents horizontal, while 1 represents vertical.}

\hspace{0.2cm} layerName layerDirection layerMinLength 

\hspace{0.2cm} \textcolor{blue}{\#\# Routing capacities of GCell edges at the $0$-th layer}

\hspace{0.2cm} \textcolor{blue}{\#\#\# Capacities of GCell at [x(0), y(0)], Capacities of GCell at [x(1), y(0)], ...}

\hspace{0.2cm} 10 10 10 \cdots

\hspace{0.2cm} \textcolor{blue}{\#\#\# Capacities of GCell at [x(0), y(1)], Capacities of GCell at [x(1), y(1)], ...}

\hspace{0.2cm} 10 10 10 \cdots

\hspace{0.2cm} \cdots \cdots \cdots

\hspace{0.2cm} \textcolor{blue}{\#\# Information for the $1$-th layer}

\hspace{0.2cm} \cdots \cdots \cdots

## Evaluation
### Evaluation Metrics

### Benchmarks
