# Gloabl Routing

## Background

Global routing (GR) techniques, which establish coarse-grain routing paths for signal nets throughout a Very Large Scale Integration (VLSI) circuit, have a wide range of applications spanning various stages of the modern VLSI design flow. GR significantly impacts circuit timing, power consumption, and overall routability. As an NP-hard problem, GR has been addressed through various heuristics over the past decades, often trading off some degree of optimality for practical efficiency.

## Problem Description

In global routing, a 3D routing space is defined using global routing cells (GCells), created by a regular grid of horizontal and vertical lines. This configuration results in the formation of a grid graph  $\( \mathcal{G}(\mathcal{V}, \mathcal{E}) \)$ where each GCell is treated as a vertex $\( (v \in \mathcal{V}) \)$ and edges $\( (e \in \mathcal{E}) \)$ connect adjacent GCells within the same layer (GCell edges) or between GCells in neighboring layers (via edges). It's important to note that each layer has a preferred routing direction, which means GCell edges can be either horizontal or vertical. The essence of the global routing challenge is to establish concrete path for each net within the grid graph. This process ensures the interconnection of all pins without overflow while minimizing total wire length and the number of vias.

### Inputs
There are two input files for each circuit: a routing resource file (with a \texttt{.cap} extension) and a net information file (with a \texttt{.net} extension). The routing resource file offers a detailed representation of the GCell grid graph and the routing resources it encompasses. Meanwhile, the net information file shows the access points for all the pins within each net.


## Evaluation
### Evaluation Metrics

### Benchmarks
