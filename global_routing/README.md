# Global Routing

## Background

Global routing (GR) techniques, which establish coarse-grain routing paths for signal nets throughout a Very Large Scale Integration (VLSI) circuit, have a wide range of applications spanning various stages of the modern VLSI design flow. GR significantly impacts circuit timing, power consumption, and overall routability. As an NP-hard problem, GR has been addressed through various heuristics over the past decades, often trading off some degree of optimality for practical efficiency.

## Formalization

In global routing, a 3D routing space is defined using GR cells (GCells), created by a regular grid of horizontal and vertical lines. This configuration results in the formation of a grid graph  $\mathcal{G}(\mathcal{V}, \mathcal{E})$ where each GCell is treated as a vertex $v \in \mathcal{V}$ and edges $e \in \mathcal{E}$ connect adjacent GCells within the same layer (GCell edges) or between GCells in neighboring layers (via edges). It's important to note that each layer has a preferred routing direction, which means GCell edges can be either horizontal or vertical. The essence of the global routing challenge is to establish concrete path for each net within the grid graph. This process ensures the interconnection of all pins without overflow while minimizing total wire length and the number of vias.

The global routing solution is measured by the weighted sum of the following metrics: total wire length, via utilization and routing overflow:

    Score = UnitLengthWireCost*TotalWL + UnitViaCost*ViaCount + OverflowScore

TotalWL and ViaCount denote the sum of the wire length for all nets and the total number of vias, respectively. UnitLengthWireCost and UnitViaCost are defined in the .cap file. In our evaluation, UnitLengthWireCost and UnitViaCost are set to be $0.00131579$ and $4$, respectively.

The OverflowScore is computed as the total overflow cost across all GCell edges. For a GCell edge at layer $l$ with routing capacity $c$ and demand $d$, the overflow cost is defined as:
$OverflowCost(c,d,l) = OFWeight[l] * (\exp^{0.5(d-c)})$.

$OFWeight[l]$ is overflow weight for GCell edges at the $l$-th layer, which is defined in the .cap file.
The smaller the weighted score, the better.

## Input Format
There are two input files for each testcase: a routing resource file (with a .cap extension) and a net information file (with a .net extension). The routing resource file offers a detailed representation of the GCell grid graph and the routing resources it encompasses. Meanwhile, the net information file shows the access points for all the pins within each net.

### Routing Resource File (.cap)

The routing resource file follows this format, with each value on a new line (line-by-line breakdown):

#### Line 1: Grid Dimensions
```
nLayers xSize ySize
```
- `nLayers` (integer): Number of metal layers in the routing grid
- `xSize` (integer): Width of the GCell grid (number of GCells in the x direction)
- `ySize` (integer): Height of the GCell grid (number of GCells in the y direction)

#### Line 2: Cost Weights
```
UnitLengthWireCost UnitViaCost OFWeight[0] OFWeight[1] OFWeight[2] ... OFWeight[nLayers-1]
```
- `UnitLengthWireCost` (float): Cost per unit length of wire
- `UnitViaCost` (float): Cost per via
- `OFWeight[i]` (float): Overflow weight for the i-th layer (array of nLayers float values)

#### Line 3: Horizontal GCell Edge Lengths
```
HorizontalGCellEdgeLengths[0] HorizontalGCellEdgeLengths[1] ... HorizontalGCellEdgeLengths[xSize-2]
```
- `HorizontalGCellEdgeLengths[i]` (integer): Length of the i-th horizontal GCell edge (array of xSize-1 integer values)

#### Line 4: Vertical GCell Edge Lengths
```
VerticalGCellEdgeLengths[0] VerticalGCellEdgeLengths[1] ... VerticalGCellEdgeLengths[ySize-2]
```
- `VerticalGCellEdgeLengths[i]` (integer): Length of the i-th vertical GCell edge (array of ySize-1 integer values)

#### Layer Information (for each layer from 0 to nLayers-1)
For each layer, there will be 1 + ySize lines:

##### Layer Header
```
layerName layerDirection layerMinLength
```
- `layerName` (string): Name of the layer
- `layerDirection` (integer): Preferred routing direction (0 = horizontal, 1 = vertical)
- `layerMinLength` (float): Minimum length of a wire at this metal layer (not used in this problem)

##### Capacity Grid (one line for each y-coordinate)
For each y from 0 to ySize-1:
```
capacity[0,y] capacity[1,y] ... capacity[xSize-1,y]
```
- `capacity[x,y]` (float): Routing capacity of the GCell edge at position [x,y] for the current layer

### Net Information File (.net)

The net information file follows this format:

```
# Net name
Net0
(
# Access point locations (layer, x, y) for pin 0
[(0, 202, 347), (0, 202, 348), ...]
# Access point locations for pin 1
[(1, 5, 6), (2, 5, 6), ...]
...
)
Net1
(
...
)
```

Each net consists of:
- Net name (string)
- Opening bracket '('
- List of pins, where each pin has one or more access points
- Each access point is formatted as `(layer, x, y)` where:
  - `layer` (integer): Layer index
  - `x` (integer): x-coordinate
  - `y` (integer): y-coordinate
- Access points for a pin are enclosed in square brackets and separated by spaces
- Closing bracket ')'

## Output Format
The GR solution is described in the GCell coordinate system. To enhance routability and ensure pin accessibility during the subsequent detailed routing process, we enforce following constraints:
1. Metal 1 (the 0-th layer) is not employed for net routing. To reach pins on Metal 1, vias must be utilized to establish connections from higher layers.
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

## Note
The problem presented here is a simplified version of the ISPD 2024 Global Routing Contest [1], with certain constraints relaxed and the computation of performance metrics simplified.

The ISPD 2024 Global Routing Contest comprises seven RTL designs, each paired with two placed netlists. For the same RTL design, the two netlists differ slightly in netlist structure and implementation settings, such as core density and macro placement. The benchmarks are derived from the open-source TILOS macro placement suite [2] and are synthesized using the NanGate 45nm technology node. The largest design contains approximately 50 million cells. Some testcases feature macros that restrict access to certain routing resources. To simplify the setup, power grid and clock tree routing are excluded.

Below table details the statistics of the test cases. 

|Design | #std cells | #macros | #nets | #pins | density (\%) | GCell grid dimensions (nLayers xSize ySize) |
|  ----  | ----  | ----  | ----  | ---- | ---- | ---- | 
Ariane_sample | 122K | 133 | 129K | 420K | 51 | 10 * 844 * 1144|
MemPool-Tile_sample | 129K | 20 | 136K | 500K | 51 | 10 * 475 * 644|
NVDLA_sample | 166K | 128 | 177K | 630K | 51 | 10 * 1240 * 1682|
BlackParrot_sample | 715K | 220 | 770K | 2.9M | 68 | 10 * 1532 * 2077|
MemPool-Group_sample | 3.1M | 320 | 3.3M | 10.9M | 68 | 10 * 1782 * 2417|
MemPool-Cluster_sample | 9.9M | 1296 | 10.6M | 40.2M | 68 | 10 * 3511 * 4764|
TeraPool-Cluster_sample | 49.7M | 4192 | 59.3M | 213M | 68 | 10 * 7891 * 10708|
|  ----  | ----  | ----  | ----  | ---- | ---- | ---- | 
Ariane_rank | 121K | 133 | 128K | 435K | 68 | 10 * 716 * 971|
MemPool-Tile_rank | 128K | 20 | 136K | 483K | 68 | 10 * 429 * 581|
NVDLA_rank | 164K | 128 | 174K | 610K | 68 | 10 * 908 * 1682|
BlackParrot_rank | 780K | 220 | 825K | 2.9M | 68 | 10 * 1532 * 2077|
MemPool-Group_rank | 3.0M | 320 | 3.2M | 10.9M | 68 | 10 * 1782 * 2417|
MemPool-Cluster_rank | 9.9M | 1296 | 10.6M | 40.2M | 51 | 10 * 4113 * 5580|
TeraPool-Cluster_rank | 49.7M | 4192 | 59.3M | 213M | 51 | 10 * 9245 * 12544|

Parsers for the input files can be found in [evaluator.cpp](https://drive.google.com/drive/u/2/folders/1Ckqd9Fq-CpqVwAlaSObMmv0Uvqbx3IVf)
* [Testcases](https://drive.google.com/drive/u/2/folders/1bon65UEAx8cjSvVhYJ-lgC8QMDX0fvUm)
* [Evaluation Scripts](https://drive.google.com/drive/u/2/folders/1Ckqd9Fq-CpqVwAlaSObMmv0Uvqbx3IVf)
* [Example Outputs](https://drive.google.com/drive/u/2/folders/1FKbYnYVHoroDp9kulaTBWkfKkdly1rdn)

## References
1. Liang, Rongjian, et al. "Gpu/ml-enhanced large scale global routing contest." International Symposium on Physical Design. 2024.
2. Cheng, Chung-Kuan, et al. "Assessment of reinforcement learning for macro placement." International Symposium on Physical Design. 2023.
