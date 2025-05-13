# Global Routing for VLSI Circuits

## 1. Background

Global routing is a critical step in Very Large Scale Integration (VLSI) physical design. It involves determining the approximate paths for interconnecting signal nets across a chip layout. The quality of global routing significantly impacts circuit performance metrics such as timing delay, power consumption, and overall manufacturability (routability). The chip layout is typically represented as a 3D grid, and the task is to find paths for each net within this grid, connecting all its required points (pins) while respecting resource limitations and design rules. Due to the massive number of nets and complex interactions, global routing is computationally challenging (NP-hard) and relies on effective heuristics to find high-quality solutions within reasonable runtime.

## 2. Formalization

### 2.1 Routing Grid Graph

The routing space is modeled as a 3D grid graph $\mathcal{G}(\mathcal{V}, \mathcal{E})$.

* **Grid Dimensions:** Defined by the number of layers ($nLayers$), width ($xSize$), and height ($ySize$).
* **Vertices (GCells):** $\mathcal{V} = \{ (x, y, z) \mid 0 \le x < xSize, 0 \le y < ySize, 0 \le z < nLayers \}$. Each vertex represents a Global Cell (GCell).
* **Edges:** $\mathcal{E} = \mathcal{E}_H \cup \mathcal{E}_V \cup \mathcal{E}_Z$, representing connections between adjacent GCells.
    * **Horizontal Edges ($\mathcal{E}_H$):** An edge $e = ((x, y, z), (x+1, y, z))$ exists if and only if layer $z$ has a **horizontal** preferred routing direction (specified in the input `.cap` file) and $0 \le x < xSize-1$, $0 \le y < ySize$.
    * **Vertical Edges ($\mathcal{E}_V$):** An edge $e = ((x, y, z), (x, y+1, z))$ exists if and only if layer $z$ has a **vertical** preferred routing direction (specified in the input `.cap` file) and $0 \le x < xSize$, $0 \le y < ySize-1$.
    * **Via Edges ($\mathcal{E}_Z$):** An edge $e = ((x, y, z), (x, y, z+1))$ exists for all $0 \le x < xSize$, $0 \le y < ySize$, $0 \le z < nLayers-1$. These represent potential locations for vias connecting adjacent layers.

### 2.2 Edge Attributes

Each edge $e \in \mathcal{E}$ has associated properties:

* **Capacity ($c(e)$):**
    * For a horizontal edge $e = ((x, y, z), (x+1, y, z)) \in \mathcal{E}_H$, its capacity $c(e)$ is provided by the `capacity[x][y]` value for layer $z$ in the input `.cap` file.
    * For a vertical edge $e = ((x, y, z), (x, y+1, z)) \in \mathcal{E}_V$, its capacity $c(e)$ is provided by the `capacity[x][y]` value for layer $z$ in the input `.cap` file.
    * For a via edge $e \in \mathcal{E}_Z$, its capacity $c(e) = \infty$. Via usage is penalized by cost, not capacity limit.
    * Edges in non-preferred directions are strictly disallowed (effectively $c(e)=0$ and they are not included in $\mathcal{E}_H$ or $\mathcal{E}_V$).
* **Length ($len(e)$):**
    * For $e = ((x, y, z), (x+1, y, z)) \in \mathcal{E}_H$, $len(e) = HorizontalGCellEdgeLengths[x]$ (from `.cap` file).
    * For $e = ((x, y, z), (x, y+1, z)) \in \mathcal{E}_V$, $len(e) = VerticalGCellEdgeLengths[y]$ (from `.cap` file).
    * For $e \in \mathcal{E}_Z$, $len(e) = 0$. Wire length cost applies only to horizontal and vertical segments.
* **Layer Index ($l(e)$):** For $e \in \mathcal{E}_H \cup \mathcal{E}_V$, $l(e)$ is the layer index $z$ associated with the edge.

### 2.3 Nets and Pins

* The input defines a set of nets $\mathcal{N}$.
* Each net $N \in \mathcal{N}$ consists of a set of pins $P_N = \{p_1, p_2, ..., p_{k_N}\}$.
* Each pin $p_i$ has one or more possible **access points**, represented as a set of GCells $A_{p_i} \subset \mathcal{V}$. An access point is specified by its coordinates $(z, x, y)$, corresponding to the GCell $(x, y, z)$.

### 2.4 Routing Solution

A routing solution consists of assigning a path $Path_N \subseteq \mathcal{G}$ to each net $N \in \mathcal{N}$. $Path_N$ is represented by the set of edges $E(Path_N) \subseteq \mathcal{E}$ used by the net.

### 2.5 Constraints

A valid routing solution must satisfy the following constraints:

1.  **Connectivity:** For each net $N \in \mathcal{N}$, the set of edges $E(Path_N)$ must form a single connected component (e.g., a Steiner tree topology) in $\mathcal{G}$. Furthermore, for *every* pin $p_i \in P_N$, the set of vertices touched by $Path_N$ must include *at least one* of its access points. That is, $V(Path_N) \cap A_{p_i} \neq \emptyset$ for all $p_i \in P_N$. Any access point from $A_{p_i}$ can be chosen for connection.
2.  **Layer 0 Restriction (Metal 1):** Layer 0 (metal layer with index $z=0$) cannot be used for routing segments. No horizontal ($e \in \mathcal{E}_H$ with $z=0$) or vertical ($e \in \mathcal{E}_V$ with $z=0$) edges can be part of any $Path_N$. Pins with access points on layer 0 (e.g., $(x, y, 0) \in A_{p_i}$) *must* be connected to the rest of the net via a via edge starting at that location, i.e., using the edge $((x, y, 0), (x, y, 1))$.
3.  **Preferred Direction:** Paths must only utilize edges present in the graph $\mathcal{G}$ as defined in Section 2.1, inherently respecting the preferred routing direction of each layer.

### 2.6 Scoring Objective

The goal is to find a valid routing solution that minimizes the total weighted score $S$:

$$S = C_{wire} \times TotalWL + C_{via} \times ViaCount + Score_{overflow}$$

Where:

* **$C_{wire}$ (UnitLengthWireCost):** Cost per unit length of wire (provided in `.cap`).
* **$C_{via}$ (UnitViaCost):** Cost per via (provided in `.cap`).
* **$TotalWL$ (Total Wire Length):** The sum of lengths of all horizontal and vertical edges used across all nets.
    $$TotalWL = \sum_{N \in \mathcal{N}} \sum_{e \in E(Path_N) \cap (\mathcal{E}_H \cup \mathcal{E}_V)} len(e)$$
* **$ViaCount$ (Total Via Count):** The total number of via edges used across all nets. Each via edge connects adjacent layers at the same (x,y) coordinate.
    $$ViaCount = \sum_{N \in \mathcal{N}} |\{ e \in E(Path_N) \cap \mathcal{E}_Z \}|$$
* **$Score_{overflow}$ (Total Overflow Score):** The penalty for exceeding edge capacities.
    * **Demand $d(e)$:** For any edge $e \in \mathcal{E}_H \cup \mathcal{E}_V$, the demand is the number of nets routed through it: $d(e) = |\{ N \in \mathcal{N} \mid e \in E(Path_N) \}|$.
    * **Edge Overflow Cost $O(e)$:** Let $c(e)$ be the capacity of edge $e$, and $l(e)$ be its layer index. Let $OFWeight[l(e)]$ be the overflow weight for that layer (from `.cap`).
        $$O(e) = \begin{cases} OFWeight[l(e)] \times \exp(0.5 \times (d(e) - c(e))) & \text{if } d(e) > c(e) \\ 0 & \text{if } d(e) \le c(e) \end{cases}$$
    * **Total Overflow Score:** Sum of overflow costs over all horizontal and vertical edges.
        $$Score_{overflow} = \sum_{e \in \mathcal{E}_H \cup \mathcal{E}_V} O(e)$$

The objective is to find the set of paths $\{Path_N\}_{N \in \mathcal{N}}$ that satisfies all constraints and yields the minimum score $S$.

## 3. Input Format

Two input files are provided for each test case: a routing resource file (`.cap`) and a net information file (`.net`).

### 3.1 Routing Resource File (`.cap`)

This file describes the grid graph structure and resources. Values are newline-separated unless specified otherwise.

1.  `nLayers xSize ySize` (Integers: Number of layers, grid width, grid height)
2.  `UnitLengthWireCost UnitViaCost OFWeight[0] OFWeight[1] ... OFWeight[nLayers-1]` (Floats: Cost factors and per-layer overflow weights, space-separated)
3.  `HorizontalGCellEdgeLengths[0] ... HorizontalGCellEdgeLengths[xSize-2]` (Integers: Lengths of horizontal edges between $x$ and $x+1$, space-separated. $xSize-1$ values.)
4.  `VerticalGCellEdgeLengths[0] ... VerticalGCellEdgeLengths[ySize-2]` (Integers: Lengths of vertical edges between $y$ and $y+1$, space-separated. $ySize-1$ values.)
5.  **Layer Information Blocks:** Repeated for each layer $l$ from 0 to $nLayers-1$. Each block contains:
    * `layerName layerDirection layerMinLength` (String, Integer {0=Horizontal, 1=Vertical}, Float (unused))
    * **Capacity Grid (ySize lines):** For each $y$ from 0 to $ySize-1$, a line follows:
        `capacity[0][y] capacity[1][y] ... capacity[xSize-1][y]` (Floats: Capacity values, space-separated)

        **Important Capacity Mapping:**
        * The `capacity[x][y]` value for layer $l$ defines the capacity $c(e)$ for the edge originating at GCell $(x, y, l)$ in its **preferred direction**.
        * If `layerDirection` is 0 (Horizontal), `capacity[x][y]` applies to edge $((x, y, l), (x+1, y, l))$ for $0 \le x < xSize-1$. The capacity values provided for $x = xSize-1$ (the last column in the input line) are **ignored** for horizontal layers.
        * If `layerDirection` is 1 (Vertical), `capacity[x][y]` applies to edge $((x, y, l), (x, y+1, l))$ for $0 \le y < ySize-1$. The capacity values provided for $y = ySize-1$ (the last row of capacity data for this layer) are **ignored** for vertical layers.

### 3.2 Net Information File (`.net`)

This file lists the nets and their pin access points.

```
# Name of Net0 (Example: boot_addr_i[63])
NetName0
(
# Access point locations for pin 0 of Net0
[(layer1, x1, y1), (layer2, x2, y2), ...]
# Access point locations for pin 1 of Net0
[(layer3, x3, y3), ...]
...
)
# Name of Net1
NetName1
(
# Access point locations for pin 0 of Net1
[(layerA, xA, yA), ...]
...
)
...
```

* Each net starts with its name on a new line.
* The pins and their access points are enclosed in parentheses `(...)`.
* Each line inside the parentheses defines one pin, starting with `[`.
* Inside the `[...]`, one or more access points are listed as tuples `(layer, x, y)`, separated by commas.
* `layer`, `x`, `y` are integer coordinates corresponding to GCell $(x, y, layer)$.

### 3.3 Example Input Files

#### Example `.cap` file (simplified):

```
3 4 3
0.1 2.0 1.0 1.0 1.0
10 10 10
10 10
M0 0 0.0
10 10 10 0
10 10 10 0
10 10 10 0
M1 1 0.0
15 15 15 15
15 15 15 15
0 0 0 0
M2 0 0.0
12 12 12 0
12 12 12 0
12 12 12 0
```

* **Line 1:** 3 layers, grid size 4x3 (xSize=4, ySize=3).
* **Line 2:** `UnitLengthWireCost=0.1`, `UnitViaCost=2.0`, `OFWeight` for layer 0, 1, 2 are all `1.0`.
* **Line 3:** Horizontal edge lengths: `Length(x=0 to x=1)=10`, `Length(x=1 to x=2)=10`, `Length(x=2 to x=3)=10`.
* **Line 4:** Vertical edge lengths: `Length(y=0 to y=1)=10`, `Length(y=1 to y=2)=10`.
* **Layer 0 (M0):** Horizontal. `layerMinLength` is unused.
    * Capacities `capacity[x][y]` for layer 0. For this horizontal layer, last column (`capacity[3][y]`, which is `0`) is ignored.
* **Layer 1 (M1):** Vertical.
    * Capacities `capacity[x][y]` for layer 1. For this vertical layer, last row (`capacity[x][2]`, which is `0 0 0 0`) is ignored.
* **Layer 2 (M2):** Horizontal.
    * Capacities `capacity[x][y]` for layer 2. Last column (`0`) ignored.

#### Example `.net` file:

```
NetA
(
[(0, 0, 0)]
[(2, 3, 2)]
)
NetB
(
[(1, 1, 1), (1, 2, 1)]
[(1, 0, 1)]
)
```

* **NetA:** Has two pins.
    * Pin 1 has one access point: GCell $(0,0,0)$ (layer 0, x=0, y=0).
    * Pin 2 has one access point: GCell $(3,2,2)$ (layer 2, x=3, y=2).
* **NetB:** Has two pins.
    * Pin 1 has two access points: GCell $(1,1,1)$ and GCell $(2,1,1)$.
    * Pin 2 has one access point: GCell $(0,1,1)$.

## 4. Output Format

The output file should describe the routing path for each net as a sequence of connected segments.

```
# Net name
NetName0
(
# Segment 1: xl yl zl xh yh zh
x1l y1l z1l x1h y1h z1h
# Segment 2: xl yl zl xh yh zh
x2l y2l z2l x2h y2h z2h
...
)
# Net name
NetName1
(
...
)
...
```

* Each net's routing solution starts with its name on a new line, followed by `(`.
* Each subsequent line defines one **segment** of the route using 6 integer coordinates: `$x_l$ $y_l$ $z_l$ $x_h$ $y_h$ $z_h$`.
* It must hold that $x_l \le x_h$, $y_l \le y_h$, and $z_l \le z_h$.
* A segment represents a straight line connection in the 3D grid and must correspond to one of the following types:
    1.  **Horizontal Segment:** $y_l=y_h$, $z_l=z_h$, $x_l < x_h$. This implies layer $z_l$ has a horizontal preferred direction. This segment represents the path utilizing horizontal edges $((x, y_l, z_l), (x+1, y_l, z_l))$ for $x$ from $x_l$ to $x_h-1$.
    2.  **Vertical Segment:** $x_l=x_h$, $z_l=z_h$, $y_l < y_h$. This implies layer $z_l$ has a vertical preferred direction. This segment represents the path utilizing vertical edges $((x_l, y, z_l), (x_l, y+1, z_l))$ for $y$ from $y_l$ to $y_h-1$.
    3.  **Via Segment/Stack:** $x_l=x_h$, $y_l=y_h$, $z_l < z_h$. This segment represents the path utilizing via edges $((x_l, y_l, z), (x_l, y_l, z+1))$ for $z$ from $z_l$ to $z_h-1$. Each individual via edge in this stack contributes to the `ViaCount`.
* The collection of segments for a given net must form a single connected component satisfying all constraints defined in Section 2.5.
* The output ends with `)` for each net.

### 4.1 Example Output

For `NetA` from the example `.net` file, assuming Layer 0 and 2 are Horizontal, and Layer 1 is Vertical.
A possible path connecting pin $(0,0,0)$ on Layer 0 to pin $(3,2,2)$ on Layer 2 could be:

```
NetA
(
0 0 0 0 0 1
0 0 1 3 0 1
3 0 1 3 0 2
3 0 2 3 2 2
)
```

* `0 0 0 0 0 1`: Via from GCell $(0,0,0)$ to $(0,0,1)$. (Connects Layer 0 pin to Layer 1)
* `0 0 1 3 0 1`: Horizontal segment on Layer 1 from GCell $(0,0,1)$ to $(3,0,1)$. (This assumes Layer 1 allows horizontal routing, if it's vertical, this segment would be invalid. For this example, let's assume Layer 1 is horizontal for this segment for demonstration, or the path needs to change, e.g., go up to Layer 2 earlier if Layer 2 is horizontal).
    * *Correction based on typical layer direction assumptions for example:* If Layer 1 is Vertical, and Layer 2 is Horizontal:
        A more valid path respecting Layer 0 via, vertical Layer 1, horizontal Layer 2:
        ```
        NetA
        (
        0 0 0 0 0 1
        0 0 1 0 2 1
        0 2 1 0 2 2
        0 2 2 3 2 2
        )
        ```
        * `0 0 0 0 0 1`: Via from $(0,0,0)$ to $(0,0,1)$.
        * `0 0 1 0 2 1`: Vertical segment on Layer 1 (Vertical preferred) from $(0,0,1)$ to $(0,2,1)$.
        * `0 2 1 0 2 2`: Via from $(0,2,1)$ to $(0,2,2)$.
        * `0 2 2 3 2 2`: Horizontal segment on Layer 2 (Horizontal preferred) from $(0,2,2)$ to $(3,2,2)$.
* The sequence of segments connects the specified access points and forms a continuous path.