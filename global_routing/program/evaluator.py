#!/usr/bin/env python3
from typing import Union, List
import math
from dataclasses import dataclass
from enum import Enum, auto


class Direction(Enum):
    HORIZONTAL = auto()
    VERTICAL = auto()
    BOTH = auto()
    NONE = auto()


@dataclass
class Point:
    x: int
    y: int


@dataclass
class Point3D:
    x: int
    y: int
    z: int


@dataclass
class Box:
    lo: Point
    hi: Point

    def hpwl(self) -> int:
        return self.width() + self.height()

    def width(self) -> int:
        return self.hi.x - self.lo.x

    def height(self) -> int:
        return self.hi.y - self.lo.y


@dataclass
class Pin:
    accesses: List[Point3D]


@dataclass
class Net:
    name: str
    idx: int
    pins: List[Pin]
    box: Box


@dataclass
class GCell:
    demand: int = 0
    capacity: float = 0.0


@dataclass
class GridGraph2D:
    name: str = ""
    direction: Direction = Direction.NONE
    min_length: float = 0.0
    unit_length_cost: float = 0.0
    unit_via_cost: float = 0.0
    unit_overflow_cost: float = 0.0
    num_gridx: int = 0
    num_gridy: int = 0
    gcells: List[List[GCell]] = None

    def is_routing_layer(self) -> bool:
        return self.direction != Direction.NONE

    def is_hor(self) -> bool:
        return self.direction == Direction.HORIZONTAL

    def is_ver(self) -> bool:
        return self.direction == Direction.VERTICAL


@dataclass
class GridGraph:
    num_gridx: int = 0
    num_gridy: int = 0
    num_layer: int = 0
    planes: List[GridGraph2D] = None
    x_coords: List[int] = None
    y_coords: List[int] = None

    def cell_width(self, x: int) -> int:
        return self.x_coords[x + 1] - self.x_coords[x]

    def cell_height(self, y: int) -> int:
        return self.y_coords[y + 1] - self.y_coords[y]


class GlobalRoutingDB:
    def __init__(self):
        self.nets: List[Net] = []
        self.graph: GridGraph = GridGraph()
        self.layer_directions: List[int] = []
        self.error_message: str = ""

    def read_files(self, input_file: str, net_file: str, solution_file: str) -> bool:
        if not self.read_graph(input_file):
            return False
        if not self.read_nets(net_file):
            return False
        if not self.read_gr_solution(solution_file):
            return False
        return True

    def read_graph(self, input_file: str) -> bool:
        with open(input_file, "r") as f:
            # Read dimensions
            num_layers, num_gridx, num_gridy = map(int, f.readline().split())
            self.graph.num_layer = num_layers
            self.graph.num_gridx = num_gridx
            self.graph.num_gridy = num_gridy
            self.graph.planes = [GridGraph2D() for _ in range(num_layers)]

            # Read costs
            unit_length_cost, unit_via_cost = map(float, f.readline().split())
            overflow_costs = list(map(float, f.readline().split()))

            # Read GCell edge lengths
            self.graph.x_coords = [0] + list(map(int, f.readline().split()))
            self.graph.y_coords = [0] + list(map(int, f.readline().split()))

            # Read layer information
            for z in range(num_layers):
                layer_info = f.readline().split()
                layer_name = layer_info[0]
                direction = int(layer_info[1])
                min_length = float(layer_info[2])

                plane = self.graph.planes[z]
                plane.name = layer_name
                plane.min_length = min_length
                plane.direction = (
                    Direction.HORIZONTAL if direction == 0 else Direction.VERTICAL
                )
                plane.set_unit_cost(unit_length_cost, unit_via_cost, overflow_costs[z])

                if z != 0 and plane.is_routing_layer():
                    plane.init(num_gridx, num_gridy)

                # Read capacities
                for y in range(num_gridy):
                    capacities = list(map(float, f.readline().split()))
                    for x in range(num_gridx):
                        if plane.is_routing_layer():
                            plane.get_gcell(x, y).set_capacity(capacities[x])

        return True

    def read_nets(self, net_file: str) -> bool:
        with open(net_file, "r") as f:
            current_net = None
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if "(" not in line and ")" not in line and len(line) > 1:
                    # New net
                    current_net = Net(
                        name=line,
                        idx=len(self.nets),
                        pins=[],
                        box=Box(
                            Point(float("inf"), float("inf")),
                            Point(float("-inf"), float("-inf")),
                        ),
                    )
                    self.nets.append(current_net)
                elif "[" in line:
                    # New pin
                    if current_net:
                        pin = Pin(accesses=[])
                        current_net.pins.append(pin)
                        # Parse access points
                        points = (
                            line.replace("[", "")
                            .replace("]", "")
                            .replace("(", "")
                            .replace(")", "")
                            .split()
                        )
                        for i in range(0, len(points), 3):
                            z, x, y = map(int, points[i : i + 3])
                            pin.accesses.append(Point3D(x, y, z))
                            current_net.box.lo.x = min(current_net.box.lo.x, x)
                            current_net.box.lo.y = min(current_net.box.lo.y, y)
                            current_net.box.hi.x = max(current_net.box.hi.x, x)
                            current_net.box.hi.y = max(current_net.box.hi.y, y)
        return True

    def read_gr_solution(self, solution_file: str) -> bool:
        with open(solution_file, "r") as f:
            net_mapper = {net.name: net for net in self.nets}
            net_completed = {net.name: False for net in self.nets}

            total_opens = 0
            total_vias = [0] * self.graph.num_layer
            flag = [
                [
                    [-1 for _ in range(self.graph.num_gridy)]
                    for _ in range(self.graph.num_gridx)
                ]
                for _ in range(self.graph.num_layer)
            ]
            wire_counter = [
                [
                    [0 for _ in range(self.graph.num_gridy)]
                    for _ in range(self.graph.num_gridx)
                ]
                for _ in range(self.graph.num_layer)
            ]
            stacked_via_counter = [
                [
                    [0 for _ in range(self.graph.num_gridy)]
                    for _ in range(self.graph.num_gridx)
                ]
                for _ in range(self.graph.num_layer)
            ]

            current_net = None
            via_loc = []
            has_connectivity_violation = False

            for line in f:
                line = line.strip()
                if not line:
                    continue

                if not current_net:
                    current_net = net_mapper.get(line)
                    has_connectivity_violation = False
                elif line[0] == "(":
                    continue
                elif line[0] == ")":
                    self.update_stacked_via_counter(
                        current_net.idx, via_loc, flag, stacked_via_counter
                    )
                    if has_connectivity_violation:
                        total_opens += 1
                    else:
                        if not self.check_connectivity(current_net, flag):
                            total_opens += 1
                        else:
                            net_completed[current_net.name] = True
                    current_net = None
                    via_loc = []
                else:
                    # Parse wire/via
                    xl, yl, zl, xh, yh, zh = map(int, line.split())
                    if zh != zl:  # via
                        if xh == xl and yh == yl:
                            for z in range(zl, zh):
                                total_vias[z] += 1
                                via_loc.append(Point3D(xl, yl, z))
                        else:
                            has_connectivity_violation = True
                    else:  # wire
                        plane = self.graph.planes[zl]
                        if plane.is_hor():
                            if xh > xl and yh == yl:
                                for x in range(xl, xh + 1):
                                    flag[zl][x][yl] = current_net.idx
                                    wire_counter[zl][x][yl] += 1
                            else:
                                has_connectivity_violation = True
                        elif plane.is_ver():
                            if yh > yl and xh == xl:
                                for y in range(yl, yh + 1):
                                    flag[zl][xl][y] = current_net.idx
                                    wire_counter[zl][xl][y] += 1
                            else:
                                has_connectivity_violation = True
                        else:
                            has_connectivity_violation = True

            # Calculate costs
            wl_cost = 0
            via_cost = 0
            overflow_cost = 0
            overflow_slope = 0.5

            for z in range(self.graph.num_layer):
                plane = self.graph.planes[z]
                if not plane.is_routing_layer():
                    via_cost += total_vias[z] * plane.unit_via_cost()
                    continue

                total_wl = 0
                layer_overflows = 0
                for x in range(self.graph.num_gridx):
                    for y in range(self.graph.num_gridy):
                        cell = plane.get_gcell(x, y)
                        demand = 2 * wire_counter[z][x][y]
                        cell.demand = demand

                        if cell.capacity > 0.001:
                            overflow = cell.demand - 2 * cell.capacity
                            layer_overflows += self.overflow_loss_func(
                                overflow / 2, overflow_slope
                            )
                        elif cell.capacity >= 0 and cell.demand > 0:
                            layer_overflows += self.overflow_loss_func(
                                cell.demand / 2, overflow_slope
                            )
                        elif cell.capacity < 0:
                            self.error_message = f"Capacity error ({x}, {y}, {z})"
                            return False

                        if plane.is_hor():
                            total_wl += wire_counter[z][x][y] * self.graph.cell_width(x)
                        elif plane.is_ver():
                            total_wl += wire_counter[z][x][y] * self.graph.cell_height(
                                y
                            )

                overflow_cost += layer_overflows * plane.unit_overflow_cost()
                via_cost += total_vias[z] * plane.unit_via_cost()
                wl_cost += total_wl * plane.unit_length_cost()

            total_incompleted = sum(
                1 for completed in net_completed.values() if not completed
            )
            self.total_cost = overflow_cost + via_cost + wl_cost

            if total_opens > 0:
                self.error_message = f"Number of open nets: {total_opens}"
                return False
            if total_incompleted > 0:
                self.error_message = f"Number of incompleted nets: {total_incompleted}"
                return False

            return True

    def update_stacked_via_counter(
        self,
        net_idx: int,
        via_loc: List[Point3D],
        flag: List[List[List[int]]],
        stacked_via_counter: List[List[List[int]]],
    ) -> None:
        for pp in via_loc:
            if flag[pp.z][pp.x][pp.y] != net_idx:
                flag[pp.z][pp.x][pp.y] = net_idx

                direction = self.layer_directions[pp.z]
                if direction == 0:  # horizontal
                    if pp.x > 0 and pp.x < self.graph.num_gridx - 1:
                        stacked_via_counter[pp.z][pp.x - 1][pp.y] += 1
                        stacked_via_counter[pp.z][pp.x][pp.y] += 1
                    elif pp.x > 0:
                        stacked_via_counter[pp.z][pp.x - 1][pp.y] += 2
                    elif pp.x < self.graph.num_gridx - 1:
                        stacked_via_counter[pp.z][pp.x][pp.y] += 2
                elif direction == 1:  # vertical
                    if pp.y > 0 and pp.y < self.graph.num_gridy - 1:
                        stacked_via_counter[pp.z][pp.x][pp.y - 1] += 1
                        stacked_via_counter[pp.z][pp.x][pp.y] += 1
                    elif pp.y > 0:
                        stacked_via_counter[pp.z][pp.x][pp.y - 1] += 2
                    elif pp.y < self.graph.num_gridy - 1:
                        stacked_via_counter[pp.z][pp.x][pp.y] += 2

        for pp in via_loc:
            flag[pp.z][pp.x][pp.y] = net_idx
            flag[pp.z + 1][pp.x][pp.y] = net_idx

    def check_connectivity(self, net: Net, flag: List[List[List[int]]]) -> bool:
        mark = net.idx
        traced_mark = net.idx + len(self.nets)
        stack = []

        # Start from first pin's access points
        for ac in net.pins[0].accesses:
            if flag[ac.z][ac.x][ac.y] == mark:
                flag[ac.z][ac.x][ac.y] = traced_mark
                stack.append(ac)

        # Propagate connectivity
        while stack:
            pp = stack.pop()
            plane = self.graph.planes[pp.z]

            if plane.is_hor():
                if pp.x > 0 and flag[pp.z][pp.x - 1][pp.y] == mark:
                    flag[pp.z][pp.x - 1][pp.y] = traced_mark
                    stack.append(Point3D(pp.x - 1, pp.y, pp.z))
                if (
                    pp.x < self.graph.num_gridx - 1
                    and flag[pp.z][pp.x + 1][pp.y] == mark
                ):
                    flag[pp.z][pp.x + 1][pp.y] = traced_mark
                    stack.append(Point3D(pp.x + 1, pp.y, pp.z))
            elif plane.is_ver():
                if pp.y > 0 and flag[pp.z][pp.x][pp.y - 1] == mark:
                    flag[pp.z][pp.x][pp.y - 1] = traced_mark
                    stack.append(Point3D(pp.x, pp.y - 1, pp.z))
                if (
                    pp.y < self.graph.num_gridy - 1
                    and flag[pp.z][pp.x][pp.y + 1] == mark
                ):
                    flag[pp.z][pp.x][pp.y + 1] = traced_mark
                    stack.append(Point3D(pp.x, pp.y + 1, pp.z))

            if pp.z > 0 and flag[pp.z - 1][pp.x][pp.y] == mark:
                flag[pp.z - 1][pp.x][pp.y] = traced_mark
                stack.append(Point3D(pp.x, pp.y, pp.z - 1))
            if pp.z < self.graph.num_layer - 1 and flag[pp.z + 1][pp.x][pp.y] == mark:
                flag[pp.z + 1][pp.x][pp.y] = traced_mark
                stack.append(Point3D(pp.x, pp.y, pp.z + 1))

        # Check if all pins are connected
        for i in range(1, len(net.pins)):
            connected = False
            for ac in net.pins[i].accesses:
                if flag[ac.z][ac.x][ac.y] == traced_mark:
                    connected = True
                    break
            if not connected:
                return False

        return True

    def overflow_loss_func(self, overflow: float, slope: float) -> float:
        return math.exp(overflow * slope)


def evaluate(cap_file: str, net_file: str, solution_file: str) -> Union[int, float]:
    """
    Cost calculation function: calculates the cost.
    Suppose the input has been verified by the verifier, which means the input to this function is always valid.
    Please do NOT change the function name and arguments.
    It is used by the agent to evaluate the cost of the generated solution.

    Args:
        cap_file: Path to the routing resource file
        net_file: Path to the net information file
        solution_file: Path to the output file generated by the solver

    Returns:
        Union[int, float]: The final cost
    """
    db = GlobalRoutingDB()
    if not db.read_files(cap_file, net_file, solution_file):
        if db.error_message:
            print(db.error_message)
        return float("inf")

    return db.total_cost
