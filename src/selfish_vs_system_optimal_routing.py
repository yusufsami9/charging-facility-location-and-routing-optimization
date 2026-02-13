# q_2_2_a.py
# Part 2.2(a) — Selfish routing vs system-optimal routing (given stations from 2.1(d))


import time
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB


# -----------------------------
# USER INPUT: stations from (d)
# -----------------------------
Q_CAP = 10.0  # per unit capacity, as stated in the assignment
STATION_UNITS = {
    # Grand coalition (ABC) locations from your output in 2.1(d)
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 7: 1, 12: 1, 15: 1, 22: 1, 31: 1, 33: 1
}
STATIONS = set(STATION_UNITS.keys())


# -----------------------------
# I/O helpers
# -----------------------------
def read_network(network_file: str) -> Tuple[List[int], List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    """
    network.txt format: i j distance
    Interpreted as directed arc (i,j) with travel time = distance.
    """
    arcs = []
    nodes = set()
    w = {}

    with open(network_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            i, j = int(parts[0]), int(parts[1])
            dist = float(parts[2])
            arcs.append((i, j))
            nodes.add(i)
            nodes.add(j)
            w[(i, j)] = dist

    N = sorted(nodes)
    return N, arcs, w


def read_pairs(pairs_file: str) -> List[Tuple[float, int, int]]:
    """
    pairsX.txt format: volume origin destination
    (first line may be header)
    """
    pairs = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                vol = float(parts[0])
                o = int(parts[1])
                d = int(parts[2])
            except ValueError:
                continue
            pairs.append((vol, o, d))
    return pairs


def load_all_companies() -> List[Tuple[float, int, int, str]]:
    """
    Returns list of OD demands across A,B,C:
    (volume, origin, destination, company_label)
    """
    all_pairs = []
    for X in ["A", "B", "C"]:
        pf = f"data/pairs{X}.txt"
        pairs = read_pairs(pf)
        for (vol, o, d) in pairs:
            all_pairs.append((vol, o, d, X))
    return all_pairs


# -----------------------------
# Shortest charge-feasible path (Selfish)
# -----------------------------
def dijkstra_charge_feasible(
    N: List[int],
    out_arcs: Dict[int, List[int]],
    w: Dict[Tuple[int, int], float],
    origin: int,
    dest: int,
    stations: set
) -> Tuple[float, List[int]]:

    INF = 1e100
    dist = {v: INF for v in N}
    prev = {v: None for v in N}

    dist[origin] = 0.0
    pq = [(0.0, origin)]

    while pq:
        du, u = heapq.heappop(pq)
        if du != dist[u]:
            continue
        if u == dest:
            break

        for v in out_arcs[u]:
            # enforce charge-feasible intermediate nodes
            if (v != dest) and (v not in stations):
                continue

            nd = du + w[(u, v)]
            if nd < dist[v] - 1e-12:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dist[dest] >= INF / 2:
        raise ValueError(f"No charge-feasible path from {origin} to {dest} with given stations.")

    # reconstruct path
    path = []
    cur = dest
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return dist[dest], path


def selfish_routing(
    N: List[int],
    arcs: List[Tuple[int, int]],
    w: Dict[Tuple[int, int], float],
    demands: List[Tuple[float, int, int, str]],
    stations: set,
    station_units: Dict[int, int],
    Q: float
):

    out_arcs = defaultdict(list)
    for (i, j) in arcs:
        out_arcs[i].append(j)

    total_tt = 0.0
    station_load = defaultdict(float)  # total recharge load at node j

    paths_info = []  # for debugging/reporting if needed

    for idx, (vol, o, d, lab) in enumerate(demands):
        dist_od, path = dijkstra_charge_feasible(N, out_arcs, w, o, d, stations)

        total_tt += vol * dist_od

        # charging load counts arrivals at station nodes that are not destination
        # path nodes: [o, ..., d]; every node except origin is an arrival
        for t in range(1, len(path)):
            node = path[t]
            if node in stations and node != d:
                station_load[node] += vol

        paths_info.append((lab, vol, o, d, dist_od, path))

    # capacity check
    violations = {}
    for j in stations:
        cap = Q * station_units.get(j, 0)
        load = station_load.get(j, 0.0)
        if load > cap + 1e-9:
            violations[j] = (load, cap, load - cap)

    return total_tt, station_load, violations, paths_info


# -----------------------------
# System-optimal routing (min-cost flow with station capacity)
# -----------------------------
def system_optimal_routing(
    N: List[int],
    arcs: List[Tuple[int, int]],
    w: Dict[Tuple[int, int], float],
    demands: List[Tuple[float, int, int, str]],
    stations: set,
    station_units: Dict[int, int],
    Q: float,
    output_flag: int = 0
):
    """
    Minimize total travel time subject to:
      - flow conservation for each OD commodity
      - only allowed to use arcs (i,j) in A
      - charging capacity at stations: sum_{k: dest!=j} sum_{i} x_{k,i,j} <= Q * units(j)
    This is a splittable multicommodity min-cost flow.
    """
    out_arcs = defaultdict(list)
    in_arcs = defaultdict(list)
    for (i, j) in arcs:
        out_arcs[i].append(j)
        in_arcs[j].append(i)

    K = list(range(len(demands)))
    V = {k: demands[k][0] for k in K}
    O = {k: demands[k][1] for k in K}
    D = {k: demands[k][2] for k in K}

    m = gp.Model("SystemOptimal_MCF")
    m.Params.OutputFlag = output_flag

    # x[k,i,j] only for arcs to keep model smaller
    Aset = set(arcs)
    x = m.addVars(K, arcs, vtype=GRB.CONTINUOUS, lb=0.0, name="x")  # indexed by (k,(i,j))

    # objective: sum_k sum_(i,j) dist_ij * x
    m.setObjective(gp.quicksum(w[(i, j)] * x[k, (i, j)] for k in K for (i, j) in arcs), GRB.MINIMIZE)

    # flow conservation
    for k in K:
        ok, dk, vk = O[k], D[k], V[k]
        for i in N:
            rhs = 0.0
            if i == ok:
                rhs = vk
            elif i == dk:
                rhs = -vk

            m.addConstr(
                gp.quicksum(x[k, (i, j)] for j in out_arcs[i] if (i, j) in Aset) -
                gp.quicksum(x[k, (j, i)] for j in in_arcs[i] if (j, i) in Aset)
                == rhs,
                name=f"flow_k{k}_i{i}"
            )

    # capacity at station nodes only (non-stations effectively cap=0)
    for j in N:
        if j not in stations:
            continue
        cap = Q * station_units.get(j, 0)

        expr = gp.LinExpr()
        for k in K:
            if D[k] == j:
                continue  # destination: no charging counted
            for i in in_arcs[j]:
                if (i, j) in Aset:
                    expr += x[k, (i, j)]
        m.addConstr(expr <= cap, name=f"cap_{j}")

    # solve
    t0 = time.time()
    m.optimize()
    t1 = time.time()

    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or m.SolCount == 0:
        raise RuntimeError(f"System-optimal routing did not solve. Status={m.Status}")

    return float(m.ObjVal), (t1 - t0)


# -----------------------------
# Main
# -----------------------------
def main():
    NETWORK_FILE = "data/network.txt"

    N, arcs, w = read_network(NETWORK_FILE)
    all_demands = load_all_companies()  # 60 OD pairs

    print("===== PART 2.2(a) — SELFISH ROUTING vs SYSTEM-OPTIMAL =====")
    print(f"Network: {NETWORK_FILE} | |N|={len(N)} nodes | |A|={len(arcs)} arcs")
    print(f"Stations given (from 2.1(d)): {sorted(STATIONS)}")
    print(f"Capacity per unit Q = {Q_CAP} | Units installed = {sum(STATION_UNITS.values())}")
    print(f"Total OD pairs (A+B+C): {len(all_demands)}")
    print()

    # 1) System-optimal routing (central control)
    sys_tt, sys_time = system_optimal_routing(
        N=N, arcs=arcs, w=w,
        demands=all_demands,
        stations=STATIONS,
        station_units=STATION_UNITS,
        Q=Q_CAP,
        output_flag=0  # set 1 if you want Gurobi log
    )

    # 2) Selfish routing (shortest charge-feasible paths)
    selfish_tt, station_load, violations, _paths = selfish_routing(
        N=N, arcs=arcs, w=w,
        demands=all_demands,
        stations=STATIONS,
        station_units=STATION_UNITS,
        Q=Q_CAP
    )

    print("----- TOTAL TRAVEL TIME -----")
    print(f"System-optimal total travel time: {sys_tt:.4f}")
    print(f"Selfish routing total travel time: {selfish_tt:.4f}")
    print(f"Difference (selfish - optimal): {selfish_tt - sys_tt:.4f}")
    print()

    # 3) Capacity violations under selfish routing
    print("----- CAPACITY CHECK UNDER SELFISH ROUTING -----")
    any_violation = False
    for j in sorted(STATIONS):
        load = station_load.get(j, 0.0)
        cap = Q_CAP * STATION_UNITS.get(j, 0)
        if load > cap + 1e-9:
            any_violation = True
            print(f"Node {j}: load={load:.4f} > cap={cap:.4f} | violation={load-cap:.4f}")
        else:
            print(f"Node {j}: load={load:.4f} <= cap={cap:.4f}")

    if not any_violation:
        print("No capacity violations under selfish routing.")

    print()
    print("----- COMPUTING TIMES -----")
    print(f"System-optimal solve time (s): {sys_time:.4f}")
    print("Selfish routing time: depends on shortest-path runs (not timed here).")


if __name__ == "__main__":
    main()
