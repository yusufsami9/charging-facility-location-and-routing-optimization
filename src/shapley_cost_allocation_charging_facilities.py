# q_2_1_d.py
# Part 2(d) — Shapley value cost allocation for joint Charging Facility Location (CFL)

import time
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, FrozenSet

import gurobipy as gp
from gurobipy import GRB


# ---------------------------------------------------------
# 1) Read implicit network: arcs (i,j) with distance (ignored in model)
# ---------------------------------------------------------
def read_network(network_file: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    network.txt format (each line): i  j  distance
    Each line is interpreted as a directed arc (i,j) in the implicit network A.
    """
    arcs: List[Tuple[int, int]] = []
    nodes_set = set()

    with open(network_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            i, j = int(parts[0]), int(parts[1])
            arcs.append((i, j))
            nodes_set.add(i)
            nodes_set.add(j)

    N = sorted(nodes_set)
    return N, arcs


# ---------------------------------------------------------
# 2) Read OD pairs: (volume, origin, destination)
# ---------------------------------------------------------
def read_pairs(pairs_file: str) -> List[Tuple[float, int, int]]:
    """
    pairsX.txt format:
      volume origin destination
    First line may be a header; we skip non-numeric lines.
    """
    pairs: List[Tuple[float, int, int]] = []
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


# ---------------------------------------------------------
# 3) Solve capacitated CFL for a given coalition
# ---------------------------------------------------------
def solve_cfl_capacitated(
    N: List[int],
    arcs: List[Tuple[int, int]],
    pairs: List[Tuple[float, int, int]],
    Q: float = 10.0,
    output_flag: int = 0,
) -> Dict:
    """
    Capacitated CFL model:

      y_j ∈ Z_{>=0}  : number of charging units installed at node j
      x_{ij}^k ≥ 0   : flow of OD-pair k on arc (i,j), only for (i,j) in A

    min  sum_j y_j

    Flow conservation (each k, each node i):
      sum_{(i,j) in A} x_{ij}^k - sum_{(j,i) in A} x_{ji}^k = b_i^k
      b_{O_k}^k = V_k, b_{D_k}^k = -V_k, else 0

    Capacity at node j:
      sum_{k: D_k != j} sum_{(i,j) in A} x_{ij}^k <= Q * y_j
    """
    out_arcs = defaultdict(list)
    in_arcs = defaultdict(list)
    for (i, j) in arcs:
        out_arcs[i].append(j)
        in_arcs[j].append(i)

    K = list(range(len(pairs)))
    V = {k: pairs[k][0] for k in K}
    O = {k: pairs[k][1] for k in K}
    D = {k: pairs[k][2] for k in K}

    m = gp.Model("CFL_Capacitated")
    m.Params.OutputFlag = output_flag

    # Vars
    y = m.addVars(N, vtype=GRB.INTEGER, lb=0, name="y")

    # Create x only on existing arcs, but access as x[k,i,j]
    A = gp.tuplelist(arcs)
    x = m.addVars(K, A, vtype=GRB.CONTINUOUS, lb=0.0, name="x")

    # Objective
    m.setObjective(gp.quicksum(y[j] for j in N), GRB.MINIMIZE)

    # Flow conservation
    for k in K:
        ok, dk, vk = O[k], D[k], V[k]
        for i in N:
            rhs = 0.0
            if i == ok:
                rhs = vk
            elif i == dk:
                rhs = -vk

            m.addConstr(
                gp.quicksum(x[k, i, j] for j in out_arcs[i]) -
                gp.quicksum(x[k, j, i] for j in in_arcs[i]) == rhs,
                name=f"flow_k{k}_i{i}"
            )

    # Capacity constraints
    for j in N:
        expr = gp.LinExpr()
        for k in K:
            if D[k] == j:
                continue
            for i in in_arcs[j]:
                expr += x[k, i, j]
        m.addConstr(expr <= Q * y[j], name=f"cap_{j}")

    m.optimize()

    res = {
        "status": m.Status,
        "runtime_sec": float(m.Runtime),
        "obj_total_units": None,
        "open_locations": None,
        "open_locations_with_units": None,
    }

    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and m.SolCount > 0:
        total_units = sum(int(round(y[j].X)) for j in N)
        loc_units = [(j, int(round(y[j].X))) for j in N if y[j].X > 1e-6]
        loc_units.sort(key=lambda t: (-t[1], t[0]))
        locations = [j for (j, _) in loc_units]

        res["obj_total_units"] = float(total_units)
        res["open_locations"] = locations
        res["open_locations_with_units"] = loc_units

    return res


# ---------------------------------------------------------
# 4) Shapley value for 3 players given coalition costs c(S)
# ---------------------------------------------------------
def shapley_three_players(cost: Dict[FrozenSet[str], float]) -> Dict[str, float]:
    """
    Shapley value for players A,B,C. Assumes cost[frozenset()] = 0.
    For n=3:
      weights by |S| are: 0 -> 1/3, 1 -> 1/6, 2 -> 1/3
    """
    players = ["A", "B", "C"]
    weights = {0: 1/3, 1: 1/6, 2: 1/3}

    phi = {}
    for i in players:
        others = [p for p in players if p != i]
        val = 0.0
        for r in range(0, len(others) + 1):
            for S_tuple in combinations(others, r):
                S = frozenset(S_tuple)
                val += weights[len(S)] * (cost[S | {i}] - cost[S])
        phi[i] = val
    return phi


# ---------------------------------------------------------
# 5) Main
# ---------------------------------------------------------
def main():
    NETWORK_FILE = "data/network.txt"
    Q_CAP = 10.0

    N, arcs = read_network(NETWORK_FILE)

    pairs_company = {
        "A": read_pairs("data/pairsA.txt"),
        "B": read_pairs("data/pairsB.txt"),
        "C": read_pairs("data/pairsC.txt"),
    }

    coalitions = [
        frozenset(["A"]),
        frozenset(["B"]),
        frozenset(["C"]),
        frozenset(["A", "B"]),
        frozenset(["A", "C"]),
        frozenset(["B", "C"]),
        frozenset(["A", "B", "C"]),
    ]

    print("===== PART 2(d) — SHAPLEY VALUE COST ALLOCATION =====")
    print(f"Network: {NETWORK_FILE} | |N|={len(N)} nodes | |A|={len(arcs)} arcs")
    print(f"Facility capacity: Q = {Q_CAP}")
    print()

    coalition_cost: Dict[FrozenSet[str], float] = {frozenset(): 0.0}
    coalition_runtime: Dict[FrozenSet[str], float] = {}
    coalition_locations: Dict[FrozenSet[str], List[int]] = {}

    for S in coalitions:
        label = "".join(sorted(S))
        merged_pairs: List[Tuple[float, int, int]] = []
        for comp in sorted(S):
            merged_pairs.extend(pairs_company[comp])

        print(f"--- Coalition {label} ---")
        print(f"Companies: {sorted(S)} | #OD pairs total = {len(merged_pairs)}")

        res = solve_cfl_capacitated(
            N=N,
            arcs=arcs,
            pairs=merged_pairs,
            Q=Q_CAP,
            output_flag=0,  # set 1 if you want Gurobi log
        )

        if res["open_locations_with_units"] is None:
            print(f"Status: {res['status']} (no solution)")
            print(f"Computing time (s): {res['runtime_sec']:.4f}")
            print()
            continue

        coalition_cost[S] = float(res["obj_total_units"])
        coalition_runtime[S] = float(res["runtime_sec"])
        coalition_locations[S] = res["open_locations"]

        print("Status: solved")
        print(f"Computing time (s): {res['runtime_sec']:.4f}")
        print(f"Optimal cost (total units): {int(res['obj_total_units'])}")
        print(f"Optimal charging locations: {res['open_locations']}")
        print()

    # Check all needed coalitions exist
    needed = [
        frozenset(),
        frozenset(["A"]), frozenset(["B"]), frozenset(["C"]),
        frozenset(["A", "B"]), frozenset(["A", "C"]), frozenset(["B", "C"]),
        frozenset(["A", "B", "C"]),
    ]
    for S in needed:
        if S not in coalition_cost:
            raise RuntimeError(f"Missing cost for coalition {S}. Cannot compute Shapley.")

    phi = shapley_three_players(coalition_cost)

    print("===== COALITION OPTIMAL COSTS c(S) =====")
    for S in needed:
        label = "∅" if len(S) == 0 else "".join(sorted(S))
        print(f"c({label}) = {coalition_cost[S]:.2f}")
    print()

    print("===== SHAPLEY COST ALLOCATION =====")
    for p in ["A", "B", "C"]:
        print(f"phi_{p} = {phi[p]:.4f}")
    print(f"Sum phi = {sum(phi.values()):.4f} | c(ABC) = {coalition_cost[frozenset(['A','B','C'])]:.4f}")
    print()

    print("===== SAVINGS VS. BUILDING ALONE =====")
    for p in ["A", "B", "C"]:
        standalone = coalition_cost[frozenset([p])]
        savings = standalone - phi[p]
        print(f"{p}: standalone c({p})={standalone:.2f} | pays phi={phi[p]:.4f} | savings={savings:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
