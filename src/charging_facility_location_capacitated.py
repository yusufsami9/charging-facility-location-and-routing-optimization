# q_2_1_c.py
# Part 2(c) — Capacitated Charging Facility Location (CFL)


import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
from typing import Dict, List, Tuple


# ---------------------------------------------------------
# 1) Read implicit network: arcs (i,j) with distance (ignored in model)
# ---------------------------------------------------------
def read_network(network_file: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    network.txt format (each line): i  j  distance
    We interpret each line as a directed arc (i,j) in the implicit network A.
    """
    arcs = []
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


# ---------------------------------------------------------
# 3) Solve capacitated CFL for one company
# ---------------------------------------------------------
def solve_cfl_capacitated(
    N: List[int],
    arcs: List[Tuple[int, int]],
    pairs: List[Tuple[float, int, int]],
    Q: float = 10.0,
    output_flag: int = 0,
) -> Dict:

    # adjacency for building constraints fast
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

    # Decision variables
    y = m.addVars(N, vtype=GRB.INTEGER, lb=0, name="y")

    # IMPORTANT FIX: create x with 3 indices (k,i,j)
    x = m.addVars(K, N, N, vtype=GRB.CONTINUOUS, lb=0.0, name="x")

    # Disallow arcs not in A by fixing x[k,i,j] = 0 if (i,j) not in arcs
    # (This keeps indexing simple.)
    Aset = set(arcs)
    for k in K:
        for i in N:
            for j in N:
                if (i, j) not in Aset:
                    x[k, i, j].UB = 0.0

    # Objective: minimize total installed charging units
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

    # Capacity constraints at charging nodes
    for j in N:
        expr = gp.LinExpr()
        for k in K:
            if D[k] == j:
                continue  # destination: no charging counted
            for i in in_arcs[j]:  # i such that (i,j) in A
                expr += x[k, i, j]
        m.addConstr(expr <= Q * y[j], name=f"cap_{j}")

    # Optimize
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

        res["obj_total_units"] = total_units
        res["open_locations"] = locations
        res["open_locations_with_units"] = loc_units

    return res


# ---------------------------------------------------------
# 4) Run A, B, C and print required outputs
# ---------------------------------------------------------
def main():
    NETWORK_FILE = "data/network.txt"
    Q_CAP = 10.0

    N, arcs = read_network(NETWORK_FILE)

    print("===== PART 2(c) — CAPACITATED CFL PER COMPANY =====")
    print(f"Network: {NETWORK_FILE} | |N|={len(N)} nodes | |A|={len(arcs)} arcs")
    print(f"Facility capacity: Q = {Q_CAP}")
    print()

    for X in ["A", "B", "C"]:
        pairs_file = f"data/pairs{X}.txt"
        pairs = read_pairs(pairs_file)

        print(f"--- Company {X} ---")
        print(f"Pairs file: {pairs_file} | #OD pairs = {len(pairs)}")

        res = solve_cfl_capacitated(
            N=N,
            arcs=arcs,
            pairs=pairs,
            Q=Q_CAP,
            output_flag=0,  # set 1 to see Gurobi log
        )

        if res["status"] == GRB.INFEASIBLE:
            print("Status: INFEASIBLE")
            print(f"Computing time (s): {res['runtime_sec']:.4f}")
            print()
            continue

        if res["open_locations_with_units"] is None:
            print(f"Status: {res['status']} (no solution found)")
            print(f"Computing time (s): {res['runtime_sec']:.4f}")
            print()
            continue



        print("Status: solved")
        print(f"Computing time (s): {res['runtime_sec']:.4f}")
        print(f"Optimal number of charging facilities (total units): {res['obj_total_units']}")
        print(f"Optimal charging locations (nodes with y>0): {res['open_locations']}")
        print("Locations with units (node: units):")
        print("  " + ", ".join([f"{j}: {u}" for (j, u) in res["open_locations_with_units"]]))
        print()

    print("Done.")


if __name__ == "__main__":
    main()
