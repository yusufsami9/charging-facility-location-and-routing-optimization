# q_1_2_c.py
# Part 1.2(c) — Scenario-based two-index CVRP + cutting planes (up to 5 iterations)


import gurobipy as gp
from gurobipy import GRB
import random
import math
import json
from typing import Dict, List, Tuple, Any



def read_instance_txt(path: str, n_customers: int = 25):

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) < 2 + (n_customers + 1):
        raise ValueError("instance.txt seems too short / malformed.")

    Q = int(float(lines[0].split()[0]))

    demand_tokens = lines[1].split()
    if len(demand_tokens) != n_customers:
        raise ValueError(f"Expected {n_customers} demands, got {len(demand_tokens)}.")

    q_nom = {0: 0}
    for i in range(1, n_customers + 1):
        q_nom[i] = int(float(demand_tokens[i - 1]))

    N = list(range(0, n_customers + 1))
    c = {}
    matrix_lines = lines[2: 2 + (n_customers + 1)]
    if len(matrix_lines) != (n_customers + 1):
        raise ValueError("Distance matrix seems incomplete.")

    for i, row in enumerate(matrix_lines):
        vals = row.split()
        if len(vals) != (n_customers + 1):
            raise ValueError(f"Distance row {i} expected {n_customers + 1} values, got {len(vals)}.")
        for j, v in enumerate(vals):
            c[(i, j)] = int(float(v))

    return N, Q, q_nom, c


# -----------------------------
# 2) Extract routes from x_ij
# -----------------------------
def extract_routes_from_x(x, N: List[int], depot: int = 0) -> List[List[int]]:
    """
    Build routes by following arcs starting from depot.
    Assumes each customer has exactly 1 outgoing arc.
    """
    succ = {}
    for i in N:
        for j in N:
            if i != j and x[i, j].X > 0.5:
                succ[i] = j
                break

    starts = [j for j in N if j != depot and x[depot, j].X > 0.5]

    routes = []
    for s in starts:
        route = [depot, s]
        cur = s
        while cur != depot:
            nxt = succ.get(cur, None)
            if nxt is None:
                break
            route.append(nxt)
            cur = nxt
        routes.append(route)

    # Make route listing stable/readable: sort by first customer after depot
    routes.sort(key=lambda r: r[1] if len(r) > 1 else 10**9)
    return routes


def route_load(route: List[int], demand: Dict[int, int]) -> int:
    return sum(demand[node] for node in route if node != 0)


def total_capacity_violation(routes: List[List[int]], demand: Dict[int, int], Q: int) -> int:
    """
    TOTAL violation for one scenario = sum over routes max(0, load(route) - Q)
    """
    tot = 0
    for r in routes:
        load = route_load(r, demand)
        tot += max(0, load - Q)
    return tot


def print_routes(routes: List[List[int]], q_nom: Dict[int, int], Q: int) -> None:
    """
    Prints routes + planned (nominal) load, to help you later pick a violating route/scenario.
    """
    for k, r in enumerate(routes, 1):
        load = route_load(r, q_nom)
        print(f"    Route {k}: {r} | nominal load = {load}/{Q}")


# -----------------------------
# 3) Scenario-based model
# -----------------------------
def solve_scenario_based_model(
    N: List[int],
    Q: int,
    c: Dict[Tuple[int, int], int],
    scenarios: List[Dict[int, int]],
    num_vehicles: int = 5,
    time_limit: int = 600,
    output_flag: int = 1,
):
    """
    Scenario-based two-index CVRP using MTZ load constraints, scenario-indexed.

    For each scenario s:
        u[s,i] - u[s,j] + Q*x[i,j] <= Q - q_s[j]   for all i in N, j in N\{0}, i!=j

    Routing variables x[i,j] are shared across scenarios.
    """
    depot = 0
    customers = [i for i in N if i != depot]
    S_idx = list(range(len(scenarios)))

    m = gp.Model("CVRP_ScenarioBased")
    m.Params.OutputFlag = output_flag
    m.Params.TimeLimit = time_limit

    # Routing vars
    x = m.addVars(N, N, vtype=GRB.BINARY, name="x")
    for i in N:
        x[i, i].UB = 0

    # Scenario-dependent load vars: u[s,i]
    u = m.addVars(S_idx, N, vtype=GRB.CONTINUOUS, name="u")
    for s in S_idx:
        u[s, depot].LB = 0
        u[s, depot].UB = 0
        for i in customers:
            u[s, i].LB = scenarios[s][i]
            u[s, i].UB = Q

    # Objective: minimize total distance
    m.setObjective(gp.quicksum(c[i, j] * x[i, j] for i in N for j in N), GRB.MINIMIZE)

    # Degree constraints for customers
    for i in customers:
        m.addConstr(gp.quicksum(x[i, j] for j in N if j != i) == 1, name=f"out_{i}")
    for j in customers:
        m.addConstr(gp.quicksum(x[i, j] for i in N if i != j) == 1, name=f"in_{j}")

    # Fleet size at depot
    m.addConstr(gp.quicksum(x[depot, j] for j in customers) == num_vehicles, name="depot_out")
    m.addConstr(gp.quicksum(x[i, depot] for i in customers) == num_vehicles, name="depot_in")

    # Scenario-based MTZ constraints
    for s in S_idx:
        q_s = scenarios[s]
        for i in N:
            for j in customers:
                if i != j:
                    m.addConstr(
                        u[s, i] - u[s, j] + Q * x[i, j] <= Q - q_s[j],
                        name=f"mtz_s{s}_{i}_{j}"
                    )

    m.optimize()
    return m, x


# -----------------------------
# 4) Cutting-plane loop (up to 5 iterations)
# -----------------------------
def cutting_plane_cvrp(
    instance_file: str = "instance.txt",
    n_customers: int = 25,
    num_vehicles: int = 5,
    time_limit: int = 600,
    iterations: int = 5,
    k_sim: int = 1000,
    seed: int = 42,
    output_flag: int = 0,
    save_json_path: str = "cp_solutions_1_2c.json",
    print_routes_each_iter: bool = True,
):
    N, Q, q_nom, c = read_instance_txt(instance_file, n_customers=n_customers)

    # Scenario set S: start with nominal demand only
    S: List[Dict[int, int]] = [q_nom.copy()]

    rng = random.Random(seed)

    results: List[Dict[str, Any]] = []

    print(f"Loaded instance: customers={n_customers}, Q={Q}, vehicles={num_vehicles}")
    print(f"Cutting-plane settings: iterations={iterations}, k_sim={k_sim}, seed={seed}")
    print("-" * 98)
    print(f"{'iter':>4} | {'|S|':>3} | {'cost':>8} | {'gap':>7} | {'time(s)':>8} | "
          f"{'viol_scen/1000':>14} | {'avg_viol':>8} | {'max_viol':>8}")
    print("-" * 98)

    for it in range(1, iterations + 1):
        # 1) Solve scenario-based model with current S
        m, x = solve_scenario_based_model(
            N=N, Q=Q, c=c, scenarios=S,
            num_vehicles=num_vehicles,
            time_limit=time_limit,
            output_flag=output_flag
        )

        if m.SolCount == 0:
            print(f"{it:4d} | {len(S):3d} | {'-':>8} | {'-':>7} | {m.Runtime:8.2f} | "
                  f"{'-':>14} | {'-':>8} | {'-':>8}")
            print("No feasible solution found.")
            break

        cost = float(m.ObjVal)
        gap = float(m.MIPGap) if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else float("nan")
        runtime = float(m.Runtime)
        status = int(m.status)

        # Extract routes for this iteration
        routes = extract_routes_from_x(x, N, depot=0)

        if print_routes_each_iter:
            print(f"  Iteration {it}: selected routes")
            print_routes(routes, q_nom=q_nom, Q=Q)

        # 2) Simulate demand realizations and compute TOTAL violation per scenario
        violations: List[int] = []
        best_violation = -1
        best_scenario: Dict[int, int] | None = None

        for _ in range(k_sim):
            q_sim = {0: 0}
            for i in range(1, n_customers + 1):
                lo = math.floor(0.9 * q_nom[i])
                hi = math.ceil(1.1 * q_nom[i])
                q_sim[i] = rng.randint(lo, hi)

            v = total_capacity_violation(routes, q_sim, Q)
            violations.append(v)
            if v > best_violation:
                best_violation = v
                best_scenario = q_sim

        num_viol_scen = sum(1 for v in violations if v > 0)
        avg_viol = sum(violations) / len(violations)
        max_viol = max(violations)

        print(f"{it:4d} | {len(S):3d} | {cost:8.2f} | {gap:7.2%} | {runtime:8.2f} | "
              f"{num_viol_scen:14d} | {avg_viol:8.2f} | {max_viol:8d}")

        # Save iteration result (so you can do Part 1.2(d) without rerunning)
        routes_with_nom_load = [
            {"route": r, "nominal_load": int(route_load(r, q_nom))}
            for r in routes
        ]
        results.append({
            "iter": it,
            "S_size": len(S),
            "status": status,
            "objective_cost": cost,
            "mip_gap": gap,
            "runtime_sec": runtime,
            "viol_scenarios_out_of_k": num_viol_scen,
            "avg_total_violation": avg_viol,
            "max_total_violation": int(max_viol),
            "routes": routes_with_nom_load,
        })

        # 3) Add most violated scenario (if any)
        if max_viol > 0 and best_scenario is not None:
            S.append(best_scenario)
        else:
            print("No violating scenario found in simulation; stopping early.")
            break

    # Persist results to JSON
    payload = {
        "instance_file": instance_file,
        "n_customers": n_customers,
        "capacity_Q": Q,
        "num_vehicles": num_vehicles,
        "iterations_requested": iterations,
        "k_sim": k_sim,
        "seed": seed,
        "results": results,
    }
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved solutions to: {save_json_path}")
    return payload


if __name__ == "__main__":
    cutting_plane_cvrp(
        instance_file="data/instance.txt",
        n_customers=25,
        num_vehicles=5,
        time_limit=600,
        iterations=5,
        k_sim=1000,
        seed=42,
        output_flag=0,                 # keep console clean
        save_json_path="report/cp_solutions_1_2c.json",
        print_routes_each_iter=True,   # IMPORTANT: prints routes per iteration
    )
