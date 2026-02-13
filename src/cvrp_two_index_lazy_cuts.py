# QUESTION 1.2(a)


import gurobipy as gp
from gurobipy import GRB
import math
import time



def read_instance(filename: str):
    """
    Reads instance.txt with the following format:
      - First number: vehicle capacity Q
      - Next 25 numbers: demands of customers 1..25 (depot demand is 0)
      - Next (26 x 26) numbers: distance matrix for nodes 0..25

    Returns:
      Q          : int
      demands    : list of length 26, demands[0] = 0
      dist_matrix: dict {(i,j): distance}
      n_nodes    : int (26)
    """
    print(f"Reading {filename}...")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
            tokens = content.split()
            it = iter(tokens)

        # Capacity
        Q = int(next(it))

        # Fixed size in this assignment
        n_customers = 25
        n_nodes = n_customers + 1  # includes depot 0

        # Demands: customers 1..25, prepend depot demand 0
        demands_raw = [int(next(it)) for _ in range(n_customers)]
        demands = [0] + demands_raw

        # Distance matrix
        dist_matrix = {}
        for i in range(n_nodes):
            for j in range(n_nodes):
                dist_matrix[(i, j)] = float(next(it))

        return Q, demands, dist_matrix, n_nodes

    except Exception as e:
        print(f"ERROR while reading file: {e}")
        return None, None, None, None


# ============================================================
# 2) CALLBACK FOR LAZY CONSTRAINTS
# ============================================================
def subtour_and_capacity_callback(model, where):
    """
    This callback is triggered during the MIP solving process.

    The important line is:
        if where == GRB.Callback.MIPSOL:
    Meaning:
      - "MIPSOL" event happens whenever Gurobi finds a new *integer feasible* solution.
      - At that moment, we inspect the current solution and add "lazy constraints"
        if we detect:
          (1) subtours not connected to the depot (illegal routes),
          (2) depot-connected routes that violate capacity (demand > Q).

    Lazy constraints are constraints we do not add upfront; instead we add them only
    when we see they are violated by an incumbent integer solution.
    """
    if where == GRB.Callback.MIPSOL:
        # Get current integer solution values for x[i,j]
        vals = model.cbGetSolution(model._vars)

        # Extract selected directed edges (arcs) where x[i,j] = 1
        selected_edges = gp.tuplelist((i, j) for (i, j) in model._vars.keys() if vals[i, j] > 0.5)

        # Build adjacency list: adj[i] = [j] such that i -> j is chosen
        adj = {i: [] for i in range(model._n)}
        for i, j in selected_edges:
            adj[i].append(j)

        # ------------------------------------------------------------
        # STEP 1) SUBTOUR CHECK: nodes not reachable from the depot
        # ------------------------------------------------------------
        # BFS from depot 0 to find all nodes reachable in this solution graph
        visited_from_depot = {0}
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if v not in visited_from_depot:
                    visited_from_depot.add(v)
                    queue.append(v)

        # If not all nodes are reachable from the depot, we have disconnected cycles/subtours
        if len(visited_from_depot) < model._n:
            unvisited = set(range(model._n)) - visited_from_depot

            # For each disconnected component, trace a cycle and add a subtour elimination cut
            while unvisited:
                start = next(iter(unvisited))
                cycle = [start]
                curr = start

                while True:
                    if curr in unvisited:
                        unvisited.remove(curr)

                    if not adj[curr]:
                        break  # defensive; should not happen in a proper degree solution

                    curr = adj[curr][0]
                    if curr == start or curr in cycle:
                        break
                    cycle.append(curr)

                # Classic subtour elimination cut for directed arcs:
                # sum_{u in S} sum_{v in S, v!=u} x_uv <= |S| - 1
                model.cbLazy(
                    gp.quicksum(model._vars[u, v] for u in cycle for v in cycle if u != v)
                    <= len(cycle) - 1
                )

        # ------------------------------------------------------------
        # STEP 2) CAPACITY CHECK: depot-started routes exceeding Q
        # ------------------------------------------------------------
        # Each arc leaving the depot corresponds to one route start.
        if 0 in adj:
            for start_node in adj[0]:
                route_nodes = [start_node]
                curr = start_node
                current_load = model._demands[curr]

                # Follow the route until it returns to depot (0)
                while True:
                    if not adj[curr]:
                        break
                    next_node = adj[curr][0]
                    if next_node == 0:
                        break  # route ends at depot

                    route_nodes.append(next_node)
                    current_load += model._demands[next_node]
                    curr = next_node

                # If route demand exceeds capacity, add a rounded capacity inequality
                if current_load > model._Q:
                    r_S = math.ceil(current_load / model._Q)

                    # Rounded capacity cut:
                    # sum_{u in S} sum_{v in S, v!=u} x_uv <= |S| - r(S)
                    model.cbLazy(
                        gp.quicksum(model._vars[u, v] for u in route_nodes for v in route_nodes if u != v)
                        <= len(route_nodes) - r_S
                    )


# ============================================================
# 3) BUILD & SOLVE MODEL (FLEET SIZE FIXED TO 5)
# ============================================================
def solve_part_a():
    filename = "data/instance.txt"
    Q, demands, dists, n = read_instance(filename)
    if Q is None:
        return

    NUM_VEHICLES = 5

    print(f"\nInstance loaded: {n - 1} customers, capacity Q={Q}")
    print(f"Fleet size fixed to: {NUM_VEHICLES}")

    m = gp.Model("CVRP_TwoIndex_LazyCuts")

    # Decision variables: x[i,j] = 1 if we travel from i to j
    x = m.addVars(dists.keys(), vtype=GRB.BINARY, name="x")

    # Objective: minimize total distance
    m.setObjective(gp.quicksum(dists[i, j] * x[i, j] for (i, j) in dists.keys()), GRB.MINIMIZE)

    # 1) Degree constraints for customers: exactly 1 outgoing and 1 incoming
    for i in range(1, n):
        m.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1, name=f"out_{i}")
        m.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == 1, name=f"in_{i}")

    # 2) Depot balance: number of arcs leaving depot equals number entering depot
    m.addConstr(
        gp.quicksum(x[0, j] for j in range(1, n)) == gp.quicksum(x[j, 0] for j in range(1, n)),
        name="depot_balance",
    )

    # 3) Fleet size: exactly NUM_VEHICLES routes leave the depot
    m.addConstr(gp.quicksum(x[0, j] for j in range(1, n)) == NUM_VEHICLES, name="FleetSize")

    # 4) Disallow self-loops
    for i in range(n):
        x[i, i].LB = 0
        x[i, i].UB = 0

    # Attach data to model for callback usage
    m._vars = x
    m._n = n
    m._Q = Q
    m._demands = demands

    # Enable lazy constraints (required if we use cbLazy)
    m.Params.LazyConstraints = 1

    # Optional: set a time limit (uncomment if needed)
    # m.setParam("TimeLimit", 600)

    print("\nStarting optimization...")
    start_time = time.time()
    m.optimize(subtour_and_capacity_callback)
    end_time = time.time()

    # -------------------------
    # Reporting
    # -------------------------
    if m.SolCount > 0:
        status_str = "optimal" if m.status == GRB.OPTIMAL else ("time_limit" if m.status == GRB.TIME_LIMIT else str(m.status))

        print("\n" + "=" * 50)
        print("PART 1.2(a) - RESULTS")
        print("=" * 50)
        print(f"Status:            {status_str}")
        print(f"Objective (cost):  {m.objVal:.2f}")
        print(f"Runtime (s):       {end_time - start_time:.4f}")
        print(f"Optimality gap:    {m.MIPGap * 100:.2f}%")

        # Extract chosen arcs
        vals = m.getAttr("x", x)
        selected = gp.tuplelist((i, j) for (i, j) in vals.keys() if vals[i, j] > 0.5)

        # Print routes
        print("\nOptimal routes:")
        starts = [j for (i, j) in selected if i == 0]

        for idx, start_node in enumerate(starts, 1):
            route = [0, start_node]
            curr = start_node
            load = demands[start_node]

            while curr != 0:
                found = False
                for (u, v) in selected:
                    if u == curr:
                        route.append(v)
                        curr = v
                        if v != 0:
                            load += demands[v]
                        found = True
                        break
                if not found:
                    break

            print(f"  Route {idx}: {route} | Load: {load}/{Q}")

        # Quick consistency check: should be exactly NUM_VEHICLES starts
        if len(starts) != NUM_VEHICLES:
            print(f"\nWARNING: Found {len(starts)} depot starts, expected {NUM_VEHICLES}. Check route extraction.")

    else:
        print(f"\nNo feasible solution found. Status: {m.status}")


if __name__ == "__main__":
    solve_part_a()
