
# QUESTION 1.1(a)

import gurobipy as gp
from gurobipy import GRB


def solve_route_based_cvrp(routes_file, num_vehicles=5):


    # --- 1. DATA LOADING ---
    print(f"Loading routes from {routes_file}...")

    routes = []
    all_customers = set()
    # temp_customer_routes[customer_id] = [route_idx1, route_idx2...]
    temp_customer_routes = {}

    try:
        with open(routes_file, 'r') as f:
            for r_idx, line in enumerate(f):
                tokens = line.strip().split()

                # Safety check: Skip empty lines
                if not tokens:
                    continue

                # Parse cost (Index 0) and path (Index 2 to end)
                # Note: Index 1 is Load, which we skip.
                cost = float(tokens[0])
                path_nodes = [int(t) for t in tokens[2:]]

                # Exclude 0 (Depot) to get only customers
                customers_in_route = [node for node in path_nodes if node != 0]

                routes.append({'id': r_idx, 'cost': cost, 'customers': customers_in_route})

                # Update mapping: Customer -> List of Routes serving it
                for cust in customers_in_route:
                    all_customers.add(cust)
                    if cust not in temp_customer_routes:
                        temp_customer_routes[cust] = []
                    temp_customer_routes[cust].append(r_idx)

        print(f"Successfully loaded {len(routes)} routes.")

        # Sort customers for consistent ordering
        sorted_customers = sorted(list(all_customers))
        num_customers_found = len(sorted_customers)
        print(f"Detected {num_customers_found} unique customers: {sorted_customers}")

        # Warning if customer count does not match the assignment expectation
        if num_customers_found != 25:
            print(f"WARNING: Assignment expects 25 customers, but found {num_customers_found}!")

    except FileNotFoundError:
        print(f"Error: File {routes_file} not found.")
        return None
    except ValueError as e:
        print(f"Error processing file data: {e}")
        return None

    # --- 2. MODEL FORMULATION ---
    model = gp.Model("CVRP_Route_Based_Part_A")

    # Decision Variables: x[r] = 1 if route r is selected
    x = model.addVars(len(routes), vtype=GRB.BINARY, name="x")

    # Objective: Minimize Total Cost
    model.setObjective(gp.quicksum(routes[r]['cost'] * x[r] for r in range(len(routes))), GRB.MINIMIZE)

    # Constraint 1: Set Partitioning
    # Each detected customer must be visited exactly once
    for cust_id in sorted_customers:
        model.addConstr(
            gp.quicksum(x[r] for r in temp_customer_routes[cust_id]) == 1,
            name=f"Cover_Customer_{cust_id}"
        )

    # Constraint 2: Vehicle Limit (Exactly 5 vehicles)
    model.addConstr(x.sum() == num_vehicles, name="Vehicle_Count")

    # --- 3. OPTIMIZATION ---
    model.setParam('TimeLimit', 600)
    print("\nStarting optimization...")
    model.optimize()

    # --- 4. REPORTING ---
    # Check if at least one feasible solution was found
    if model.SolCount > 0:
        print("\n" + "=" * 30)
        print(" RESULTS - Part 1.1 (a)")
        print("=" * 30)

        obj_val = model.ObjVal
        runtime = model.Runtime
        gap = model.MIPGap

        selected_route_indices = [r for r in range(len(routes)) if x[r].X > 0.5]
        selected_costs = [routes[r]['cost'] for r in selected_route_indices]

        # Calculate Range (Fairness metric for Part B comparison)
        if selected_costs:
            range_val = max(selected_costs) - min(selected_costs)
        else:
            range_val = 0.0

        print(f"Objective Value (z*):         {obj_val:.2f}")
        print(f"Runtime:                      {runtime:.4f} seconds")
        print(f"Optimality Gap:               {gap:.4f}")
        print(f"Range (Max - Min):            {range_val:.2f}")
        print("-" * 30)
        print(f"Selected Route Costs: {selected_costs}")
        print("-" * 30)

        return obj_val
    else:
        print("\nNo feasible solution found within the time limit.")
        return None


if __name__ == "__main__":
    solve_route_based_cvrp("data/routes.txt")
