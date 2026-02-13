
# QUESTION 1.1(b)

import gurobipy as gp
from gurobipy import GRB


def solve_fairness_cvrp(routes_file, optimal_z_star=8831.0, epsilon=0.05, num_vehicles=5):
    """
    Solves the CVRP using a Vehicle-Index formulation to minimize Range (Fairness).
    Part 1.1 (b) of the assignment.

    Parameters:
    - optimal_z_star: The optimal cost found in Part A (8831).
    - epsilon: The allowable budget relaxation (5% = 0.05).
    """

    # --- 1. DATA LOADING (Robust Version) ---
    print(f"Loading routes from {routes_file}...")

    routes = []
    all_customers = set()
    # temp_customer_routes[customer_id] = [route_idx1, route_idx2...]
    temp_customer_routes = {}

    try:
        with open(routes_file, 'r') as f:
            for r_idx, line in enumerate(f):
                tokens = line.strip().split()

                # Safety: Skip empty lines
                if not tokens:
                    continue

                # Parse cost and path
                cost = float(tokens[0])
                path_nodes = [int(t) for t in tokens[2:]]

                # Exclude Depot (0)
                customers_in_route = [node for node in path_nodes if node != 0]

                routes.append({'id': r_idx, 'cost': cost, 'customers': customers_in_route})

                # Update mapping
                for cust in customers_in_route:
                    all_customers.add(cust)
                    if cust not in temp_customer_routes:
                        temp_customer_routes[cust] = []
                    temp_customer_routes[cust].append(r_idx)

        print(f"Successfully loaded {len(routes)} routes.")

        sorted_customers = sorted(list(all_customers))
        print(f"Detected {len(sorted_customers)} unique customers.")

    except FileNotFoundError:
        print(f"Error: File {routes_file} not found.")
        return None
    except ValueError as e:
        print(f"Error processing file data: {e}")
        return None

    # --- 2. MODEL FORMULATION (Vehicle-Index) ---
    model = gp.Model("CVRP_Fairness_Part_B")

    # Budget Calculation: (1 + epsilon) * z*
    max_budget = (1 + epsilon) * optimal_z_star
    print(f"Budget Limit set to: {max_budget:.2f} (Base: {optimal_z_star}, +{epsilon * 100}%)")

    # Decision Variables
    # x[r, k] = 1 if route r is assigned to vehicle k
    # This creates (Num_Routes * Num_Vehicles) binary variables
    x = model.addVars(len(routes), num_vehicles, vtype=GRB.BINARY, name="x")

    # Auxiliary Variables for Fairness (Eq 4a, 4e in slide)
    # w_max = max workload (eta), w_min = min workload (gamma)
    w_max = model.addVar(vtype=GRB.CONTINUOUS, name="w_max")
    w_min = model.addVar(vtype=GRB.CONTINUOUS, name="w_min")

    # Objective: Minimize Range (Eq 4a)
    model.setObjective(w_max - w_min, GRB.MINIMIZE)

    # Constraint 1: Set Partitioning (Eq 4b)
    # Every customer must be visited exactly once by EXACTLY ONE vehicle
    for cust_id in sorted_customers:
        relevant_routes = temp_customer_routes[cust_id]
        model.addConstr(
            gp.quicksum(x[r, k] for r in relevant_routes for k in range(num_vehicles)) == 1,
            name=f"Cover_Customer_{cust_id}"
        )

    # Constraint 2: One Route Per Vehicle (Eq 4d)
    # Each vehicle k must select exactly one route from the pool
    for k in range(num_vehicles):
        model.addConstr(
            gp.quicksum(x[r, k] for r in range(len(routes))) == 1,
            name=f"One_Route_Vehicle_{k}"
        )

    # Constraint 3: Budget Constraint (Eq 4c)
    # Total cost of all selected routes must not exceed (1 + epsilon) * z*
    total_cost_expr = gp.quicksum(routes[r]['cost'] * x[r, k]
                                  for r in range(len(routes))
                                  for k in range(num_vehicles))

    model.addConstr(total_cost_expr <= max_budget, name="Budget_Constraint")

    # Constraint 4: Define Max and Min Workload (Eq 4e)
    # For every vehicle k, its load must be <= w_max and >= w_min
    for k in range(num_vehicles):
        vehicle_load = gp.quicksum(routes[r]['cost'] * x[r, k] for r in range(len(routes)))

        model.addConstr(vehicle_load <= w_max, name=f"Max_Load_Vehicle_{k}")
        model.addConstr(vehicle_load >= w_min, name=f"Min_Load_Vehicle_{k}")

    # --- 3. OPTIMIZATION ---
    # Set time limit (Fairness models are harder to solve than Part A)
    model.setParam('TimeLimit', 600)
    print("\nStarting optimization for Fairness...")
    model.optimize()

    # --- 4. REPORTING ---
    if model.SolCount > 0:
        print("\n" + "=" * 30)
        print(" RESULTS - Part 1.1 (b) Fairness")
        print("=" * 30)

        obj_val = model.ObjVal  # This is the Range (w_max - w_min)
        runtime = model.Runtime
        gap = model.MIPGap

        # Retrieve individual vehicle loads
        vehicle_loads = []
        total_actual_cost = 0

        print("\nVehicle Assignments:")
        for k in range(num_vehicles):
            # Find which route this vehicle selected
            for r in range(len(routes)):
                if x[r, k].X > 0.5:
                    cost = routes[r]['cost']
                    vehicle_loads.append(cost)
                    total_actual_cost += cost
                    print(f"  Vehicle {k}: Route {r} (Cost: {cost})")
                    break  # Since each vehicle has only 1 route

        min_load = min(vehicle_loads)
        max_load = max(vehicle_loads)
        range_val = max_load - min_load

        print("-" * 30)
        print(f"Objective (Range):            {obj_val:.2f}")
        print(f"Calculated Range (Check):     {range_val:.2f}")
        print(f"Total Cost:                   {total_actual_cost:.2f} (Limit: {max_budget:.2f})")
        print(f"Optimality Gap:               {gap:.4f}")
        print(f"Runtime:                      {runtime:.4f} seconds")
        print("-" * 30)
        print(f"Loads: {vehicle_loads}")
        print("-" * 30)

        return obj_val
    else:
        print("\nNo feasible solution found within the time limit.")
        return None


if __name__ == "__main__":
    # Ensure you use the exact optimal value from Part A here (8831)
    solve_fairness_cvrp("data/routes.txt", optimal_z_star=8831.0)
