# QUESTION 1.1(e) - Last-customer formulation with summary table output

import gurobipy as gp
from gurobipy import GRB


def solve_last_customer_fairness_summary(
    routes_file: str,
    z_star: float,
    eps_list,
    num_vehicles: int = 5,
    depot_id: int = 0,
    time_limit: int = 600,
):
    # -----------------------
    # 1) DATA LOADING
    # -----------------------
    print(f"Loading routes from {routes_file}...")

    routes = []
    all_customers = set()
    temp_customer_routes = {}  # customer -> list of routes that contain customer

    try:
        with open(routes_file, "r", encoding="utf-8") as f:
            for r_idx, line in enumerate(f):
                tokens = line.strip().split()
                if not tokens:
                    continue

                cost = float(tokens[0])
                path_nodes = [int(t) for t in tokens[2:]]

                customers_in_route = [node for node in path_nodes if node != depot_id]
                if len(customers_in_route) == 0:
                    continue

                routes.append({"id": r_idx, "cost": cost, "customers": customers_in_route})

                for cust in customers_in_route:
                    all_customers.add(cust)
                    temp_customer_routes.setdefault(cust, []).append(r_idx)

        customers = sorted(all_customers)

        print(f"Loaded {len(routes)} routes.")
        print(f"Detected {len(customers)} customers: {customers}")
        if len(customers) != 25:
            print(f"WARNING: Assignment expects 25 customers, but found {len(customers)}!")

    except FileNotFoundError:
        print(f"Error: File {routes_file} not found.")
        return None
    except ValueError as e:
        print(f"Error processing file data: {e}")
        return None

    # Big-M: safe upper bound on route cost
    M = max(r["cost"] for r in routes)

    R = range(len(routes))

    # Build helper: for each customer i, list of routes where i is the last customer
    last_routes_of_customer = {i: [] for i in customers}
    for r in R:
        last_cust = routes[r]["customers"][-1]
        if last_cust in last_routes_of_customer:
            last_routes_of_customer[last_cust].append(r)

    # -----------------------
    # 2) Solve for each epsilon and store results
    # -----------------------
    results = []

    for eps in eps_list:
        budget = (1.0 + eps) * z_star

        model = gp.Model(f"CVRP_LastCustomer_eps_{eps:.3f}")
        model.setParam("TimeLimit", time_limit)
        model.setParam("OutputFlag", 0)  # silence Gurobi log

        x = model.addVars(R, vtype=GRB.BINARY, name="x")
        eta = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="eta")
        gamma = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="gamma")

        model.setObjective(eta - gamma, GRB.MINIMIZE)

        # Cover each customer exactly once
        for cust_id in customers:
            model.addConstr(
                gp.quicksum(x[r] for r in temp_customer_routes[cust_id]) == 1,
                name=f"Cover_{cust_id}",
            )

        # Exactly K routes
        model.addConstr(x.sum() == num_vehicles, name="Vehicle_Count")

        # Budget
        total_cost_expr = gp.quicksum(routes[r]["cost"] * x[r] for r in R)
        model.addConstr(total_cost_expr <= budget, name="Budget")

        # (1e) eta definition
        for i in customers:
            model.addConstr(
                gp.quicksum(routes[r]["cost"] * x[r] for r in last_routes_of_customer[i]) <= eta,
                name=f"EtaDef_{i}",
            )

        # (1f) gamma definition via big-M
        for i in customers:
            sum_bir_x = gp.quicksum(x[r] for r in last_routes_of_customer[i])
            sum_pr_bir_x = gp.quicksum(routes[r]["cost"] * x[r] for r in last_routes_of_customer[i])
            model.addConstr(M * (1 - sum_bir_x) + sum_pr_bir_x >= gamma, name=f"GammaDef_{i}")

        model.optimize()

        if model.SolCount == 0:
            results.append(
                {
                    "eps": eps,
                    "budget": budget,
                    "feasible": False,
                    "total_cost": None,
                    "range": None,
                    "gap": None,
                    "runtime": model.Runtime,
                }
            )
            continue

        selected_route_indices = [r for r in R if x[r].X > 0.5]
        selected_costs = [routes[r]["cost"] for r in selected_route_indices]

        total_cost = sum(selected_costs)
        range_val = max(selected_costs) - min(selected_costs)

        results.append(
            {
                "eps": eps,
                "budget": budget,
                "feasible": True,
                "total_cost": total_cost,
                "range": range_val,
                "gap": model.MIPGap,
                "runtime": model.Runtime,
            }
        )

    # -----------------------
    # 3) Print summary table
    # -----------------------
    print("\nSUMMARY (Part 1.1(e) - Last-customer formulation)")
    print(f"z* = {z_star:.2f}, vehicles = {num_vehicles}, big-M = {M:.2f}, time limit = {time_limit}s\n")

    header = f"{'eps':>7} | {'budget':>10} | {'total_cost':>11} | {'range':>7} | {'gap':>7} | {'runtime(s)':>10} | status"
    print(header)
    print("-" * len(header))

    for row in results:
        if not row["feasible"]:
            print(
                f"{row['eps']:7.3f} | {row['budget']:10.2f} | {'-':>11} | {'-':>7} | {'-':>7} | {row['runtime']:10.2f} | infeasible/no-sol"
            )
        else:
            print(
                f"{row['eps']:7.3f} | {row['budget']:10.2f} | {row['total_cost']:11.2f} | {row['range']:7.2f} | {row['gap']:7.4f} | {row['runtime']:10.2f} | ok"
            )

    return results


if __name__ == "__main__":
    Z_STAR = 8831.0
    EPS_LIST = [0.01, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15]

    solve_last_customer_fairness_summary(
        routes_file="data/routes.txt",
        z_star=Z_STAR,
        eps_list=EPS_LIST,
        num_vehicles=5,
        depot_id=0,
        time_limit=600,
    )
