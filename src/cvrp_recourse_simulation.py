# QUESTION 1.2(e)

import json
import math
import random
from typing import Dict, List, Tuple


# ------------------------------------------------------------
# 1) Read instance.txt (to get Q, nominal demands, distance matrix)
# ------------------------------------------------------------
def read_instance_txt(path: str, n_customers: int = 25):
    """
    instance.txt format:
      line 1: capacity Q
      line 2: n demands (customers 1..n)
      next n+1 lines: distance matrix rows for nodes 0..n (size (n+1)x(n+1))
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    Q = int(float(lines[0].split()[0]))

    demand_tokens = lines[1].split()
    if len(demand_tokens) != n_customers:
        raise ValueError(f"Expected {n_customers} demands on line 2, got {len(demand_tokens)}.")

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
            # keep as int costs (as your other scripts did)
            c[(i, j)] = int(float(v))

    return N, Q, q_nom, c


# ------------------------------------------------------------
# 2) Load cp_solutions_1_2c.json (your saved output structure)
# ------------------------------------------------------------
def load_cp_solutions(json_path: str) -> List[dict]:
    """
    Expected structure (as in your posted JSON):
      {
        ...,
        "results": [
           {"iter":1, "objective_cost":..., "routes":[{"route":[...], ...}, ...]},
           ...
        ]
      }
    Returns the list under "results".
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        return data["results"]

    raise ValueError("Unrecognized JSON structure in cp_solutions_1_2c.json")


# ------------------------------------------------------------
# 3) Demand sampling: integer demand in [0.9 q_i, 1.1 q_i]
# ------------------------------------------------------------
def sample_demands(q_nom: Dict[int, int], rng: random.Random, n_customers: int) -> Dict[int, int]:
    """
    Draw integer demands: q~_i in [0.9*q_i, 1.1*q_i].
    Use floor/ceil on bounds, then randint to get an integer in the interval.
    """
    q_real = {0: 0}
    for i in range(1, n_customers + 1):
        lo = math.floor(0.9 * q_nom[i])
        hi = math.ceil(1.1 * q_nom[i])
        q_real[i] = rng.randint(lo, hi)
    return q_real


# ------------------------------------------------------------
# 4) Cost BEFORE recourse: planned cost using the route arcs
# ------------------------------------------------------------
def planned_cost(routes: List[List[int]], c: Dict[Tuple[int, int], int]) -> int:
    """
    Sum of arc costs along each planned route (including depot returns).
    """
    total = 0
    for r in routes:
        for t in range(len(r) - 1):
            total += c[(r[t], r[t + 1])]
    return total


# ------------------------------------------------------------
# 5) Cost AFTER recourse: FAILURE-BASED recourse policy (corrected)
# ------------------------------------------------------------
def recourse_cost_failure_based(
    routes: List[List[int]],
    q_real: Dict[int, int],
    Q: int,
    c: Dict[Tuple[int, int], int],
) -> int:
    """
    Failure-based recourse policy:

    For each planned route, follow the same customer order.
    Let remaining capacity start at Q at the depot.

    For each next customer nxt in the planned order:
      1) You TRAVEL to nxt (pay cost current -> nxt)
      2) Upon arrival, you observe realized demand d = q_real[nxt]
      3) If remaining < d:
            You cannot serve -> go back to depot and return:
                pay nxt -> 0 + 0 -> nxt
            Refill, then serve nxt:
                remaining = Q - d
         Else:
            Serve directly:
                remaining -= d
      4) Continue to the next customer in the same planned order

    After the last customer, return to depot (pay last -> 0).
    """
    total = 0

    for route in routes:
        remaining = Q
        current = 0  # start at depot

        # route is like [0, a, b, ..., 0], customers are route[1:-1]
        for idx in range(1, len(route) - 1):
            nxt = route[idx]
            d = q_real[nxt]

            # Always travel to nxt first (information is revealed at arrival)
            total += c[(current, nxt)]

            if remaining < d:
                # Failure at nxt: go to depot and back to nxt, then serve
                total += c[(nxt, 0)]
                total += c[(0, nxt)]
                remaining = Q - d
            else:
                remaining -= d

            current = nxt

        # Return to depot at end of route
        total += c[(current, 0)]

    return total


# ------------------------------------------------------------
# 6) Main simulation
# ------------------------------------------------------------
def main():
    INSTANCE_FILE = "data/instance.txt"
    CP_JSON = "report/cp_solutions_1_2c.json"

    N_CUSTOMERS = 25
    K_SIM = 1000
    SEED = 42

    _, Q, q_nom, c = read_instance_txt(INSTANCE_FILE, n_customers=N_CUSTOMERS)
    results = load_cp_solutions(CP_JSON)

    rng = random.Random(SEED)

    print("===== PART 1.2(e) — RECOURSE SIMULATION =====")
    print(f"Simulations per solution: k={K_SIM}, seed={SEED}\n")
    print(f"{'iter':>4} | {'cost(from c)':>11} | {'avg cost BEFORE':>15} | {'avg cost AFTER':>14} | {'avg extra':>10}")
    print("-" * 67)

    for res in results:
        it = res.get("iter", None)
        base_cost = float(res.get("objective_cost", 0.0))

        # Extract routes from JSON
        routes = []
        for r_obj in res.get("routes", []):
            routes.append(r_obj["route"])

        # Planned cost from the stored routes (should match base_cost, but we keep both)
        planned = planned_cost(routes, c)

        before_sum = 0.0
        after_sum = 0.0

        for _ in range(K_SIM):
            q_real = sample_demands(q_nom, rng, N_CUSTOMERS)

            # BEFORE recourse: planned routing cost (does not depend on q_real)
            before_sum += planned

            # AFTER recourse: apply failure-based policy under q_real
            after_sum += recourse_cost_failure_based(routes, q_real, Q, c)

        avg_before = before_sum / K_SIM
        avg_after = after_sum / K_SIM
        avg_extra = avg_after - avg_before

        print(f"{it:4d} | {base_cost:11.2f} | {avg_before:15.2f} | {avg_after:14.2f} | {avg_extra:10.2f}")


if __name__ == "__main__":
    main()
