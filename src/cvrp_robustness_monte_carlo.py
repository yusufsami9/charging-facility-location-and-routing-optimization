# QUESTION 1.2(b)

import random
import math

# ===============================
# Fixed routes from 1.2(a)
# ===============================
routes = [
    [0, 1, 11, 22, 7, 14, 25, 0],
    [0, 9, 0],
    [0, 20, 16, 12, 19, 18, 15, 0],
    [0, 21, 5, 3, 24, 2, 13, 0],
    [0, 23, 4, 17, 6, 10, 8, 0]
]

Q = 478
K = 1000   # number of simulations
random.seed(42)

# ===============================
# Read deterministic demands
# ===============================
def read_demands(filename="data/instance.txt"):
    with open(filename, "r") as f:
        tokens = f.read().split()
        it = iter(tokens)

        Q_file = int(next(it))  # capacity (same as Q)
        n_customers = 25

        demands_raw = [int(next(it)) for _ in range(n_customers)]
        demands = {0: 0}
        for i in range(1, n_customers + 1):
            demands[i] = demands_raw[i - 1]

    return demands

q = read_demands()

# ===============================
# Monte Carlo simulation
# ===============================
violation_counts = 0
total_violations = []
max_violation = 0

for k in range(K):

    # Step 1: draw random demands
    q_tilde = {}
    for i in q:
        if i == 0:
            q_tilde[i] = 0
        else:
            low = math.floor(0.9 * q[i])
            high = math.ceil(1.1 * q[i])
            q_tilde[i] = random.randint(low, high)

    # Step 2–3: compute total violation for this scenario
    total_violation = 0

    for r in routes:
        load = sum(q_tilde[i] for i in r if i != 0)
        violation = max(0, load - Q)
        total_violation += violation

    total_violations.append(total_violation)

    if total_violation > 0:
        violation_counts += 1

    if total_violation > max_violation:
        max_violation = total_violation

# ===============================
# Results
# ===============================
average_violation = sum(total_violations) / K

print("\n===== ROBUSTNESS SIMULATION RESULTS =====")
print(f"Number of scenarios with violation: {violation_counts} / {K}")
print(f"Average total capacity violation:   {average_violation:.2f}")
print(f"Maximum total capacity violation:   {max_violation}")
