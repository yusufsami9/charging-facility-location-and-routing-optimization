# Charging Facility Location & Robust CVRP — Optimization Project (Gurobi)

This repository contains my implementations and experiments for a graduate-level Operations Research / Optimization assignment focused on:

- Capacitated Vehicle Routing Problem (CVRP)
- Fairness-aware routing (range minimization)
- Robust CVRP under demand uncertainty (Monte Carlo simulation + cutting planes)
- Capacitated Charging Facility Location (CFL)
- Cooperative game theory (Shapley value for cost allocation)
- Selfish routing vs. system-optimal routing under fixed charging infrastructure

The original assignment description is not included due to copyright restrictions.  
All formulations and experiments are implemented in Python using Gurobi.

---

## 📁 Repository Structure
```
.
├── data/
│ ├── instance.txt # CVRP instance (demands + distance matrix)
│ ├── network.txt # Directed road network (i, j, distance)
│ ├── pairsA.txt # OD pairs for company A
│ ├── pairsB.txt # OD pairs for company B
│ ├── pairsC.txt # OD pairs for company C
│ └── routes.txt # Candidate routes for route-based CVRP
│
├── src/
│ ├── q_1_1_a.py # CVRP (route-based set partitioning)
│ ├── q_1_1_b.py # Fair CVRP (minimize range with budget constraint)
│ ├── q_1_1_e.py # Last-customer fairness formulation + epsilon study
│ ├── q_1_2_a.py # Two-index CVRP with lazy cuts (subtour + capacity)
│ ├── q_1_2_b.py # Robustness via Monte Carlo simulation
│ ├── q_1_2_c.py # Scenario-based robust CVRP (cutting-plane loop)
│ ├── q_1_2_e.py # Failure-based recourse cost simulation
│ ├── q_2_1_c.py # Capacitated Charging Facility Location (single firms)
│ ├── q_2_1_d.py # Shapley value cost allocation (coalitions A, B, C)
│ └── q_2_2_a.py # Selfish vs system-optimal routing with fixed stations
│
├── report/
│ └── Final_Report.pdf # Full mathematical models, results, discussion
│
├── README.md
├── .gitignore
└── LICENSE
```
---

## 🧠 Problem Overview

### Part 1 — Capacitated Vehicle Routing Problem (CVRP)
- Route-based CVRP with set-partitioning formulation
- Fairness-aware routing by minimizing the workload range between vehicles
- Budget-constrained fairness (epsilon-relaxation of optimal cost)
- Two-index CVRP with lazy constraints (subtour elimination + rounded capacity cuts)
- Robust CVRP under demand uncertainty:
  - Monte Carlo simulation
  - Scenario-based optimization
  - Cutting-plane approach
  - Failure-based recourse policy and cost analysis

### Part 2 — Charging Facility Location & Routing
- Capacitated Charging Facility Location (CFL) for individual companies
- Coalition formation and joint infrastructure planning
- Shapley value for fair cost allocation among companies
- Comparison of:
  - Selfish routing (shortest charge-feasible paths)
  - System-optimal routing (multicommodity min-cost flow)
- Capacity violation analysis under decentralized routing behavior

---

## 🚀 How to Run

> ⚠️ Requires **Gurobi** with a valid license.

1. Create environment and install dependencies:
```bash
pip install gurobipy
```
Run any part from src/, for example:
```
python src/q_1_2_a.py
python src/q_2_1_d.py
python src/q_2_2_a.py
```
All scripts assume data files are available under data/.

## Outputs

Each script prints:

- Objective values

- Runtime

- MIP gap (if applicable)

- Selected routes / facility locations

- Robustness and recourse statistics

- Coalition costs and Shapley allocations

- System-optimal vs selfish routing comparison

Detailed results and analysis are documented in:

📄 report/ProjectReport.pdf

## Methods & Tools

- Optimization: Gurobi (MIP, lazy constraints, multicommodity flow)

- Robust optimization: scenario generation, cutting-plane loop

- Simulation: Monte Carlo demand sampling

- Game theory: Shapley value for cooperative cost allocation

- Graph algorithms: shortest paths with charging feasibility
