import pandas as pd
from gurobipy import *

def parse_data(input_file):
    df = pd.read_excel(input_file)
    days_map = {
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
    }

    L = {}

    for _, row in df.iterrows():
        i = row["origin"]
        j = row["destination"]
        for day_name, t in days_map.items():
            L[(i, j, t)] = int(row[day_name])
    return L

def build_and_solve_model(L):
    airports = ["A", "B", "C"]
    days = [1, 2, 3, 4, 5]
    arcs = [(i, j) for i in airports for j in airports if i != j]


    F = 1200 # Fleet size
    h = 10 # Cost of holding cargo

    # Empty reposition costs
    c = {
        ("A", "B"): 7, ("B", "A"): 7,
        ("A", "C"): 3, ("C", "A"): 3,
        ("B", "C"): 6, ("C", "B"): 6,
    }

    m = Model("ExpressAir")

    x = m.addVars(arcs, days, vtype=GRB.INTEGER, name="x")
    y = m.addVars(arcs, days, vtype=GRB.INTEGER, name="y")
    u = m.addVars(arcs, days, vtype=GRB.INTEGER, name="u")
    s = m.addVars(airports, days, vtype=GRB.INTEGER, name="s")

    objExpr = LinExpr()
    for (i, j) in arcs:
        for t in days:
            objExpr += h * u[i, j, t]
            objExpr += c[(i, j)] * y[i, j, t]

    m.setObjective(objExpr, GRB.MINIMIZE)

    # Fleet size constraint
    for t in days:
        expr = LinExpr()
        for i in airports:
            expr += s[i, t]
        m.addConstr(expr == F, name=f"fleet_{t}")

    # Aircraft dispatch capacity
    for i in airports:
        for t in days:
            expr = LinExpr()
            for j in airports:
                if j != i:
                    expr += x[i, j, t]
                    expr += y[i, j, t]
            m.addConstr(expr <= s[i, t], name=f"cap_{i}_{t}")

    # Aircraft flow balance
    for i in airports:
        for t in [2, 3, 4, 5]:
            inflow = LinExpr()
            outflow = LinExpr()

            for j in airports:
                if j != i:
                    inflow += x[j, i, t - 1]
                    inflow += y[j, i, t - 1]
                    outflow += x[i, j, t - 1]
                    outflow += y[i, j, t - 1]

            m.addConstr(
                s[i, t] == s[i, t - 1] + inflow - outflow,
                name=f"air_balance_{i}_{t}"
            )

    # Friday -> Monday cycle
    for i in airports:
        inflow = LinExpr()
        outflow = LinExpr()

        for j in airports:
            if j != i:
                inflow += x[j, i, 5]
                inflow += y[j, i, 5]
                outflow += x[i, j, 5]
                outflow += y[i, j, 5]

        m.addConstr(
            s[i, 1] == s[i, 5] + inflow - outflow,
            name=f"air_balance_{i}_1"
        )

    # Cargo balance
    for (i, j) in arcs:
        for t in [2, 3, 4, 5]:
            m.addConstr(
                u[i, j, t] == u[i, j, t-1] + L[(i, j, t)] - x[i, j, t],
                name=f"cargo_balance_{i}_{j}_{t}"
            )

    # Friday -> Monday cargo cycle
    for (i, j) in arcs:
        m.addConstr(
            u[i, j, 1] == u[i, j, 5] + L[(i, j, 1)] - x[i, j, 1],
            name=f"cargo_balance_{i}_{j}_1"
        )

    # Cargo availability
    for (i, j) in arcs:
        m.addConstr(x[i, j, 1] <= u[i, j, 5] + L[(i, j, 1)])
        for t in [2, 3, 4, 5]:
            m.addConstr(x[i, j, t] <= u[i, j, t-1] + L[(i, j, t)])

    m.optimize()

    return m, x, y, u, s

def main():
    input_file = "data.xlsx"
    L = parse_data(input_file)

    model, x, y, u, s = build_and_solve_model(L)

    if model.Status == GRB.OPTIMAL:
        print("Optimal objective value:", model.ObjVal)


if __name__ == "__main__":
    main()