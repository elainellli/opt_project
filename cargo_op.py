import pandas as pd
from gurobipy import Model, GRB, LinExpr

from typing import Dict, Tuple

def parse_data(input_file):
    df = pd.read_excel(input_file)
    df.columns = df.columns.str.strip()
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

def _days_columns():
    return [("Monday", 1), ("Tuesday", 2), ("Wednesday", 3), ("Thursday", 4), ("Friday", 5)]

def _var_arcs_to_dataframe(var_dict, arcs, days):
    # Output format: origin, destination, Monday..Friday (same as data.xlsx)
    rows = []
    for (i, j) in arcs:
        row = {"origin": i, "destination": j}
        for day_name, t in _days_columns():
            val = var_dict[i, j, t].X
            row[day_name] = int(round(val))
        rows.append(row)
    return pd.DataFrame(rows)

def _var_airport_to_dataframe(var_dict, airports, days):
    # Output format: airport, Monday..Friday
    rows = []
    for i in airports:
        row = {"airport": i}
        for day_name, t in _days_columns():
            val = var_dict[i, t].X
            row[day_name] = int(round(val))
        rows.append(row)
    return pd.DataFrame(rows)

def export_results_to_excel(output_file, x, y, u, s):
    airports = ["A", "B", "C"]
    days = [1, 2, 3, 4, 5]
    arcs = [(i, j) for i in airports for j in airports if i != j]

    df_x = _var_arcs_to_dataframe(x, arcs, days)
    df_y = _var_arcs_to_dataframe(y, arcs, days)
    df_u = _var_arcs_to_dataframe(u, arcs, days)
    df_s = _var_airport_to_dataframe(s, airports, days)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_x.to_excel(writer, sheet_name="x_loaded", index=False)
        df_y.to_excel(writer, sheet_name="y_empty", index=False)
        df_u.to_excel(writer, sheet_name="u_backlog", index=False)
        df_s.to_excel(writer, sheet_name="s_aircraft", index=False)

def main():
    input_file = "data.xlsx"
    L = parse_data(input_file)

    model, x, y, u, s = build_and_solve_model(L)

    if model.Status == GRB.OPTIMAL:
        print("Optimal objective value:", model.ObjVal)
        export_results_to_excel("results.xlsx", x, y, u, s)
        print("Wrote results to results.xlsx")
        print("Optimal values for x (loaded cargo):")
        for key in x.keys():
            if x[key].X > 0:
                print(f"x{key} = {x[key].X}")
        print("Optimal values for y (empty repositioning):")
        for key in y.keys():
            if y[key].X > 0:
                print(f"y{key} = {y[key].X}")
        print("Optimal values for u (held cargo):")
        for key in u.keys():
            if u[key].X > 0:
                print(f"u{key} = {u[key].X}")
        print("Optimal values for s (aircraft stationed):")
        for key in s.keys():
            if s[key].X > 0:
                print(f"s{key} = {s[key].X}")



if __name__ == "__main__":
    main()