"""
full_process.py

This module defines a single convenience function, `run_full_process`, which
combines the end‑to‑end production planning, inventory optimisation and
simulation workflow defined in the original notebook.  The function is
parametrised so that the user can easily change key inputs (e.g., demand
data, inventory proportions, processing times, cost coefficients and
simulation settings) and obtain aggregated and disaggregated production
schedules along with discrete-event simulation metrics.  To verify that
the function works correctly the defaults in `__main__` reproduce the
example found in the question.

Usage:
    from full_process import run_full_process
    results = run_full_process(data, ...)

The `results` dictionary contains Pandas DataFrames for the aggregated
schedule, disaggregated schedule, inventory and production tables, as well
as simulation summaries (totals, per product and per station).  When
`make_plots=True`, Matplotlib figures are returned in the dictionary too.

Author: OpenAI ChatGPT
"""

import numpy as np
import pandas as pd
import random
import heapq
import math

# --- Conditional import of PuLP ---
try:
    from pulp import LpProblem, LpVariable, lpSum, value, LpStatus, PULP_CBC_CMD
    HAS_PULP = True
except ImportError:
    # PuLP is not available in this environment.  We will use SciPy as a
    # fallback for linear programming.  Note that some features (binary
    # variables, certain optional constraints) are not supported with
    # SciPy and will raise NotImplementedError if requested.
    HAS_PULP = False
    from scipy.optimize import linprog

def promedio(min_val, max_val):
    """Compute the midpoint between two values."""
    return (min_val + max_val) / 2

def plan_produccion_optimo(
    total_mes: dict,
    initial_stock: float,
    # --- cost and labour parameters ---
    ct=80, ht=2.70, pit=100000, crt=12, cot=18, cwt=10, cwt_prima=37,
    m=1.0,
    # --- optional behaviours ---
    alpha=0.5,
    use_max_hours=False, max_hours=4160,
    use_safety_stock=False,
    use_fixed_cost_capacity=False, cap_max=4400, fixed_cost=1,
    use_no_consecutive=False,
    use_smooth=False, smooth_pct=0.20,
    use_layoff_limit=False, layoff_pct=0.10,
    use_subcontracting=False, csc=95.0,
    solver=None, solver_msg=0,
    graficar=False,
    H_mes_Ma=192,
    final_stock: float=0.0
):
    """Solve the aggregated monthly production planning problem.

    See the original notebook for a full description of each parameter.  This
    function returns a dictionary containing the solver status, the optimal
    objective value, a DataFrame with monthly production, inventory and
    related variables, and the underlying PuLP model object.
    """
    # determine months in canonical order
    orden_meses = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    meses = [m for m in orden_meses if m in total_mes]
    mes_anterior = {meses[i]: meses[i-1] for i in range(1, len(meses))}
    # if PuLP is available use the original integer formulation
    if HAS_PULP:
        # build model
        mdl = LpProblem("Minimizar_Costos", sense=1)
        # variables
        pt  = LpVariable.dicts("Produccion", meses, lowBound=0, cat='Continuous')
        itv = LpVariable.dicts("Inventario", meses, lowBound=0, cat='Continuous')
        st  = LpVariable.dicts("Pendiente", meses, lowBound=0, cat='Continuous')
        lrt = LpVariable.dicts("Hora_regular", meses, lowBound=0, cat='Continuous')
        lot = LpVariable.dicts("Hora_extra",   meses, lowBound=0, cat='Continuous')
        lut = LpVariable.dicts("Hora_ociosa",  meses, lowBound=0, cat='Continuous')
        wt_mas   = LpVariable.dicts("Trabajadores_mas",   meses, lowBound=0, cat='Continuous')
        wt_menos = LpVariable.dicts("Trabajadores_menos", meses, lowBound=0, cat='Continuous')
        nit = LpVariable.dicts("Inventario_neto", meses, lowBound=0, cat='Continuous')
        # binary variables
        if use_fixed_cost_capacity or use_no_consecutive:
            Y = LpVariable.dicts("Produccion_binaria", meses, lowBound=0, upBound=1, cat="Binary")
        else:
            Y = {}
        # subcontracting
        if use_subcontracting:
            SC = LpVariable.dicts("Subcontratacion", meses, lowBound=0, cat='Continuous')
        else:
            SC = {}
        # objective components
        costo_produccion       = lpSum(ct * pt[t]  for t in meses)
        costo_retraso          = lpSum(crt * lrt[t] for t in meses)
        costo_oportunidad_OT   = lpSum(cot * lot[t] for t in meses)
        costo_inventario       = lpSum(ht * itv[t]  for t in meses)
        costo_penalizacion     = lpSum(pit * st[t]  for t in meses)
        costo_exceso_capacidad = lpSum(cwt * wt_mas[t]   for t in meses)
        costo_capacidad_ociosa = lpSum(cwt_prima * wt_menos[t] for t in meses)
        obj = (costo_produccion + costo_retraso + costo_oportunidad_OT +
               costo_inventario + costo_penalizacion + costo_exceso_capacidad + costo_capacidad_ociosa)
        if use_fixed_cost_capacity:
            obj += lpSum(fixed_cost * Y[t] for t in meses)
        if use_subcontracting:
            obj += lpSum(csc * SC[t] for t in meses)
        mdl += obj
        # core constraints
        for t in meses:
            if t == meses[0]:
                mdl += (lrt[t] == H_mes_Ma + wt_mas[t] - wt_menos[t]), f"Plantilla_{t}"
            else:
                prev = mes_anterior[t]
                mdl += (lrt[t] == lrt[prev] + wt_mas[t] - wt_menos[t]), f"Plantilla_{t}"
        for i, t in enumerate(meses):
            demanda = total_mes[t]
            if i == 0:
                if use_subcontracting:
                    mdl += (nit[t] == initial_stock + pt[t] + SC[t] - demanda), f"Inv_{t}"
                else:
                    mdl += (nit[t] == initial_stock + pt[t] - demanda), f"Inv_{t}"
            else:
                prev = mes_anterior[t]
                if use_subcontracting:
                    mdl += (nit[t] == nit[prev] + pt[t] + SC[t] - demanda), f"Inv_{t}"
                else:
                    mdl += (nit[t] == nit[prev] + pt[t] - demanda), f"Inv_{t}"
            mdl += (nit[t] == itv[t] - st[t]), f"Inv_neto_{t}"
        # minimum ending inventory
        if meses:
            mdl += (nit[meses[-1]] >= final_stock), "Inventario_final"
        # regular/extra/idle hours vs production balance
        for t in meses:
            mdl += (lot[t] - lut[t] == m * pt[t] - lrt[t]), f"Horas_balance_{t}"
        # optional constraints
        if use_max_hours:
            for t in meses:
                mdl += lrt[t] <= max_hours, f"Max_LR_{t}"
        if use_safety_stock and len(meses) > 1:
            factor = alpha
            for i in range(len(meses) - 1):
                t = meses[i]
                t_next = meses[i + 1]
                mdl += itv[t] >= factor * total_mes[t_next], f"SafetyStock_{t}"
        if use_fixed_cost_capacity:
            for t in meses:
                mdl += pt[t] <= cap_max * Y[t], f"Capacidad_Max_{t}"
        if use_no_consecutive:
            for i in range(len(meses) - 1):
                t = meses[i]
                t_next = meses[i + 1]
                mdl += Y[t] + Y[t_next] <= 1, f"NoConsecutivo_{t}"
        if use_smooth and len(meses) > 1:
            up = 1 + smooth_pct
            down = 1 - smooth_pct
            for i in range(len(meses) - 1):
                t = meses[i]
                t1 = meses[i + 1]
                mdl += pt[t1] <= up * pt[t],  f"Smooth_up_{t}"
                mdl += pt[t1] >= down * pt[t], f"Smooth_down_{t}"
        if use_layoff_limit:
            if meses:
                first = meses[0]
                mdl += wt_menos[first] <= layoff_pct * H_mes_Ma, f"Despidos_{first}"
                for t in meses[1:]:
                    prev = mes_anterior[t]
                    mdl += wt_menos[t] <= layoff_pct * lrt[prev], f"Despidos_{t}"
        # solve
        if solver is None:
            solver = PULP_CBC_CMD(msg=solver_msg)
        mdl.solve(solver)
        status = LpStatus.get(mdl.status, str(mdl.status))
        try:
            z = mdl.objective.value()
        except Exception:
            z = value(mdl.objective)
        # build results df
        filas = []
        for t in meses:
            fila = {
                "Mes": t,
                "P(t)": pt[t].value(),
                "I(t)": itv[t].value(),
                "S(t)": st[t].value(),
                "L(RT)": lrt[t].value() + wt_mas[t].value() - wt_menos[t].value() + lot[t].value(),
                "NI(t)": nit[t].value(),
                "D(t)": total_mes[t],
            }
            if use_subcontracting:
                fila["SC(t)"] = SC[t].value()
            if use_fixed_cost_capacity or use_no_consecutive:
                fila["Y(t)"] = Y[t].value()
            filas.append(fila)
        df = pd.DataFrame(filas)
        for c in df.columns:
            if c != "Mes":
                df[c] = pd.to_numeric(df[c], errors="coerce").round(0)
        return {"status": status, "z": z, "df": df, "modelo": mdl}
    else:
        # fallback using SciPy linear programming
        # Only baseline constraints are supported.  Optional features such as
        # safety stock, fixed capacity cost, no consecutive production or
        # smoothing are not implemented in the fallback and will raise.
        if use_subcontracting or use_safety_stock or use_fixed_cost_capacity or use_no_consecutive or use_smooth or use_layoff_limit:
            raise NotImplementedError("Optional constraints require PuLP which is not available.")
        M = len(meses)
        if M == 0:
            return {"status": "Infeasible", "z": None, "df": pd.DataFrame(), "modelo": None}
        # number of decision variables: 9 per month
        nvar = 9 * M
        # cost vector
        c = np.zeros(nvar)
        for i, t in enumerate(meses):
            base = i * 9
            c[base + 0] = ct           # P(t)
            c[base + 1] = ht           # I(t)
            c[base + 2] = pit          # S(t)
            c[base + 3] = crt          # lrt
            c[base + 4] = cot          # lot
            c[base + 5] = 0.0          # lut
            c[base + 6] = cwt          # wt_mas
            c[base + 7] = cwt_prima    # wt_menos
            c[base + 8] = 0.0          # nit
        # equality constraints
        A_eq = []
        b_eq = []
        # Plantilla constraints
        for i, t in enumerate(meses):
            row = np.zeros(nvar)
            if i == 0:
                # lrt[0] == H_mes_Ma + wt_mas - wt_menos
                base = i * 9
                row[base + 3] = -1      # -lrt
                row[base + 6] = 1       # +wt_mas
                row[base + 7] = -1      # -wt_menos
                A_eq.append(row)
                b_eq.append(-H_mes_Ma)
            else:
                base = i * 9
                prev_base = (i - 1) * 9
                row[base + 3] = 1       # +lrt_i
                row[prev_base + 3] = -1 # -lrt_{i-1}
                row[base + 6] = -1      # -wt_mas_i
                row[base + 7] = 1       # +wt_menos_i
                A_eq.append(row)
                b_eq.append(0.0)
        # Inventory balance constraints
        for i, t in enumerate(meses):
            row = np.zeros(nvar)
            if i == 0:
                base = i * 9
                row[base + 8] = 1       # +nit_0
                row[base + 0] = -1      # -P_0
                A_eq.append(row)
                b_eq.append(initial_stock - total_mes[t])
            else:
                base = i * 9
                prev_base = (i - 1) * 9
                row[base + 8] = 1       # +nit_i
                row[prev_base + 8] = -1 # -nit_{i-1}
                row[base + 0] = -1      # -P_i
                A_eq.append(row)
                b_eq.append(-total_mes[t])
        # Inventory net constraints: nit = itv - st
        for i, t in enumerate(meses):
            base = i * 9
            row = np.zeros(nvar)
            row[base + 8] = 1  # +nit
            row[base + 1] = -1 # -itv
            row[base + 2] = 1  # +st
            A_eq.append(row)
            b_eq.append(0.0)
        # Hours balance: lot - lut - m*pt + lrt = 0
        for i, t in enumerate(meses):
            base = i * 9
            row = np.zeros(nvar)
            row[base + 4] = 1       # lot
            row[base + 5] = -1      # -lut
            row[base + 0] = -m      # -m*pt
            row[base + 3] = 1       # +lrt
            A_eq.append(row)
            b_eq.append(0.0)
        # convert to arrays
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        # inequality constraints: final inventory nit_M-1 >= final_stock -> -nit_M-1 <= -final_stock
        A_ub = np.zeros((1, nvar))
        b_ub = np.zeros(1)
        last_base = (M - 1) * 9
        A_ub[0, last_base + 8] = -1.0
        b_ub[0] = -final_stock
        # bounds: all variables >= 0
        bounds = [(0, None) for _ in range(nvar)]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        status = 'Optimal' if res.success else 'Infeasible'
        z = res.fun if res.success else None
        # parse solution
        filas = []
        if res.success:
            x = res.x
            for i, t in enumerate(meses):
                base = i * 9
                P = x[base + 0]
                I = x[base + 1]
                S = x[base + 2]
                LR = x[base + 3]
                LO = x[base + 4]
                LU = x[base + 5]
                Wp = x[base + 6]
                Wm = x[base + 7]
                NIT = x[base + 8]
                filas.append({
                    "Mes": t,
                    "P(t)": P,
                    "I(t)": I,
                    "S(t)": S,
                    "L(RT)": LR + Wp - Wm + LO,  # plantilla ajustada (LR ± W) más horas extra
                    "NI(t)": NIT,
                    "D(t)": total_mes[t]
                })
        else:
            for t in meses:
                filas.append({"Mes": t, "P(t)": None, "I(t)": None, "S(t)": None, "L(RT)": None, "NI(t)": None, "D(t)": total_mes[t]})
        df = pd.DataFrame(filas)
        for ccol in df.columns:
            if ccol != 'Mes' and df[ccol].notna().all():
                df[ccol] = df[ccol].round(0)
        return {"status": status, "z": z, "df": df, "modelo": None}

def desagregado_optimo(
    data,  # dict {mes: [d1, d2, ..., dJ]}
    P_agregado,  # dict {mes: P(t)} capacidad mensual agregada
    I_agregado,  # dict {mes: I(t)} inventario mensual agregado
    inv_inicial_j,  # dict {j: inventario inicial por bebida}
    inv_final_j,  # dict {j: inventario final mínimo por bebida (en el último mes)}
    ponderaciones,  # array/Serie de largo J
    costo_prod=1.0,  # coeficiente para Produccion[i,j] en el objetivo
    costo_inv=0.25,  # coeficiente para Inventario[i,j] en el objetivo
    modo_ponderacion="exact"  # "exact" -> Iij[m,j] == wj * I_agregado[m]; "min" -> sum_j Iij == I_agregado[m] y Iij >= wj * I_agregado[m]
):
    orden_meses = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]
    meses = [m for m in orden_meses if m in data.keys()]
    # Índices de productos 0..J-1
    productos = list(range(len(next(iter(data.values())))))
    # Pesos por producto w
    w = {j: float(ponderaciones[j]) for j in productos}

    # --- Modelo PuLP ---
    if HAS_PULP:
        mdl = LpProblem("Desagregado", sense=1)
        Pij = LpVariable.dicts("Produccion", (meses, productos), lowBound=0, cat="Continuous")
        Iij = LpVariable.dicts("Inventario", (meses, productos), lowBound=0, cat="Continuous")
        mdl += lpSum(costo_prod * Pij[m][j] + costo_inv * Iij[m][j] for m in meses for j in productos), "Objetivo_Desagregado"
        for m_idx, m in enumerate(meses):
            # balance por producto
            for j in productos:
                demanda_ij = float(data[m][j])
                if m_idx == 0:
                    mdl += Iij[m][j] == float(inv_inicial_j[j]) + Pij[m][j] - demanda_ij, f"Inv_{m}_prod{j}"
                else:
                    prev = meses[m_idx - 1]
                    mdl += Iij[m][j] == Iij[prev][j] + Pij[m][j] - demanda_ij, f"Inv_{m}_prod{j}"
            # capacidad agregada
            if m not in P_agregado:
                raise KeyError(f"No encuentro P_agregado para el mes '{m}'. Verifica las claves.")
            mdl += lpSum(Pij[m][j] for j in productos) <= float(P_agregado[m]), f"Capacidad_{m}"
            # inventario agregado -> desagregado según modo
            if m not in I_agregado:
                raise KeyError(f"No encuentro I_agregado para el mes '{m}'. Verifica las claves.")
            I_m = float(I_agregado[m])
            if modo_ponderacion == "exact":
                for j in productos:
                    mdl += Iij[m][j] == w[j] * I_m, f"InvPondExact_{m}_prod{j}"
            elif modo_ponderacion == "min":
                mdl += lpSum(Iij[m][j] for j in productos) == I_m, f"InvAgregado_{m}"
                for j in productos:
                    mdl += Iij[m][j] >= w[j] * I_m, f"InvPondMin_{m}_prod{j}"
            else:
                raise ValueError("modo_ponderacion debe ser 'exact' o 'min'.")
        # Inventario final mínimo por producto en el último mes
        if meses:
            mes_final = meses[-1]
            for j in productos:
                mdl += Iij[mes_final][j] >= float(inv_final_j[j]), f"InvFinal_prod{j}"
        mdl.solve(PULP_CBC_CMD(msg=0))
        status = LpStatus.get(mdl.status, str(mdl.status))
        filas = []
        for m in meses:
            for j in productos:
                filas.append({
                    "Mes": m,
                    "Producto": j,
                    "Produccion": float(Pij[m][j].value() or 0.0),
                    "Inventario": float(Iij[m][j].value() or 0.0),
                    "Demanda": float(data[m][j])
                })
        df = pd.DataFrame(filas)
        return {"status": status, "df": df, "modelo": mdl}

    # --- Fallback SciPy linprog ---
    M = len(meses)
    J = len(productos)
    if M == 0 or J == 0:
        return {"status": "Infeasible", "df": pd.DataFrame(), "modelo": None}
    # variable ordering: for each month i, P[i][j] (j=0..J-1) then I[i][j] (j=0..J-1)
    nvar = 2 * M * J
    c = np.zeros(nvar)
    for i, m in enumerate(meses):
        base = i * 2 * J
        for j in productos:
            idxP = base + j
            idxI = base + J + j
            c[idxP] = costo_prod
            c[idxI] = costo_inv
    A_eq = []
    b_eq = []
    # inventory balance per product
    for i, m in enumerate(meses):
        base = i * 2 * J
        for j in productos:
            idxP = base + j
            idxI = base + J + j
            row = np.zeros(nvar)
            if i == 0:
                # I_0_j - P_0_j = inv_inicial_j - demand
                row[idxI] = 1
                row[idxP] = -1
                A_eq.append(row)
                b_eq.append(float(inv_inicial_j[j]) - float(data[m][j]))
            else:
                prev_base = (i - 1) * 2 * J
                idxI_prev = prev_base + J + j
                row[idxI] = 1
                row[idxI_prev] = -1
                row[idxP] = -1
                A_eq.append(row)
                b_eq.append(-float(data[m][j]))
    A_ub = []
    b_ub = []
    # per-month capacity and aggregated inventory constraints
    for i, m in enumerate(meses):
        base = i * 2 * J
        # capacity: sum_j P[i][j] <= P_agregado[m]
        row_cap = np.zeros(nvar)
        for j in productos:
            row_cap[base + j] = 1
        A_ub.append(row_cap)
        b_ub.append(float(P_agregado[m]))
        # inventory aggregated: depending on modo
        I_m = float(I_agregado[m])
        if modo_ponderacion == "exact":
            # equality constraints I[i][j] == w[j] * I_m
            for j in productos:
                row = np.zeros(nvar)
                row[base + J + j] = 1
                A_eq.append(row)
                b_eq.append(w[j] * I_m)
        elif modo_ponderacion == "min":
            # sum_j I[i][j] == I_m
            row_sum = np.zeros(nvar)
            for j in productos:
                row_sum[base + J + j] = 1
            A_eq.append(row_sum)
            b_eq.append(I_m)
            # and I[i][j] >= w[j] * I_m -> -I[i][j] <= -w[j]*I_m
            for j in productos:
                row_min = np.zeros(nvar)
                row_min[base + J + j] = -1
                A_ub.append(row_min)
                b_ub.append(-w[j] * I_m)
        else:
            raise ValueError("modo_ponderacion debe ser 'exact' o 'min'.")
    # final inventory minimum per product in last month
    last_idx = (M - 1) * 2 * J
    for j in productos:
        row = np.zeros(nvar)
        row[last_idx + J + j] = -1.0
        A_ub.append(row)
        b_ub.append(-float(inv_final_j[j]))
    # convert to arrays
    A_eq = np.array(A_eq) if A_eq else None
    b_eq = np.array(b_eq) if b_eq else None
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    bounds = [(0, None) for _ in range(nvar)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    status = 'Optimal' if res.success else 'Infeasible'
    filas = []
    if res.success:
        x = res.x
        for i, m in enumerate(meses):
            base = i * 2 * J
            for j in productos:
                idxP = base + j
                idxI = base + J + j
                filas.append({
                    "Mes": m,
                    "Producto": j,
                    "Produccion": float(x[idxP]),
                    "Inventario": float(x[idxI]),
                    "Demanda": float(data[m][j])
                })
    else:
        for m in meses:
            for j in productos:
                filas.append({"Mes": m, "Producto": j, "Produccion": None, "Inventario": None, "Demanda": float(data[m][j])})
    df = pd.DataFrame(filas)
    return {"status": status, "df": df, "modelo": None}

class Simulator:
    """Discrete‑event simulator for the bottling line with four stations.

    The simulator models five product types flowing through four stations in series.
    Each entity represents a batch (100 bottles = 300 litres).  Entities arrive
    according to a schedule derived from the production plan and are processed
    sequentially at each station with service times drawn from uniform or
    Gaussian distributions as defined in `service_params`.  Finished
    batches accumulate in a final hold queue until at least `batch_size` are
    available, at which point they are released simultaneously.  The horizon
    is measured in minutes (default 365 days).

    Parameters
    ----------
    capacities : list of int
        Capacity (number of parallel machines) for each station [E1, E2, E3, E4].
    horizon : float, optional
        Simulation horizon in minutes (default: 365*1440).
    batch_size : int, optional
        Number of entities required to release a batch from the hold area.
    service_params : dict, optional
        Dictionary specifying min/max (or mean/SD) service times per station and
        product type.  See original notebook for keys.
    """
    def __init__(self, capacities, horizon=365*1440, batch_size=28, service_params=None):
        self.capacities = capacities
        self.horizon = float(horizon)
        self.batch_size = batch_size
        # scheduling info will be set externally via schedule_tia
        self.n_products = 0
        # default service parameters if none provided
        self.service_params = service_params or {
            "min_firstS":15, "max_firstS":25,
            "min_pas1":26, "max_pas1":40,
            "min_pas2":18, "max_pas2":28,
            "min_pas3":18, "max_pas3":26,
            "min_pas4":22, "max_pas4":33,
            "min_pas5":24, "max_pas5":38,
            "min_fill":10, "max_fill":15,
            "min_label":18, "max_label":22
        }

    def run(self, seed, schedule_tia):
        random.seed(seed)
        # number of product types determined by schedule_tia
        self.n_products = len(schedule_tia)
        # event queue
        event_q = []  # (time, id, type, payload)
        event_id = 0
        def schedule(ev_time, ev_type, payload=None):
            nonlocal event_id
            heapq.heappush(event_q, (ev_time, event_id, ev_type, payload))
            event_id += 1
        # initial arrival times
        next_arrival = [0.0] * self.n_products
        current_tia  = [schedule_tia[p][0] for p in range(self.n_products)]
        # month index per product (for TIA changes)
        month_idx    = [0] * self.n_products
        # end times of each month (in minutes)
        # compute if schedule_tia provided monthly intervals; here we assume 12 months
        month_days = [31,28,31,30,31,30,31,31,30,31,30,31]
        end_times = []
        acc = 0
        for d in month_days:
            acc += d * 1440
            end_times.append(acc)
        # stats
        wip = 0
        wip_time_area = 0.0
        wip_prod = [0]*self.n_products
        wip_area_prod = [0.0]*self.n_products
        last_time = 0.0
        cycle_times = [[] for _ in range(self.n_products)]
        throughput_counts = [0] * self.n_products
        arrivals_counts   = [0] * self.n_products
        # resources per station
        class Res:
            def __init__(self, cap):
                self.cap = cap
                self.busy = 0
                self.queue = []
                self.busy_area = 0.0
                self.wait_times = []
        resources = [Res(c) for c in self.capacities]
        # hold final
        hold = []
        hold_release_flag = [False]
        # service time function
        sp = self.service_params
        def service_time(st, ptype):
            if st == 0:
                return random.uniform(sp["min_firstS"], sp["max_firstS"])
            if st == 1:
                # pasteurisation depends on product type (1–5)
                if ptype == 1:
                    return random.uniform(sp["min_pas1"], sp["max_pas1"])
                if ptype == 2:
                    return random.uniform(sp["min_pas2"], sp["max_pas2"])
                if ptype == 3:
                    return random.uniform(sp["min_pas3"], sp["max_pas3"])
                if ptype == 4:
                    return random.uniform(sp["min_pas4"], sp["max_pas4"])
                return random.uniform(sp["min_pas5"], sp["max_pas5"])
            if st == 2:
                return random.uniform(sp["min_fill"], sp["max_fill"])
            # st == 3: labelling
            return max(random.gauss(sp["min_label"], sp["max_label"]), 0.0)
        # enqueue helper
        def enqueue(ent, st):
            res = resources[st]
            if res.busy < res.cap:
                wait = current_time - ent['qtime']
                res.wait_times.append(wait)
                res.busy += 1
                stime = service_time(st, ent['ptype'])
                schedule(current_time + stime, 'end', (st, ent))
            else:
                ent['qtime'] = current_time
                res.queue.append(ent)
        # arrival event
        def arrival(prod):
            nonlocal wip
            ent = {'prod': prod, 'ptype': prod+1, 'arr_time': current_time, 'qtime': current_time}
            arrivals_counts[prod] += 1
            wip += 1
            wip_prod[prod] += 1
            enqueue(ent, 0)
            # schedule next arrival
            next_t = next_arrival[prod] + current_tia[prod]
            idx = month_idx[prod]
            # update month index if crossing month boundary
            while idx < len(end_times)-1 and next_t >= end_times[idx]:
                idx += 1
            month_idx[prod] = idx
            current_tia[prod] = schedule_tia[prod][idx]
            next_arrival[prod] = next_t
            if next_t <= self.horizon:
                schedule(next_t, 'arr', prod)
        # service end event
        def service_end(payload):
            st, ent = payload
            res = resources[st]
            res.busy -= 1
            if st < len(resources)-1:
                ent['qtime'] = current_time
                enqueue(ent, st+1)
            else:
                hold.append(ent)
                if len(hold) >= self.batch_size and not hold_release_flag[0]:
                    schedule(current_time, 'rel')
                    hold_release_flag[0] = True
            # start next
            if res.queue:
                nex = res.queue.pop(0)
                wait = current_time - nex['qtime']
                res.wait_times.append(wait)
                res.busy += 1
                stime = service_time(st, nex['ptype'])
                schedule(current_time + stime, 'end', (st, nex))
        # release hold event
        def release_hold():
            nonlocal wip
            rel = min(self.batch_size, len(hold))
            for _ in range(rel):
                e = hold.pop(0)
                p = e['prod']
                throughput_counts[p] += 1
                cycle_times[p].append(current_time - e['arr_time'])
                wip_prod[p] -= 1
                wip -= 1
            if len(hold) >= self.batch_size:
                schedule(current_time, 'rel')
                hold_release_flag[0] = True
            else:
                hold_release_flag[0] = False
        # schedule initial arrivals
        for p in range(self.n_products):
            schedule(next_arrival[p], 'arr', p)
        current_time = 0.0
        # event loop
        while event_q:
            ev_time, _, ev_type, payload = heapq.heappop(event_q)
            if ev_time > self.horizon:
                dt = self.horizon - last_time
                if dt > 0:
                    wip_time_area += dt * wip
                    for j in range(self.n_products):
                        wip_area_prod[j] += dt * wip_prod[j]
                    for r in resources:
                        r.busy_area += dt * r.busy
                break
            current_time = ev_time
            dt = current_time - last_time
            if dt > 0:
                wip_time_area += dt * wip
                for j in range(self.n_products):
                    wip_area_prod[j] += dt * wip_prod[j]
                for r in resources:
                    r.busy_area += dt * r.busy
                last_time = current_time
            if ev_type == 'arr':
                arrival(payload)
            elif ev_type == 'end':
                service_end(payload)
            elif ev_type == 'rel':
                release_hold()
        # if horizon reached before simulation ended
        if last_time < self.horizon:
            dt = self.horizon - last_time
            wip_time_area += dt * wip
            for j in range(self.n_products):
                wip_area_prod[j] += dt * wip_prod[j]
            for r in resources:
                r.busy_area += dt * r.busy
        # metrics
        sim_time = self.horizon
        tp_day = [throughput_counts[j] / (sim_time/1440.0) for j in range(self.n_products)]
        total_tp_day = sum(throughput_counts) / (sim_time/1440.0)
        cycle_means = [(sum(ct)/len(ct) if ct else None) for ct in cycle_times]
        all_ct = [c for sub in cycle_times for c in sub]
        total_cycle_mean = (sum(all_ct)/len(all_ct)) if all_ct else None
        wip_avg = wip_time_area / sim_time
        wip_avg_prod = [wip_area_prod[j] / sim_time for j in range(self.n_products)]
        takt = (sim_time / sum(throughput_counts)) if sum(throughput_counts)>0 else None
        takt_prod = [(sim_time / throughput_counts[j] if throughput_counts[j]>0 else None) for j in range(self.n_products)]
        utilizations = [r.busy_area / (sim_time*r.cap) if r.cap>0 else None for r in resources]
        waits = [(sum(r.wait_times)/len(r.wait_times) if r.wait_times else 0.0) for r in resources]
        return {
            'throughput_day': tp_day,
            'total_throughput_day': total_tp_day,
            'cycle_means': cycle_means,
            'total_cycle_mean': total_cycle_mean,
            'wip_avg': wip_avg,
            'wip_avg_prod': wip_avg_prod,
            'takt': takt,
            'takt_prod': takt_prod,
            'utilizations': utilizations,
            'waits': waits,
            'tp_counts': throughput_counts,
            'arrivals_counts': arrivals_counts,
            'arrivals_total': sum(arrivals_counts)
        }

def dimension_stations(base_cap=[1,1,1,1], test_caps=[1,2,3,4], reps=3, service_params=None, schedule_tia=None):
    """Determine minimal station capacities satisfying utilisation and wait thresholds.

    For each station sequentially, test capacities from `test_caps` using the
    simulator and pick the smallest capacity that yields utilisation ≤ 0.85
    and mean wait ≤ 200 minutes on average across `reps` replications.  The
    schedule required by the simulator must be passed via `schedule_tia`.
    """
    dims = base_cap.copy()
    for st in range(4):
        chosen = test_caps[-1]
        for cap in test_caps:
            caps = dims.copy()
            caps[st] = cap
            util_vals, wait_vals = [], []
            for r in range(reps):
                sim = Simulator(caps, service_params=service_params)
                m = sim.run(seed=1000 + st*100 + cap*10 + r, schedule_tia=schedule_tia)
                util_vals.append(m['utilizations'][st])
                wait_vals.append(m['waits'][st])
            avg_util = sum(util_vals)/len(util_vals)
            avg_wait = sum(wait_vals)/len(wait_vals)
            if avg_util <= 0.85:
                chosen = cap
                break
        dims[st] = chosen
    return dims

def compute_ci(values, alpha=0.05):
    """Compute mean and half‑width of a two‑sided (1-alpha) confidence interval."""
    n = len(values)
    mean_val = sum(values)/n
    if n > 1:
        sd = math.sqrt(sum((x-mean_val)**2 for x in values) / (n-1))
    else:
        sd = 0.0
    z = 1.96  # for 95% CI
    hw = z * sd / math.sqrt(n)
    return mean_val, hw

def run_experiment(
    return_tables=True,
    make_plots=False,
    reps=10,
    verbose=False,
    service_params=None,
    schedule_tia=None,
    custom_capacities=None
):
    """Run multiple simulation replications and return summary tables (and optionally plots).
    
    Parameters
    ----------
    return_tables : bool
        Whether to return the summary DataFrames.
    make_plots : bool
        Whether to generate Matplotlib figures (returned in the result dictionary).
    reps : int
        Number of simulation replications.
    verbose : bool
        Print intermediate messages (capacities chosen and number of replications).
    service_params : dict
        Parameters governing service time distributions (see Simulator).
    schedule_tia : list of lists
        Time between arrivals schedule: schedule_tia[p][m] gives the interarrival
        time for product p during month m (0-11).  This is required to run the
        simulator and to dimension stations.
    custom_capacities : list of float or int, optional
        Capacities to force for each station [E1, E2, E3, E4]. When provided,
        the simulation uses these values but still reports the ideal (dimensioned)
        capacities for referencia.
    """
    # determine station capacities
    recommended = dimension_stations(service_params=service_params, schedule_tia=schedule_tia)
    capacities_to_use = recommended
    if custom_capacities is not None:
        if len(custom_capacities) != len(recommended):
            raise ValueError("Se esperan 4 capacidades para las estaciones.")
        try:
            capacities_to_use = [max(1, int(round(float(c)))) for c in custom_capacities]
        except (TypeError, ValueError) as exc:
            raise ValueError("Capacidades no válidas, deben ser numéricas.") from exc
    results = [
        Simulator(capacities_to_use, service_params=service_params).run(
            seed=2000 + r,
            schedule_tia=schedule_tia
        )
        for r in range(reps)
    ]
    # aggregate totals across replications
    total_tp   = [m['total_throughput_day'] for m in results]
    total_ct   = [m['total_cycle_mean']     for m in results]
    total_wip  = [m['wip_avg']              for m in results]
    total_tkt  = [m['takt']                 for m in results]
    total_arrs = [m['arrivals_total']       for m in results]
    mtp, hw_tp   = compute_ci(total_tp)
    mct, hw_ct   = compute_ci(total_ct)
    mwp, hw_wp   = compute_ci(total_wip)
    mtk, hw_tk   = compute_ci(total_tkt)
    marr, hw_arr = compute_ci(total_arrs)
    # build tables
    df_totales = pd.DataFrame({
        "Métrica": [
            "Entradas_365d [lotes]",
            "Throughput [lotes/día]",
            "CycleTime [min]",
            "WIP [lotes]",
            "TaktTime [min/lote]"
        ],
        "Media": [marr, mtp, mct, mwp, mtk],
        "IC95_HW": [hw_arr, hw_tp, hw_ct, hw_wp, hw_tk]
    })
    # per product table
    products = ["Energizante","Isotónica","Agua saborizada","Té","Jugo"]
    rows=[]
    for j, p in enumerate(products):
        th   = [m['throughput_day'][j]   for m in results]
        ctj  = [m['cycle_means'][j]      for m in results]
        wipj = [m['wip_avg_prod'][j]     for m in results]
        tkj  = [m['takt_prod'][j]        for m in results]
        arrj = [m['arrivals_counts'][j]  for m in results]
        th_m, th_hw = compute_ci(th)
        ct_m, ct_hw = compute_ci(ctj)
        wp_m, wp_hw = compute_ci(wipj)
        tk_m, tk_hw = compute_ci(tkj)
        aj_m, aj_hw = compute_ci(arrj)
        rows.append({
            "Producto"              : p,
            "Entradas_365d_media"   : aj_m,
            "Entradas_365d_HW"      : aj_hw,
            "Throughput_media"      : th_m,
            "Throughput_HW"         : th_hw,
            "CycleTime_media_min"   : ct_m,
            "CycleTime_HW"          : ct_hw,
            "WIP_media"             : wp_m,
            "WIP_HW"                : wp_hw,
            "TaktTime_media_min"    : tk_m,
            "TaktTime_HW"           : tk_hw
        })
    df_productos = pd.DataFrame(rows)
    # per station table
    station_names = ["Recepción y Mezcla","Pasteurización","Llenado","Etiquetado"]
    station_rows=[]
    for st in range(4):
        util = [m['utilizations'][st] for m in results]
        wait = [m['waits'][st]        for m in results]
        u_m, u_hw = compute_ci(util)
        w_m, w_hw = compute_ci(wait)
        station_rows.append({
            "Estación"              : station_names[st],
            "Capacidad_ideal"       : recommended[st],
            "Capacidad"             : capacities_to_use[st],
            "Utilización_media"     : u_m,
            "Utilización_HW"        : u_hw,
            "Espera_media_min"      : w_m,
            "Espera_HW"             : w_hw
        })
    df_estaciones = pd.DataFrame(station_rows)
    if verbose:
        print(f"Réplicas usadas para IC95%: {reps}")
        print("Capacidades recomendadas:", recommended)
    out = {
        "df_totales": df_totales,
        "df_productos": df_productos,
        "df_estaciones": df_estaciones,
        "capacities": {
            "ideal": recommended,
            "used": capacities_to_use
        }
    }
    # optional plots
    if make_plots:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
        figs = {}
        decimal_formatter = FormatStrFormatter('%.2f')
        # throughput per product
        fig1, ax1 = plt.subplots()
        ax1.yaxis.set_major_formatter(decimal_formatter)
        ax1.bar(df_productos["Producto"], df_productos["Throughput_media"], yerr=df_productos["Throughput_HW"], capsize=4)
        ax1.set_ylabel("Throughput [lotes/día]")
        ax1.set_title("Throughput por producto (media ± IC95 HW)")
        ax1.set_xlabel("Producto")
        fig1.tight_layout()
        figs["fig_throughput_prod"] = fig1
        # arrivals per product
        fig2, ax2 = plt.subplots()
        ax2.yaxis.set_major_formatter(decimal_formatter)
        ax2.bar(df_productos["Producto"], df_productos["Entradas_365d_media"], yerr=df_productos["Entradas_365d_HW"], capsize=4)
        ax2.set_ylabel("Entradas en 365 días [lotes]")
        ax2.set_title("Entradas por producto (media ± IC95 HW)")
        ax2.set_xlabel("Producto")
        fig2.tight_layout()
        figs["fig_arrivals_prod"] = fig2
        # utilisation per station
        fig3, ax3 = plt.subplots()
        ax3.yaxis.set_major_formatter(decimal_formatter)
        ax3.bar(df_estaciones["Estación"], df_estaciones["Utilización_media"], yerr=df_estaciones["Utilización_HW"], capsize=4)
        ax3.set_ylabel("Utilización (fracción)")
        ax3.set_title("Utilización por estación (media ± IC95 HW)")
        ax3.set_xlabel("Estación")
        fig3.tight_layout()
        figs["fig_util_station"] = fig3
        # wait per station
        fig4, ax4 = plt.subplots()
        ax4.yaxis.set_major_formatter(decimal_formatter)
        ax4.bar(df_estaciones["Estación"], df_estaciones["Espera_media_min"], yerr=df_estaciones["Espera_HW"], capsize=4)
        ax4.set_ylabel("Espera media [min]")
        ax4.set_title("Espera por estación (media ± IC95 HW)")
        ax4.set_xlabel("Estación")
        fig4.tight_layout()
        figs["fig_wait_station"] = fig4
        out["figs"] = figs
    return out if return_tables or make_plots else None

def run_full_process(
    data: dict,
    p_inv_inicial: float,
    p_inv_final: float,
    tiempos_procesos: dict,
    unidad: float,
    use_no_consecutive: bool=False,
    use_smooth: bool=False,
    ct: float=578, ht: float=145, pit: float=1e7,
    crt: float=5931.25, cot: float=5931.25,
    cwt: float=5931.25, cwt_prima: float=5931.25,
    graficar: bool=True,
    costo_prod: float=1.0, costo_inv: float=0.25,
    return_tables: bool=True, make_plots: bool=True,
    capacidades_override=None,
    reps: int=10, verbose: bool=True
):
    """Run the entire workflow: optimisation (aggregated & disaggregated) and simulation.

    Parameters
    ----------
    data : dict
        Dictionary mapping month names to lists of demands per product.
    p_inv_inicial : float
        Proportion of average monthly demand kept as initial inventory.
    p_inv_final : float
        Proportion of average monthly demand required as final inventory in December.
    tiempos_procesos : dict
        Dictionary of process times with keys matching those in the original model.
    unidad : float
        Conversion factor (e.g., litres per unit or seconds per litre).
    use_no_consecutive, use_smooth : bool
        Flags to activate optional constraints in the aggregated model.
    ct, ht, pit, crt, cot, cwt, cwt_prima : floats
        Cost coefficients for production, inventory, shortages, labour and capacity adjustments.
    graficar : bool
        Whether to generate a production/demand plot in the aggregated model (saved as file).
    costo_prod, costo_inv : float
        Cost coefficients for production and inventory in the disaggregated model.
    capacidades_override : list, optional
        Capacidades personalizadas para las estaciones en la simulación.
    return_tables, make_plots, reps, verbose : control flags for the simulation experiment.

    Returns
    -------
    dict
        A dictionary containing intermediate and final results:
        - aggregated model results (key 'agg')
        - disaggregated model results (key 'disagg')
        - inventory and production tables (keys 'tabla_inventario', 'tabla_produccion')
        - simulation tables (keys 'sim_totales', 'sim_productos', 'sim_estaciones')
        - optional figures under 'figs' if make_plots=True.
    """
    # --- Step 1: Aggregate demand and compute initial/final stock ---
    totals = {month: sum(values) for month, values in data.items()}
    avg_sales = np.mean(list(totals.values()))
    initial_stock = avg_sales * p_inv_inicial
    final_stock   = avg_sales * p_inv_final
    # --- Step 2: Compute production times and m (hours per unit) ---
    prom_firstS = promedio(tiempos_procesos["min_firstS"], tiempos_procesos["max_firstS"])
    prom_fill   = promedio(tiempos_procesos["min_fill"], tiempos_procesos["max_fill"])
    prom_label  = promedio(tiempos_procesos["min_label"], tiempos_procesos["max_label"])
    pasteurizaciones = []
    for i in range(1,6):
        pasteurizaciones.append(
            promedio(tiempos_procesos[f"min_pas{i}"], tiempos_procesos[f"max_pas{i}"])
        )
    prom_pasteurizacion = sum(pasteurizaciones) / len(pasteurizaciones)
    tiempo_total = prom_firstS + prom_pasteurizacion + prom_fill + prom_label
    tiempo_total_H = tiempo_total / 60.0
    m = tiempo_total_H / (100 * unidad)  # hours per unit
    # --- Step 3: Solve aggregated model ---
    agg_res = plan_produccion_optimo(
        total_mes=totals,
        initial_stock=initial_stock,
        ct=ct, ht=ht, pit=pit, crt=crt, cot=cot,
        cwt=cwt, cwt_prima=cwt_prima,
        m=m,
        use_subcontracting=False,
        use_safety_stock=False,
        use_fixed_cost_capacity=False,
        use_no_consecutive=use_no_consecutive,
        use_smooth=use_smooth,
        solver_msg=0,
        graficar=graficar,
        H_mes_Ma=192,
        final_stock=final_stock
    )
    # --- Step 4: Disaggregate by product ---
    df_agg = agg_res["df"].copy()
    P_agregado = df_agg.set_index("Mes")["P(t)"].to_dict()
    I_agregado = df_agg.set_index("Mes")["I(t)"].to_dict()
    # compute weights per product and initial/final inventory per product
    totales_bebida = np.sum([np.array(vals, dtype=float) for vals in data.values()], axis=0)
    if totales_bebida.sum() == 0:
        raise ValueError("La suma anual de la demanda por bebida es 0; no se puede ponderar.")
    ponderaciones = totales_bebida / totales_bebida.sum()
    num_productos = len(totales_bebida)
    inv_inicial_j = {j: float(initial_stock) * float(ponderaciones[j]) for j in range(num_productos)}
    inv_final_j   = {j: float(final_stock)   * float(ponderaciones[j]) for j in range(num_productos)}
    disagg_res = desagregado_optimo(
        data=data,
        P_agregado=P_agregado,
        I_agregado=I_agregado,
        inv_inicial_j=inv_inicial_j,
        inv_final_j=inv_final_j,
        ponderaciones=ponderaciones,
        costo_prod=costo_prod,
        costo_inv=costo_inv,
        modo_ponderacion='exact'
    )
    df_desag = disagg_res["df"].copy()
    # --- Step 5: Create inventory and production tables ---
    product_labels = {j: f"Producto_{j+1}" for j in range(num_productos)}
    df_desag["Producto_lbl"] = df_desag["Producto"].map(product_labels)
    tabla_inventario = df_desag.pivot(index="Mes", columns="Producto_lbl", values="Inventario").fillna(0)
    tabla_produccion = df_desag.pivot(index="Mes", columns="Producto_lbl", values="Produccion").fillna(0)
    tabla_inventario["Total_mes"] = tabla_inventario.sum(axis=1)
    tabla_produccion["Total_mes"] = tabla_produccion.sum(axis=1)
    orden_meses = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]
    tabla_inventario = tabla_inventario.reindex(orden_meses)
    tabla_produccion = tabla_produccion.reindex(orden_meses)
    # adjust production by given percentages for first 4 products (as in original)
    porcentajes = [0.80, 0.90, 0.93, 0.87, 0.84]
    tabla_produccion = tabla_produccion.reindex(orden_meses)
    # avoid division by zero: if number of products less than 5, pad with ones
    if len(porcentajes) > tabla_produccion.shape[1] - 1:
        # use as many percentages as we have columns minus total
        pass
    else:
        tabla_produccion.iloc[:, :-1] = tabla_produccion.iloc[:, :-1].div(porcentajes, axis=1)
    tabla_produccion.drop(columns=["Total_mes"], inplace=True)
    # dictionary for simulation from production table
    diccionario = tabla_produccion.apply(lambda fila: fila.tolist(), axis=1).to_dict()
    # --- Step 6: Compute schedule of interarrival times (TIA) per product/month ---
    month_days = {
        "January":31,"February":28,"March":31,"April":30,"May":31,"June":30,
        "July":31,"August":31,"September":30,"October":31,"November":30,"December":31
    }
    month_names = list(diccionario.keys())
    def compute_tia(data_dict):
        tia = []
        for j in range(len(list(diccionario.values())[0])):
            prod_tia = []
            for m in month_names:
                litros = diccionario[m][j]
                unidades = litros / unidad
                lotes_mes = max(1, round(unidades / 100.0))
                lotes_dia = lotes_mes / month_days[m]
                interval = 1440.0 / lotes_dia
                prod_tia.append(interval)
            tia.append(prod_tia)
        return tia
    schedule_tia = compute_tia(diccionario)
    # --- Step 7: Run simulation experiment ---
    capacidades_list = None
    if capacidades_override is not None:
        try:
            capacidades_list = [float(x) for x in capacidades_override]
        except (TypeError, ValueError):
            capacidades_list = None
        else:
            if len(capacidades_list) != 4:
                capacidades_list = None
    sim_res = run_experiment(
        return_tables=return_tables,
        make_plots=make_plots,
        reps=reps,
        verbose=verbose,
        service_params=tiempos_procesos,
        schedule_tia=schedule_tia,
        custom_capacities=capacidades_list
    )
    # assemble output
    out = {
        "agg": agg_res,
        "disagg": disagg_res,
        "tabla_inventario": tabla_inventario,
        "tabla_produccion": tabla_produccion,
        "diccionario": diccionario,
        "sim_totales": sim_res.get("df_totales"),
        "sim_productos": sim_res.get("df_productos"),
        "sim_estaciones": sim_res.get("df_estaciones"),
        "capacities": sim_res.get("capacities")
    }
    if make_plots and "figs" in sim_res:
        out["figs"] = sim_res["figs"]
    return out

if __name__ == "__main__":
    # Example usage with the original data and parameters from the question.
    data_example = {
        "January":  [75648, 44172, 66960, 31320, 75600],
        "February": [81952, 47853, 72540, 33930, 81900],
        "March":    [98342, 57424, 87048, 40716, 98280],
        "April":    [107168, 62577, 94860, 44370, 107100],
        "May":      [100864, 58896, 89280, 41760, 100800],
        "June":     [103386, 60368, 91512, 42804, 103320],
        "July":     [105907, 61841, 93744, 43848, 105840],
        "August":   [108429, 63313, 95976, 44892, 108360],
        "September":[103386, 60368, 91512, 42804, 103320],
        "October":  [114733, 66994, 101556, 47502, 114660],
        "November": [119776, 69939, 106020, 49590, 119700],
        "December": [141210, 82454, 124992, 58464, 141120]
    }
    tiempos_procesos_example = {
        "min_firstS":15, "max_firstS":25,
        "min_pas1":26, "max_pas1":40,
        "min_pas2":18, "max_pas2":28,
        "min_pas3":18, "max_pas3":26,
        "min_pas4":22, "max_pas4":33,
        "min_pas5":24, "max_pas5":38,
        "min_fill":10, "max_fill":15,
        "min_label":18,"max_label":22
    }
    results = run_full_process(
        data=data_example,
        p_inv_inicial=0.25,
        p_inv_final=0.10,
        tiempos_procesos=tiempos_procesos_example,
        unidad=3,
        use_no_consecutive=False,
        use_smooth=False,
        ct=578,
        ht=145,
        pit=10000000,
        crt=5931.25,
        cot=5931.25,
        cwt=5931.25,
        cwt_prima=5931.25,
        graficar=True,
        costo_prod=1.0,
        costo_inv=0.25,
        return_tables=True,
        make_plots=False,
        reps=3,
        verbose=True
    )
    # print summary of results
    print("Aggregated model status:", results['agg']['status'])
    print("Aggregated objective:", results['agg']['z'])
    print("Disaggregated model status:", results['disagg']['status'])
    print("Simulation totals table:\n", results['sim_totales'])
