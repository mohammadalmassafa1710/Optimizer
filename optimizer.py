import pandas as pd
import math
from pulp import *
from collections import defaultdict

def run_cutting_stock_optimization(orders_list, params):
    """
    Runs the Cutting Stock MILP Optimization.
    
    Parameters:
    - orders_list: list of dicts, keys: ['Customer', 'Product', 'GSM', 'Roll Width', 'Quantity']
    - params: dict with keys:
        - deckle (int)
        - min_fill (int)
        - max_rolls (int)
        - prod_per_run (float)
        - tolerance (float)
        - mip_gap (float)
        - time_limit (int)
        
    Returns:
    - status: str (solver status)
    - summary: dict
    - production_plan: list of dicts
    - deckle_detail: list of dicts
    - remaining_by_customer: list of dicts
    - widths_used: list of dicts
    """
    DECKLE = int(params.get('deckle', 3800))
    MIN_FILL = int(params.get('min_fill', 3500))
    MAX_ROLLS = int(params.get('max_rolls', 3))
    PROD_PER_RUN = float(params.get('prod_per_run', 10.0))
    TOL = float(params.get('tolerance', 0.10))
    MIP_GAP = float(params.get('mip_gap', 0.005))
    TIME_LIMIT = int(params.get('time_limit', 120))

    # Load orders
    df_orig = pd.DataFrame(orders_list)
    if df_orig.empty:
        return "Empty Orders", {}, [], [], [], []
        
    # Standardize columns
    df_orig['Roll Width'] = df_orig['Roll Width'].astype(int)
    df_orig['Quantity'] = df_orig['Quantity'].astype(float)
    
    demand = df_orig.groupby('Roll Width')['Quantity'].sum().to_dict()
    widths = sorted(demand.keys())
    W = len(widths)
    w_idx = {w: i for i, w in enumerate(widths)}

    width_meta = df_orig.sort_values('Quantity', ascending=False).groupby('Roll Width').first().reset_index()
    width_to_customer = dict(zip(width_meta['Roll Width'], width_meta['Customer']))
    width_to_product = dict(zip(width_meta['Roll Width'], width_meta['Product']))
    width_to_gsm = dict(zip(width_meta['Roll Width'], width_meta['GSM']))

    # Pattern generation
    def generate_patterns(widths, deckle, min_fill, max_rolls):
        results = []
        def recurse(idx, rem_mm, rem_rolls, cur):
            if idx == len(widths):
                if deckle - rem_mm >= min_fill and cur:
                    results.append(dict(cur))
                return
            w = widths[idx]
            for cnt in range(0, min(int(rem_mm // w), rem_rolls) + 1):
                if cnt > 0: 
                    cur[w] = cnt
                else:       
                    cur.pop(w, None)
                recurse(idx + 1, rem_mm - w * cnt, rem_rolls - cnt, cur)
            cur.pop(w, None)
        recurse(0, deckle, max_rolls, {})
        return results

    raw_patterns = generate_patterns(widths, DECKLE, MIN_FILL, MAX_ROLLS)
    if not raw_patterns:
        return "No valid cutting patterns could be generated with the current Deckle and Min Fill parameters.", {}, [], [], [], []

    # Build A matrix
    def build_vectors(patterns):
        A, pat_waste, pat_total = [], [], []
        for pat in patterns:
            total_w = sum(w * c for w, c in pat.items())
            tons_row = [0.0] * W
            for w, c in pat.items():
                tons_row[w_idx[w]] = PROD_PER_RUN * (w * c) / total_w
            A.append(tons_row)
            pat_waste.append(DECKLE - total_w)
            pat_total.append(total_w)
        return A, pat_waste, pat_total

    A, pat_waste, pat_total = build_vectors(raw_patterns)

    # Dominance pruning
    def prune(patterns, A, pat_waste):
        n, dominated = len(patterns), set()
        for a in range(n):
            if a in dominated: continue
            for b in range(n):
                if b == a or b in dominated: continue
                if pat_waste[a] <= pat_waste[b] and all(A[a][i] >= A[b][i] for i in range(W)):
                    dominated.add(b)
        keep = [i for i in range(n) if i not in dominated]
        return keep

    keep = prune(raw_patterns, A, pat_waste)
    raw_patterns = [raw_patterns[i] for i in keep]
    A = [A[i] for i in keep]
    pat_waste = [pat_waste[i] for i in keep]
    pat_total = [pat_total[i] for i in keep]
    P = len(raw_patterns)

    # Per-pattern upper bounds (BOX constraints)
    UB = []
    for p in range(P):
        ub = 100_000
        for i, w in enumerate(widths):
            if A[p][i] > 1e-9:
                ub = min(ub, math.floor((1 + TOL) * demand[w] / A[p][i]))
        UB.append(max(ub, 0))

    # LP Relaxation
    lp = LpProblem("LP", LpMaximize)
    y = [LpVariable(f"y{p}", lowBound=0, upBound=UB[p], cat='Continuous') for p in range(P)]
    lp += lpSum(y[p] * PROD_PER_RUN for p in range(P))
    for i, w in enumerate(widths):
        produced = lpSum(A[p][i] * y[p] for p in range(P))
        lp += produced >= (1 - TOL) * demand[w]
        lp += produced <= (1 + TOL) * demand[w]

    lp.solve(PULP_CBC_CMD(msg=0))
    lp_bound = value(lp.objective) or 0.0

    # MILP
    milp = LpProblem("MILP", LpMaximize)
    x = [LpVariable(f"x{p}", lowBound=0, upBound=UB[p], cat='Integer') for p in range(P)]
    milp += lpSum(x[p] * PROD_PER_RUN for p in range(P))

    for i, w in enumerate(widths):
        produced = lpSum(A[p][i] * x[p] for p in range(P))
        milp += produced >= (1 - TOL) * demand[w]
        milp += produced <= (1 + TOL) * demand[w]

    # Solve MILP
    # Note: msg=0 to keep logs clean in server stdout, but can be 1. We will use 0.
    milp.solve(PULP_CBC_CMD(msg=0, timeLimit=TIME_LIMIT, gapRel=MIP_GAP, warmStart=True))

    status_str = LpStatus[milp.status]
    if status_str not in ['Optimal', 'Closed']:
        # If infeasible or status is not optimal, return immediately with status info
        # to prevent math crashes below.
        # But wait, let's still compile whatever we got if it's feasible but not optimal.
        if milp.status == LpStatusInfeasible or milp.status == LpStatusUndefined:
            return status_str, {"Solver Status": status_str}, [], [], [], []

    ip_production = value(milp.objective) or 0.0
    ip_runs = sum(int(round(x[p].varValue or 0)) for p in range(P))
    gap_pct = (lp_bound - ip_production) / lp_bound * 100 if lp_bound > 0 else 0

    # Build output structures
    plan_rows, detail_rows = [], []
    produced_by_width = defaultdict(float)
    deckle_no = 1

    for p in range(P):
        val = x[p].varValue
        if not val or val < 0.5:
            continue
        pat = raw_patterns[p]
        deckle_count = int(round(val))
        total_width = pat_total[p]
        waste = pat_waste[p]
        total_rolls = sum(pat.values())
        plan_total = 0.0
        slots = []

        for w, repeat in sorted(pat.items()):
            ppr = round(PROD_PER_RUN * w * repeat / total_width, 4)
            tp = round(ppr * deckle_count, 4)
            produced_by_width[w] += tp
            plan_total += tp
            detail_rows.append({
                "Deckle No.": deckle_no,
                "Product": width_to_product.get(w, ""),
                "GSM": width_to_gsm.get(w, ""),
                "Customer": width_to_customer.get(w, ""),
                "Roll Width (mm)": w,
                "Repeat in Deckle": repeat,
                "Produced per Run (ton)": ppr,
                "Deckle Count": deckle_count,
                "Total Produced (ton)": tp
            })
            slots.append(f"{w}mm" + (f"x{repeat}" if repeat > 1 else ""))

        dom = max(pat.keys(), key=lambda w: demand.get(w, 0))
        plan_rows.append({
            "Deckle No.": deckle_no,
            "Product": width_to_product.get(dom, ""),
            "GSM": width_to_gsm.get(dom, ""),
            "Roll Widths Used": " + ".join(slots),
            "Total Width (mm)": total_width,
            "Waste (mm)": waste,
            "Rolls in Deckle": total_rolls,
            "Deckle Count": deckle_count,
            "Total Production (ton)": round(plan_total, 4),
            "Customers": ", ".join(
                f"{width_to_customer.get(w,'?')[:15]}@{w}mm"
                for w in sorted(pat.keys())
            )
        })
        deckle_no += 1

    # Remaining by customer
    remaining_rows = []
    total_orders = len(df_orig)
    closed_orders = 0
    ord_by_width = df_orig.groupby('Roll Width')['Quantity'].sum().to_dict()

    for _, r in df_orig.iterrows():
        w = r['Roll Width']
        share = r['Quantity'] / ord_by_width[w] if ord_by_width[w] else 0
        produced = round(produced_by_width.get(w, 0.0) * share, 4)
        rem = round(r['Quantity'] - produced, 4)
        closed = rem <= r['Quantity'] * TOL
        if closed: 
            closed_orders += 1
        remaining_rows.append({
            "Customer": r['Customer'],
            "Product": r['Product'],
            "GSM": r['GSM'],
            "Roll Width (mm)": w,
            "Ordered Qty (ton)": round(r['Quantity'], 2),
            "Max Allowed (ton)": round(r['Quantity'] * (1 + TOL), 2),
            "Produced (ton)": produced,
            "Remaining Qty (ton)": rem,
            "Status": "CLOSED" if closed else "OPEN",
            "Reason": "" if closed else "Below minimum coverage threshold."
        })

    # Width utilization
    plan_df = pd.DataFrame(plan_rows)
    ws_rows = []
    if not plan_df.empty:
        ws = plan_df.groupby("Total Width (mm)").agg(
            DC=("Deckle Count", "sum"), TP=("Total Production (ton)", "sum")
        ).reset_index()
        total_dc = ws["DC"].sum()
        total_tp = ws["TP"].sum()
        
        for _, row in ws.iterrows():
            ws_rows.append({
                "Total Width (mm)": int(row["Total Width (mm)"]),
                "Deckle Count": int(row["DC"]),
                "% of Total Deckles": round(row["DC"] / total_dc * 100, 2) if total_dc > 0 else 0,
                "Total Production (ton)": round(row["TP"], 2),
                "% of Total Production": round(row["TP"] / total_tp * 100, 2) if total_tp > 0 else 0
            })
        ws_rows.sort(key=lambda x: x['Deckle Count'], reverse=True)

    total_production = plan_df["Total Production (ton)"].sum() if not plan_df.empty else 0.0
    total_waste_mm = int((plan_df["Waste (mm)"] * plan_df["Deckle Count"]).sum()) if not plan_df.empty else 0
    total_deckles = int(plan_df["Deckle Count"].sum()) if not plan_df.empty else 0
    closure_pct = closed_orders / total_orders * 100 if total_orders else 0
    
    # Calculate waste percentage
    # Total mm produced = total_deckles * DECKLE
    total_mm = total_deckles * DECKLE
    waste_pct = (total_waste_mm / total_mm * 100) if total_mm > 0 else 0.0

    summary = {
        "Solver Status": status_str,
        "LP Upper Bound (tons)": round(lp_bound, 2),
        "MILP Total Production (ton)": round(ip_production, 2),
        "Optimality Gap (%)": round(gap_pct, 2),
        "Total Deckles": total_deckles,
        "Total Production (ton)": round(total_production, 2),
        "Total Waste (mm)": total_waste_mm,
        "Waste Percentage (%)": round(waste_pct, 2),
        "Max Rolls per Deckle": MAX_ROLLS,
        "Min Fill (mm)": MIN_FILL,
        "Tolerance (+/-)": f"{int(TOL*100)}%",
        "Total Orders": total_orders,
        "Closed Orders": closed_orders,
        "Overall Closure Rate (%)": round(closure_pct, 2)
    }

    return status_str, summary, plan_rows, detail_rows, remaining_rows, ws_rows
