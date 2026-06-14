"""
Comparison of Two MILP Formulations for EV Charging Scheduling
================================================================
Solved with REAL IBM CPLEX (Community Edition, via docplex).

Formulation 1 (Iarochen et al. 2026): assignment + sequencing (delta) vars
  Variables: x[j,l,r] binary, S[j] integer, C[j] integer, T[j] real,
             delta[j,k,l,r] binary
  Processing time: p[j][l] = ceil(e_j / (tau * w_l))  [paper Eq. 4]

Formulation 2 (M3-APD, Unlu & Mason 2010 adapted for EVCSP):
  Variables: u[j,l,r,pos] binary (job j at type-l charger r, position pos),
             c[l,r,pos] continuous (completion time at position pos on charger r
             of type l), C[j] continuous, T[j] real
  Implements constraints (16)-(21) from Unlu & Mason (2010) extended to
  the unrelated parallel machine (Rm) environment with release dates.

Fixes applied vs. initial draft:
  * math.ceil() used for all processing times  (paper Eq. p_jl = ceil(e_j/tau*w_l))
  * S[j], C[j] declared as integer_var in Iarochen model  (paper domain (10))
  * M3 variables indexed by (j, l, r, pos) — charger type l is explicit
  * Safe error handling around each solver call

NOTE on CPLEX Community Edition limits:
  The free CE enforces a hard cap of 1000 variables AND 1000 constraints.
  POS_MAX=4 keeps M3-APD well within these limits for the n=10 instance.
"""

import math
import time
from typing import Any

from docplex.mp.model import Model

# ─────────────────────────────────────────────────────────────────────────────
# 1. TEST INSTANCE  (identical to the earlier CBC run)
# ─────────────────────────────────────────────────────────────────────────────
tau = 1
charger_types = {0: 11, 1: 22, 2: 43}  # type -> power (kW)
chargers_per_type = {0: 4, 1: 3, 2: 2}  # type -> count

charger_list, charger_type, charger_power = [], {}, {}
cid = 0
for l, cnt in chargers_per_type.items():
    for _ in range(cnt):
        charger_list.append(cid)
        charger_type[cid] = l
        charger_power[cid] = charger_types[l]
        cid += 1
m = len(charger_list)

R = {l: [ci for ci in charger_list if charger_type[ci] == l] for l in charger_types}
L = list(charger_types.keys())

demands = [
    (0, 0, 5, 22),
    (1, 0, 4, 44),
    (2, 1, 6, 33),
    (3, 0, 3, 11),
    (4, 2, 8, 55),
    (5, 1, 5, 22),
    (6, 0, 7, 66),
    (7, 3, 9, 33),
    (8, 1, 6, 44),
    (9, 2, 7, 22),
]
n = len(demands)
J = list(range(n))
a = {j: demands[j][1] for j in J}
d = {j: demands[j][2] for j in J}
e = {j: demands[j][3] for j in J}

# Processing time p[j][l] = ceil(e_j / (tau * w_l)) — paper Eq. 4 / domain (10)
# Keyed by charger index i; all chargers of the same type share the same value.
p = {
    j: {i: math.ceil(e[j] / (tau * charger_power[i])) for i in charger_list}
    for j in J
}
# Also build type-indexed view p_l[j][l] used by M3 (release-date constraint)
p_l = {
    j: {l: math.ceil(e[j] / (tau * charger_types[l])) for l in L}
    for j in J
}

# BIG_M: upper bound on any completion time — uses ceiling processing times
BIG_M = sum(p[j][0] for j in J) + max(a.values()) + 10

print("=" * 68)
print(f"INSTANCE: n={n} EVs, m={m} chargers (11kWx4, 22kWx3, 43kWx2)")
print(f"BIG_M = {BIG_M:.2f}")
print("=" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FORMULATION 1 — IAROCHEN (docplex / CPLEX)
# ─────────────────────────────────────────────────────────────────────────────
def solve_iarochen_cplex():
    mdl = Model(name="Iarochen_EVCSP_CPLEX")

    x = {
        (j, l, r): mdl.binary_var(name=f"x_{j}_{l}_{r}")
        for j in J
        for l in L
        for r in R[l]
    }
    # Paper domain (10): S_j, C_j ∈ Z+  →  integer_var
    S = {j: mdl.integer_var(lb=0, name=f"S_{j}") for j in J}
    C = {j: mdl.integer_var(lb=0, name=f"C_{j}") for j in J}
    T = {j: mdl.continuous_var(lb=0, name=f"T_{j}") for j in J}
    delta = {
        (j, k, l, r): mdl.binary_var(name=f"d_{j}_{k}_{l}_{r}")
        for j in J
        for k in J
        if j < k
        for l in L
        for r in R[l]
    }

    mdl.minimize(mdl.sum(T[j] for j in J))

    for j in J:
        mdl.add_constraint(mdl.sum(x[j, l, r] for l in L for r in R[l]) == 1)

    for j in J:
        mdl.add_constraint(S[j] >= a[j])

    for j in J:
        mdl.add_constraint(
            C[j] == S[j] + mdl.sum(p[j][r] * x[j, l, r] for l in L for r in R[l])
        )

    for j in J:
        mdl.add_constraint(T[j] >= C[j] - d[j])

    for j in J:
        for k in J:
            if j >= k:
                continue
            for l in L:
                for r in R[l]:
                    mdl.add_constraint(
                        S[j] + p[j][r] * x[j, l, r]
                        <= S[k]
                        + BIG_M * (3 - delta[j, k, l, r] - x[j, l, r] - x[k, l, r])
                    )
                    mdl.add_constraint(
                        S[k] + p[k][r] * x[k, l, r]
                        <= S[j]
                        + BIG_M * (2 + delta[j, k, l, r] - x[j, l, r] - x[k, l, r])
                    )

    n_vars = mdl.number_of_variables
    n_cons = mdl.number_of_constraints
    print(f"\n[Iarochen] vars={n_vars}, constraints={n_cons}")

    t0 = time.time()
    sol = mdl.solve(log_output=False)
    elapsed = time.time() - t0

    status = mdl.solve_details.status if sol else "INFEASIBLE/NO SOL"
    obj = sol.get_objective_value() if sol else None

    schedule = {}
    if sol:
        for j in J:
            for l in L:
                for r in R[l]:
                    if sol.get_value(x[j, l, r]) > 0.5:
                        schedule[j] = {
                            "charger": r,
                            "type": l,
                            "start": round(sol.get_value(S[j]), 4),
                            "end": round(sol.get_value(C[j]), 4),
                            "tard": round(sol.get_value(T[j]), 4),
                        }
    return status, obj, elapsed, schedule, n_vars, n_cons


# ─────────────────────────────────────────────────────────────────────────────
# 3. FORMULATION 2 — M3-APD  (Unlu & Mason 2010, adapted for EVCSP)
#    Variables indexed explicitly by charger type l  (as required by brief)
#    Implements constraints (16)-(21) from Unlu & Mason (2010).
# ─────────────────────────────────────────────────────────────────────────────
def solve_m3_cplex(pos_max: int = 4) -> tuple[str, Any, float, dict, int, int]:
    """M3 Assignment-and-Positional-Date formulation.

    Decision variables
    ------------------
    u[j, l, r, pos] : binary  — job j assigned to position pos on charger r
                                of type l  (charger type l is EXPLICIT per brief)
    c[l, r, pos]    : continuous ≥ 0 — completion time at position pos on
                                charger r of type l
    C[j]            : continuous ≥ 0 — completion time of job j
    T[j]            : continuous ≥ 0 — tardiness of job j

    Constraints from Unlu & Mason (2010) M3:
      (16) each job → exactly one (charger, position)
      (17) each (charger, position) → at most one job
      (18) first-position completion time ≥ processing time
      (19) subsequent positions chain off previous completion time
      (21) release-date: c[l,r,pos] ≥ (a_j + p_jl) * u[j,l,r,pos]  for all j
      (20) link positional date to job completion via big-M
    """
    POS = list(range(1, pos_max + 1))

    mdl = Model(name="M3_EV_APD_CPLEX")

    # u[j, l, r, pos]: job j at type-l charger r, position pos
    u = {
        (j, l, r, pos): mdl.binary_var(name=f"u_{j}_{l}_{r}_{pos}")
        for j in J
        for l in L
        for r in R[l]
        for pos in POS
    }
    # c[l, r, pos]: completion time of the job at position pos on charger r (type l)
    c = {
        (l, r, pos): mdl.continuous_var(lb=0, name=f"c_{l}_{r}_{pos}")
        for l in L
        for r in R[l]
        for pos in POS
    }
    C = {j: mdl.continuous_var(lb=0, name=f"C_{j}") for j in J}
    T = {j: mdl.continuous_var(lb=0, name=f"T_{j}") for j in J}

    mdl.minimize(mdl.sum(T[j] for j in J))

    # (16) each demand -> exactly one (type, charger, position)
    for j in J:
        mdl.add_constraint(
            mdl.sum(u[j, l, r, pos] for l in L for r in R[l] for pos in POS) == 1,
            ctname=f"assign_{j}",
        )

    # (17) at most one EV per (charger, position) slot
    for l in L:
        for r in R[l]:
            for pos in POS:
                mdl.add_constraint(
                    mdl.sum(u[j, l, r, pos] for j in J) <= 1,
                    ctname=f"slot_{l}_{r}_{pos}",
                )

    # (18) first position: c[l,r,1] >= p_jl * u[j,l,r,1]  for all j
    # (21) release date:   c[l,r,1] >= (a_j + p_jl) * u[j,l,r,1]  for all j
    # Note: (21) dominates (18) since a_j >= 0, so only (21) is added.
    for l in L:
        for r in R[l]:
            mdl.add_constraint(
                c[l, r, 1] >= mdl.sum(
                    (a[j] + p_l[j][l]) * u[j, l, r, 1] for j in J
                ),
                ctname=f"pos1_rel_{l}_{r}",
            )

    # (19) subsequent positions: chain completion times
    # (21) also applies: c[l,r,pos] >= (a_j + p_jl) * u[j,l,r,pos]
    for l in L:
        for r in R[l]:
            for pos in POS:
                if pos == 1:
                    continue
                # (19): c[l,r,pos] >= c[l,r,pos-1] + sum_j(p_jl * u[j,l,r,pos])
                mdl.add_constraint(
                    c[l, r, pos] >= c[l, r, pos - 1]
                    + mdl.sum(p_l[j][l] * u[j, l, r, pos] for j in J),
                    ctname=f"chain_{l}_{r}_{pos}",
                )
                # (21): c[l,r,pos] >= (a_j + p_jl) * u[j,l,r,pos]  for each j
                mdl.add_constraint(
                    c[l, r, pos] >= mdl.sum(
                        (a[j] + p_l[j][l]) * u[j, l, r, pos] for j in J
                    ),
                    ctname=f"reldate_{l}_{r}_{pos}",
                )

    # (20) link positional completion date to job completion time (big-M)
    for j in J:
        for l in L:
            for r in R[l]:
                for pos in POS:
                    mdl.add_constraint(
                        C[j] >= c[l, r, pos] - BIG_M * (1 - u[j, l, r, pos]),
                        ctname=f"link_{j}_{l}_{r}_{pos}",
                    )

    # Tardiness: T_j >= C_j - d_j
    for j in J:
        mdl.add_constraint(T[j] >= C[j] - d[j], ctname=f"tard_{j}")

    n_vars = mdl.number_of_variables
    n_cons = mdl.number_of_constraints
    print(f"[M3-APD]   vars={n_vars}, constraints={n_cons}  (POS_MAX={pos_max})")

    t0 = time.time()
    sol = mdl.solve(log_output=False)
    elapsed = time.time() - t0

    status = mdl.solve_details.status if sol else "INFEASIBLE/NO SOL"
    obj = sol.get_objective_value() if sol else None

    schedule: dict = {}
    if sol:
        for j in J:
            for l in L:
                for r in R[l]:
                    for pos in POS:
                        if sol.get_value(u[j, l, r, pos]) > 0.5:
                            c_val = sol.get_value(c[l, r, pos])
                            start_val = max(0.0, c_val - p_l[j][l])  # clamp -0.0 artefact
                            schedule[j] = {
                                "charger": r,
                                "type": l,
                                "position": pos,
                                "start": round(start_val, 4),
                                "end": round(c_val, 4),
                                "tard": round(sol.get_value(T[j]), 4),
                            }
    return status, obj, elapsed, schedule, n_vars, n_cons


# ─────────────────────────────────────────────────────────────────────────────
# 4. RUN  (with graceful error handling for missing CPLEX runtime)
# ─────────────────────────────────────────────────────────────────────────────
print("\n>>> Solving Iarochen formulation with real CPLEX (Community Edition)...")
try:
    st1, obj1, t1, sch1, v1, c1 = solve_iarochen_cplex()
except Exception as exc:  # noqa: BLE001
    print(f"  [ERROR] Iarochen solve failed: {exc}")
    st1, obj1, t1, sch1, v1, c1 = "ERROR", None, 0.0, {}, 0, 0

print("\n>>> Solving M3-APD formulation with real CPLEX (Community Edition)...")
try:
    st2, obj2, t2, sch2, v2, c2 = solve_m3_cplex(pos_max=4)
except Exception as exc:  # noqa: BLE001
    print(f"  [ERROR] M3-APD solve failed: {exc}")
    st2, obj2, t2, sch2, v2, c2 = "ERROR", None, 0.0, {}, 0, 0


def _fmt_obj(val: Any) -> str:
    """Format objective value safely — returns 'N/A' when no solution exists."""
    return f"{val:>15.4f}" if val is not None else f"{'N/A':>15}"


SEP = "=" * 68
print(f"\n{SEP}")
print("RESULTS — SOLVED WITH IBM CPLEX (Community Edition v22.2)")
print(SEP)
print(f"{'':28s} {'Iarochen':>15} {'M3-APD':>15}")
print(f"{'Solver status':28s} {st1:>15} {st2:>15}")
print(f"{'Total tardiness':28s}{_fmt_obj(obj1)}{_fmt_obj(obj2)}")
print(f"{'CPLEX solve time (s)':28s} {t1:>15.4f} {t2:>15.4f}")
print(f"{'Variables':28s} {v1:>15} {v2:>15}")
print(f"{'Constraints':28s} {c1:>15} {c2:>15}")
print(SEP)

if sch1:
    print("\nSCHEDULE — Iarochen (CPLEX)")
    print(
        f"{'EV':>3} {'a':>3} {'d':>3} {'chgr':>5} {'typ':>4} {'start':>6} {'end':>6} {'tard':>5}"
    )
    for j in J:
        s = sch1[j]
        print(
            f"{j:>3} {a[j]:>3} {d[j]:>3} {s['charger']:>5} {s['type']:>4} "
            f"{s['start']:>6.2f} {s['end']:>6.2f} {s['tard']:>5.2f}"
        )
else:
    print("\nSCHEDULE — Iarochen: no solution to display.")

if sch2:
    print("\nSCHEDULE — M3-APD (CPLEX)")
    print(
        f"{'EV':>3} {'a':>3} {'d':>3} {'chgr':>5} {'typ':>4} {'pos':>4} {'start':>6} {'end':>6} {'tard':>5}"
    )
    for j in J:
        s = sch2[j]
        print(
            f"{j:>3} {a[j]:>3} {d[j]:>3} {s['charger']:>5} {s['type']:>4} {s['position']:>4} "
            f"{s['start']:>6.2f} {s['end']:>6.2f} {s['tard']:>5.2f}"
        )
else:
    print("\nSCHEDULE — M3-APD: no solution to display.")
