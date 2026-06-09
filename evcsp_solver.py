"""
Electric Vehicle Charging Scheduling Problem (EVCSP)
=====================================================
Minimizes total tardiness across all EVs using three methods:
  1. MILP  – exact model via docplex (exported to LP, solved with CBC / CPLEX)
  2. Greedy Heuristic – priority-based urgency-index construction
  3. GVNS  – General Variable Neighborhood Search (Shake + VND)

Reference:
  Iarochen et al., "Scheduling Electric Vehicle Charging to Minimize Total
  Tardiness", EvoApplications 2026, LNCS 16525, pp. 85-99.
"""

from __future__ import annotations

import math
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ── PuLP / CBC for solving (fallback when CPLEX runtime is absent) ───────────
import pulp

# ── PuLP is used for MILP model building ──────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ChargerType:
    """Represents one type of charger at the station."""

    type_id: int  # 1-indexed type label
    power_kw: float  # Charging power in kW  (w_l)
    count: int  # Number of chargers of this type  (m_l)


@dataclass
class EV:
    """Represents one EV charging demand."""

    job_id: int  # 1-indexed job label
    arrival: float  # Arrival time  (a_j)  [time slots]
    deadline: float  # Desired departure time  (d_j)  [time slots]
    energy_kwh: float  # Required energy  (e_j)  [kWh]


@dataclass
class ChargingStation:
    """
    A charging station with K charger types.
    Internally, chargers are indexed 0..m-1 in a flat list; each carries
    its type_id so that processing times can be looked up quickly.
    """

    charger_types: List[ChargerType]
    tau: float = 1.0  # Duration of one time slot in hours

    # Derived attributes, populated in __post_init__
    chargers: List[int] = field(default_factory=list)  # flat list of type_ids
    total_chargers: int = 0
    _type_dict: dict = field(default_factory=dict)

    def __post_init__(self):
        self.chargers = []
        for ct in self.charger_types:
            self.chargers.extend([ct.type_id] * ct.count)
            self._type_dict[ct.type_id] = ct
        self.total_chargers = len(self.chargers)

    def processing_time(self, ev: EV, type_id: int) -> float:
        """p_{jl} = ceil(e_j / (tau * w_l)) – integer number of time slots."""
        ct = self._type_dict[type_id]
        return math.ceil(ev.energy_kwh / (self.tau * ct.power_kw))

    def charger_type(self, charger_idx: int) -> int:
        """Return type_id for a 0-based charger index."""
        return self.chargers[charger_idx]


# ── Solution representation ──────────────────────────────────────────────────


@dataclass
class Assignment:
    """Stores the full schedule for one EV."""

    job_id: int
    charger_idx: int  # 0-based charger index
    start: float
    end: float
    tardiness: float


@dataclass
class Schedule:
    """Complete schedule: list of assignments + aggregated metrics."""

    assignments: List[Assignment] = field(default_factory=list)

    @property
    def total_tardiness(self) -> float:
        return sum(a.tardiness for a in self.assignments)

    @property
    def max_tardiness(self) -> float:
        return max((a.tardiness for a in self.assignments), default=0.0)

    @property
    def pct_tardy(self) -> float:
        n = len(self.assignments)
        if n == 0:
            return 0.0
        return 100.0 * sum(1 for a in self.assignments if a.tardiness > 0) / n

    def summary(self) -> dict:
        return {
            "total_tardiness": round(self.total_tardiness, 4),
            "max_tardiness": round(self.max_tardiness, 4),
            "pct_tardy_evs": round(self.pct_tardy, 2),
            "n_assigned": len(self.assignments),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  INSTANCE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════


def generate_instance(
    n_evs: int,
    station: ChargingStation,
    mode: str = "loose",
    tight_fraction: float = 0.5,
    seed: Optional[int] = None,
) -> List[EV]:
    """
    Synthetic instance generator following Zaidi et al. (2024) protocol.

    Parameters
    ----------
    n_evs        : fleet size
    station      : charging station definition (used for reference power)
    mode         : 'loose' or 'tight'
    tight_fraction: fraction of EVs with alpha=0.1 in tight mode
    seed         : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    evs: List[EV] = []

    ref_power = station.charger_types[0].power_kw  # 11 kW reference charger

    for j in range(1, n_evs + 1):
        arrival = float(rng.uniform(0, 0.2 * n_evs))
        energy = float(rng.uniform(5.5, 66.0))  # kWh
        p1j = math.ceil(energy / (station.tau * ref_power))  # slots on type-1

        if mode == "loose":
            alpha = float(rng.uniform(0.1, 1.0))
        else:  # tight
            if rng.random() < tight_fraction:
                alpha = 0.1
            else:
                alpha = 0.2

        deadline = arrival + (1 + alpha) * p1j
        evs.append(EV(job_id=j, arrival=arrival, deadline=deadline, energy_kwh=energy))

    return evs


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  SCHEDULE BUILDER  (shared by heuristic and GVNS)
# ═══════════════════════════════════════════════════════════════════════════════


def build_schedule_from_order(
    order: List[int],  # job_ids in processing order
    ev_dict: dict,  # job_id -> EV
    station: ChargingStation,
    horizon: float,
) -> Schedule:
    """
    Greedy schedule builder: given an ordered permutation of EVs, assigns
    each EV (in order) to the earliest-available compatible charger slot
    that starts no earlier than the EV's arrival time.

    This is NOT the priority heuristic itself; it is the shared scheduling
    kernel used by both the greedy heuristic (after sorting by urgency) and
    the GVNS (after permutation moves).
    """
    # charger_end[r] = earliest time charger r becomes free
    charger_end = [0.0] * station.total_chargers

    assignments: List[Assignment] = []
    charger_types = [station.charger_type(r) for r in range(station.total_chargers)]

    for jid in order:
        ev = ev_dict[jid]
        best: Optional[Tuple[float, int]] = None  # (start_time, charger_idx)

        p_times = {}
        for ct in station.charger_types:
            p_times[ct.type_id] = station.processing_time(ev, ct.type_id)

        for r, type_id in enumerate(charger_types):
            p = p_times[type_id]
            # Earliest start: max(arrival, charger available)
            start = max(ev.arrival, charger_end[r])
            end = start + p
            if end > horizon:
                continue  # charger can't fit before horizon
            if best is None or start < best[0]:
                best = (start, r)

        if best is None:
            # Fall back: assign to charger that minimises tardiness ignoring horizon
            best_tard = math.inf
            fallback = None
            for r, type_id in enumerate(charger_types):
                p = p_times[type_id]
                start = max(ev.arrival, charger_end[r])
                end = start + p
                tard = max(0.0, end - ev.deadline)
                if tard < best_tard or fallback is None:
                    best_tard = tard
                    fallback = (start, r)
            best = fallback  # type: ignore[assignment]

        start_t, r = best  # type: ignore[misc]
        type_id = charger_types[r]
        p = p_times[type_id]
        end_t = start_t + p
        tard = max(0.0, end_t - ev.deadline)

        charger_end[r] = end_t
        assignments.append(
            Assignment(
                job_id=jid, charger_idx=r, start=start_t, end=end_t, tardiness=tard
            )
        )

    return Schedule(assignments=assignments)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  GREEDY PRIORITY HEURISTIC
# ═══════════════════════════════════════════════════════════════════════════════


class GreedyHeuristic:
    """
    Priority-based greedy heuristic (Section 4.1).

    Urgency index  K_{jl} = p_{jl} / (d_j - a_j)
    EVs are sorted in descending urgency; ties broken by tightest window.
    """

    def __init__(self, station: ChargingStation):
        self.station = station

    def solve(self, evs: List[EV], horizon: float) -> Tuple[Schedule, float]:
        t0 = time.perf_counter()

        ev_dict = {ev.job_id: ev for ev in evs}

        # Build priority list: (urgency, job_id, type_id)
        priority: List[Tuple[float, int, int]] = []
        for ev in evs:
            window = max(ev.deadline - ev.arrival, 1e-9)
            for ct in self.station.charger_types:
                p = self.station.processing_time(ev, ct.type_id)
                urgency = p / window
                priority.append((urgency, ev.job_id, ct.type_id))

        # Sort descending by urgency
        priority.sort(key=lambda x: -x[0])

        # Assign each EV exactly once in urgency order
        assigned = set()
        order: List[int] = []

        for _, jid, _type in priority:
            if jid not in assigned:
                assigned.add(jid)
                order.append(jid)

        sched = build_schedule_from_order(order, ev_dict, self.station, horizon)
        elapsed = time.perf_counter() - t0
        return sched, elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  GVNS METAHEURISTIC
# ═══════════════════════════════════════════════════════════════════════════════


class GVNS:
    """
    General Variable Neighborhood Search (Section 4.2).

    Three neighbourhoods on the EV permutation:
      N1 – Swap  (exchange positions of two EVs)
      N2 – Insert (remove EV, reinsert at different position)
      N3 – 2-opt reverse (reverse a sub-sequence)

    Shaking: repeated 2-opt (N3) reversal, h times per iteration.
    VND: sequential N1 → N2 → N3, first-improvement, restart from N1.
    """

    # ── GVNS hyper-parameters (Table 1) ──────────────────────────────────────
    MAX_ITER = 100
    MAX_NO_IMPROVE = 50
    H_MIN = 2
    H_MAX = 15
    DELTA_H = 1
    MAX_TIME = 10.0  # seconds per VND call
    MAX_RUNTIME = 900.0  # seconds total

    def __init__(self, station: ChargingStation, seed: Optional[int] = None):
        self.station = station
        self.rng = random.Random(seed)

    # ── neighbourhood operators ───────────────────────────────────────────────

    def _swap(self, perm: List[int], i: int, j: int) -> List[int]:
        p = perm[:]
        p[i], p[j] = p[j], p[i]
        return p

    def _insert(self, perm: List[int], i: int, j: int) -> List[int]:
        p = perm[:]
        ev = p.pop(i)
        p.insert(j, ev)
        return p

    def _reverse(self, perm: List[int], i: int, j: int) -> List[int]:
        p = perm[:]
        p[i : j + 1] = p[i : j + 1][::-1]
        return p

    # ── shaking ───────────────────────────────────────────────────────────────

    def _shake(self, perm: List[int], h: int) -> List[int]:
        n = len(perm)
        p = perm[:]
        for _ in range(h):
            if n < 2:
                break
            i = self.rng.randint(0, n - 2)
            j = self.rng.randint(i + 1, n - 1)
            p = self._reverse(p, i, j)
        return p

    # ── VND ───────────────────────────────────────────────────────────────────

    def _vnd(
        self,
        perm: List[int],
        ev_dict: dict,
        horizon: float,
        max_trials: int,
        max_time: float,
    ) -> List[int]:
        best_perm = perm[:]
        best_cost = build_schedule_from_order(
            best_perm, ev_dict, self.station, horizon
        ).total_tardiness
        n = len(perm)
        k = 1  # neighbourhood index 1-3
        t_start = time.perf_counter()

        while k <= 3:
            improved = False
            for _ in range(max_trials):
                if time.perf_counter() - t_start > max_time:
                    return best_perm

                i = self.rng.randint(0, n - 1)
                j = self.rng.randint(0, n - 1)
                if i == j:
                    continue

                if k == 1:  # Swap
                    candidate = self._swap(best_perm, i, j)
                elif k == 2:  # Insert
                    i2 = self.rng.randint(0, n - 1)
                    j2 = self.rng.randint(0, n - 1)
                    if i2 == j2:
                        continue
                    candidate = self._insert(best_perm, i2, j2)
                else:  # 2-opt reverse (k == 3)
                    lo, hi = (i, j) if i < j else (j, i)
                    candidate = self._reverse(best_perm, lo, hi)

                cost = build_schedule_from_order(
                    candidate, ev_dict, self.station, horizon
                ).total_tardiness

                if cost < best_cost:
                    best_perm = candidate
                    best_cost = cost
                    improved = True
                    k = 1  # restart from N1 (pipe strategy)
                    break

            if not improved:
                k += 1

        return best_perm

    # ── main GVNS loop ────────────────────────────────────────────────────────

    def solve(
        self,
        evs: List[EV],
        horizon: float,
        warm_start: Optional[Schedule] = None,
    ) -> Tuple[Schedule, float]:
        t0 = time.perf_counter()

        ev_dict = {ev.job_id: ev for ev in evs}
        n = len(evs)
        max_trials = min(300, int(1.5 * n))

        # ── Initial solution ──────────────────────────────────────────────────
        if warm_start is not None:
            best_perm = [
                a.job_id for a in sorted(warm_start.assignments, key=lambda a: a.start)
            ]
        else:
            # Use greedy heuristic order as warm start
            heuristic = GreedyHeuristic(self.station)
            h_sched, _ = heuristic.solve(evs, horizon)
            best_perm = [
                a.job_id for a in sorted(h_sched.assignments, key=lambda a: a.start)
            ]

        best_sched = build_schedule_from_order(
            best_perm, ev_dict, self.station, horizon
        )
        best_cost = best_sched.total_tardiness

        h = self.H_MIN
        no_improve = 0
        itr = 0

        while itr < self.MAX_ITER and no_improve < self.MAX_NO_IMPROVE:
            if time.perf_counter() - t0 > self.MAX_RUNTIME:
                break

            # Shaking
            perm_shaken = self._shake(best_perm, h)

            # VND intensification
            perm_improved = self._vnd(
                perm_shaken, ev_dict, horizon, max_trials, self.MAX_TIME
            )
            new_sched = build_schedule_from_order(
                perm_improved, ev_dict, self.station, horizon
            )
            new_cost = new_sched.total_tardiness

            if new_cost < best_cost:
                best_perm = perm_improved
                best_sched = new_sched
                best_cost = new_cost
                h = self.H_MIN
                no_improve = 0
            else:
                h = min(h + self.DELTA_H, self.H_MAX)
                no_improve += 1

            itr += 1

        elapsed = time.perf_counter() - t0
        return best_sched, elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MILP MODEL  (docplex formulation, solved with PuLP/CBC)
# ═══════════════════════════════════════════════════════════════════════════════


class MILPSolver:
    """
    Exact MILP formulation (Section 3.2).

    Variables
    ---------
    x[j,r]      binary  – EV j assigned to charger r
    S[j]        integer – start time of EV j
    C[j]        integer – completion time of EV j
    T[j]        real    – tardiness of EV j
    delta[j,k,r] binary – EV j before EV k on charger r

    This model is built natively with PuLP and solved with CBC.
    """

    def __init__(self, station: ChargingStation, time_limit: float = 300.0):
        self.station = station
        self.time_limit = time_limit

    def solve(self, evs: List[EV], horizon: float) -> Tuple[Optional[Schedule], float]:
        t0 = time.perf_counter()

        n = len(evs)
        ev_dict = {ev.job_id: ev for ev in evs}
        R = list(range(self.station.total_chargers))  # 0..m-1
        M = horizon * 3  # big-M

        mdl = pulp.LpProblem("EVCSP_TotalTardiness", pulp.LpMinimize)

        # ── Decision variables ────────────────────────────────────────────────
        x = pulp.LpVariable.dicts("x", ((ev.job_id, r) for ev in evs for r in R), cat=pulp.LpBinary)
        S = pulp.LpVariable.dicts("S", (ev.job_id for ev in evs), lowBound=0, cat=pulp.LpInteger)
        C = pulp.LpVariable.dicts("C", (ev.job_id for ev in evs), lowBound=0, cat=pulp.LpContinuous)
        T = pulp.LpVariable.dicts("T", (ev.job_id for ev in evs), lowBound=0, cat=pulp.LpContinuous)

        ev_ids_sorted = sorted([ev.job_id for ev in evs])
        delta = pulp.LpVariable.dicts("d", 
                                      ((j, k, r) for ri, r in enumerate(R) 
                                       for ii, j in enumerate(ev_ids_sorted) 
                                       for k in ev_ids_sorted[ii + 1:]), 
                                      cat=pulp.LpBinary)

        # ── Objective (1): minimise sum of tardiness ──────────────────────────
        mdl += pulp.lpSum(T[ev.job_id] for ev in evs)

        # ── Constraints ───────────────────────────────────────────────────────

        for ev in evs:
            # (2) Each EV assigned to exactly one charger
            mdl += pulp.lpSum(x[(ev.job_id, r)] for r in R) == 1, f"assign_{ev.job_id}"
            # (3) Start >= arrival
            mdl += S[ev.job_id] >= ev.arrival, f"arrival_{ev.job_id}"

            # (4) Completion = start + processing_time (on assigned charger)
            for r in R:
                type_id = self.station.charger_type(r)
                p = self.station.processing_time(ev, type_id)
                mdl += C[ev.job_id] >= S[ev.job_id] + p * x[(ev.job_id, r)] - M * (1 - x[(ev.job_id, r)]), f"comp_lb_{ev.job_id}_{r}"
                mdl += C[ev.job_id] <= S[ev.job_id] + p * x[(ev.job_id, r)] + M * (1 - x[(ev.job_id, r)]), f"comp_ub_{ev.job_id}_{r}"

            # (5)-(6) Tardiness = max(0, C_j - d_j)
            mdl += T[ev.job_id] >= C[ev.job_id] - ev.deadline, f"tard_{ev.job_id}"

        # (7)-(8) Non-overlap on each charger (big-M disjunctive)
        for r in R:
            type_id = self.station.charger_type(r)
            for ii, j in enumerate(ev_ids_sorted):
                ev_j = ev_dict[j]
                p_j = self.station.processing_time(ev_j, type_id)
                for k in ev_ids_sorted[ii + 1 :]:
                    ev_k = ev_dict[k]
                    p_k = self.station.processing_time(ev_k, type_id)
                    mdl += S[j] + p_j * x[(j, r)] <= S[k] + M * (3 - delta[(j, k, r)] - x[(j, r)] - x[(k, r)]), f"seq1_{j}_{k}_{r}"
                    mdl += S[k] + p_k * x[(k, r)] <= S[j] + M * (2 + delta[(j, k, r)] - x[(j, r)] - x[(k, r)]), f"seq2_{j}_{k}_{r}"

        # ── Solve ─────────────────────────────────────────────────────────────
        try:
            status = mdl.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=int(self.time_limit)))
        except Exception as exc:
            warnings.warn(f"MILP solver error: {exc}")
            status = pulp.LpStatusNotSolved

        elapsed = time.perf_counter() - t0

        if pulp.LpStatus[status] not in ("Optimal", "Feasible"):
            return None, elapsed

        # ── Extract schedule ──────────────────────────────────────────────────
        assignments = []
        for ev in evs:
            chosen_r = None
            for r in R:
                val = pulp.value(x[(ev.job_id, r)])
                if val is not None and val > 0.5:
                    chosen_r = r
                    break

            if chosen_r is None:
                continue

            start_val = pulp.value(S[ev.job_id]) or 0.0
            comp_val = pulp.value(C[ev.job_id]) or 0.0
            tard_val = pulp.value(T[ev.job_id]) or 0.0
            tard_val = max(0.0, tard_val)

            assignments.append(
                Assignment(
                    job_id=ev.job_id,
                    charger_idx=chosen_r,
                    start=float(start_val),
                    end=float(comp_val),
                    tardiness=float(tard_val),
                )
            )

        return Schedule(assignments=assignments), elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  BENCHMARKING RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


def run_benchmark(
    fleet_sizes: List[int] = (10, 30, 50, 100),
    modes: List[str] = ("loose", "tight"),
    seed: int = 42,
    milp_time_limit: float = 120.0,
    run_milp: bool = True,
) -> pd.DataFrame:
    """
    Run all three solvers for each (fleet_size, mode) combination and
    return a summary DataFrame matching Table 2 / Table 3 in the paper.
    """
    # Station: 9 chargers – 4×11 kW, 3×22 kW, 2×43 kW  (Section 5)
    station = ChargingStation(
        charger_types=[
            ChargerType(type_id=1, power_kw=11.0, count=4),
            ChargerType(type_id=2, power_kw=22.0, count=3),
            ChargerType(type_id=3, power_kw=43.0, count=2),
        ],
        tau=1.0,
    )

    heuristic_solver = GreedyHeuristic(station)
    gvns_solver = GVNS(station, seed=seed)
    milp_solver = MILPSolver(station, time_limit=milp_time_limit)

    rows = []

    for n in fleet_sizes:
        for mode in modes:
            evs = generate_instance(n, station, mode=mode, seed=seed + n)
            horizon = max(ev.deadline for ev in evs) * 2  # generous horizon

            print(f"  [{mode:5s}] n={n:4d} | ", end="", flush=True)

            # ── Greedy Heuristic ──────────────────────────────────────────────
            h_sched, h_time = heuristic_solver.solve(evs, horizon)
            h_sum = h_sched.summary()
            print(f"Heuristic OK ({h_time:.3f}s) | ", end="", flush=True)

            # ── GVNS ─────────────────────────────────────────────────────────
            g_sched, g_time = gvns_solver.solve(evs, horizon, warm_start=h_sched)
            g_sum = g_sched.summary()
            print(f"GVNS OK ({g_time:.1f}s) | ", end="", flush=True)

            # ── MILP ─────────────────────────────────────────────────────────
            m_sum = {
                "total_tardiness": float("nan"),
                "max_tardiness": float("nan"),
                "pct_tardy_evs": float("nan"),
                "n_assigned": 0,
            }
            m_time = float("nan")
            if run_milp and n <= 100:  # MILP only practical for small instances
                try:
                    m_sched, m_time = milp_solver.solve(evs, horizon)
                    if m_sched is not None:
                        m_sum = m_sched.summary()
                    print(f"MILP OK ({m_time:.1f}s)")
                except Exception as exc:
                    print(f"MILP error: {exc}")
            else:
                print("MILP skipped (too large)")

            rows.append(
                {
                    "mode": mode,
                    "n_evs": n,
                    # Heuristic
                    "H_total_tard": h_sum["total_tardiness"],
                    "H_max_tard": h_sum["max_tardiness"],
                    "H_pct_tardy": h_sum["pct_tardy_evs"],
                    "H_runtime_s": round(h_time, 4),
                    # GVNS
                    "G_total_tard": g_sum["total_tardiness"],
                    "G_max_tard": g_sum["max_tardiness"],
                    "G_pct_tardy": g_sum["pct_tardy_evs"],
                    "G_runtime_s": round(g_time, 2),
                    # MILP
                    "M_total_tard": m_sum["total_tardiness"],
                    "M_max_tard": m_sum["max_tardiness"],
                    "M_pct_tardy": m_sum["pct_tardy_evs"],
                    "M_runtime_s": round(m_time, 2)
                    if not math.isnan(m_time)
                    else float("nan"),
                }
            )

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  EVCSP – Minimize Total Tardiness")
    print("  Methods: MILP (PuLP + CBC)  |  Greedy Heuristic  |  GVNS")
    print("=" * 70)

    print("\nRunning benchmark (small to medium instances) …\n")
    df = run_benchmark(
        fleet_sizes=[10, 30, 50, 100, 200],
        modes=["loose", "tight"],
        seed=2024,
        milp_time_limit=120.0,
        run_milp=True,
    )

    print("\n\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    # ── Table 2 style: Total Tardiness + Runtime ──────────────────────────────
    print("\n── Table A: Total Tardiness & Runtime ──")
    cols_a = [
        "mode",
        "n_evs",
        "M_total_tard",
        "M_runtime_s",
        "H_total_tard",
        "H_runtime_s",
        "G_total_tard",
        "G_runtime_s",
    ]
    print(df[cols_a].to_string(index=False))

    # ── Table 3 style: Max Tardiness + % Tardy ───────────────────────────────
    print("\n── Table B: Max Tardiness & % Tardy EVs ──")
    cols_b = [
        "mode",
        "n_evs",
        "M_max_tard",
        "M_pct_tardy",
        "H_max_tard",
        "H_pct_tardy",
        "G_max_tard",
        "G_pct_tardy",
    ]
    print(df[cols_b].to_string(index=False))

    # ── Optimality gap (where MILP is available and non-zero) ────────────────
    print("\n── Table C: Optimality Gap GVNS vs MILP (where applicable) ──")
    gap_rows = []
    for _, row in df.iterrows():
        m_t = row["M_total_tard"]
        g_t = row["G_total_tard"]
        if not math.isnan(m_t) and m_t > 0:
            gap = 100.0 * (g_t - m_t) / m_t
            gap_rows.append(
                {
                    "mode": row["mode"],
                    "n_evs": int(row["n_evs"]),
                    "MILP": m_t,
                    "GVNS": g_t,
                    "gap_%": round(gap, 2),
                }
            )
        elif not math.isnan(m_t) and m_t == 0.0:
            gap_rows.append(
                {
                    "mode": row["mode"],
                    "n_evs": int(row["n_evs"]),
                    "MILP": 0.0,
                    "GVNS": g_t,
                    "gap_%": "n/a (÷0)",
                }
            )
    if gap_rows:
        print(pd.DataFrame(gap_rows).to_string(index=False))
    else:
        print("  No MILP solutions available for gap computation.")

    # ── Save to CSV ────────────────────────────────────────────────────────────
    out_path = "evcsp_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to: {out_path}")
    print("\nDone.")
