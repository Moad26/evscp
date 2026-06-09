import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import pulp

@dataclass
class ChargerType:
    type_id: int
    power_kw: float
    count: int

@dataclass
class EV:
    job_id: int
    arrival: float
    deadline: float
    energy_kwh: float

@dataclass
class ChargingStation:
    charger_types: List[ChargerType]
    tau: float = 1.0

    chargers: List[int] = field(default_factory=list)
    total_chargers: int = 0
    _type_dict: dict = field(default_factory=dict)

    def __post_init__(self):
        self.chargers = []
        for ct in self.charger_types:
            self.chargers.extend([ct.type_id] * ct.count)
            self._type_dict[ct.type_id] = ct
        self.total_chargers = len(self.chargers)

    def processing_time(self, ev: EV, type_id: int) -> float:
        ct = self._type_dict[type_id]
        return math.ceil(ev.energy_kwh / (self.tau * ct.power_kw))

    def charger_type(self, r: int) -> int:
        return self.chargers[r]

@dataclass
class Assignment:
    job_id: int; charger_idx: int; start: float; end: float; tardiness: float

@dataclass
class Schedule:
    assignments: List[Assignment]

class MILPSolver:
    def __init__(self, station: ChargingStation, time_limit: float = 300.0):
        self.station = station
        self.time_limit = time_limit

    def solve(self, evs: List[EV], horizon: float):
        import time
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
            mdl += pulp.lpSum(x[(ev.job_id, r)] for r in R) == 1, f"assign_{ev.job_id}"
            mdl += S[ev.job_id] >= ev.arrival, f"arrival_{ev.job_id}"
            for r in R:
                type_id = self.station.charger_type(r)
                p = self.station.processing_time(ev, type_id)
                mdl += C[ev.job_id] >= S[ev.job_id] + p * x[(ev.job_id, r)] - M * (1 - x[(ev.job_id, r)]), f"comp_lb_{ev.job_id}_{r}"
                mdl += C[ev.job_id] <= S[ev.job_id] + p * x[(ev.job_id, r)] + M * (1 - x[(ev.job_id, r)]), f"comp_ub_{ev.job_id}_{r}"
            mdl += T[ev.job_id] >= C[ev.job_id] - ev.deadline, f"tard_{ev.job_id}"

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
        status = mdl.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=int(self.time_limit)))
        elapsed = time.perf_counter() - t0

        if pulp.LpStatus[status] not in ("Optimal", "Feasible"):
            return None, elapsed

        assignments = []
        for ev in evs:
            chosen_r = None
            for r in R:
                val = pulp.value(x[(ev.job_id, r)])
                if val is not None and val > 0.5:
                    chosen_r = r
                    break
            if chosen_r is None: continue
            start_val = pulp.value(S[ev.job_id]) or 0.0
            comp_val = pulp.value(C[ev.job_id]) or 0.0
            tard_val = pulp.value(T[ev.job_id]) or 0.0
            assignments.append(Assignment(ev.job_id, chosen_r, float(start_val), float(comp_val), float(max(0.0, tard_val))))
        return Schedule(assignments), elapsed

s = ChargingStation([ChargerType(1, 11, 2)])
e = [EV(1, 0, 5, 20), EV(2, 1, 6, 20)]
sol = MILPSolver(s).solve(e, 20)
print(sol[0].assignments)
