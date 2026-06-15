"""
generate_graphs.py
==================
Generates 7 publication-quality figures for the EV scheduling MILP comparison.
All results come from running the real CPLEX solver — nothing is hardcoded.

Graphs produced (saved to graphs/):
  1. gantt_iarochen.png          — Gantt chart for the Iarochen CPLEX schedule
  2. gantt_m3_apd.png            — Gantt chart for the M3-APD CPLEX schedule
  3. model_size_comparison.png   — Variables & constraints grouped bar chart
  4. solve_time_comparison.png   — Solve time bar chart
  5. instance_data_table.png     — Rendered instance input-data table
  6. processing_time_heatmap.png — p_{jl} matrix heatmap
  7. scalability_theoretical.png — Theoretical complexity growth vs CPLEX CE cap

Run with:
  uv run --with matplotlib --with numpy --with docplex --with cplex python3 generate_graphs.py
"""

import os
import sys
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

os.makedirs("graphs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Import the solver module (suppress its top-level print output while
#           importing, then re-enable for the actual solve calls).
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 68)
print("Importing ev_schedule_cplex (suppressing top-level prints)…")
print("=" * 68)

# ev_schedule_cplex.py has module-level print + solve calls; we only want to
# run the two solver functions ourselves.  We import via importlib so we can
# redirect stdout during the module load (which triggers the top-level solves).
import importlib.util, io, contextlib

spec = importlib.util.spec_from_file_location(
    "ev_schedule_cplex",
    os.path.join(os.path.dirname(__file__), "ev_schedule_cplex.py"),
)
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    evscp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evscp)

# Re-use the instance data already built inside the module
J              = evscp.J
a              = evscp.a
d              = evscp.d
e              = evscp.e
p_l            = evscp.p_l
L              = evscp.L
charger_list   = evscp.charger_list
charger_type   = evscp.charger_type   # charger-id -> type-int
charger_types  = evscp.charger_types  # type-int   -> power kW
chargers_per_type = evscp.chargers_per_type

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Run the solvers and capture live results
# ─────────────────────────────────────────────────────────────────────────────
print("\n>>> Solving Iarochen formulation with CPLEX…")
try:
    st1, obj1, t1, sch_iarochen, v1, c1 = evscp.solve_iarochen_cplex()
    print(f"    status={st1}  obj={obj1}  time={t1:.4f}s  vars={v1}  cons={c1}")
except Exception as exc:
    print(f"  [ERROR] Iarochen solve failed: {exc}")
    st1, obj1, t1, sch_iarochen, v1, c1 = "ERROR", None, 0.0, {}, 0, 0

print("\n>>> Solving M3-APD formulation with CPLEX…")
try:
    st2, obj2, t2, sch_m3, v2, c2 = evscp.solve_m3_cplex(pos_max=4)
    print(f"    status={st2}  obj={obj2}  time={t2:.4f}s  vars={v2}  cons={c2}")
except Exception as exc:
    print(f"  [ERROR] M3-APD solve failed: {exc}")
    st2, obj2, t2, sch_m3, v2, c2 = "ERROR", None, 0.0, {}, 0, 0

results = {
    "Iarochen": {"vars": v1, "cons": c1, "time": t1, "obj": obj1, "status": st1},
    "M3-APD":   {"vars": v2, "cons": c2, "time": t2, "obj": obj2, "status": st2},
}

print("\n" + "=" * 68)
print("Live CPLEX results")
print("=" * 68)
for name, r in results.items():
    print(f"  {name:10s}: status={r['status']}  obj={r['obj']}  "
          f"time={r['time']:.4f}s  vars={r['vars']}  cons={r['cons']}")
print("=" * 68)

# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
TYPE_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}
TYPE_LABELS = {0: "Type L0 (11 kW)", 1: "Type L1 (22 kW)", 2: "Type L2 (43 kW)"}
FMT_COLORS  = {"Iarochen": "#4C72B0", "M3-APD": "#DD8452"}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — dark-theme Gantt chart
# ─────────────────────────────────────────────────────────────────────────────
def draw_gantt(schedule: dict, title: str, filename: str, show_position: bool = False):
    if not schedule:
        print(f"  [SKIP] No solution for {filename}")
        return

    chargers = sorted({s["charger"] for s in schedule.values()})
    charger_row = {c: i for i, c in enumerate(chargers)}
    n_rows = len(chargers)

    fig, ax = plt.subplots(figsize=(13, max(5, n_rows * 0.65 + 1.5)))
    fig.patch.set_facecolor("#1E1E2E")
    ax.set_facecolor("#2A2A3E")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555577")

    legend_handles: dict = {}

    for j, s in schedule.items():
        row   = charger_row[s["charger"]]
        start = s["start"]
        end   = s["end"]
        dur   = max(end - start, 0.05)           # guard against 0-width bars
        typ   = s["type"]
        tard  = s["tard"]
        color = TYPE_COLORS[typ]

        ax.barh(row, dur, left=start, height=0.55,
                color=color, alpha=0.88, edgecolor="white", linewidth=0.6)

        if tard > 0:
            ax.barh(row, tard, left=d[j], height=0.55,
                    color="#E74C3C", alpha=0.6, edgecolor="none")

        label = f"EV{j}" if not show_position else f"EV{j}\np{s['position']}"
        ax.text(start + dur / 2, row, label,
                va="center", ha="center", fontsize=7.5,
                color="white", fontweight="bold")

        # Arrival marker
        ax.annotate("▼", xy=(a[j], row + 0.36), fontsize=7,
                    color="#A0E0FF", ha="center", va="bottom")

        # Deadline marker
        ax.axvline(x=d[j], ymin=row / n_rows, ymax=(row + 1) / n_rows,
                   color="#FF6B6B", linewidth=1.0, linestyle="--", alpha=0.6)

        if typ not in legend_handles:
            legend_handles[typ] = mpatches.Patch(color=color, label=TYPE_LABELS[typ])

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [f"Charger {c}\n({charger_types[charger_type[c]]} kW)" for c in chargers],
        color="white", fontsize=9,
    )
    ax.set_xlabel("Time (slots)", color="white")
    ax.set_title(title, color="white", fontsize=14, pad=12)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")

    handles = [legend_handles[t] for t in sorted(legend_handles)]
    handles += [
        mpatches.Patch(color="#A0E0FF", label="▼ Arrival time"),
        mpatches.Patch(color="#FF6B6B", label="-- Deadline"),
    ]
    if any(s["tard"] > 0 for s in schedule.values()):
        handles.append(mpatches.Patch(color="#E74C3C", alpha=0.6, label="Tardiness"))

    ax.legend(handles=handles, loc="upper right",
              facecolor="#2A2A3E", edgecolor="#555577", labelcolor="white", fontsize=8)
    ax.set_xlim(left=0)
    ax.invert_yaxis()
    plt.tight_layout()

    path = f"graphs/{filename}"
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1 & 2 — GANTT CHARTS  (live schedule from CPLEX)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating Gantt charts…")
draw_gantt(
    sch_iarochen,
    f"Schedule — Iarochen et al. (2026)  [Assignment + Sequencing]  "
    f"obj={obj1}  time={t1:.4f}s",
    "gantt_iarochen.png",
)
draw_gantt(
    sch_m3,
    f"Schedule — M3-APD (Unlu & Mason 2010)  [Assignment + Positional Date]  "
    f"obj={obj2}  time={t2:.4f}s",
    "gantt_m3_apd.png",
    show_position=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3 — MODEL SIZE COMPARISON  (live vars/cons from CPLEX)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating model size comparison…")
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor("#1E1E2E")
ax.set_facecolor("#2A2A3E")

formulations = ["Iarochen", "M3-APD"]
metrics      = ["Variables", "Constraints"]
vals         = [[results[f]["vars"], results[f]["cons"]] for f in formulations]

x = np.arange(len(metrics))
w = 0.32
for i, (f, v) in enumerate(zip(formulations, vals)):
    b = ax.bar(x + (i - 0.5) * w, v, width=w,
               color=FMT_COLORS[f], label=f, alpha=0.88,
               edgecolor="white", linewidth=0.6)
    for rect, val in zip(b, v):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 8,
                str(val), ha="center", va="bottom", color="white",
                fontsize=10, fontweight="bold")

CAP = 1000
ax.axhline(CAP, color="#E74C3C", linewidth=1.6, linestyle="--",
           label="CPLEX CE cap (1,000)")
ax.text(1.5, CAP + 15, "CPLEX Community Edition cap",
        color="#E74C3C", fontsize=9, ha="right")

ax.set_xticks(x)
ax.set_xticklabels(metrics, color="white")
ax.set_ylabel("Count", color="white")
ax.set_title(
    f"Model Size Comparison — Iarochen vs. M3-APD  (n={len(J)}, m={len(charger_list)})",
    color="white", pad=12,
)
ax.tick_params(colors="white")
ax.legend(facecolor="#2A2A3E", edgecolor="#555577", labelcolor="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#555577")

plt.tight_layout()
plt.savefig("graphs/model_size_comparison.png", dpi=300,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  ✓ graphs/model_size_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4 — SOLVE TIME COMPARISON  (live times from CPLEX)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating solve time comparison…")
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor("#1E1E2E")
ax.set_facecolor("#2A2A3E")

times = [results[f]["time"] for f in formulations]
bars  = ax.bar(formulations, times, color=[FMT_COLORS[f] for f in formulations],
               alpha=0.88, edgecolor="white", linewidth=0.7, width=0.45)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{t:.4f} s", ha="center", va="bottom", color="white",
            fontsize=11, fontweight="bold")

ax.set_ylabel("CPLEX solve time (seconds)", color="white")
ax.set_title(
    f"Solve Time — Iarochen vs. M3-APD  (n={len(J)}, m={len(charger_list)})",
    color="white", pad=12,
)
ax.tick_params(colors="white")
ax.set_ylim(0, max(times) * 1.4 if max(times) > 0 else 0.1)
for spine in ax.spines.values():
    spine.set_edgecolor("#555577")

plt.tight_layout()
plt.savefig("graphs/solve_time_comparison.png", dpi=300,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  ✓ graphs/solve_time_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5 — INSTANCE DATA TABLE  (from module's instance data)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating instance data table…")
fig, ax = plt.subplots(figsize=(11, 4.2))
fig.patch.set_facecolor("#1E1E2E")
ax.set_facecolor("#1E1E2E")
ax.axis("off")

col_labels = [
    "EV j", "Arrival aⱼ", "Deadline dⱼ", "Energy eⱼ (kWh)",
    "p_{j,L0} (11kW)", "p_{j,L1} (22kW)", "p_{j,L2} (43kW)",
]
rows = [[j, a[j], d[j], e[j], p_l[j][0], p_l[j][1], p_l[j][2]] for j in J]

table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.65)

for (r, c_), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#4C72B0")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#2E2E4E")
        cell.set_text_props(color="white")
    else:
        cell.set_facecolor("#252540")
        cell.set_text_props(color="white")
    cell.set_edgecolor("#555577")

n_types = ", ".join(f"{cnt}×{charger_types[l]}kW" for l, cnt in chargers_per_type.items())
ax.set_title(
    f"Test Instance: {len(J)} EVs, {len(charger_list)} Chargers ({n_types})",
    color="white", fontsize=13, pad=14, y=0.95,
)

plt.tight_layout()
plt.savefig("graphs/instance_data_table.png", dpi=300,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  ✓ graphs/instance_data_table.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6 — PROCESSING TIME HEATMAP  (from module's p_l data)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating processing time heatmap…")
matrix = np.array([[p_l[j][l] for l in L] for j in J])

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor("#1E1E2E")
ax.set_facecolor("#2A2A3E")

im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(L)))
ax.set_xticklabels([f"L{l} ({charger_types[l]} kW)" for l in L], color="white", fontsize=11)
ax.set_yticks(range(len(J)))
ax.set_yticklabels([f"EV {j}" for j in J], color="white", fontsize=10)
ax.set_xlabel("Charger Type", color="white")
ax.set_ylabel("EV Job j", color="white")
ax.set_title(
    r"Processing-Time Matrix  $p_{jl} = \lceil e_j / (\tau \cdot w_l) \rceil$",
    color="white", fontsize=13, pad=12,
)
ax.tick_params(colors="white")

for i in range(len(J)):
    for k in range(len(L)):
        ax.text(k, i, str(matrix[i, k]),
                ha="center", va="center",
                color="black" if matrix[i, k] < matrix.max() * 0.65 else "white",
                fontsize=11, fontweight="bold")

cbar = plt.colorbar(im, ax=ax)
cbar.ax.yaxis.set_tick_params(color="white")
cbar.set_label("Processing time (slots)", color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

plt.tight_layout()
plt.savefig("graphs/processing_time_heatmap.png", dpi=300,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  ✓ graphs/processing_time_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7 — THEORETICAL SCALABILITY  (derived from the formulation structure)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating theoretical scalability chart…")
m_val = len(charger_list)   # total chargers (from module)
P_val = 4                   # POS_MAX used in solve_m3_cplex

n_range = np.arange(2, 50)

def iarochen_vars(n):
    return n * m_val + 3 * n + (n * (n - 1) // 2) * m_val

def iarochen_cons(n):
    return n + n + n + n + 2 * (n * (n - 1) // 2) * m_val

def m3_vars(n):
    return n * m_val * P_val + m_val * P_val + 2 * n

def m3_cons(n):
    return (n + m_val * P_val + m_val
            + m_val * (P_val - 1) + m_val * (P_val - 1)
            + n * m_val * P_val + n)

# Verify the formulas reproduce the live counts at n=len(J)
n0 = len(J)
print(f"  Formula check at n={n0}:")
print(f"    Iarochen vars  formula={iarochen_vars(n0)}  live={v1}")
print(f"    Iarochen cons  formula={iarochen_cons(n0)}  live={c1}")
print(f"    M3-APD   vars  formula={m3_vars(n0)}  live={v2}")
print(f"    M3-APD   cons  formula={m3_cons(n0)}  live={c2}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
fig.patch.set_facecolor("#1E1E2E")
CAP = 1000

panel_data = [
    (axes[0], "Variables vs. n",   iarochen_vars, m3_vars,   v1, v2),
    (axes[1], "Constraints vs. n", iarochen_cons, m3_cons,   c1, c2),
]

for ax, subtitle, fn_i, fn_m, live_i, live_m in panel_data:
    ax.set_facecolor("#2A2A3E")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555577")

    ax.plot(n_range, fn_i(n_range), color=FMT_COLORS["Iarochen"],
            linewidth=2.5, label="Iarochen")
    ax.plot(n_range, fn_m(n_range), color=FMT_COLORS["M3-APD"],
            linewidth=2.5, label="M3-APD")
    ax.axhline(CAP, color="#E74C3C", linewidth=1.8, linestyle="--",
               label="CPLEX CE cap (1,000)")

    for fn, col, live, tag in [
        (fn_i, FMT_COLORS["Iarochen"], live_i, "Iarochen"),
        (fn_m, FMT_COLORS["M3-APD"],   live_m, "M3-APD"),
    ]:
        cross = n_range[fn(n_range) > CAP]
        if len(cross):
            ax.axvline(cross[0], color=col, linewidth=1.2, linestyle=":", alpha=0.8)
            ax.text(cross[0] + 0.3, CAP * 0.55,
                    f"{tag} hits cap\nat n={cross[0]}", color=col, fontsize=8.5)

        # Dot at the actual n used
        ax.scatter([n0], [fn(n0)], color=col, s=70, zorder=5)
        ax.annotate(f"n={n0}: {live}", (n0, fn(n0)),
                    textcoords="offset points",
                    xytext=(6, 6 if tag == "Iarochen" else -16),
                    color="white", fontsize=8)

    ax.set_xlabel("Number of EVs (n)")
    ax.set_ylabel("Count")
    ax.set_title(subtitle, color="white")
    ax.legend(facecolor="#2A2A3E", edgecolor="#555577", labelcolor="white", fontsize=9)

fig.suptitle(
    f"Theoretical Model Growth vs. CPLEX Community Edition Cap  "
    f"(m={m_val} chargers, POS_MAX={P_val})\n"
    "Iarochen: O(n²m)   |   M3-APD: O(nmP)",
    color="white", fontsize=12, y=1.01,
)
plt.tight_layout()
plt.savefig("graphs/scalability_theoretical.png", dpi=300,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  ✓ graphs/scalability_theoretical.png")

print("\n✅  All 7 graphs saved to graphs/")
