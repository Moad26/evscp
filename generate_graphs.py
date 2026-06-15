"""
generate_graphs.py
==================
Generates 7 publication-quality figures for the EV scheduling MILP comparison.

Graphs produced (saved to graphs/):
  1. gantt_iarochen.png          — Gantt chart for Iarochen schedule
  2. gantt_m3_apd.png            — Gantt chart for M3-APD schedule
  3. model_size_comparison.png   — Variables & constraints grouped bar chart
  4. solve_time_comparison.png   — Solve time bar chart
  5. instance_data_table.png     — Rendered instance input-data table
  6. processing_time_heatmap.png — p_{jl} matrix heatmap
  7. scalability_theoretical.png — Theoretical complexity growth vs CPLEX CE cap

Run with:  uv run --with matplotlib --with numpy python3 generate_graphs.py
"""

import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

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
# INSTANCE DATA  (same as ev_schedule_cplex.py)
# ─────────────────────────────────────────────────────────────────────────────
tau = 1
charger_types = {0: 11, 1: 22, 2: 43}   # type -> power (kW)
chargers_per_type = {0: 4, 1: 3, 2: 2}  # type -> count

charger_list, charger_type_map, charger_power = [], {}, {}
cid = 0
for l, cnt in chargers_per_type.items():
    for _ in range(cnt):
        charger_list.append(cid)
        charger_type_map[cid] = l
        charger_power[cid] = charger_types[l]
        cid += 1

L = list(charger_types.keys())
R = {l: [ci for ci in charger_list if charger_type_map[ci] == l] for l in L}

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
J = list(range(len(demands)))
a = {j: demands[j][1] for j in J}
d = {j: demands[j][2] for j in J}
e = {j: demands[j][3] for j in J}

p_l = {j: {l: math.ceil(e[j] / (tau * charger_types[l])) for l in L} for j in J}

# ─────────────────────────────────────────────────────────────────────────────
# CPLEX RESULTS (hardcoded from last run)
# ─────────────────────────────────────────────────────────────────────────────
results = {
    "Iarochen": {"vars": 525, "cons": 850, "time": 0.0563, "obj": 0.0},
    "M3-APD":   {"vars": 416, "cons": 479, "time": 0.0784, "obj": 0.0},
}

# Schedule: j -> {charger, type, start, end, tard}
sch_iarochen = {
    0: {"charger": 4, "type": 1, "start": 0.0, "end": 1.0, "tard": 0.0},
    1: {"charger": 2, "type": 0, "start": 0.0, "end": 4.0, "tard": 0.0},
    2: {"charger": 6, "type": 1, "start": 1.0, "end": 3.0, "tard": 0.0},
    3: {"charger": 0, "type": 0, "start": 0.0, "end": 1.0, "tard": 0.0},
    4: {"charger": 5, "type": 1, "start": 2.0, "end": 5.0, "tard": 0.0},
    5: {"charger": 3, "type": 0, "start": 1.0, "end": 3.0, "tard": 0.0},
    6: {"charger": 7, "type": 2, "start": 0.0, "end": 2.0, "tard": 0.0},
    7: {"charger": 7, "type": 2, "start": 3.0, "end": 4.0, "tard": 0.0},
    8: {"charger": 1, "type": 0, "start": 1.0, "end": 5.0, "tard": 0.0},
    9: {"charger": 8, "type": 2, "start": 2.0, "end": 3.0, "tard": 0.0},
}

sch_m3 = {
    0: {"charger": 0, "type": 0, "position": 1, "start": 0.0, "end": 2.0, "tard": 0.0},
    1: {"charger": 8, "type": 2, "position": 4, "start": 1.0, "end": 3.0, "tard": 0.0},
    2: {"charger": 5, "type": 1, "position": 4, "start": 1.0, "end": 3.0, "tard": 0.0},
    3: {"charger": 2, "type": 0, "position": 4, "start": 2.0, "end": 3.0, "tard": 0.0},
    4: {"charger": 3, "type": 0, "position": 4, "start": 2.0, "end": 7.0, "tard": 0.0},
    5: {"charger": 1, "type": 0, "position": 3, "start": 1.0, "end": 3.0, "tard": 0.0},
    6: {"charger": 6, "type": 1, "position": 4, "start": 0.0, "end": 3.0, "tard": 0.0},
    7: {"charger": 7, "type": 2, "position": 4, "start": 3.0, "end": 4.0, "tard": 0.0},
    8: {"charger": 4, "type": 1, "position": 4, "start": 1.0, "end": 3.0, "tard": 0.0},
    9: {"charger": 0, "type": 0, "position": 3, "start": 2.0, "end": 4.0, "tard": 0.0},
}

# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
TYPE_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}   # L0 blue, L1 orange, L2 green
TYPE_LABELS = {0: "Type L0 (11 kW)", 1: "Type L1 (22 kW)", 2: "Type L2 (43 kW)"}
FMT_COLORS  = {"Iarochen": "#4C72B0", "M3-APD": "#DD8452"}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: draw a Gantt chart
# ─────────────────────────────────────────────────────────────────────────────
def draw_gantt(schedule, title, filename, show_position=False):
    chargers = sorted({s["charger"] for s in schedule.values()})
    charger_row = {c: i for i, c in enumerate(chargers)}
    n_rows = len(chargers)

    fig, ax = plt.subplots(figsize=(13, max(5, n_rows * 0.65 + 1.5)))
    fig.patch.set_facecolor("#1E1E2E")
    ax.set_facecolor("#2A2A3E")

    for spine in ax.spines.values():
        spine.set_edgecolor("#555577")

    legend_handles = {}

    for j, s in schedule.items():
        row     = charger_row[s["charger"]]
        start   = s["start"]
        end     = s["end"]
        dur     = end - start
        typ     = s["type"]
        color   = TYPE_COLORS[typ]
        tard    = s["tard"]

        # Main bar
        bar = ax.barh(row, dur, left=start, height=0.55,
                      color=color, alpha=0.88, edgecolor="white", linewidth=0.6)

        # Tardiness overlay
        if tard > 0:
            ax.barh(row, tard, left=d[j], height=0.55,
                    color="#E74C3C", alpha=0.6, edgecolor="none")

        # Label inside bar: EV index (and position for M3)
        label = f"EV{j}" if not show_position else f"EV{j}\np{s['position']}"
        ax.text(start + dur / 2, row, label,
                va="center", ha="center", fontsize=7.5,
                color="white", fontweight="bold")

        # Arrival marker ▼
        ax.annotate("▼", xy=(a[j], row + 0.36), fontsize=7,
                    color="#A0E0FF", ha="center", va="bottom")

        # Deadline marker |
        ax.axvline(x=d[j], ymin=(row) / n_rows,
                   ymax=(row + 1) / n_rows,
                   color="#FF6B6B", linewidth=1.0, linestyle="--", alpha=0.6)

        if typ not in legend_handles:
            legend_handles[typ] = mpatches.Patch(color=color, label=TYPE_LABELS[typ])

    # Axes formatting
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [f"Charger {c}\n({charger_types[charger_type_map[c]]} kW)" for c in chargers],
        color="white", fontsize=9
    )
    ax.set_xlabel("Time (slots)", color="white")
    ax.set_title(title, color="white", fontsize=14, pad=12)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")

    # Legend
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
# 1 & 2 — GANTT CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Gantt charts…")
draw_gantt(sch_iarochen, "Schedule — Iarochen et al. (2026)  [Assignment + Sequencing]",
           "gantt_iarochen.png")
draw_gantt(sch_m3,       "Schedule — M3-APD (Unlu & Mason 2010)  [Assignment + Positional Date]",
           "gantt_m3_apd.png", show_position=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3 — MODEL SIZE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("Generating model size comparison…")
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor("#1E1E2E")
ax.set_facecolor("#2A2A3E")

formulations = ["Iarochen", "M3-APD"]
metrics      = ["Variables", "Constraints"]
vals         = [[results[f]["vars"], results[f]["cons"]] for f in formulations]

x    = np.arange(len(metrics))
w    = 0.32
bars = []
for i, (f, v) in enumerate(zip(formulations, vals)):
    b = ax.bar(x + (i - 0.5) * w, v, width=w,
               color=FMT_COLORS[f], label=f, alpha=0.88,
               edgecolor="white", linewidth=0.6)
    bars.append(b)
    for rect, val in zip(b, v):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 10,
                str(val), ha="center", va="bottom", color="white",
                fontsize=10, fontweight="bold")

# CPLEX CE cap line
ax.axhline(1000, color="#E74C3C", linewidth=1.6, linestyle="--", label="CPLEX CE cap (1,000)")
ax.text(1.5, 1015, "CPLEX Community Edition cap", color="#E74C3C", fontsize=9, ha="right")

ax.set_xticks(x)
ax.set_xticklabels(metrics, color="white")
ax.set_ylabel("Count", color="white")
ax.set_title("Model Size Comparison — Iarochen vs. M3-APD  (n=10, m=9)", color="white", pad=12)
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
# 4 — SOLVE TIME COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("Generating solve time comparison…")
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor("#1E1E2E")
ax.set_facecolor("#2A2A3E")

times = [results[f]["time"] for f in formulations]
colors_bar = [FMT_COLORS[f] for f in formulations]
bars = ax.bar(formulations, times, color=colors_bar, alpha=0.88,
              edgecolor="white", linewidth=0.7, width=0.45)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{t:.4f} s", ha="center", va="bottom", color="white",
            fontsize=11, fontweight="bold")

ax.set_ylabel("CPLEX solve time (seconds)", color="white")
ax.set_title("Solve Time — Iarochen vs. M3-APD  (n=10, m=9)", color="white", pad=12)
ax.tick_params(colors="white")
ax.set_ylim(0, max(times) * 1.35)
for spine in ax.spines.values():
    spine.set_edgecolor("#555577")
ax.xaxis.label.set_color("white")

plt.tight_layout()
plt.savefig("graphs/solve_time_comparison.png", dpi=300,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  ✓ graphs/solve_time_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5 — INSTANCE DATA TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("Generating instance data table…")
fig, ax = plt.subplots(figsize=(11, 4.2))
fig.patch.set_facecolor("#1E1E2E")
ax.set_facecolor("#1E1E2E")
ax.axis("off")

col_labels = ["EV j", "Arrival aⱼ", "Deadline dⱼ", "Energy eⱼ (kWh)",
              "p_{j,L0} (11kW)", "p_{j,L1} (22kW)", "p_{j,L2} (43kW)"]
rows = [[j, a[j], d[j], e[j], p_l[j][0], p_l[j][1], p_l[j][2]] for j in J]

table = ax.table(cellText=rows, colLabels=col_labels,
                 loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.65)

for (r, c), cell in table.get_celld().items():
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

ax.set_title("Test Instance: 10 EVs, 9 Chargers (4×11kW, 3×22kW, 2×43kW)",
             color="white", fontsize=13, pad=14, y=0.95)

plt.tight_layout()
plt.savefig("graphs/instance_data_table.png", dpi=300,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  ✓ graphs/instance_data_table.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6 — PROCESSING TIME HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
print("Generating processing time heatmap…")
matrix = np.array([[p_l[j][l] for l in L] for j in J])

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor("#1E1E2E")
ax.set_facecolor("#2A2A3E")

im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

ax.set_xticks(range(len(L)))
ax.set_xticklabels(["L0 (11 kW)", "L1 (22 kW)", "L2 (43 kW)"], color="white", fontsize=11)
ax.set_yticks(range(len(J)))
ax.set_yticklabels([f"EV {j}" for j in J], color="white", fontsize=10)
ax.set_xlabel("Charger Type", color="white")
ax.set_ylabel("EV Job j", color="white")
ax.set_title("Processing-Time Matrix  $p_{jl} = \\lceil e_j / (τ \\cdot w_l) \\rceil$",
             color="white", fontsize=13, pad=12)
ax.tick_params(colors="white")

# Annotate cells
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
# 7 — THEORETICAL SCALABILITY CHART
# ─────────────────────────────────────────────────────────────────────────────
print("Generating theoretical scalability chart…")
m_val = 9    # total chargers
P_val = 4    # POS_MAX for M3-APD

n_range = np.arange(2, 45)

def iarochen_vars(n):
    """x[j,l,r] + S,C,T per job + delta[j<k,l,r]"""
    return n * m_val + 3 * n + (n * (n - 1) // 2) * m_val

def iarochen_cons(n):
    """assign + release + completion + tardiness + 2*disjunctive"""
    return n + n + n + n + 2 * (n * (n - 1) // 2) * m_val

def m3_vars(n):
    """u[j,l,r,pos] + c[l,r,pos] + C[j] + T[j]"""
    return n * m_val * P_val + m_val * P_val + 2 * n

def m3_cons(n):
    """assign(n) + slot(m*P) + pos1(m) + chain(m*(P-1)) + reldate(m*(P-1)) + link(n*m*P) + tard(n)"""
    return (n + m_val * P_val + m_val
            + m_val * (P_val - 1) + m_val * (P_val - 1)
            + n * m_val * P_val + n)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
fig.patch.set_facecolor("#1E1E2E")

CAP = 1000

for ax in axes:
    ax.set_facecolor("#2A2A3E")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555577")

# Left: Variables
ax = axes[0]
ax.plot(n_range, iarochen_vars(n_range), color=FMT_COLORS["Iarochen"],
        linewidth=2.5, label="Iarochen — Variables")
ax.plot(n_range, m3_vars(n_range), color=FMT_COLORS["M3-APD"],
        linewidth=2.5, label="M3-APD — Variables")
ax.axhline(CAP, color="#E74C3C", linewidth=1.8, linestyle="--", label="CPLEX CE cap (1,000)")

# Mark crossings
n_cross_i = n_range[iarochen_vars(n_range) > CAP]
n_cross_m = n_range[m3_vars(n_range) > CAP]
if len(n_cross_i):
    ax.axvline(n_cross_i[0], color=FMT_COLORS["Iarochen"], linewidth=1.2,
               linestyle=":", alpha=0.8)
    ax.text(n_cross_i[0] + 0.3, CAP * 0.55,
            f"Iarochen hits cap\nat n={n_cross_i[0]}",
            color=FMT_COLORS["Iarochen"], fontsize=8.5)
if len(n_cross_m):
    ax.axvline(n_cross_m[0], color=FMT_COLORS["M3-APD"], linewidth=1.2,
               linestyle=":", alpha=0.8)
    ax.text(n_cross_m[0] + 0.3, CAP * 0.25,
            f"M3-APD hits cap\nat n={n_cross_m[0]}",
            color=FMT_COLORS["M3-APD"], fontsize=8.5)

# Observed point n=10
ax.scatter([10], [iarochen_vars(10)], color=FMT_COLORS["Iarochen"], s=60, zorder=5)
ax.scatter([10], [m3_vars(10)], color=FMT_COLORS["M3-APD"], s=60, zorder=5)
ax.annotate(f"n=10: {iarochen_vars(10)}", (10, iarochen_vars(10)),
            textcoords="offset points", xytext=(6, 6), color="white", fontsize=8)
ax.annotate(f"n=10: {m3_vars(10)}", (10, m3_vars(10)),
            textcoords="offset points", xytext=(6, -14), color="white", fontsize=8)

ax.set_xlabel("Number of EVs (n)")
ax.set_ylabel("Count")
ax.set_title("Variables vs. n", color="white")
ax.legend(facecolor="#2A2A3E", edgecolor="#555577", labelcolor="white", fontsize=9)

# Right: Constraints
ax = axes[1]
ax.plot(n_range, iarochen_cons(n_range), color=FMT_COLORS["Iarochen"],
        linewidth=2.5, label="Iarochen — Constraints")
ax.plot(n_range, m3_cons(n_range), color=FMT_COLORS["M3-APD"],
        linewidth=2.5, label="M3-APD — Constraints")
ax.axhline(CAP, color="#E74C3C", linewidth=1.8, linestyle="--", label="CPLEX CE cap (1,000)")

n_cross_ic = n_range[iarochen_cons(n_range) > CAP]
n_cross_mc = n_range[m3_cons(n_range) > CAP]
if len(n_cross_ic):
    ax.axvline(n_cross_ic[0], color=FMT_COLORS["Iarochen"], linewidth=1.2,
               linestyle=":", alpha=0.8)
    ax.text(n_cross_ic[0] + 0.3, CAP * 0.55,
            f"Iarochen hits cap\nat n={n_cross_ic[0]}",
            color=FMT_COLORS["Iarochen"], fontsize=8.5)
if len(n_cross_mc):
    ax.axvline(n_cross_mc[0], color=FMT_COLORS["M3-APD"], linewidth=1.2,
               linestyle=":", alpha=0.8)
    ax.text(n_cross_mc[0] + 0.3, CAP * 0.25,
            f"M3-APD hits cap\nat n={n_cross_mc[0]}",
            color=FMT_COLORS["M3-APD"], fontsize=8.5)

ax.scatter([10], [iarochen_cons(10)], color=FMT_COLORS["Iarochen"], s=60, zorder=5)
ax.scatter([10], [m3_cons(10)], color=FMT_COLORS["M3-APD"], s=60, zorder=5)
ax.annotate(f"n=10: {iarochen_cons(10)}", (10, iarochen_cons(10)),
            textcoords="offset points", xytext=(6, 6), color="white", fontsize=8)
ax.annotate(f"n=10: {m3_cons(10)}", (10, m3_cons(10)),
            textcoords="offset points", xytext=(6, -14), color="white", fontsize=8)

ax.set_xlabel("Number of EVs (n)")
ax.set_title("Constraints vs. n", color="white")
ax.legend(facecolor="#2A2A3E", edgecolor="#555577", labelcolor="white", fontsize=9)

fig.suptitle(
    "Theoretical Model Growth vs. CPLEX Community Edition Cap  (m=9 chargers, POS_MAX=4)\n"
    "Iarochen: O(n²m)   |   M3-APD: O(nmP)",
    color="white", fontsize=12, y=1.01
)
plt.tight_layout()
plt.savefig("graphs/scalability_theoretical.png", dpi=300,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  ✓ graphs/scalability_theoretical.png")

print("\n✅  All 7 graphs saved to graphs/")
