"""Consistent main-text/SI figures and tables from the retained application data."""
import csv
import json
import math
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.special import xlogy
from scipy.stats import t

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MAN = HERE.parents[1] / "01_manuscript_tex_and_source"
if not MAN.is_dir():
    MAN = HERE / "generated_tex"
    MAN.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
COLOR = {"poor": BLUE, "middle": GREEN, "rich": ORANGE}
STARTS = ("poor", "middle", "rich")
TEMPS = [.59, .60, .61, .62, .63, .64]
plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.labelsize": 9,
    "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": .7, "lines.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False})


def read(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


rows = read("trajectory_statistics.csv")
states = read("state_composition_and_acceptance.csv")
comparison = read("initial_state_comparison.csv")
boundary = read("phase_boundaries.csv")
sensitivity = read("fit_sensitivity.csv")
structure = read("structure_identification.csv")
models = json.loads((DATA / "fitted_models.json").read_text())
schema = json.loads((DATA / "snapshot_schema.json").read_text())["columns"]
inventory = json.loads((DATA / "inventory.json").read_text(encoding="utf-8"))["temperatures"]
spec = importlib.util.spec_from_file_location("wf", HERE / "workflow_phase_diagram.py")
wf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wf)
for r in comparison:
    families = []
    for family in (r["left"], r["right"]):
        g = [{**s, **{k: float(s[k]) for k in ("mean_c", "sem_c", "df_c")}} for s in rows
             if s["T"] == r["T"] and s["phase"] == r["phase"] and s["mu"] == r["mu"] and s["start"] == family]
        families.append(wf.aggregate_replicas(g))
    a, b = families
    variance = a[1]**2+b[1]**2
    denominator = a[1]**4/a[2]+b[1]**4/b[2]
    df = variance**2/denominator if denominator else 1
    r["welch_df"] = df
    r["p_two_sided"] = float(2*t.sf(abs(a[0]-b[0])/math.sqrt(variance), df)) if variance else 1.0
previous = 0.0
for i, r in enumerate(sorted(comparison, key=lambda r: r["p_two_sided"])):
    previous = min(1.0, max(previous, (len(comparison)-i)*r["p_two_sided"]))
    r["p_holm"] = previous
if "--main-only" not in sys.argv:
    with (DATA / "initial_state_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
if "--diagnostics-only" in sys.argv:
    print("Nominal and Holm-adjusted diagnostics restored without changing figures.", flush=True)
    raise SystemExit(0)


def group(T, phase, mu=None):
    return [r for r in rows if float(r["T"]) == T and r["phase"] == phase and (mu is None or float(r["mu"]) == mu)]


def data(r):
    z = np.load(DATA / "snapshots" / (r["case"]+".npz"))
    return {k: z["values"][:, i] for i, k in enumerate(schema)}, z["stage"] == 1


def panel(ax, label):
    ax.text(0, 1.025, f"({label})", transform=ax.transAxes, va="bottom", fontweight="bold")
    ax.grid(axis="y", alpha=.4, color="#D8D8D8", lw=.4)
    ax.tick_params(direction="out", length=3, width=.65)


def save(fig, name):
    fig.savefig(MAN / (name+".pdf"), bbox_inches="tight", pad_inches=.035)
    fig.savefig(HERE / (name+".png"), dpi=600, bbox_inches="tight", pad_inches=.035)
    plt.close(fig)


def trace(ax, T, phase, mu, field="c", all_seeds=False):
    for start in STARTS:
        g = sorted([r for r in group(T, phase, mu) if r["start"] == start], key=lambda r: int(r["seed"]))
        for j, r in reversed(list(enumerate(g if all_seeds else g[:1]))):
            a, mc = data(r)
            use = np.ones(len(mc), dtype=bool) if field == "c" else mc
            ax.plot(a["md_step"][use]/1e6, a[field][use], color=COLOR[start],
                    lw=.85 if j == 0 else .45, alpha=.95 if j == 0 else .3,
                    ls=("-", "-.", "--")[STARTS.index(start)])
    ax.axvspan(1.92, 2.4, facecolor="#DDE5D0", alpha=.7, zorder=0)
    ax.set_xlim(0, 2.4)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel(r"MD steps / $10^6$")
    ax.set_ylabel(r"$c_B$" if field == "c" else r"$U/(N\epsilon_{AA})$")


def gibbs(c, T, coeff, offset=0):
    c = np.asarray(c)
    value = offset+T*(xlogy(c, c)+xlogy(1-c, 1-c))
    for k, v in enumerate(coeff):
        value += v*c**(k+1)/(k+1)
    return value


# The main figure retains the original four-stage narrative.
fig, ax = plt.subplots(2, 2, figsize=(7.15, 5.6))
for a, phase, letter in ((ax[0, 0], "solid", "a"), (ax[0, 1], "liquid", "b")):
    trace(a, .64, phase, -2.4)
    panel(a, letter)
    a.set_title(rf"{'FCC solid' if phase == 'solid' else 'Liquid'}: $T=0.64$, $\Delta\mu=-2.4$")
legend = [Line2D([], [], color=COLOR[s], ls=("-", "-.", "--")[i], label=s.capitalize()+" start") for i, s in enumerate(STARTS)]
fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, 1.015), ncol=3, frameon=False)
b = next(r for r in boundary if float(r["T"]) == .64)
T, mu, cs, cl = .64, float(b["mu_eq"]), float(b["c_s"]), float(b["c_l"])
ff = models["0.64"]
intercept = float(b["intercept"])
for phase, color, label in (("solid", BLUE, "FCC solid"), ("liquid", ORANGE, "Liquid")):
    fit = ff[phase]
    c = np.linspace(fit["c_min"], min(fit["c_max"], .055), 600)
    y = gibbs(c, T, fit["coefficients"], 0 if phase == "solid" else float(b["offset"]))-intercept-mu*c
    ax[1, 0].plot(c, y, color=color, label=label)
ax[1, 0].axhline(0, color=".4", ls="--", lw=.8, label="Common tangent")
ax[1, 0].scatter([cs, cl], [0, 0], c=[BLUE, ORANGE], s=20, zorder=4)
ax[1, 0].annotate(r"$c_s$", (cs, 0), xytext=(.006, .0007), arrowprops={"arrowstyle": "-", "lw": .5})
ax[1, 0].annotate(r"$c_l$", (cl, 0), xytext=(cl+.004, .0012), arrowprops={"arrowstyle": "-", "lw": .5})
ax[1, 0].set(xlabel=r"$c_B$", ylabel=r"$\widetilde G_\phi/\epsilon_{AA}$", xlim=(0, .055), ylim=(-.0004, .017))
ax[1, 0].set_title(r"Common-tangent construction: $T=0.64$")
ax[1, 0].legend(frameon=False, loc="upper right", fontsize=7.5)
panel(ax[1, 0], "c")
T = np.array([float(r["T"]) for r in boundary])
melting_reference = (0.0, 0.65)
last = int(np.argmax(T))
endpoint_guides = {}
for field, color, label in (("c_s", BLUE, "Solidus"), ("c_l", ORANGE, "Liquidus")):
    x = np.array([float(r[field]) for r in boundary])
    lo = np.array([float(r[field+"_lo"]) for r in boundary])
    hi = np.array([float(r[field+"_hi"]) for r in boundary])
    ax[1, 1].errorbar(x, T, xerr=[x-lo, hi-x], fmt="o-", color=color, ms=3, capsize=2, label=label)
    # Connect to the prescribed endpoint without adding a simulated point or interval.
    ax[1, 1].plot([x[last], melting_reference[0]], [T[last], melting_reference[1]], color=color)
    endpoint_guides[field] = [[float(x[last]), float(T[last])], list(melting_reference)]
ax[1, 1].plot([0], [.65], "*", color=".2", ms=7, zorder=5, label=r"Pure-A $T_m$")
ax[1, 1].set(xlabel=r"$c_B$", ylabel=r"$T\;(\epsilon_{AA}/k_B)$", xlim=(-.005, .167), ylim=(.585, .654))
ax[1, 1].set_title("Solid--liquid coexistence")
ax[1, 1].legend(frameon=False, loc="upper right", fontsize=7.5)
inset = ax[1, 1].inset_axes([.22, .12, .32, .29])
x = np.array([float(r["c_s"]) for r in boundary])
inset.errorbar(x, T, xerr=[x-np.array([float(r["c_s_lo"]) for r in boundary]), np.array([float(r["c_s_hi"]) for r in boundary])-x], fmt="o-", color=BLUE, ms=2, lw=.7)
inset.plot([x[last], 0], [T[last], .65], color=BLUE, lw=.7)
inset.plot([0], [.65], "*", color=".2", ms=4, zorder=5)
inset.set(xlim=(-.0002, .0035), ylim=(.585, .654))
inset.text(.97, .96, "Solidus", transform=inset.transAxes, ha="right", va="top", fontsize=6)
inset.tick_params(labelsize=6, length=2)
inset.set_xticks([0, .0015, .003])
inset.set_yticks([.59, .62, .65])
panel(ax[1, 1], "d")
fig.tight_layout(rect=(0, 0, 1, .96), h_pad=2, w_pad=1.8)
save(fig, "Figure_8")
(HERE / "figure8_endpoint_guides.json").write_text(json.dumps({
    "simulated_temperatures": T.tolist(),
    "prescribed_reference": list(melting_reference),
    "guide_segments": endpoint_guides,
    "additional_binary_observations": 0,
    "computed_coexistence_points_and_intervals_unchanged": True
}, indent=2), encoding="utf-8")
if "--main-only" in sys.argv:
    print("Main figure regenerated without changing SI assets.", flush=True)
    raise SystemExit(0)

# Reference enthalpy and offset.
rp = read("enthalpy_mean_points.csv")
ref = json.loads((DATA / "analysis_summary.json").read_text())
fig, ax = plt.subplots(1, 2, figsize=(7.15, 2.8))
for phase, color in (("solid", BLUE), ("liquid", ORANGE)):
    g = [r for r in rp if r["phase"] == phase]
    xx = np.array([float(r["T"]) for r in g])
    ax[0].errorbar(xx, [float(r["H"]) for r in g], yerr=[float(r["ci95_H"]) for r in g], fmt="o", color=color, ms=3, capsize=2, label=phase.capitalize())
    ax[0].plot(xx, np.polynomial.polynomial.polyval(xx, ref["enthalpy_coefficients"][phase]), color=color)
ax[0].set(xlabel=r"$T\;(\epsilon_{AA}/k_B)$", ylabel=r"$H/(N\epsilon_{AA})$")
ax[0].legend(frameon=False)
ax[1].plot([float(r["T"]) for r in boundary], [float(r["offset"]) for r in boundary], "o-", color=GREEN, ms=3)
ax[1].plot([.65], [0], "*", color=".2", ms=7)
ax[1].set(xlabel=r"$T\;(\epsilon_{AA}/k_B)$", ylabel=r"$(G_l^A-G_s^A)/\epsilon_{AA}$")
for i, a in enumerate(ax): panel(a, "ab"[i])
fig.tight_layout()
save(fig, "Figure_S9")

fig, axes = plt.subplots(6, 2, figsize=(7.15, 9.0))
for i, T in enumerate(TEMPS):
    for j, phase in enumerate(("solid", "liquid")):
        a = axes[i, j]
        for start in STARTS:
            for mu in sorted({float(r["mu"]) for r in group(T, phase)}):
                g = [r for r in group(T, phase, mu) if r["start"] == start]
                if not g: continue
                mean = np.mean([float(r["mean_c"]) for r in g])
                sem = max(math.sqrt(sum(float(r["sem_c"])**2 for r in g))/len(g), np.std([float(r["mean_c"]) for r in g], ddof=1)/math.sqrt(len(g)) if len(g)>1 else 0)
                a.errorbar(mean, mu, xerr=sem, fmt={"poor":"o", "middle":"^", "rich":"s"}[start], color=COLOR[start], mfc="white", ms=2.4, mew=.6, lw=.55, capsize=1)
        f = models[str(T)][phase]
        c = np.geomspace(f["c_min"], f["c_max"], 300)
        a.plot(c, T*np.log(c/(1-c))+np.polynomial.polynomial.polyval(c, f["coefficients"]), color=".25", lw=.8, zorder=0)
        if phase == "solid": a.set_xscale("log")
        a.set_title(f"T = {T:.2f}; {'FCC solid' if phase=='solid' else 'liquid'}", fontsize=8)
        a.set_ylabel(r"$\Delta\mu/\epsilon_{AA}$", fontsize=8)
        a.yaxis.set_major_locator(MaxNLocator(3))
        a.tick_params(labelsize=7)
        panel(a, chr(97+2*i+j))
for a in axes[-1]: a.set_xlabel(r"$c_B$")
fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, .975), h_pad=.5)
save(fig, "Figure_S10")

fig, axes = plt.subplots(6, 2, figsize=(7.15, 9))
for i, T in enumerate(TEMPS):
    for j, phase in enumerate(("solid", "liquid")):
        a = axes[i, j]
        g = sorted([r for r in comparison if float(r["T"]) == T and r["phase"] == phase and r["left"] == "poor" and r["right"] == "rich"], key=lambda r: float(r["mu"]))
        for r in g:
            color = "#9C4142" if r["zero_in_ci95"] != "True" else BLUE
            a.errorbar(float(r["mu"]), float(r["difference_B"]), yerr=float(r["ci95_half_B"]), fmt="o", color=color, ms=2.5, capsize=2, lw=.65)
        a.axhline(0, color=".4", ls="--", lw=.6)
        a.set_title(f"T = {T:.2f}; {phase}", fontsize=8)
        a.set_ylabel(r"$N(\bar c_{\rm poor}-\bar c_{\rm rich})$", fontsize=8)
        a.xaxis.set_major_locator(MaxNLocator(4))
        a.yaxis.set_major_locator(MaxNLocator(3))
        a.tick_params(labelsize=7)
        panel(a, chr(97+2*i+j))
for a in axes[-1]: a.set_xlabel(r"$\Delta\mu/\epsilon_{AA}$")
fig.tight_layout(h_pad=.8)
save(fig, "Figure_S11")

for field, filename in (("c", "Figure_S12"), ("U_per_atom", "Figure_S13")):
    fig, axes = plt.subplots(3, 2, figsize=(7.15, 6.9))
    for i, (T, mu) in enumerate(((.59, -1.0), (.62, -2.4), (.64, -2.4))):
        for j, phase in enumerate(("solid", "liquid")):
            a = axes[i, j]
            trace(a, T, phase, mu, field, all_seeds=True)
            a.set_title(rf"{phase.capitalize()}: $T={T:.2f}$, $\Delta\mu={mu:.1f}$")
            panel(a, chr(97+2*i+j))
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .96), h_pad=1.4, w_pad=1.8)
    save(fig, filename)

fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.1))
rng = np.random.default_rng(42)
for phase, color, mark in (("solid", BLUE, "o"), ("liquid", ORANGE, "s")):
    g = [r for r in structure if r["stage"] == "cycle_0060" and r["phase"] == phase]
    axes[0].scatter([float(r["T"])+rng.uniform(-.0018, .0018) for r in g], [float(r["FCC_0.15"]) for r in g], s=7, color=color, marker=mark, alpha=.6, label=phase.capitalize())
    g = [r for r in rows if r["phase"] == phase]
    axes[1].scatter([float(r["mean_c"]) for r in g], [float(r["mean_CSP"]) for r in g], s=7, color=color, marker=mark, alpha=.6)
axes[0].set_yscale("symlog", linthresh=.001)
axes[0].set(xlabel=r"$T\;(\epsilon_{AA}/k_B)$", ylabel="Final FCC fraction")
axes[0].set_yticks([0, .001, .01, .1, 1])
axes[0].set_yticklabels(["0", "0.001", "0.01", "0.1", "1"])
axes[0].legend(frameon=False, loc="center right")
axes[1].set_xscale("log")
axes[1].set(xlabel=r"$\bar c_B$", ylabel="Mean centrosymmetry parameter")
for i, a in enumerate(axes): panel(a, "ab"[i])
fig.tight_layout(w_pad=2)
save(fig, "Figure_S14")

variant_labels = {"tail30": "Final 30%", "tail50": "Final 50%", "poor_start_only": "Poor starts only", "rich_start_only": "Rich starts only",
                  "quadratic_excess": "Quadratic excess", "quartic_excess": "Quartic excess", "quadratic_enthalpy": "Quadratic enthalpy"}
colors = [BLUE, ORANGE, GREEN, "#CC79A7", "#56B4E9", "#8A6B30", "#777777"]
fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.7))
for k, (variant, label) in enumerate(variant_labels.items()):
    g = sorted([r for r in sensitivity if r["variant"] == variant], key=lambda r: float(r["T"]))
    for a, key in zip(axes, ("delta_cs_B", "delta_cl_B")):
        a.plot([float(r["T"]) for r in g], [float(r[key]) for r in g], color=colors[k], marker=("o", "s", "^", "v", "D", "+", "x")[k], ms=3, lw=.7, label=label)
for i, a in enumerate(axes):
    a.axhline(0, color=".3", ls="--", lw=.6)
    a.set_xlabel(r"$T\;(\epsilon_{AA}/k_B)$")
    a.set_ylabel(r"$N\,\delta c_s$" if i==0 else r"$N\,\delta c_l$")
    panel(a, "ab"[i])
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, .78), w_pad=2)
save(fig, "Figure_S15")

# Tables use numerical outputs only; parameter/protocol prose is authored in SI.
lines = []
for item in inventory:
    T = item["T"]
    g = group(T, "solid")
    liq = group(T, "liquid")
    dense = sorted({float(r["mu"]) for r in g if r["start"] == "middle"})
    mus = ", ".join(f"{v:.1f}" for v in item["mu"])
    lines.append(f"{T:.2f} & {mus} & {len(g)} / {len(liq)} & " + (", ".join(f"{v:.1f}" for v in dense) or "---") + r" \\")
(MAN / "application_grid_rows.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
lines = []
acceptance_summary = []
for T in TEMPS:
    for phase in ("solid", "liquid"):
        g = group(T, phase)
        pooled = sum(int(r["accepts"]) for r in g)/sum(int(r["attempts"]) for r in g)
        lo, hi = min(float(r["acceptance"]) for r in g), max(float(r["acceptance"]) for r in g)
        acceptance_summary.append({"T": T, "phase": phase, "pooled": pooled, "min": lo, "max": hi})
        lines.append(f"{T:.2f} & {phase.capitalize()} & {100*pooled:.4f} & {100*lo:.4f}--{100*hi:.4f} & "
                     f"{min(float(r['mc_ESS']) for r in g):.1f} & {min(float(r['cycle_ESS']) for r in g):.1f}" + r" \\")
(MAN / "application_acceptance_rows.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
lines = []
for b in boundary:
    lines.append(f"{float(b['T']):.2f} & {float(b['c_s']):.6f} [{float(b['c_s_lo']):.6f}, {float(b['c_s_hi']):.6f}] & "
                 f"{float(b['c_l']):.6f} [{float(b['c_l_lo']):.6f}, {float(b['c_l_hi']):.6f}] & {float(b['mu_eq']):.5f}"+r" \\")
(MAN / "application_boundary_rows.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
exceptions = [r for r in comparison if r["zero_in_ci95"] != "True"]
lines = [f"{float(r['T']):.2f} & {r['phase'].capitalize()} & {float(r['mu']):.1f} & {float(r['difference_B']):.3f} & {float(r['ci95_half_B']):.3f} & {r['p_two_sided']:.4f} & {r['p_holm']:.4f}"+r" \\" for r in exceptions]
(MAN / "application_exception_rows.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
metrics = {"acceptance": acceptance_summary, "initial_comparisons": len(comparison), "unadjusted_nonzero_intervals": len(exceptions),
           "minimum_Holm_adjusted_p": min(r["p_holm"] for r in comparison),
           "max_initial_difference_c": max(abs(float(r["difference_B"]))/6912 for r in comparison),
           "max_solid_initial_difference_B": max(abs(float(r["difference_B"])) for r in comparison if r["phase"] == "solid"),
           "figure_files": ["Figure_8"]+[f"Figure_S{i}" for i in range(9,16)]}
for variant in variant_labels:
    g = [r for r in sensitivity if r["variant"] == variant]
    metrics[variant] = {"max_cs_B": max(abs(float(r["delta_cs_B"])) for r in g), "max_cl_B": max(abs(float(r["delta_cl_B"])) for r in g)}
(DATA / "publication_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
table_templates = {
    "grid": (
        r"\begin{tabularx}{\textwidth}{l>{\raggedright\arraybackslash}Xl>{\raggedright\arraybackslash}p{0.21\textwidth}}",
        r"$T\;(\epsilon_{AA}/k_B)$ & $\Delta\mu/\epsilon_{AA}$ & S/L & Dense-validation $\Delta\mu/\epsilon_{AA}$ \\",
        r"\end{tabularx}"),
    "acceptance": (
        r"\begin{tabular}{llrrrr}",
        r"$T$ & Phase & Pooled acceptance (\%) & Trajectory range (\%) & Min. MC ESS & Min. cycle ESS \\",
        r"\end{tabular}"),
    "boundary": (
        r"\begin{tabular}{lccc}",
        r"$T\;(\epsilon_{AA}/k_B)$ & $c_s$ [95\% interval] & $c_l$ [95\% interval] & $\Delta\mu_{\rm eq}/\epsilon_{AA}$ \\",
        r"\end{tabular}"),
    "exception": (
        r"\begin{tabular}{llrrrrr}",
        r"$T$ & Phase & $\Delta\mu/\epsilon_{AA}$ & Mean B-count difference & 95\% half-width & $p$ & $p_{\rm Holm}$ \\",
        r"\end{tabular}"),
}
for stem, (opening, header, closing) in table_templates.items():
    body = (MAN / f"application_{stem}_rows.tex").read_text(encoding="utf-8")
    table = "\n".join([opening, r"\toprule", header, r"\midrule", body.rstrip(), r"\bottomrule", closing, ""])
    (MAN / f"application_{stem}_table.tex").write_text(table, encoding="utf-8")
print(json.dumps(metrics, indent=2), flush=True)
