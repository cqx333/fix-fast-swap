"""Publication figures and tables from the verified collected revision data."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FixedLocator, FuncFormatter
import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAN = ROOT / "01_manuscript_tex_and_source"
DATA = HERE / "data"
WORKSPACE = ROOT.parent
COLLECTED = WORKSPACE / "CPC-test_revision_unified_20261012/results/collected_all_node05"
AUDIT = DATA / "audit"
BLUE, ORANGE = "#0072B2", "#D55E00"
COLORS = ["#009E73", "#CC79A7", "#0072B2", "#D55E00"]
ORDER = ["lj_cut", "morse", "table_lj", "buck", "buck_coul_cut", "coul_dsf",
         "hybrid_overlay_lj_yukawa", "yukawa", "zbl", "ternary_lj_cut", "unequal_lj_cut"]
NAME = {"lj_cut": "LJ/cut", "morse": "Morse", "table_lj": "Tabulated LJ",
        "buck": "Buckingham", "buck_coul_cut": "Buckingham + Coul/cut", "coul_dsf": "Coul/DSF",
        "hybrid_overlay_lj_yukawa": "LJ + Yukawa", "yukawa": "Yukawa", "zbl": "ZBL",
        "ternary_lj_cut": "Ternary LJ", "unequal_lj_cut": "Unequal-cutoff LJ",
        "morse_dynamic": "Morse ($a=6$)"}
SEEDS = [24681357, 24682357, 24683357]
plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.labelsize": 9,
                    "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                    "legend.fontsize": 8, "axes.linewidth": .7, "lines.linewidth": 1.1,
                    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix",
                    "savefig.dpi": 600, "axes.spines.top": False, "axes.spines.right": False})
ASSETS, CHECKS = [], []


def copy_data():
    DATA.mkdir(exist_ok=True)
    for folder, source in (("core", COLLECTED / "analysis/core"),
                           ("incremental", COLLECTED / "incremental/analysis"), ("audit", AUDIT)):
        destination = DATA / folder
        destination.mkdir(exist_ok=True)
        for file in source.glob("*.csv"):
            target = destination / file.name
            if file.resolve() != target.resolve():
                shutil.copy2(file, target)
        if (source / "snapshots").is_dir():
            shutil.copytree(source / "snapshots", destination / "snapshots", dirs_exist_ok=True)
    for source, destination in ((COLLECTED / "analysis/collection_summary.json", DATA / "collection_summary.json"),
                                (AUDIT / "review_metrics.json", DATA / "review_metrics.json"),
                                (COLLECTED / "config.json", DATA / "core_config.json"),
                                (COLLECTED / "incremental/metadata/effective_config.json", DATA / "incremental_config.json")):
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def numeric(path):
    return [{k: float(v) if v else np.nan for k, v in row.items()} for row in read_csv(path)]


def trajectory(case, kind, rank=24, seed=SEEDS[0], exact=False, dynamic=False, half=False):
    if dynamic:
        key = f"dynamic__{case}__p{rank}__s{seed}__{kind}__dt{'half' if half else 'base'}"
        rows = numeric(DATA / "incremental/snapshots" / (key + ".csv"))
        every = 200 if half else 100
        rows = [r for r in rows if int(r["local_md_step"]) % every == 0 or r["sweeps"] == 0]
        xname = "sweeps"
    else:
        key = f"{'exact' if exact else 'parallel'}__{case}__p{1 if exact else rank}__s{seed}__{kind}"
        if not exact and case in ("ternary_lj_cut", "zbl") and rank == 24:
            rows = numeric(DATA / "incremental/snapshots" / (key + ".csv"))
            xname = "sweeps"
        else:
            rows = numeric(DATA / "core/snapshots" / (key + ".csv"))
            xname = "attempted_sweeps"
    return np.array([r[xname] for r in rows]), rows


def handles():
    return [Line2D([], [], color=BLUE, lw=1.3, label="atom/swap"),
            Line2D([], [], color=ORANGE, ls="--", lw=1.3, label="fast/swap")]


def panel(ax, letter):
    ax.text(0, 1.04, f"({letter})", transform=ax.transAxes, fontweight="bold", va="bottom")
    ax.grid(axis="y", color="#D8D8D8", lw=.4, alpha=.5)
    ax.tick_params(direction="out", length=3, width=.65)


def tidy(ax, time=False):
    ax.yaxis.set_major_locator(MaxNLocator(4))
    if time:
        ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)


def plot_series(ax, case, obs, rank=24, exact=False, dynamic=False, half=False, tail=False, all_seeds=True):
    seeds = SEEDS if all_seeds else SEEDS[:1]
    for kind, color, linestyle in (("atom", BLUE, "-"), ("fast", ORANGE, "--")):
        for index, seed in reversed(list(enumerate(seeds))):
            x, rows = trajectory(case, kind, rank, seed, exact, dynamic, half)
            y = np.array([r[obs] for r in rows])
            use = x >= .25 * x[-1] if tail else (x > 0 if exact else np.ones(len(x), dtype=bool))
            ax.plot(x[use], y[use], color=color, ls=linestyle, lw=.9 if index == 0 else .45,
                    alpha=.95 if index == 0 else .23, zorder=3 if index == 0 else 2)
            if exact and kind == "fast":
                xx, yy = x[use], y[use]
                positions = np.arange(0, len(xx), max(1, len(xx) // 14))
                ax.plot(xx[positions], yy[positions], "o", ms=2.2, mfc="white", mec=color, mew=.65, zorder=4)
    if not exact and not tail:
        ax.axvspan(0, .25 * x[-1], facecolor="#E8E8E8", zorder=0)
    ax.set_xlim(.25*x[-1] if tail else 0, x[-1])
    ax.margins(y=.12)
    tidy(ax, True)


def save(fig, filename, description):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_overlap = []
    for legend in fig.legends:
        bbox = legend.get_window_extent(renderer)
        for index, ax in enumerate(fig.axes):
            if bbox.overlaps(ax.get_window_extent(renderer)):
                legend_overlap.append(index)
    if legend_overlap:
        raise ValueError(f"Legend overlaps data axes: {filename}: {legend_overlap}")
    annotations = []
    for ax in fig.axes:
        annotations += [ax.title, ax._left_title, ax._right_title, *ax.texts]
    annotations += list(fig.texts)
    annotations = [text for text in annotations if text.get_visible() and text.get_text()]
    collisions = []
    for index, left in enumerate(annotations):
        a = left.get_window_extent(renderer)
        for right in annotations[index+1:]:
            b = right.get_window_extent(renderer)
            if min(a.x1,b.x1)-max(a.x0,b.x0) > .8 and min(a.y1,b.y1)-max(a.y0,b.y0) > .8:
                collisions.append([left.get_text(), right.get_text()])
    if collisions:
        raise ValueError(f"Annotation overlap: {filename}: {collisions}")
    fig.savefig(MAN / (filename + ".pdf"), bbox_inches="tight", pad_inches=.055)
    fig.savefig(HERE / (filename + ".png"), dpi=350, bbox_inches="tight", pad_inches=.055)
    CHECKS.append({"file": filename, "axes": len(fig.axes), "legend_data_axes_overlaps": legend_overlap,
                   "annotation_overlaps": collisions})
    ASSETS.append({"file": filename + ".pdf", "description": description})
    plt.close(fig)


def main_validation():
    fig = plt.figure(figsize=(7.1, 6.2))
    grid = fig.add_gridspec(6, 2, height_ratios=[1, 1, .72, .72, .72, .72], hspace=.64, wspace=.30)
    a, b = fig.add_subplot(grid[:2, 0]), fig.add_subplot(grid[:2, 1])
    for ax, obs, label, letter in ((a, "U_per_atom", r"Potential energy per atom, $U/N$", "a"),
                                   (b, "x_type_2", r"Composition, $x_B$", "b")):
        plot_series(ax, "lj_cut", obs, rank=1, exact=True, all_seeds=False)
        panel(ax, letter)
        ax.set_xlabel("MC sweeps")
        ax.set_ylabel(label)
        ax.set_title("LJ/cut; $N=864$, 1 rank", pad=12)
    for row, model in enumerate(("lj_cut", "morse", "buck_coul_cut", "zbl")):
        for col, obs in enumerate(("U_per_atom", "x_type_2")):
            ax = fig.add_subplot(grid[row+2, col])
            plot_series(ax, model, obs, tail=True)
            ax.set_title(NAME[model], loc="right", fontsize=8, pad=2)
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(labelsize=7.3)
            if row == 0:
                panel(ax, "c" if col == 0 else "d")
            if row == 1:
                ax.set_ylabel(r"$U/N$" if col == 0 else r"$x_B$")
            if row == 3:
                ax.set_xlabel("MC sweeps")
    fig.subplots_adjust(left=.105, right=.985, bottom=.065, top=.90)
    fig.legend(handles=handles(), loc="upper center", bbox_to_anchor=(.52, 1.0), ncol=2, frameon=False)
    save(fig, "Figure_2", "Frozen single-rank LJ paths and retained 24-rank raw trajectories.")


def main_mobile():
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.95))
    for col, case in enumerate(("lj_cut", "morse_dynamic")):
        for row, obs in enumerate(("U_per_atom", "x_type_2")):
            ax = axes[row, col]
            plot_series(ax, case, obs, rank=2, dynamic=True)
            panel(ax, chr(97 + row*2 + col))
            if row == 0:
                ax.set_title(NAME[case] + "; $N=512$, 2 ranks", pad=13)
            if col == 0:
                ax.set_ylabel(r"$U/N$" if row == 0 else r"Composition, $x_B$")
            if row == 1:
                ax.set_xlabel("MC sweeps")
    fig.subplots_adjust(left=.10, right=.985, bottom=.11, top=.86, hspace=.45, wspace=.28)
    fig.legend(handles=handles(), loc="upper center", bbox_to_anchor=(.52, 1.0), ncol=2, frameon=False)
    save(fig, "Figure_3", "Finite-step MD/MC trajectories for LJ/cut and alpha=6 Morse.")


def repeated(case, rank, n, kind):
    selected = [r for r in TIMING_RAW if r["case"] == case and int(r["ranks"]) == rank and int(r["natoms"]) == n and r["kind"] == kind]
    return np.array([float(r["time_per_actual_sweep_s"]) for r in sorted(selected, key=lambda r: int(r["repeat"]))])


def uncertainty(values):
    return float(np.mean(values)), float(t.ppf(.975, len(values)-1) * np.std(values, ddof=1) / np.sqrt(len(values)))


def ratio(a, b):
    point = float(a.mean() / b.mean())
    gradient = a/a.mean() - b/b.mean()
    half = t.ppf(.975, len(a)-1) * np.std(gradient, ddof=1) / np.sqrt(len(a))
    return point, point*np.exp(-half), point*np.exp(half)


def main_size():
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.45, 4.8))
    series = {}
    for rank, marker, ls in ((1, "o", "-"), (24, "s", "--")):
        rows = sorted([r for r in TIMING if r["case"] == "lj_cut" and int(r["ranks"]) == rank and
                       not (rank == 1 and int(r["natoms"]) == 24576)], key=lambda r: int(r["natoms"]))
        series[rank] = rows
        x = np.array([int(r["natoms"]) for r in rows])
        for kind, color in (("atom", BLUE), ("fast", ORANGE)):
            values = [uncertainty(repeated("lj_cut", rank, n, kind)) for n in x]
            y, error = np.array(values).T
            a.errorbar(x, y, yerr=error, color=color, marker=marker, ms=3.3, ls=ls, capsize=2, lw=1)
        rates = [ratio(repeated("lj_cut", rank, n, "atom"), repeated("lj_cut", rank, n, "fast")) for n in x]
        y, low, high = np.array(rates).T
        b.errorbar(x, y, yerr=[y-low, high-y], color=BLUE if rank == 1 else ORANGE,
                   marker=marker, ls=ls, ms=3.6, capsize=2, lw=1)
    x = np.array([int(r["natoms"]) for r in series[1]])
    for kind, power in (("atom", 2), ("fast", 1)):
        y = np.array([float(r[kind+"_mean_s"]) for r in series[1]])
        coefficient = np.exp(np.mean(np.log(y) - power*np.log(x)))
        a.plot(x, coefficient*x**power, color="#777777", lw=.9, ls=":" if power == 1 else "-.")
    a.set_xscale("log"); a.set_yscale("log"); b.set_xscale("log")
    for ax, letter in ((a, "a"), (b, "b")):
        panel(ax, letter)
        ax.set_xlim(420, 47000)
        ax.set_xticks([500, 2000, 10000, 40000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_xlabel("Number of atoms, $N$")
    a.set_ylabel("Wall time per sweep (s)")
    b.set_ylabel("Same-rank speedup")
    upper = max(ratio(repeated("lj_cut", rank, int(row["natoms"]), "atom"),
                      repeated("lj_cut", rank, int(row["natoms"]), "fast"))[2]
                for rank, rows in series.items() for row in rows)
    ymax = 250 * math.ceil((upper*1.07)/250)
    b.set_ylim(0, ymax); b.set_yticks(np.arange(0, ymax+1, 250))
    hs = [Line2D([], [], color=BLUE, marker="o", ms=3, label="atom, 1 rank"),
          Line2D([], [], color=ORANGE, marker="o", ms=3, label="fast, 1 rank"),
          Line2D([], [], color=BLUE, marker="s", ls="--", ms=3, label="atom, 24 ranks"),
          Line2D([], [], color=ORANGE, marker="s", ls="--", ms=3, label="fast, 24 ranks"),
          Line2D([], [], color="#777777", ls="-.", label=r"$N^2$ guide"),
          Line2D([], [], color="#777777", ls=":", label=r"$N$ guide")]
    fig.subplots_adjust(left=.20, right=.965, bottom=.09, top=.84, hspace=.48)
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(.54, 1.0), ncol=2, frameon=False, fontsize=7.4)
    b.legend(handles=[Line2D([], [], color=BLUE, marker="o", label="1 rank"),
                      Line2D([], [], color=ORANGE, marker="s", ls="--", label="24 ranks")],
             loc="upper left", frameon=False, fontsize=7.5)
    save(fig, "Figure_4", "Repeated LJ size scaling; guides are not fitted exponents.")


def main_strong():
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.45, 4.8))
    ranks = [1, 2, 4, 8, 12, 16, 24]
    for kind, color, marker in (("atom", BLUE, "o"), ("fast", ORANGE, "s")):
        values = [uncertainty(repeated("lj_cut", rank, 24576, kind)) for rank in ranks]
        mean, ci = np.array(values).T
        a.errorbar(ranks, mean, yerr=ci, color=color, marker=marker, ms=3.5, capsize=2)
        rates = [ratio(repeated("lj_cut", 1, 24576, kind), repeated("lj_cut", rank, 24576, kind)) for rank in ranks]
        y, low, high = np.array(rates).T
        b.errorbar(ranks, y, yerr=[y-low, high-y], color=color, marker=marker, ms=3.5, capsize=2)
    a.set_yscale("log"); a.set_ylabel("Wall time per sweep (s)")
    b.plot(ranks, ranks, color="#777777", ls=":", lw=.9)
    b.set_ylabel("Speedup relative to 1 rank")
    b.set_ylim(0, 27); b.set_yticks([0, 6, 12, 18, 24])
    for ax, letter in ((a, "a"), (b, "b")):
        panel(ax, letter); ax.set_xlim(.5, 25)
        ax.set_xticks(ranks); ax.set_xlabel("MPI ranks")
    fig.subplots_adjust(left=.20, right=.97, bottom=.09, top=.86, hspace=.48)
    fig.legend(handles=handles()+[Line2D([], [], color="#777777", ls=":", label="Ideal scaling")],
               loc="upper center", bbox_to_anchor=(.56, 1.0), ncol=1, frameon=False)
    save(fig, "Figure_5", "LJ strong scaling, common node/state, 3 repeated timings.")


def main_styles():
    cases = ORDER[:-1]
    fig, ax = plt.subplots(figsize=(3.5, 4.9))
    positions = np.arange(len(cases))
    for offset, rank, n, color in ((-.16, 1, 864, BLUE), (.16, 24, 6144, ORANGE)):
        rates = [ratio(repeated(case, rank, n, "atom"), repeated(case, rank, n, "fast")) for case in cases]
        mean, low, high = np.array(rates).T
        ax.barh(positions+offset, mean, height=.27, color=color, alpha=.85)
        ax.errorbar(mean, positions+offset, xerr=[mean-low, high-mean], fmt="none", ecolor="#333333", capsize=2, lw=.6)
    ax.set_yticks(positions)
    ax.set_yticklabels([NAME[case].replace("Buckingham + Coul/cut", "Buck. + Coul/cut") for case in cases], fontsize=8.6)
    ax.invert_yaxis()
    upper = max(ratio(repeated(case, rank, n, "atom"), repeated(case, rank, n, "fast"))[2]
                for case in cases for rank, n in ((1, 864), (24, 6144)))
    xmax = 250 * math.ceil(upper * 1.06 / 250)
    ax.set_xlim(0, xmax); ax.set_xticks(np.arange(0, xmax+1, 250))
    ax.set_xlabel("Same-rank speedup")
    ax.grid(axis="x", color="#DDDDDD", lw=.4); ax.set_axisbelow(True)
    fig.subplots_adjust(left=.38, right=.985, bottom=.09, top=.90)
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=6, label="1 rank; $N=864$"),
                        Line2D([], [], color=ORANGE, lw=6, label="24 ranks; $N=6144$")],
               loc="upper center", bbox_to_anchor=(.60, 1.0), frameon=False, ncol=1, fontsize=8)
    save(fig, "Figure_6", "Cross-style paired timings at explicitly stated sizes.")


def si_closure():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    rows = read_csv(DATA / "core/single_move_closure.csv")
    for index, case in enumerate(ORDER):
        values = [float(r["residual_global_minus_local"])*1e12 for r in rows if r["case"] == case]
        axes[0].scatter(values, index+np.linspace(-.18, .18, len(values)), s=8, c=BLUE, alpha=.65, edgecolors="none")
    axes[0].set_yticks(range(len(ORDER))); axes[0].set_yticklabels([NAME[c].replace("Buckingham + Coul/cut", "Buck. + Coul/cut") for c in ORDER], fontsize=7.9)
    axes[0].invert_yaxis()
    rows = [r for r in read_csv(DATA / "incremental/mobile_snapshot_closure.csv") if r["snapshot_source"].startswith("dynamic__")]
    keys = list(dict.fromkeys(r["snapshot_source"] for r in rows))
    labels = []
    for index, key in enumerate(keys):
        values = [float(r["residual"])*1e12 for r in rows if r["snapshot_source"] == key]
        axes[1].scatter(values, index+np.linspace(-.12, .12, len(values)), s=14, c=ORANGE, alpha=.8, edgecolors="none")
        labels.append(("Morse" if "morse" in key else "LJ")+ (": reference state" if "__atom__" in key else ": optimized state"))
    axes[1].set_yticks(range(4)); axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(axis="y", color="#D8D8D8", lw=.4, alpha=.5)
        ax.axvline(0, color="#777777", ls=":", lw=.8)
        ax.set_xlim(-12, 12); ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_xlabel(r"Closure residual / $10^{-12}$")
    axes[0].set_title("(a) Fixed configurations\n660 trials", loc="left", pad=12)
    axes[1].set_title("(b) Mobile snapshots\n96 trials", loc="left", pad=12)
    fig.subplots_adjust(left=.20, right=.985, bottom=.12, top=.88, wspace=.85)
    save(fig, "Figure_S1", "Independent force-path/pair-local closure; new data only.")


def dense_trajectories(filename, cases, exact=False, dynamic=False):
    rows = math.ceil(len(cases)/2)
    fig, axes = plt.subplots(rows, 4, figsize=(7.15, 2.03*rows), squeeze=False)
    fig.subplots_adjust(left=.07, right=.985, bottom=.075, top=.91, hspace=.70, wspace=.64)
    for index, configuration in enumerate(cases):
        case, rank, half = configuration if isinstance(configuration, tuple) else (configuration, 24, False)
        row, start = index//2, 2*(index%2)
        for col, obs in enumerate(("U_per_atom", "x_type_2")):
            ax = axes[row, start+col]
            plot_series(ax, case, obs, rank=rank, exact=exact, dynamic=dynamic, half=half, all_seeds=not exact)
            panel(ax, chr(97 + index*2 + col))
            ax.set_ylabel(r"$U/N$" if col == 0 else r"$x_B$", fontsize=8)
            ax.tick_params(labelsize=7)
            if row == rows-1:
                ax.set_xlabel("MC sweeps", fontsize=8)
        left, right = axes[row, start].get_position(), axes[row, start+1].get_position()
        title = NAME[case]
        if dynamic:
            title += f"; {rank} rank" + ("s" if rank > 1 else "") + (r"; $\Delta t/2$" if half else "")
        fig.text((left.x0+right.x1)/2, left.y1+.037, title, ha="center", fontsize=8.5, fontweight="bold")
    if len(cases)%2:
        axes[-1, 2].set_visible(False); axes[-1, 3].set_visible(False)
    fig.legend(handles=handles(), loc="upper center", bbox_to_anchor=(.52, 1.005), ncol=2, frameon=False)
    save(fig, filename, "Raw trajectories with unchanged method colors and external legends.")


def si_lj_variants():
    cases = ["lj_cut", "unequal_lj_cut", "ternary_lj_cut"]
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 5.9))
    for row, case in enumerate(cases):
        for col, obs in enumerate(("U_per_atom", "x_type_2")):
            ax = axes[row, col]
            plot_series(ax, case, obs)
            if case == "ternary_lj_cut" and col == 1:
                for kind, color, ls in (("atom", BLUE, "-"), ("fast", ORANGE, "--")):
                    x, data = trajectory(case, kind)
                    ax.plot(x, [r["x_type_3"] for r in data], color=color, ls=ls, lw=.9)
            panel(ax, chr(97+row*2+col)); ax.set_ylabel(r"$U/N$" if col == 0 else r"Composition")
            ax.set_title(NAME[case], loc="right", fontsize=8.5, pad=3)
            if row == 2:
                ax.set_xlabel("MC sweeps")
    fig.subplots_adjust(left=.10, right=.985, top=.90, bottom=.085, wspace=.3, hspace=.6)
    fig.legend(handles=handles(), loc="upper center", bbox_to_anchor=(.52, 1), ncol=2, frameon=False)
    save(fig, "Figure_S4", "Parallel LJ variants, including both ternary compositions.")


def si_nonlj():
    cases, ranks = ["morse", "buck_coul_cut", "zbl", "yukawa"], [2, 8, 24]
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 4.8))
    hs = []
    for case, color, marker in zip(cases, COLORS, ["o", "s", "^", "D"]):
        values = [uncertainty(repeated(case, rank, 6144, "fast")) for rank in ranks]
        mean, ci = np.array(values).T
        axes[0].errorbar(ranks, mean, yerr=ci, color=color, marker=marker, ms=4, capsize=3)
        rates = [ratio(repeated(case, rank, 6144, "atom"), repeated(case, rank, 6144, "fast")) for rank in ranks]
        mean, low, high = np.array(rates).T
        axes[1].errorbar(ranks, mean, yerr=[mean-low, high-mean], color=color, marker=marker, ms=4, capsize=3)
        hs.append(Line2D([], [], color=color, marker=marker, label=NAME[case]))
    axes[0].set_yscale("log"); axes[0].set_ylabel("fast/swap time per sweep (s)")
    axes[1].set_ylabel("Same-rank speedup")
    upper = max(ratio(repeated(case, rank, 6144, "atom"), repeated(case, rank, 6144, "fast"))[2]
                for case in cases for rank in ranks)
    axes[1].set_ylim(0, 250 * math.ceil(upper * 1.07 / 250))
    for ax, letter in zip(axes, "ab"):
        panel(ax, letter); ax.set_xticks(ranks); ax.set_xlim(.7, 25.5); ax.set_xlabel("MPI ranks")
    fig.subplots_adjust(left=.12, right=.985, top=.84, bottom=.10, hspace=.48)
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(.55, 1), ncol=2, frameon=False)
    save(fig, "Figure_S5", "Non-LJ repeated performance at fixed N=6144; no false p1 baseline.")


def si_unequal():
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    for group, rank, n in ((0, 1, 864), (1, 24, 6144)):
        for kind, color, offset in (("atom", BLUE, -.13), ("fast", ORANGE, .13)):
            values = repeated("unequal_lj_cut", rank, n, kind)
            mean, ci = uncertainty(values)
            ax.scatter(group+offset+np.linspace(-.025, .025, 3), values, color=color, s=25, zorder=3)
            ax.errorbar(group+offset, mean, yerr=ci, color=color, marker="_", ms=15, capsize=4, zorder=2)
        speed = ratio(repeated("unequal_lj_cut", rank, n, "atom"), repeated("unequal_lj_cut", rank, n, "fast"))[0]
        ax.text(group, 1.02, f"Same-rank speedup: {speed:.2f}", transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
    ax.set_yscale("log"); ax.set_xlim(-.5, 1.5); ax.set_ylim(.10, 6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["1 rank; $N=864$", "24 ranks; $N=6144$"])
    ax.set_ylabel("Wall time per MC sweep (s)")
    ax.grid(axis="y", color="#DDDDDD", lw=.4)
    fig.subplots_adjust(left=.12, right=.98, top=.79, bottom=.13)
    fig.legend(handles=handles(), loc="upper center", bbox_to_anchor=(.55, 1), ncol=2, frameon=False)
    save(fig, "Figure_S6", "Unequal-cutoff boundary timings with all 3 repetitions.")


def si_repeats():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
    ranks = [1, 2, 4, 8, 12, 16, 24]
    hs = []
    for ax, kind, color, letter in zip(axes, ("atom", "fast"), (BLUE, ORANGE), "ab"):
        for repetition, marker in enumerate(("o", "s", "^")):
            values = [repeated("lj_cut", rank, 24576, kind)[repetition]/repeated("lj_cut", rank, 24576, kind).mean() for rank in ranks]
            ax.plot(ranks, values, color=color, marker=marker, ms=4, alpha=.65, lw=.7)
        ax.axhline(1, color="#777777", ls=":", lw=.8)
        panel(ax, letter); ax.set_title(kind+"/swap", pad=12)
        ax.set_xticks(ranks); ax.set_xlim(.3, 25); ax.set_ylim(.75, 1.3)
        ax.set_xlabel("MPI ranks")
    axes[0].set_ylabel(r"Repeat time / mean time at each rank")
    hs = [Line2D([], [], color="#555555", marker=marker, lw=.7, label=f"Repeat {i+1}") for i, marker in enumerate(("o", "s", "^"))]
    fig.subplots_adjust(left=.10, right=.985, top=.78, bottom=.16, wspace=.25)
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(.54, 1), ncol=3, frameon=False)
    save(fig, "Figure_S7", "Individual strong-scaling repeats, normalized separately at each rank.")


def sci(value, digits=2):
    if value == 0:
        return "0"
    text = f"{value:.{digits}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"
    return text


def tables():
    exact = read_csv(DATA / "audit/exact_rechecked.csv")
    closure = read_csv(DATA / "core/single_move_closure.csv")
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Model & Seeds & $\max|\delta U|$ & $\max|\delta N_\alpha|$ & Closure residual \\", r"\midrule"]
    for case in ORDER:
        rows = [r for r in exact if r["case"] == case and r["stage"] == "exact"]
        residual = max(abs(float(r["residual_global_minus_local"])) for r in closure if r["case"] == case)
        lines.append(NAME[case] + rf" & 3 & ${sci(max(float(r['maximum_total_energy_difference']) for r in rows), 3)}$ & 0 & ${sci(residual, 3)}$ \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (MAN / "revision_exact_table.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
    replicas = read_csv(DATA / "audit/replica_diagnostics_rechecked.csv")
    pairs = read_csv(DATA / "audit/ensemble_differences_rechecked.csv")
    def cohort(kind, selected):
        rows = [r for r in selected if r["kind"] == kind]
        values = np.array([float(r["mean"]) for r in rows])
        within = math.sqrt(sum(float(r["se"])**2 for r in rows))/len(rows)
        se = max(within, float(np.std(values, ddof=1)/np.sqrt(len(values))))
        half = float(t.ppf(.975, min(2, *(int(r["df"]) for r in rows))) * se)
        return values.mean(), half
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Model & Quantity & atom/swap & fast/swap & Difference & Min. ESS \\", r"\midrule"]
    for case in ORDER:
        for obs in (["U_per_atom", "x_type_2", "x_type_3"] if case == "ternary_lj_cut" else ["U_per_atom", "x_type_2"]):
            pair = next(r for r in pairs if r["case"] == case and r["ranks"] == "24" and r["observable"] == obs and r["discard"] == "0.25")
            selected = [r for r in replicas if r["case"] == case and r["ranks"] == "24" and r["observable"] == obs and r["discard"] == "0.25"]
            a, a_ci = cohort("atom", selected); b, b_ci = cohort("fast", selected)
            quantity = r"$U/N$" if obs == "U_per_atom" else (r"$x_B$" if obs == "x_type_2" else r"$x_C$")
            label = NAME[case] if obs == "U_per_atom" else ""
            constant = pair["constant_observable"] == "True"
            if constant:
                cells = [f"${a:.6f}$", f"${b:.6f}$", "$"+sci(float(pair["difference_fast_minus_atom"]))+"$", "--"]
            else:
                cells = [rf"${a:.6f}\pm {sci(a_ci, 2)}$", rf"${b:.6f}\pm {sci(b_ci, 2)}$",
                         "$"+sci(float(pair["difference_fast_minus_atom"]), 3)+r"\pm "+sci(float(pair["ci95_halfwidth"]), 2)+"$",
                         f"{float(pair['minimum_ess']):.0f}"]
            lines.append(label+" & "+quantity+" & "+" & ".join(cells)+r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (MAN / "revision_parallel_table.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
    acceptance = read_csv(DATA / "audit/acceptance_rechecked.csv")
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Model & Sweeps & atom/swap & fast/swap \\", r"\midrule"]
    for case in ORDER:
        cells = []
        for kind in ("atom", "fast"):
            values = [float(r["acceptance"]) for r in acceptance if r["case"] == case and r["ranks"] == "24" and r["kind"] == kind and r["discard"] == "0.25"]
            cells.append(f"{np.mean(values):.4f} [{min(values):.4f}, {max(values):.4f}]")
        lines.append(NAME[case]+f" & {1200 if case in ('zbl','ternary_lj_cut') else 600} & "+" & ".join(cells)+r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (MAN / "revision_acceptance_table.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
    lines = [r"\begin{tabular}{llrrr}", r"\toprule", r"Model & Protocol & Quantity & Difference $\pm$ 95\% CI & Min. ESS \\", r"\midrule"]
    for case in ("lj_cut", "morse_dynamic"):
        for rank, half in ((1, False), (2, False), (1, True)):
            for obs in ("U_per_atom", "x_type_2"):
                r = next(r for r in pairs if r["case"] == case and r["ranks"] == str(rank) and r["half_dt"] == str(half) and r["observable"] == obs and r["discard"] == "0.25")
                label = NAME[case] if rank == 1 and not half and obs == "U_per_atom" else ""
                protocol = f"{rank} rank"+("s" if rank > 1 else "")+(r", $\Delta t/2$" if half else "")
                quantity = r"$U/N$" if obs == "U_per_atom" else r"$x_B$"
                lines.append(label+" & "+protocol+" & "+quantity+" & $"+sci(float(r["difference_fast_minus_atom"]),3)+r"\pm "+sci(float(r["ci95_halfwidth"]),2)+f"$ & {float(r['minimum_ess']):.1f}"+r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (MAN / "revision_dynamic_table.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
    lines = [r"\begin{tabular}{llrr}", r"\toprule", r"Model & Implementation & Quantity & $\Delta t/2-\Delta t$ mean difference $\pm$ 95\% CI \\", r"\midrule"]
    for r in read_csv(DATA / "audit/matched_duration_timestep_control.csv"):
        quantity = r"$U/N$" if r["observable"] == "U_per_atom" else r"$x_B$"
        lines.append(NAME[r["case"]]+" & "+r["kind"]+"/swap & "+quantity+" & $"+sci(float(r["half_minus_base"]),3)+r"\pm "+sci(float(r["difference_ci95_halfwidth"]),2)+r"$ \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (MAN / "revision_timestep_table.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    if not DATA.exists():
        copy_data()
    TIMING_RAW = read_csv(DATA / "core/timing_raw.csv")
    TIMING = read_csv(DATA / "audit/performance_with_hosts.csv")
    tables()
    main_validation(); main_mobile(); main_size(); main_strong(); main_styles()
    si_closure()
    nonlj = ["morse", "table_lj", "buck", "buck_coul_cut", "coul_dsf", "hybrid_overlay_lj_yukawa", "yukawa", "zbl"]
    dense_trajectories("Figure_S2", nonlj, exact=True)
    dense_trajectories("Figure_S3", nonlj)
    si_lj_variants(); si_nonlj(); si_unequal(); si_repeats()
    dense_trajectories("Figure_S8", [("lj_cut",1,False),("lj_cut",2,False),("morse_dynamic",1,False),
                       ("morse_dynamic",2,False),("lj_cut",1,True),("morse_dynamic",1,True)], dynamic=True)
    (HERE / "figure_index.json").write_text(json.dumps(ASSETS, indent=2)+"\n")
    (HERE / "figure_layout_checks.json").write_text(json.dumps(CHECKS, indent=2)+"\n")
    print("Generated", len(ASSETS), "figures and 5 statistical tables.", flush=True)
