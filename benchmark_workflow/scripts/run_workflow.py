from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

LOOP_RE = re.compile(r"Loop time of\s+([0-9.eE+-]+)")
FALLBACK_PATTERNS = (
    "parallel MC disabled",
    "FastSwap: parallel MC disabled",
    "Safe interior too small",
)

NONLJ_RICH_CASE_NAMES = {
    "morse",
    "buck_coul_cut",
    "coul_dsf",
    "hybrid_overlay_lj_yukawa",
    "zbl",
}


def load_manifest(suite: str) -> list[dict]:
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    if suite in ("all", "paper_full"):
        return manifest
    if suite == "representative":
        return [row for row in manifest if row["suite"] in ("serial_pair_styles", "parallel_pair_styles")]
    if suite == "scaling":
        return [
            row
            for row in manifest
            if row["suite"] in (
                "serial_lj_scaling",
                "parallel_lj_scaling",
                "strong_scaling",
                "size_scaling",
                "unequal_cutoff_scaling",
                "nonlj_parallel_scaling",
            )
        ]
    if suite == "strong_scaling":
        return [row for row in manifest if row["suite"] == "strong_scaling"]
    if suite == "size_scaling":
        return [row for row in manifest if row["suite"] == "size_scaling"]
    if suite == "unequal_cutoff_scaling":
        return [row for row in manifest if row["suite"] == "unequal_cutoff_scaling"]
    if suite == "nonlj_parallel_scaling":
        return [row for row in manifest if row["suite"] == "nonlj_parallel_scaling"]
    if suite == "validation":
        return [
            row
            for row in manifest
            if row["suite"] in ("validation_exact", "validation_parallel_stats", "boundary_guards")
        ]
    if suite == "validation_exact":
        return [row for row in manifest if row["suite"] == "validation_exact"]
    if suite == "validation_parallel_stats":
        return [row for row in manifest if row["suite"] == "validation_parallel_stats"]
    if suite == "boundary":
        return [row for row in manifest if row["suite"] == "boundary_guards"]
    if suite == "timeseries":
        return [row for row in manifest if row["suite"] in ("timeseries_exact", "timeseries_parallel_stats")]
    if suite == "timing_repeats":
        return [row for row in manifest if row["suite"] == "timing_repeats"]
    if suite == "rich_validation":
        return [
            row
            for row in manifest
            if row["suite"] in ("timeseries_exact", "timeseries_parallel_stats", "timing_repeats")
        ]
    if suite == "nonlj_rich_validation":
        return [
            row
            for row in manifest
            if row["suite"] in ("validation_parallel_stats", "timeseries_parallel_stats")
            and row["case"] in NONLJ_RICH_CASE_NAMES
        ]
    return [row for row in manifest if row["suite"] == suite]


def parse_loop_time(logfile: Path) -> float | None:
    if not logfile.exists():
        return None
    text = logfile.read_text(encoding="utf-8", errors="replace")
    match = None
    for match in LOOP_RE.finditer(text):
        pass
    return float(match.group(1)) if match else None


def scan_diagnostics(logfile: Path) -> dict:
    if not logfile.exists():
        return {"fallback": False, "warnings": [], "errors": []}
    warnings: list[str] = []
    errors: list[str] = []
    fallback = False
    for line in logfile.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if any(pattern in stripped for pattern in FALLBACK_PATTERNS):
            fallback = True
        if "WARNING" in stripped:
            warnings.append(stripped)
        if "ERROR" in stripped:
            errors.append(stripped)
    return {"fallback": fallback, "warnings": warnings[-5:], "errors": errors[-5:]}


def parse_final_thermo(logfile: Path) -> dict:
    if not logfile.exists():
        return {}
    header: list[str] | None = None
    rows: list[list[float]] = []
    for line in logfile.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "Step" and parts[1] == "PotEng":
            header = parts
            rows = []
            continue
        if header and len(parts) == len(header):
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    if not header or not rows:
        return {}
    return dict(zip(header, rows[-1]))


def command_for(row: dict, args: argparse.Namespace, logfile: Path) -> list[str]:
    infile = ROOT / row["input"]
    return [
        args.mpi_launcher,
        args.np_flag,
        str(row["ranks"]),
        args.lmp,
        "-screen",
        "none",
        "-log",
        str(logfile),
        "-in",
        str(infile),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmp", required=True, help="LAMMPS executable, e.g. /path/to/lmp")
    parser.add_argument("--mpi-launcher", default="mpirun", help="MPI launcher, e.g. mpirun or srun")
    parser.add_argument("--np-flag", default="-np", help="rank-count flag, e.g. -np for mpirun or -n for srun")
    parser.add_argument(
        "--suite",
        default="all",
        choices=[
            "all",
            "representative",
            "scaling",
            "strong_scaling",
            "size_scaling",
            "unequal_cutoff_scaling",
            "nonlj_parallel_scaling",
            "validation",
            "validation_exact",
            "validation_parallel_stats",
            "boundary",
            "timeseries",
            "timing_repeats",
            "rich_validation",
            "nonlj_rich_validation",
            "paper_full",
        ],
    )
    parser.add_argument("--max-ranks", type=int, default=24)
    parser.add_argument("--timeout-s", type=int, default=7200)
    parser.add_argument("--ids", nargs="*", help="optional list of run ids, case names, or suite names to execute")
    parser.add_argument("--append-status", action="store_true", help="merge new run records into an existing run_status.json by run id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = [row for row in load_manifest(args.suite) if row["ranks"] <= args.max_ranks]
    if args.ids:
        selected = set(args.ids)
        manifest = [
            row
            for row in manifest
            if row["id"] in selected or row["case"] in selected or row["suite"] in selected
        ]
    status_path = RESULTS / ("dry_run_commands.json" if args.dry_run else "run_status.json")
    run_status_by_id: dict[str, dict] = {}
    run_status_order: list[str] = []
    if args.append_status and status_path.exists():
        for old in json.loads(status_path.read_text(encoding="utf-8")):
            old_id = old.get("id")
            if not old_id:
                continue
            if old_id not in run_status_by_id:
                run_status_order.append(old_id)
            run_status_by_id[old_id] = old

    def current_status_rows() -> list[dict]:
        return [run_status_by_id[run_id] for run_id in run_status_order]

    def store_status(status: dict) -> None:
        run_id = status["id"]
        if run_id not in run_status_by_id:
            run_status_order.append(run_id)
        run_status_by_id[run_id] = status
        status_path.write_text(json.dumps(current_status_rows(), indent=2), encoding="utf-8")

    for idx, row in enumerate(manifest, 1):
        logfile = LOGS / f"log.{row['id']}.lammps"
        cmd = command_for(row, args, logfile)
        print(
            f"[{idx}/{len(manifest)}] {row['suite']} {row['case']} {row['kind']} "
            f"ranks={row['ranks']} atoms={row['natoms']}",
            flush=True,
        )
        if args.dry_run:
            print("  " + " ".join(cmd), flush=True)
            store_status({**row, "returncode": None, "log": str(logfile), "command": cmd, "dry_run": True})
            continue
        if logfile.exists():
            logfile.unlink()
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=args.timeout_s)
            elapsed = time.time() - t0
            diag = scan_diagnostics(logfile)
            loop_s = parse_loop_time(logfile) if proc.returncode == 0 else None
            final = parse_final_thermo(logfile) if proc.returncode == 0 else {}
            status = {
                **row,
                "returncode": proc.returncode,
                "elapsed_wall_s": elapsed,
                "loop_s": loop_s,
                "time_per_sweep_s": (loop_s / row["nsweeps"]) if loop_s is not None else None,
                "log": str(logfile.relative_to(ROOT)),
                "command": cmd,
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
                "fallback_detected": diag["fallback"],
                "warnings": diag["warnings"],
                "errors": diag["errors"],
                "final_thermo": final,
            }
        except subprocess.TimeoutExpired as exc:
            status = {
                **row,
                "returncode": 124,
                "elapsed_wall_s": args.timeout_s,
                "loop_s": None,
                "time_per_sweep_s": None,
                "log": str(logfile.relative_to(ROOT)),
                "command": cmd,
                "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": "TIMEOUT",
                "fallback_detected": False,
                "warnings": [],
                "errors": ["TIMEOUT"],
                "final_thermo": {},
            }
        store_status(status)
    status_path.write_text(json.dumps(current_status_rows(), indent=2), encoding="utf-8")
    print(f"Wrote {status_path}")


if __name__ == "__main__":
    main()
