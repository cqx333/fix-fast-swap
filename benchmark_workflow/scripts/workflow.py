"""Run only RNG-affected fast/swap calculations, sequentially and resumably."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from log_data import audit, parse_thermo

MARKER = "Fix fast/swap RNG revision 20261020: rank-local seeds; serial stream unchanged"
FAMILIES = ("parallel", "mdmc", "timing", "size3d")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1048576), b""):
            h.update(data)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def runtime_signature(metadata):
    fields = ("host", "executable_sha256", "libraries", "launchers", "np_flag", "source_sha256",
              "manifest_sha256", "workflow_sha256", "mpi_environment", "cmake_cache_sha256")
    # ldd virtual addresses vary across launches and are not build identity.
    return signature({k: metadata.get(k) for k in fields})


def verify_inputs():
    checks = load(ROOT / "input_integrity.json")
    for name, expected in checks.items():
        path = ROOT / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError("Bundle input changed/missing: " + name)


def select(tasks, suite="all", ids=None):
    families = set(FAMILIES) if suite == "all" else set(suite.split(","))
    if families - set(FAMILIES):
        raise ValueError("Unknown suite; use all, parallel, mdmc, timing, size3d, or comma-separated suites")
    tokens = ids.split(",") if ids else []
    chosen = {t["id"] for t in tasks if t["family"] in families and
              (not tokens or any(v in t["id"] for v in tokens))}
    if not chosen:
        raise ValueError("No tasks matched")
    if tokens and any(not any(token in key for key in chosen) for token in tokens):
        raise ValueError("At least one --ids fragment matched no selected task")
    index = {t["id"]: t for t in tasks}
    pending = list(chosen)
    while pending:
        for key in index[pending.pop()]["depends_on"]:
            if key not in chosen:
                chosen.add(key)
                pending.append(key)
    result = [t for t in tasks if t["id"] in chosen]
    if any(t["kind"] != "fast" or t["ranks"] <= 1 for t in result):
        raise ValueError("This package must never schedule atom/swap or rank-1 simulations")
    return result


def output_directory(value):
    path = (ROOT / value).resolve()
    if ROOT not in path.parents or not re.fullmatch(r"[A-Za-z0-9_./-]+", path.relative_to(ROOT).as_posix()):
        raise ValueError("Use a simple output subdirectory inside this package, e.g. results")
    if path.parts[len(ROOT.parts)] in ("baseline", "src", "scripts", "vendor", "templates", "tables", "reused_rng_regression", "tests", "tools"):
        raise ValueError("Output cannot overlap bundled evidence or code")
    return path


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def render(task, output, sweeps=None):
    text = (ROOT / task["template"]).read_text(encoding="ascii")
    raw = relative(output / "raw")
    if task["renderer"] == "mobile":
        restart = task["restart_input"].replace("@RAW@", raw)
        text, count = re.subn(r"^read_restart\s+.+$", 'read_restart "' + restart + '"', text, flags=re.M)
        if count != 1:
            raise ValueError("Mobile restart template is ambiguous")
        text = text.replace("incremental/raw/", raw + "/")
    else:
        text = text.replace("results/raw/", raw + "/")
    if task["family"] in ("timing", "size3d"):
        n = task["requested_sweeps"] if sweeps is None else sweeps
        pattern = r'(?P<head>print "REVISION_PRODUCTION_BEGIN"\s*\n)thermo \d+\s*\nrun \d+'
        text, count = re.subn(pattern, lambda m: m["head"] + f"thermo {n}\nrun {n}", text)
        if count != 1:
            raise ValueError("Timing production section is ambiguous")
    return text


def process(command, output, error, env, timeout):
    with Path(output).open("w", encoding="utf-8") as out, Path(error).open("w", encoding="utf-8") as err:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                                 start_new_session=os.name == "posix")
        try:
            return child.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name == "posix":
                os.killpg(child.pid, signal.SIGTERM)
            else:
                child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(child.pid, signal.SIGKILL)
                else:
                    child.kill()
                child.wait()
            raise


def validate(text, task, sweeps):
    if MARKER not in text:
        raise ValueError("The actual calculation did not print the corrected RNG runtime marker")
    audit_spec = dict(task, expect_parallel=False)
    result = audit(text, audit_spec)
    if result["errors"] or not result["loop_s"] or result["fallback"]:
        raise ValueError("LAMMPS error, empty production loop, or unexpected parallel fallback")
    if result.get("observed_ranks") != task["ranks"] or result.get("observed_processor_grid") != task["processors"]:
        raise ValueError("Observed MPI rank count/grid differs from the design")
    actual = result["actual_attempts"]
    if actual != sweeps * task["natoms"] or not 0 <= result["accepted_moves"] <= actual:
        raise ValueError("Actual MC attempts/acceptances differ from the intended budget")
    radius = result["observed_ghost_cutoff"]
    half = min(length / p / 2 for length, p in zip(task["box_lengths"], task["processors"]) if p > 1)
    if radius is None or half <= radius:
        raise ValueError("Observed communication cutoff violates the independence margin")
    result["observed_independence_margin"] = half - radius
    rows = parse_thermo(text)
    if task["family"] == "mdmc":
        distances = [r["c_rmin"] for r in rows]
        if not distances or min(distances) < .55 or not all(math.isfinite(x) for x in distances):
            raise ValueError("MD/MC close-contact instability")
        result["minimum_pair_distance"] = min(distances)
        events = [int(b["Step"]) for a, b in zip(rows, rows[1:]) if b["f_mc[1]"] > a["f_mc[1]"]]
        if events != list(range(100, sweeps * 100 + 1, 100)):
            raise ValueError("MD/MC actual call schedule differs from the matched reference")
    return result


def validate_raw(task, output, log):
    import numpy as np
    from legacy_common import frames
    thermo = {int(r["Step"]): r for r in parse_thermo(Path(log).read_text(errors="replace"))}
    paths = [output / "raw" / (task["id"] + ".final.dump")]
    if task["family"] == "parallel":
        paths.append(output / "raw" / (task["id"] + ".types.dump"))
    checked = 0
    for path in paths:
        count = 0
        for step, columns, data, _ in frames(path):
            count += 1
            if len(data) != task["natoms"] or step not in thermo or not np.all(np.isfinite(data)):
                raise ValueError("Truncated/nonfinite raw snapshot or snapshot step missing from thermo")
            ids = data[:, columns.index("id")]
            labels = data[:, columns.index("type")]
            if len(np.unique(ids)) != task["natoms"] or not np.all(ids == ids.astype(int)) or not np.all(labels == labels.astype(int)):
                raise ValueError("Duplicate/noninteger IDs or types in raw snapshot")
            if not np.all((labels >= 1) & (labels <= task["ntypes"])):
                raise ValueError("Raw snapshot contains invalid atom type")
            for i in range(1, task["ntypes"] + 1):
                if int(np.sum(labels == i)) != thermo[step][f"c_ctype[{i}]"]:
                    raise ValueError("Raw snapshot type counts differ from thermo")
            checked += 1
        if not count or (path.name.endswith(".final.dump") and count != 1):
            raise ValueError("Missing or invalid final snapshot")
    return {"raw_snapshot_frames_verified": checked}


@contextlib.contextmanager
def locked(output):
    path = output / "workflow.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("Output is locked. Only remove workflow.lock after confirming its process and MPI children exited")
    with os.fdopen(fd, "w") as stream:
        json.dump({"host": socket.gethostname(), "pid": os.getpid()}, stream)
    try:
        yield
    finally:
        path.unlink()


def runtime(args, env):
    binary = shutil.which(args.lmp) if not Path(args.lmp).is_file() else str(Path(args.lmp).resolve())
    launchers = {family: shlex.split(args.mpi_launcher) if args.mpi_launcher else
                 (["mpirun", "-bind-to", "core"] if family == "size3d" else ["mpirun"])
                 for family in FAMILIES}
    if not binary or any(not v or not shutil.which(v[0]) for v in launchers.values()):
        raise ValueError("LAMMPS executable or matching MPI launcher not found")
    if os.name != "posix":
        raise ValueError("Actual MPI runs require the Linux cluster; local dry-run and tests are supported")
    check = subprocess.run(["ldd", binary], env=env, capture_output=True, text=True, timeout=30)
    if check.returncode or "not found" in check.stdout + check.stderr:
        raise ValueError("Missing dynamic library; correct the environment before calculating")
    libraries = {name: {"path": str(Path(path).resolve()), "sha256": sha(path)}
                 for name, path in re.findall(r"(\S+)\s+=>\s+(/\S+)\s+\(", check.stdout)
                 if name.startswith(("liblammps", "libmpi", "libmpicxx", "libfabric"))}
    expected_dir = Path(args.expected_library_dir or Path(binary).parent).resolve()
    lammps = [v for k, v in libraries.items() if k.startswith("liblammps")]
    if not lammps or any(Path(v["path"]).parent != expected_dir for v in lammps):
        raise ValueError("Loaded liblammps is not from the requested build directory")
    known = load(ROOT / "baseline/source_datasets.json")["current_rng_library_sha256"]
    if args.expected_library_sha256 and any(v["sha256"] != args.expected_library_sha256 for v in lammps):
        raise ValueError("Loaded liblammps SHA-256 differs from --expected-library-sha256")
    if any(v["sha256"] == "7f55a477112226c840c035b252ce109b365b343072f6dc9a81e2f7d12d311373" for v in lammps):
        raise ValueError("The pre-fix library is still loaded")
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix and os.environ.get("CONDA_DEFAULT_ENV") == "lmp-fast":
        prefix = Path(prefix).resolve()
        launch_paths = [Path(shutil.which(v[0])).resolve() for v in launchers.values()]
        mpi = [v for k, v in libraries.items() if k.startswith(("libmpi", "libmpicxx"))]
        if any(prefix not in p.parents for p in launch_paths) or not mpi or any(prefix not in Path(v["path"]).parents for v in mpi):
            raise ValueError("MPI launcher/libraries do not consistently belong to lmp-fast")
    metadata = {"host": socket.gethostname(), "executable": str(Path(binary).resolve()),
                "executable_sha256": sha(binary), "libraries": libraries, "launchers": launchers,
                "known_regression_library_matches": all(v["sha256"] == known for v in lammps),
                "np_flag": args.np_flag, "ldd": check.stdout, "python": sys.version,
                "source_sha256": {p.name: sha(p) for p in (ROOT / "src/MC").glob("fix_fast_swap.*")},
                "manifest_sha256": sha(ROOT / "manifest.json"),
                "workflow_sha256": sha(Path(__file__)),
                "mpi_environment": {k: v for k, v in env.items() if k.startswith(("OMP_", "I_MPI_", "OMPI_", "MPICH_", "MPIR_", "FI_"))},
                "library_runtime_marker_required_per_task": True,
                "note": "Historical atom/rank-1 results retain their original binary and host provenance."}
    if args.cmake_cache:
        cache = Path(args.cmake_cache).resolve(strict=True)
        text = cache.read_text(errors="replace")
        source = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$", text, re.M)
        if not source:
            raise ValueError("CMake source directory missing from cache")
        for name in metadata["source_sha256"]:
            actual = Path(source[1]).parent / "src/MC" / name
            if actual.read_bytes().replace(b"\r\n", b"\n") != (ROOT / "src/MC" / name).read_bytes().replace(b"\r\n", b"\n"):
                raise ValueError("CMake source tree differs from the bundled RNG patch")
        metadata["cmake_cache_sha256"] = sha(cache)
    metadata["identity"] = runtime_signature(metadata)
    return metadata


def run(args, tasks, output):
    import numpy
    import scipy
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_DYNAMIC="FALSE")
    if args.expected_library_dir:
        env["LD_LIBRARY_PATH"] = str(Path(args.expected_library_dir).resolve()) + ":" + env.get("LD_LIBRARY_PATH", "")
    meta = runtime(args, env)
    if any(t["family"] in ("timing", "size3d") for t in tasks) and meta["host"] != "node05":
        raise ValueError("Reused timing baselines are from node05. Run timings on node05, not another host")
    for folder in ("inputs", "logs", "raw", "metadata", "analysis"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    path = output / "run_status.json"
    status = load(path) if path.exists() else {}
    failures = []
    with locked(output):
        old = output / "metadata/runtime.json"
        if old.exists() and load(old)["identity"] != meta["identity"]:
            raise ValueError("This output belongs to another host/runtime/workflow; use a new --output")
        save(old, meta)
        for family, ranks in sorted({(t["family"], t["ranks"]) for t in tasks}):
            probe_file = output / "metadata" / f"affinity_{family}_p{ranks}.jsonl"
            probe_err = probe_file.with_suffix(".stderr.txt")
            command = meta["launchers"][family] + [args.np_flag, str(ranks), sys.executable, str(ROOT / "scripts/rank_probe.py")]
            code = process(command, probe_file, probe_err, env, 60)
            records = [json.loads(line) for line in probe_file.read_text().splitlines() if line.startswith("{")]
            if code or len(records) != ranks or {r["rank"] for r in records} != set(range(ranks)) or {r["host"] for r in records} != {meta["host"]}:
                raise ValueError("MPI affinity probe failed or ranks span multiple hosts; no production jobs started")
            available = {cpu for r in records for cpu in (r["cpu_affinity"] or [])}
            if available and len(available) < ranks:
                raise ValueError("CPU affinity exposes fewer logical CPUs than ranks")
        for label, command in (("cpu", ["lscpu"]), ("load", ["uptime"]),
                               ("cpu_processes", ["ps", "-eo", "user,pid,pcpu,comm", "--sort=-pcpu"])):
            try:
                p = subprocess.run(command, capture_output=True, text=True, timeout=30)
                (output / "metadata" / (label + ".txt")).write_text(p.stdout + p.stderr, encoding="utf-8")
            except (OSError, subprocess.TimeoutExpired):
                pass

        def launch(task, suffix, amount):
            tag = task["id"] + suffix
            inp, log = output / "inputs" / ("in." + tag), output / "logs" / (tag + ".lammps")
            stdout, stderr = output / "logs" / (tag + ".stdout.txt"), output / "logs" / (tag + ".stderr.txt")
            text = render(task, output, amount)
            inp.write_bytes(text.encode("ascii"))
            command = meta["launchers"][task["family"]] + [args.np_flag, str(task["ranks"]), meta["executable"], "-screen", "none", "-log", str(log), "-in", str(inp)]
            started = time.monotonic()
            code = process(command, stdout, stderr, env, args.timeout_s)
            content = log.read_text(errors="replace") if log.exists() else ""
            if code:
                raise RuntimeError("LAMMPS exited with code " + str(code) + "; inspect " + str(log))
            result = validate(content, task, amount)
            result.update(log=relative(log), input=relative(inp), elapsed_seconds=time.monotonic() - started,
                          input_sha256=sha(inp), log_sha256=sha(log), executed_sweeps=amount)
            return result, [inp, log, stdout, stderr]

        for task in tasks:
            key = task["id"]
            try:
                previous = status.get(key, {})
                if previous.get("state") == "integrity_error" and not args.force:
                    raise ValueError("A previous integrity mismatch requires restoring files or explicitly selecting --force")
                if any(status.get(k, {}).get("state") != "complete" for k in task["depends_on"]):
                    raise ValueError("A required earlier NEW segment has not completed")
                dependencies = [ROOT / p for p in task["asset_dependencies"]]
                if task["renderer"] == "mobile":
                    dependencies += [ROOT / task["restart_input"].replace("@RAW@", relative(output / "raw"))]
                fingerprint = signature({"runtime": meta["identity"], "task": task,
                                         "dependencies": {relative(p): sha(p) for p in dependencies}})
                if previous.get("state") == "complete" and not args.force:
                    if previous["fingerprint"] != fingerprint or any(not (ROOT / p).is_file() or sha(ROOT / p) != h
                                                                    for p, h in previous["output_sha256"].items()):
                        raise ValueError("Completed task/input has changed; restore it or explicitly select --force")
                    print("SKIP", key, flush=True)
                    continue
                status[key] = {"id": key, "state": "running", "fingerprint": fingerprint}
                save(path, status)
                print(f"RUN {task['number']:03d} {key}", flush=True)
                files = []
                intervals = []
                if task["family"] in ("timing", "size3d"):
                    for label, command in (("load", ["uptime"]), ("processes", ["ps", "-eo", "user,pid,pcpu,comm", "--sort=-pcpu"])):
                        probe = subprocess.run(command, capture_output=True, text=True, timeout=30)
                        (output / "metadata" / (key + "." + label + ".txt")).write_text(probe.stdout + probe.stderr, encoding="utf-8")
                    amount = max(2, min(64, int(task.get("executed_sweeps", 20)) // 10))
                    pilot, pilot_files = launch(task, "__pilot", amount)
                    files += pilot_files
                    amount = max(2, math.ceil(amount * task["minimum_loop_seconds"] / pilot["loop_s"] * 1.2))
                    for iteration in range(5):
                        if amount > 100000:
                            raise ValueError("Required timing length exceeds 100000 sweeps; inspect performance")
                        result, outputs = launch(task, "__measure" + str(iteration), amount)
                        files += outputs
                        intervals.append({"sweeps": amount, "loop_s": result["loop_s"], "log": result["log"]})
                        if result["loop_s"] >= task["minimum_loop_seconds"]:
                            break
                        amount = max(amount + 1, math.ceil(amount * task["minimum_loop_seconds"] / result["loop_s"] * 1.2))
                    else:
                        raise ValueError("No production interval reached the required 30 seconds")
                else:
                    amount = task["requested_sweeps"]
                    result, files = launch(task, "", amount)
                    files += [output / "raw" / (key + ".final.dump")]
                    if task["family"] == "parallel":
                        files += [output / "raw" / (key + ".types.dump")]
                    if task["family"] == "mdmc":
                        files += [output / "raw" / (key + ".restart")]
                if any(not p.is_file() for p in files):
                    raise ValueError("Required raw output/checkpoint missing")
                if task["family"] in ("parallel", "mdmc"):
                    result.update(validate_raw(task, output, ROOT / result["log"]))
                record = {**task, **result, "state": "complete", "fingerprint": fingerprint,
                          "runtime_identity": meta["identity"], "source_host": meta["host"],
                          "library_hashes": meta["libraries"], "adaptive_intervals": intervals,
                          "output_sha256": {relative(p): sha(p) for p in files}}
                status[key] = record
                save(path, status)
            except KeyboardInterrupt:
                status[key] = {"id": key, "state": "interrupted"}
                save(path, status)
                raise
            except Exception as error:
                failures.append(key)
                status[key] = {"id": key, "state": "integrity_error" if previous.get("state") in ("complete", "integrity_error") else "failed",
                               "error": str(error)}
                save(path, status)
                print("FAILED", key, str(error), flush=True)
                if args.stop_on_error:
                    break
        from analyze import analyze
        analyze(output)
    if failures:
        raise RuntimeError("Failed tasks: " + ", ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp")
    parser.add_argument("--suite", default="all")
    parser.add_argument("--ids", help="Comma-separated ID fragments; earlier NEW mobile segments are included automatically")
    parser.add_argument("--output", default="results")
    parser.add_argument("--mpi-launcher", default=os.environ.get("MPI_LAUNCHER"), help="Override historical launcher settings; normally leave unset")
    parser.add_argument("--np-flag", default=os.environ.get("NP_FLAG", "-np"))
    parser.add_argument("--expected-library-dir")
    parser.add_argument("--expected-library-sha256")
    parser.add_argument("--cmake-cache")
    parser.add_argument("--timeout-s", type=int, default=14400)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    verify_inputs()
    output = output_directory(args.output)
    if args.analyze_only:
        from analyze import analyze
        analyze(output)
        return
    tasks = select(load(ROOT / "manifest.json")["tasks"], args.suite, args.ids)
    from collections import Counter
    print("Selected:", dict(Counter(t["family"] for t in tasks)), "total:", len(tasks))
    print("Only fast/swap, 2--24 ranks; no atom/swap, rank-1, phase diagram, or repeated RNG regression.")
    if args.dry_run:
        for task in tasks:
            print(f"{task['number']:03d} {task['family']:8s} p={task['ranks']:2d} N={task['natoms']:5d} {task['id']}")
            render(task, output)
        print("Input integrity and all selected templates verified. No MPI simulations launched.")
        return
    if not args.lmp:
        parser.error("--lmp is required for calculations")
    output.mkdir(parents=True, exist_ok=True)
    run(args, tasks, output)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print("ERROR:", error, file=sys.stderr)
        sys.exit(2)
