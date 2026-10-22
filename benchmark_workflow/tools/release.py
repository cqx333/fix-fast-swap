"""Run offline checks and write the upload archive without local caches/results."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verification = ROOT / "verification"
    verification.mkdir(exist_ok=True)
    commands = [[sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                [sys.executable, "scripts/workflow.py", "--dry-run"],
                [sys.executable, "scripts/workflow.py", "--suite", "mdmc", "--ids", "morse_dynamic", "--dry-run"]]
    rows = []
    for i, command in enumerate(commands, 1):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        log = verification / f"check_{i}.txt"
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
        rows.append({"command": command, "returncode": result.returncode, "log": log.relative_to(ROOT).as_posix()})
        if result.returncode:
            raise RuntimeError("Offline verification failed: " + str(log))
    for path in ROOT.glob("*.sh"):
        if b"\r\n" in path.read_bytes():
            raise ValueError("Shell scripts must use LF: " + path.name)
    report = {"date_utc": datetime.now(timezone.utc).isoformat(), "checks": rows,
              "lammps_production_runs_executed_locally": 0, "cluster_MPI_available_locally": False,
              "note": "Raw log/snapshot fixtures are archived evidence; offline validators are not new simulation results."}
    (verification / "VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                   and "results" not in p.relative_to(ROOT).parts and p.name != "SHA256SUMS.txt")
    (ROOT / "SHA256SUMS.txt").write_text("".join(digest(p) + "  " + p.relative_to(ROOT).as_posix() + "\n" for p in paths), encoding="ascii")
    paths.append(ROOT / "SHA256SUMS.txt")
    archive_path = ROOT.parent / (ROOT.name + "_upload.zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, ROOT.name + "/" + path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip():
            raise ValueError("Archive CRC failure")
        for name in archive.namelist():
            if ".." in Path(name).parts or not name.startswith(ROOT.name + "/"):
                raise ValueError("Unsafe archive path")
    print(json.dumps({"archive": str(archive_path), "bytes": archive_path.stat().st_size,
                      "files": len(paths), "offline_checks": "passed", "new_MPI_simulations": 0}, indent=2))


if __name__ == "__main__":
    main()
