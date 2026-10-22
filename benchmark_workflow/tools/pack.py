"""Package raw new results with the plan and old reference provenance."""
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = (ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results")).resolve()
    if ROOT not in results.parents or not (results / "run_status.json").exists():
        raise ValueError("Specify an existing results subdirectory inside this bundle")
    target = ROOT.parent / (ROOT.name + "_" + results.name + ".zip")
    paths = [p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and not any(x.startswith(".pytest") for x in p.parts)
             and (results in p.parents or "results" not in p.relative_to(ROOT).parts)]
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, ROOT.name + "/" + path.relative_to(ROOT).as_posix())
    print(target)


if __name__ == "__main__":
    main()
