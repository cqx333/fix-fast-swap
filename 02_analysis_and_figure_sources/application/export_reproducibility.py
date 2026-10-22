"""Export exact application input metadata and focused submission source bundles."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
import platform
from importlib.metadata import version

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT.parent
RAW = WORK / "CPC-test_revision_unified_20261012/results"
DATA = HERE / "data"
runtime = {"python":sys.version,"platform":platform.platform(),
           "packages":{name:version(name) for name in ("numpy","scipy","matplotlib","ovito")},
           "purpose":"Local numerical revision analysis, distinct from simulation build metadata in inputs."}
(DATA/"analysis_environment.json").write_text(json.dumps(runtime,indent=2)+"\n",encoding="utf-8")
MAN = ROOT / "01_manuscript_tex_and_source"
INPUT = DATA / "inputs"
INPUT.mkdir(exist_ok=True)
(INPUT / "cases").mkdir(exist_ok=True)


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


sources = {}
for row in read(DATA / "trajectory_statistics.csv"):
    folder = Path(row["source"])
    sources[row["case"]] = str(folder)
    if folder.exists():
        shutil.copy2(folder / "case.json", INPUT / "cases" / (row["case"]+".json"))
for T in [.59,.60,.61,.62,.63,.64]:
    name = f"T{T:.2f}"
    src = RAW / ("rich_lowB_20261016/" if T >= .61 else "") / name
    dest = INPUT / name
    (dest / "analysis").mkdir(parents=True, exist_ok=True)
    for name in ("effective_config.json", "environment.json", "simulation_protocol.json",
                 "rich_repair.json", "planned_trajectories.csv", "new_rich_trajectories.csv",
                 "lammps_help.txt"):
        if (src/name).exists():
            shutil.copy2(src/name, dest/name)
    for name in ("trajectory_statistics.csv", "state_convergence.csv", "initial_composition_equivalence.csv"):
        shutil.copy2(src/"analysis"/name, dest/"analysis"/name)
(INPUT / "case_sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
for name in ("temperature_workflow.py", "run_application.sh", "rich_initial_T061_T064.json"):
    src = WORK / "application_rich_repair_061_064_20261016/upload" / name
    if src.exists():
        shutil.copy2(src, HERE/name)
for name in ("rank_entry.py", "requirements.txt", "COPYING"):
    shutil.copy2(WORK/"application_temperature_8x3_20261014"/name, HERE/name)
(HERE/"potential").mkdir(exist_ok=True)
shutil.copy2(WORK/"application_temperature_8x3_20261014/potential/force.txt", HERE/"potential/force.txt")
for name in ("solid_rerun_cases.csv", "solid_phase_identity.csv"):
    shutil.copy2(WORK/"CPC_phase_all_temperatures_review_20261016"/name, DATA/("historical_"+name))
for name in ("integrity.json", "coverage_check.json", "replacement_trajectory_audit.csv", "structure_PTM.csv"):
    shutil.copy2(WORK/"CPC_phase_rich_lowB_review_20261018"/name, DATA/("replacement_"+name))

if "--package" in sys.argv:
    out = ROOT / "04_submission_packages"
    out.mkdir(exist_ok=True)
    support = [p for p in MAN.glob("*") if p.suffix in (".cls", ".sty", ".bst", ".bib")]
    common_tables = [p for p in MAN.glob("*_table.tex")]
    for kind in ("manuscript", "SI"):
        base_name = "main" if kind == "manuscript" else "supplementary_information"
        figures = [MAN/"Figure_1.png"]+[MAN/f"Figure_{i}.pdf" for i in range(2,9)] if kind=="manuscript" else [MAN/f"Figure_S{i}.pdf" for i in range(1,16)]
        for marked in (False, True):
            target = out / (kind+("_marked" if marked else "_clean")+"_sources.zip")
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
                source = (MAN/(base_name+".tex")).read_text(encoding="utf-8")
                if marked:
                    source = "\\def\\MarkChanges{1}\n"+source
                z.writestr(base_name+".tex", source.encode("utf-8"))
                for p in support+common_tables+figures:
                    z.write(p, p.name)
                if kind=="SI":
                    z.write(MAN/"application_si.tex", "application_si.tex")
                else:
                    z.write(MAN/"main.bbl", "main.bbl")
                    for p in (MAN/"thumbnails").glob("*"):
                        if p.is_file(): z.write(p, "thumbnails/"+p.name)
    with zipfile.ZipFile(out/"application_reproducibility.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in HERE.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".png", ".pyc"):
                z.write(p, p.relative_to(HERE))
    files = [p for p in out.iterdir() if p.suffix==".zip"]
    (out/"SHA256SUMS.txt").write_text("\n".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+p.name for p in files)+"\n", encoding="ascii")
print("Application inputs exported; packaging:", "--package" in sys.argv, flush=True)
