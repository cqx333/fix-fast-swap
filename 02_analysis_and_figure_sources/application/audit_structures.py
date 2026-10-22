"""Identify phase branches in the actual retained application dataset."""
import csv
import hashlib
import json
import os
from pathlib import Path
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier as PTM
import ovito

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
DATA = HERE / "data"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


cache = {}
for path in (WORK / "CPC_phase_all_temperatures_review_20261016/ptm_cache.json",
             WORK / "CPC_phase_rich_lowB_review_20261018/ptm_cache.json", DATA / "ptm_cache.json"):
    if path.exists():
        for item in json.loads(path.read_text(encoding="utf-8")).values():
            cache[item["sha256"]] = item
rows = []
trajectory = read(DATA / "trajectory_statistics.csv")
refs = read(DATA / "reference_enthalpy.csv")
targets = [(r, stage) for r in trajectory for stage in ("initial", "cycle_0060")]
targets += [(dict(r, case=Path(r["source"]).name, mu="", start="pure_A"), "reference") for r in refs]
for index, (r, stage) in enumerate(targets):
    folder = Path(r["source"])
    path = folder / ("structure.dump" if stage == "reference" else stage + "/structure.dump")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"case": r["case"], "T": r["T"], "phase": r["phase"], "mu": r.get("mu", ""),
              "start": r.get("start", ""), "stage": stage, "sha256": digest}
    if digest in cache:
        result.update({k: cache[digest][k] for k in ("FCC_0.15", "HCP_0.15", "BCC_0.15", "FCC_0.1")})
    else:
        pipeline = import_file(str(path))
        pipeline.modifiers.append(PTM(rmsd_cutoff=.15, output_rmsd=True))
        data = pipeline.compute()
        kinds = np.asarray(data.particles["Structure Type"])
        rmsd = np.asarray(data.particles["RMSD"])
        for name, kind in (("FCC", PTM.Type.FCC), ("HCP", PTM.Type.HCP), ("BCC", PTM.Type.BCC)):
            result[name+"_0.15"] = float(np.mean(kinds == int(kind)))
        result["FCC_0.1"] = float(np.mean((kinds == int(PTM.Type.FCC)) & (rmsd <= .1)))
        del pipeline, data
        cache[digest] = result
    with path.open() as stream:
        head = [next(stream) for _ in range(9)]
    lengths = [float(line.split()[1])-float(line.split()[0]) for line in head[5:8]]
    result["min_length"], result["max_length"] = min(lengths), max(lengths)
    result["cubic"] = max(lengths)-min(lengths) < 1e-7
    assert result["cubic"]
    if stage == "cycle_0060":
        with (folder / "cycle_0060/log.lammps").open(encoding="utf-8", errors="replace") as stream:
            prefix = "".join(next(stream) for _ in range(20))
        assert "2 by 2 by 2 MPI processor grid" in prefix
    rows.append(result)
    if (index+1) % 50 == 0:
        (DATA / "ptm_cache.json").write_text(json.dumps(cache), encoding="utf-8")
        print("Application structure", index+1, "/", len(targets), flush=True)
keys = list(dict.fromkeys(k for row in rows for k in row))
with (DATA / "structure_identification.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    w = csv.DictWriter(stream, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
(DATA / "ptm_cache.json").write_text(json.dumps(cache), encoding="utf-8")
final_solid = [r for r in rows if r["phase"] == "solid" and r["stage"] == "cycle_0060"]
final_liquid = [r for r in rows if r["phase"] == "liquid" and r["stage"] == "cycle_0060"]
summary = {"ovito": ovito.version_string, "structures": len(rows),
           "final_solid_FCC_range": [min(r["FCC_0.15"] for r in final_solid), max(r["FCC_0.15"] for r in final_solid)],
           "final_liquid_FCC_range": [min(r["FCC_0.15"] for r in final_liquid), max(r["FCC_0.15"] for r in final_liquid)],
           "minimum_final_box_length": min(r["min_length"] for r in rows),
           "reference_solid_FCC_range": [min(r["FCC_0.15"] for r in rows if r["phase"] == "solid" and r["stage"] == "reference"),
                                         max(r["FCC_0.15"] for r in rows if r["phase"] == "solid" and r["stage"] == "reference")],
           "reference_liquid_max_FCC": max(r["FCC_0.15"] for r in rows if r["phase"] == "liquid" and r["stage"] == "reference")}
(DATA / "structure_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
