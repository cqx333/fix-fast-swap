"""Offline contract tests; archived logs are fixtures, not new MPI results."""
import collections
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import workflow as w
import analyze as analysis

MANIFEST = w.load(ROOT / "manifest.json")
TASKS = MANIFEST["tasks"]
REFERENCES = w.load(ROOT / "baseline/records.json")


class Contracts(unittest.TestCase):
    def test_exact_scope(self):
        self.assertEqual(len(TASKS), 206)
        self.assertEqual(collections.Counter(t["family"] for t in TASKS),
                         dict(parallel=45, mdmc=36, timing=75, size3d=50))
        self.assertTrue(all(t["kind"] == "fast" and 1 < t["ranks"] <= 24 for t in TASKS))
        self.assertEqual(MANIFEST["applications"], [])

    def test_unique_task_ids_and_counterparts(self):
        self.assertEqual(len({t["id"] for t in TASKS}), len(TASKS))
        for task in TASKS:
            self.assertTrue(task["reference_keys"])
            self.assertTrue(all(REFERENCES[k]["kind"] == "atom" for k in task["reference_keys"]))

    def test_parallel_geometry(self):
        for task in TASKS:
            self.assertEqual(math.prod(task["processors"]), task["ranks"])
            half = min(length / p / 2 for length, p in zip(task["box_lengths"], task["processors"]) if p > 1)
            self.assertGreater(half, 2.7)

    def test_no_superseded_pure_flat_size(self):
        self.assertFalse(any(t.get("group_membership") == ["size"] for t in TASKS))
        self.assertEqual(len({tuple(t["box"]) for t in TASKS if t["family"] == "size3d"}), 10)

    def test_long_chains_start_clean(self):
        long = [t for t in TASKS if t["family"] == "parallel" and t["case"] in ("ternary_lj_cut", "zbl") and t["ranks"] == 24]
        self.assertEqual(len(long), 6)
        self.assertTrue(all(t["requested_sweeps"] == 1200 and not t.get("initial_data_path") for t in long))
        self.assertTrue(all("create_atoms" in w.render(t, ROOT / "results") for t in long))

    def test_timing_assets_pair_exactly(self):
        for task in TASKS:
            if task["family"] in ("timing", "size3d"):
                self.assertEqual(task["minimum_loop_seconds"], 30)
                reference = REFERENCES[task["reference_keys"][0]]
                self.assertEqual(w.sha(ROOT / task["initial_data_path"]), reference["initial_data_sha256"])

    def test_mobile_dependency_order_and_origins(self):
        mobile = [t for t in TASKS if t["family"] == "mdmc"]
        seen = set()
        for task in mobile:
            self.assertTrue(all(k in seen for k in task["depends_on"]))
            if task["start_sweep"] == 0:
                self.assertTrue(task["restart_input"].startswith("baseline/assets/prepare__"))
                self.assertEqual(task["depends_on"], [])
            else:
                self.assertTrue(task["restart_input"].startswith("@RAW@/dynamic__"))
                self.assertEqual(len(task["depends_on"]), 1)
            seen.add(task["id"])

    def test_selection_closes_dependencies(self):
        selected = w.select(TASKS, "mdmc", "lj_cut__p2__s24681357__fast__dtbase__from500")
        self.assertEqual(len(selected), 6)
        self.assertEqual([t["start_sweep"] for t in selected], [0, 100, 200, 300, 400, 500])
        with self.assertRaises(ValueError):
            w.select(TASKS, "parallel", "not_a_model")

    def test_all_inputs_are_parallel_fast(self):
        for task in TASKS:
            text = w.render(task, ROOT / "results")
            self.assertIn("all fast/swap", text)
            self.assertNotIn("all atom/swap", text)
            self.assertNotIn("incremental/raw/", text)
            self.assertIn("processors " + " ".join(map(str, task["processors"])), text)

    def test_timing_only_production_replaced(self):
        task = next(t for t in TASKS if t["family"] == "size3d")
        text = w.render(task, ROOT / "results", 12345)
        self.assertIn('print "REVISION_PRODUCTION_BEGIN"\nthermo 12345\nrun 12345', text)
        self.assertIn("run 1\n", text)

    def test_original_references_are_not_new_library(self):
        for task in TASKS:
            if task["family"] in ("timing", "size3d"):
                ref = REFERENCES[task["reference_keys"][0]]
                libraries = ref["binary_provenance"]["libraries"]
                self.assertTrue(any(v["sha256"].startswith("7f55a477") for k, v in libraries.items() if k.startswith("liblammps")))

    def test_fallback_and_wrong_marker_rejected(self):
        task = next(t for t in TASKS if t["family"] == "parallel" and t["case"] == "morse")
        record = REFERENCES[task["reference_keys"][0]]
        text = (ROOT / record["archived_log"]).read_text(errors="replace")
        with self.assertRaises(ValueError):
            w.validate(text, task, task["requested_sweeps"])
        text += "\n" + w.MARKER + "\n"
        result = w.validate(text, task, task["requested_sweeps"])
        self.assertEqual(result["actual_sweeps"], 600)
        with self.assertRaises(ValueError):
            w.validate(text + "\nFastSwap: parallel MC disabled permanently\n", task, 600)

    def test_three_d_grid_not_mistaken_for_flat(self):
        task = next(t for t in TASKS if t["family"] == "size3d")
        reference = REFERENCES[task["reference_keys"][0]]
        text = (ROOT / reference["archived_log"]).read_text(errors="replace") + "\n" + w.MARKER + "\n"
        result = w.validate(text, task, reference["executed_sweeps"])
        self.assertEqual(result["observed_processor_grid"], [2, 3, 4])

    def test_mobile_alignment_and_samples(self):
        task = next(t for t in TASKS if t["family"] == "mdmc")
        reference = REFERENCES[task["reference_keys"][0]]
        path = ROOT / reference["archived_log"]
        text = path.read_text(errors="replace") + "\n" + w.MARKER + "\n"
        result = w.validate(text, task, 100)
        self.assertEqual(result["actual_attempts"], 51200)
        rows = analysis.samples(task, path, "mdmc")
        self.assertEqual(len(rows), 100)
        self.assertEqual([r["sweeps"] for r in rows], list(range(1, 101)))

    def test_output_paths_cannot_overwrite_evidence(self):
        for value in ("baseline/new", "../outside", "src/new", "reused_rng_regression/x"):
            with self.assertRaises(ValueError):
                w.output_directory(value)
        self.assertEqual(w.output_directory("results"), ROOT / "results")

    def test_independent_bootstrap_streams(self):
        import numpy as np
        x = [np.arange(100, dtype=float)]
        self.assertFalse(np.array_equal(analysis.bootstrap_variances(x, 8), analysis.bootstrap_variances(x, 8, stream=10000)))

    def test_resume_fingerprint_excludes_ldd_addresses(self):
        a = {"host": "node05", "libraries": {"liblammps": {"sha256": "abc"}}, "ldd": "(0x11111)"}
        b = {**a, "ldd": "(0x22222)"}
        self.assertEqual(w.runtime_signature(a), w.runtime_signature(b))
        self.assertNotEqual(w.runtime_signature(a), w.runtime_signature({**a, "host": "node03"}))

    def test_real_uploaded_snapshot_crosscheck(self):
        key = "lj_fast_24681357"
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            (output / "raw").mkdir()
            for suffix in (".final.dump", ".types.dump"):
                shutil.copy2(ROOT / "reused_rng_regression" / (key + suffix), output / "raw" / (key + suffix))
            result = w.validate_raw(dict(id=key, family="parallel", natoms=6144, ntypes=2), output,
                                    ROOT / "reused_rng_regression" / (key + ".log"))
            self.assertEqual(result["raw_snapshot_frames_verified"], 26)

    def test_empty_analysis_does_not_invent_pass(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            report = analysis.analyze(Path(directory))
            self.assertEqual(report["decision"], "INCOMPLETE")
            self.assertEqual(len(report["missing_or_failed"]), 206)

    def test_fixed_charge_constant_energy_not_false_variance_failure(self):
        values = [{"sweeps": k, "U_per_atom": 1.25, "x_type_2": .5 + .001*((k % 3)-1)} for k in range(100)]
        chains = {(kind, seed): values for kind in ("atom", "fast") for seed in (1, 2, 3)}
        rows, flags, _ = analysis.compare(chains, "coul_dsf", 24, "parallel", 2, 6144)
        energy = [r for r in rows if r["observable"] == "U_per_atom"]
        self.assertTrue(all(r["expected_type_independent_constant_energy"] for r in energy))
        self.assertFalse(any("U_per_atom" in flag for flag in flags))

    def test_immutable_bundle_hashes(self):
        w.verify_inputs()


if __name__ == "__main__":
    unittest.main()
