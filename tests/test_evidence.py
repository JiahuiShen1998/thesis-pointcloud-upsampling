"""Protect reported scientific comparisons against corrupt or partial evidence."""

import csv
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "reproduce", ROOT / "tools/reproduce_results.py"
)
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        # Copy only the scientific evidence read by the reconstruction tool.
        sources = set()
        cls.classification = reproduce.modelnet_table(ROOT, sources)
        cls.geometry = reproduce.geometry_table(ROOT, sources)
        cls.detection = reproduce.kitti_table(ROOT, sources)
        for relative in sources:
            target = cls.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def corrupt(self, relative, transform, operation, message):
        target = self.root / relative
        original = target.read_text(encoding="utf-8")
        try:
            target.write_text(transform(original), encoding="utf-8")
            with self.assertRaisesRegex(reproduce.EvidenceError, message):
                operation(self.root, set())
        finally:
            target.write_text(original, encoding="utf-8")

    def test_headline_comparisons_keep_their_baselines(self):
        mn = {(r["line"], r["method"]): r for r in self.classification}
        self.assertAlmostEqual(mn[("B", "PU-Net")]["best_oa_pct"], 91.27, places=2)
        self.assertAlmostEqual(
            mn[("B", "PU-Net")]["delta_vs_reference_pp"], 0.42, places=2
        )
        kt = {(r["detector"], r["line"]): r for r in self.detection}
        sparse = kt[("PointRCNN", "B")]
        self.assertAlmostEqual(sparse["gain_vs_unadapted_ap_points"], 21.52, places=2)
        self.assertAlmostEqual(sparse["gap_to_baseline_ap_points"], -11.28, places=2)
        self.assertTrue(all(r["gap_to_baseline_ap_points"] < 0 for r in self.detection))
        self.assertTrue(all(r["samples"] == 2468 for r in self.geometry))

    def test_changed_classification_metric_is_rejected(self):
        path = (
            reproduce.MN / "pointnet2_final_runs/lineA_original_baseline/metrics.json"
        )

        def change(text):
            obj = json.loads(text)
            obj["overall_accuracy"] = 0.99
            return json.dumps(obj)

        self.corrupt(path, change, reproduce.modelnet_table, "best log OA")

    def test_incomplete_training_log_is_rejected(self):
        path = reproduce.MN / "pointnet2_final_runs/lineA_original_baseline/train.log"
        self.corrupt(
            path,
            lambda s: "\n".join(
                line for line in s.splitlines() if "Epoch 200/200" not in line
            ),
            reproduce.modelnet_table,
            "incomplete/duplicated epoch log",
        )

    def test_non_test_geometry_is_rejected(self):
        path = reproduce.MN / "reports/modelnet40_geometry_equal_n_per_sample.csv"
        self.corrupt(
            path,
            lambda s: s.replace(",test,", ",train,", 1),
            reproduce.geometry_table,
            "failed/non-test",
        )

    def test_duplicate_geometry_summary_is_rejected(self):
        path = reproduce.MN / "reports/modelnet40_geometry_equal_n_summary.csv"
        self.corrupt(
            path,
            lambda s: s + s.splitlines()[1] + "\n",
            reproduce.geometry_table,
            "summary groups",
        )

    def test_partial_kitti_validation_is_rejected(self):
        path = reproduce.KT / "convergence/final_comparison.json"

        def change(text):
            obj = json.loads(text)
            obj["phases"]["line_b_pugcn_observed_first/rcnn"][
                "validation_frames_per_epoch"
            ] = 100
            return json.dumps(obj)

        self.corrupt(path, change, reproduce.kitti_table, "full validation set")

    def test_missing_centerpoint_epoch_is_rejected(self):
        path = reproduce.KT / "convergence/centerpoint_epoch_validation_curve.csv"
        self.corrupt(
            path,
            lambda s: "\n".join(s.splitlines()[:-1]) + "\n",
            reproduce.kitti_table,
            "Incomplete CenterPoint",
        )

    def test_report_outputs_include_source_hashes(self):
        output = self.root / "outputs"
        manifest = reproduce.reproduce(self.root, output)
        self.assertEqual(manifest["modelnet40_classifier_runs"], 14)
        self.assertTrue(
            all(len(digest) == 64 for digest in manifest["sources"].values())
        )
        with (output / "kitti_adaptation.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 4)
        self.assertIn("not full experimental rerun", manifest["scope"])


if __name__ == "__main__":
    unittest.main()
