"""Check asset diagnostics without requiring PyTorch or LFS downloads."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("demo", ROOT / "tools/demo_inference.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class DemoAssetTests(unittest.TestCase):
    def test_lfs_pointer_is_not_accepted_as_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            path.write_text(
                "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n"
            )
            with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
                demo.require_asset(path)

    def test_missing_asset_has_download_guidance(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "selective Git LFS"):
                demo.require_asset(Path(directory) / "missing.npy")

    def test_label_mapping_matches_archived_class_order(self):
        labels, _ = demo.label_map()
        self.assertEqual(len(labels), 40)
        self.assertEqual(labels[0], "airplane")
        self.assertEqual(labels[39], "xbox")


if __name__ == "__main__":
    unittest.main()
