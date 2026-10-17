import json
from pathlib import Path
import tempfile
import unittest

from script.prepare_decoder_rgb import prepare


class PrepareRGBTest(unittest.TestCase):
    def fixture(self, root):
        sources = ["rgb/Scene02/a.jpg", "rgb/Scene01/b.jpg", "ai_001/a.png", "ai_002/b.png"]
        for i, relative in enumerate(sources):
            path = root / "data" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes([i]))  # Selection only; no image decoding here.
        for name, paths in [("outdoor", sources[:1]), ("val", sources[1:2]), ("indoor", sources[2:])]:
            (root / f"{name}.txt").write_text("\n".join(p + " missing_depth.png" for p in paths), encoding="utf-8")
        config = {"dataset": {"train": {"dataset_list": [
            {"name": "vkitti", "dir": "data", "filenames": str(root / "outdoor.txt")},
            {"name": "hypersim", "dir": "data", "filenames": str(root / "indoor.txt")}]}}}
        (root / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
        return {"base_data_dir": root, "output_dir": root / "prepared",
                "dataset_config": root / "config.yaml", "vkitti_val_list": root / "val.txt",
                "outdoor_train": 1, "outdoor_val": 1, "indoor_train": 1, "indoor_val": 1}

    def test_only_rgb_copied_and_scene_split_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            prepare(**args)
            train = {p.read_bytes() for p in (root / "prepared/train").iterdir()}
            val = {p.read_bytes() for p in (root / "prepared/val").iterdir()}
            self.assertEqual(len(train), 2)
            self.assertEqual(len(val), 2)
            self.assertFalse(train & val)
            with self.assertRaises(FileExistsError):
                prepare(**args)

    def test_missing_rgb_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            (root / "data/rgb/Scene01/b.jpg").unlink()
            with self.assertRaises(FileNotFoundError):
                prepare(**args)
            self.assertFalse((root / "prepared").exists())


if __name__ == "__main__":
    unittest.main()
