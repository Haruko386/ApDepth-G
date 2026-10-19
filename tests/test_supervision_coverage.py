"""Known-count fixtures exercise actual dataset decoding, masks, and the CLI."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from src.util.coverage_records import collect_records, read_record_file
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("supervision_coverage", ROOT / "script/supervision_coverage.py")
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


def make_config(root, kind, depth, *, name=None, extras=None):
    folder = root / (name or kind)
    folder.mkdir()
    if kind == "diode":
        np.save(folder / "depth.npy", depth.astype(np.float32))
        mask = np.zeros(depth.shape, dtype=np.float32)
        mask[:, ::2] = 1
        np.save(folder / "mask.npy", mask)
        entry = "missing_rgb.png depth.npy mask.npy\n"
    elif kind == "eth3d":
        depth.astype(np.float32).tofile(folder / "depth.bin")
        entry = "missing_rgb.png depth.bin\n"
    else:
        Image.fromarray(depth.astype(np.uint16)).save(folder / "depth.png")
        entry = "missing_rgb.png depth.png depth.png\n"
    files = folder / "files.txt"
    files.write_text(entry, encoding="utf-8")
    config = {"name": kind, "disp_name": name or kind, "dir": str(folder), "filenames": str(files)}
    if kind in ("vkitti", "kitti"):
        config.update(kitti_bm_crop=False, valid_mask_crop=None)
    if kind == "nyu_v2":
        config["eigen_valid_mask"] = False
    config.update(extras or {})
    return OmegaConf.create(config)


def get_counts(config, root, scale=8):
    job = coverage.CoverageDataset(config, root, [0])
    return coverage.count_regions(job[0], torch.device("cpu"), scale)


@pytest.mark.parametrize("kind,values,expected", [
    ("hypersim", [0, 1000, 65000, 65535], 16),  # strict 65 m boundary
    ("vkitti", [0, 100, 8000, 65535], 16),       # strict 80 m boundary
    ("kitti", [0, 256, 20480, 2560], 32),       # meters = PNG / 256
    ("nyu_v2", [0, 1000, 10000, 9000], 32),
    ("scannet", [0, 1000, 10000, 9000], 32),
    ("diode", [500, 500, 500, 500], 32),        # provided mask, not nominal 350 m bound
])
def test_units_and_boundaries_without_rgb(tmp_path, kind, values, expected):
    depth = np.tile(values, (8, 2))
    config = make_config(tmp_path, kind, depth)
    rows, latent = get_counts(config, tmp_path)
    assert rows[0]["pixels"] == 64
    assert rows[0]["valid_pixels"] == expected
    assert rows[0]["data_invalid_pixels"] == 64 - expected
    assert rows[0]["crop_excluded_pixels"] == 0
    assert latent["latent_cells"] == 1
    assert latent["latent_mixed_cells"] == 1


def test_eth3d_binary_decode(tmp_path, monkeypatch):
    from src.dataset.eth3d_dataset import ETH3DDataset
    monkeypatch.setattr(ETH3DDataset, "HEIGHT", 8)
    monkeypatch.setattr(ETH3DDataset, "WIDTH", 8)
    config = make_config(tmp_path, "eth3d", np.tile([0, np.inf, 1, 2], (8, 2)))
    rows, _ = get_counts(config, tmp_path)
    assert rows[0]["valid_pixels"] == 32


def test_spatial_crop_and_eval_exclusion_are_distinct(tmp_path):
    # Only the exact benchmark crop has valid data: areas outside must not enter denominator.
    depth = np.full((360, 1220), 9000, dtype=np.uint16)
    depth[8:360, 2:1218] = 100
    config = make_config(tmp_path, "vkitti", depth, extras={"kitti_bm_crop": True})
    rows, _ = get_counts(config, tmp_path)
    assert rows[0]["pixels"] == 352 * 1216
    assert rows[0]["valid_pixels"] == 352 * 1216
    config.valid_mask_crop = "eigen"
    rows, _ = get_counts(config, tmp_path)
    expected = (int(.91351351 * 352) - int(.3324324 * 352)) * (int(.96405229 * 1216) - int(.0359477 * 1216))
    assert rows[0]["valid_pixels"] == expected
    assert rows[0]["crop_excluded_pixels"] == 352 * 1216 - expected
    assert rows[0]["data_invalid_pixels"] == 0


def test_nyu_eval_crop(tmp_path):
    config = make_config(tmp_path, "nyu_v2", np.full((480, 640), 1000), extras={"eigen_valid_mask": True})
    rows, _ = get_counts(config, tmp_path)
    assert rows[0]["valid_pixels"] == 426 * 560
    assert rows[0]["crop_excluded_pixels"] == 480 * 640 - 426 * 560


def test_tar_reader_reuses_depth_decoding(tmp_path):
    config = make_config(tmp_path, "hypersim", np.full((8, 8), 1000))
    archive = tmp_path / "hypersim.tar"
    with tarfile.open(archive, "w") as handle:
        handle.add(Path(config.dir) / "depth.png", arcname="./depth.png")
    config.dir = str(archive)
    rows, latent = get_counts(config, tmp_path)
    assert rows[0]["valid_pixels"] == 64
    assert latent["latent_valid_cells"] == 1


def test_independent_sky_masks_resize_and_empty_regions(tmp_path):
    depth = np.full((8, 8), 1000)
    depth[:, 2:4] = 0
    config = make_config(tmp_path, "hypersim", depth, extras={"resize_to_hw": [4, 4]})
    sky = np.zeros((8, 8), dtype=np.uint8)
    sky[:, :4] = 255
    Image.fromarray(sky).save(tmp_path / "sky.png")
    listing = tmp_path / "sky.txt"
    listing.write_text("missing_rgb.png sky.png\n", encoding="utf-8")
    config.coverage = {"sky_mask_list": str(listing)}
    rows, latent = get_counts(config, tmp_path)
    rows = {row["region"]: row for row in rows}
    assert rows["all"]["pixels"] == 16
    assert rows["sky"]["pixels"] == 8
    assert rows["sky"]["valid_pixels"] == 4
    assert rows["non_sky"]["valid_pixels"] == 8
    assert latent["latent_cells"] == 0
    assert latent["latent_border_pixels"] == 16
    empty = coverage.aggregate([dict(region="sky", pixels=0, valid_pixels=0,
                                    data_invalid_pixels=0, crop_excluded_pixels=0)])
    assert empty["valid_percent_pixel"] is None
    assert empty["valid_percent_image"] is None


def test_latent_three_states_and_aggregation():
    valid = torch.zeros((8, 24), dtype=torch.bool)
    valid[:, :8] = True
    valid[0, 16] = True
    _, latent = coverage.count_regions({"valid": valid, "data_valid": valid}, torch.device("cpu"), 8)
    assert [latent[key] for key in ("latent_valid_cells", "latent_invalid_cells", "latent_mixed_cells")] == [1, 1, 1]
    # Unequal image sizes deliberately distinguish pixel-weighted and image-weighted means.
    result = coverage.aggregate([dict(pixels=10, valid_pixels=10, data_invalid_pixels=0, crop_excluded_pixels=0),
                                 dict(pixels=90, valid_pixels=0, data_invalid_pixels=90, crop_excluded_pixels=0)])
    assert result["valid_percent_pixel"] == 10
    assert result["valid_percent_image"] == 50


def test_fail_missing_sky_label_and_bad_diode_mask(tmp_path):
    config = make_config(tmp_path, "hypersim", np.ones((8, 8)) * 1000)
    listing = tmp_path / "sky.txt"
    listing.write_text("", encoding="utf-8")
    config.coverage = {"sky_mask_list": str(listing)}
    with pytest.raises(ValueError, match="Missing 1 sky-mask"):
        coverage.CoverageDataset(config, tmp_path, [0])
    diode = make_config(tmp_path, "diode", np.zeros((8, 8)))
    with pytest.raises(ValueError, match="nonfinite/nonpositive"):
        get_counts(diode, tmp_path)


def test_existing_mixed_config_and_cycle_detection(tmp_path):
    configs = coverage.dataset_configs(coverage.load_config("config/dataset/dataset_train.yaml"))
    assert [str(item.name) for item in configs] == ["hypersim", "vkitti"]
    assert list(configs[0].resize_to_hw) == [480, 640]
    path = tmp_path / "cyclic.yaml"
    OmegaConf.save({"base_config": [str(path)]}, path)
    with pytest.raises(ValueError, match="Circular"):
        coverage.load_config(path)


def test_two_servers_seven_records_then_plot_without_datasets(tmp_path, monkeypatch):
    from src.dataset.eth3d_dataset import ETH3DDataset
    monkeypatch.setattr(ETH3DDataset, "HEIGHT", 8)
    monkeypatch.setattr(ETH3DDataset, "WIDTH", 8)
    kinds = ["hypersim", "vkitti", "nyu_v2", "kitti", "eth3d", "scannet", "diode"]
    configs = [make_config(tmp_path, kind, np.full((8, 8), 1000)) for kind in kinds]
    training = tmp_path / "training.yaml"
    OmegaConf.save({"dataset": {"train": {"name": "mixed", "dataset_list": configs[:2]}}}, training)
    coverage.main(["--dataset_config", str(training), "--num_workers", "0",
                   "--output_dir", str(tmp_path / "training_records")])
    eval_paths = []
    for config in configs[2:]:
        path = tmp_path / f"{config.name}.yaml"
        OmegaConf.save(config, path)
        eval_paths.append(str(path))
    coverage.main(["--dataset_config", *eval_paths, "--num_workers", "0",
                   "--output_dir", str(tmp_path / "evaluation_records")])
    paths = ([tmp_path / "training_records" / f"{name}.jsonl" for name in kinds[:2]]
             + [tmp_path / "evaluation_records" / f"{name}.jsonl" for name in kinds[2:]])
    for path in paths:
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [line["kind"] for line in lines] == ["metadata", "image", "complete"]
        assert lines[1]["index"] == 0
        assert lines[1]["regions"][0]["valid_percent"] == (50 if path.stem == "diode" else 100)
    assert not list((tmp_path / "training_records").glob("*.png"))
    assert len(list((tmp_path / "training_records").iterdir())) == 2
    assert len(list((tmp_path / "evaluation_records").iterdir())) == 5
    # Make all original dataset and filename-list paths unavailable before plotting.
    for config in configs:
        folder = Path(config.dir)
        folder.rename(folder.with_name(folder.name + "_offline"))
    output = tmp_path / "local_plot"
    plot_args = [str(ROOT / "script/plot_supervision_coverage.py"), "--records",
                 *(str(path) for path in reversed(paths)), "--output_dir", str(output)]
    # Block model/dataset dependencies in the local plotting process.
    runner = """import builtins, runpy, sys
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision', 'omegaconf'} or name.startswith('src.dataset'):
        raise AssertionError('Plot tried to import a server dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ARGS
runpy.run_path(sys.argv[0], run_name='__main__')
""".replace("ARGS", repr(plot_args))
    result = subprocess.run([sys.executable, "-c", runner], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert [row["dataset"] for row in report["summary"]] == list(reversed(kinds))
    assert report["summary"][0]["valid_percent_pixel"] == 50
    assert (output / "coverage_all.png").is_file()
    assert (output / "coverage_all.pdf").is_file()
    assert (output / "coverage_latent.png").is_file()
    assert not (output / "coverage_sky.png").exists()


def test_record_order_completion_validation_and_duplicates(tmp_path):
    config = make_config(tmp_path, "hypersim", np.full((8, 8), 1000))
    Path(config.filenames).write_text("".join(f"rgb_{i}.png depth.png\n" for i in range(5)), encoding="utf-8")
    config_path = tmp_path / "dataset.yaml"
    OmegaConf.save(config, config_path)
    coverage.main(["--dataset_config", str(config_path), "--num_workers", "0", "--max_samples", "3",
                   "--output_dir", str(tmp_path / "records")])
    path = tmp_path / "records/hypersim.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    expected = sorted(np.random.default_rng(2024).choice(list(range(5)), 3, replace=False).tolist())
    assert [row["index"] for row in records if row["kind"] == "image"] == expected
    meta, summary, latent = read_record_file(path)
    assert meta["available_images"] == 5
    assert summary[0]["images"] == 3
    assert latent["latent_valid_percent_pixel"] == 100
    with pytest.raises(ValueError, match="Duplicate dataset"):
        collect_records([path, path])
    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text("\n".join(json.dumps(row) for row in records[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete"):
        read_record_file(incomplete)
    records[1]["regions"][0]["valid_pixels"] = 1
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    with pytest.raises(ValueError, match="do not sum"):
        read_record_file(corrupt)


def test_finished_dataset_survives_later_dataset_failure(tmp_path):
    first = make_config(tmp_path, "hypersim", np.full((8, 8), 1000))
    second = make_config(tmp_path, "vkitti", np.full((8, 8), 100))
    (Path(second.dir) / "depth.png").rename(Path(second.dir) / "depth_offline.png")
    configs = tmp_path / "datasets.yaml"
    OmegaConf.save([first, second], configs)
    with pytest.raises(FileNotFoundError):
        coverage.main(["--dataset_config", str(configs), "--num_workers", "0",
                       "--output_dir", str(tmp_path / "records")])
    finished = tmp_path / "records/hypersim.jsonl"
    assert read_record_file(finished)[1][0]["valid_pixels"] == 64
    assert (tmp_path / "records/vkitti.jsonl.partial").exists()
    assert not (tmp_path / "records/vkitti.jsonl").exists()


def test_cli_multiprocessing_recording(tmp_path):
    config = make_config(tmp_path, "hypersim", np.full((8, 8), 1000))
    path = tmp_path / "dataset.yaml"
    OmegaConf.save(config, path)
    result = subprocess.run([sys.executable, str(ROOT / "script/supervision_coverage.py"),
                             "--dataset_config", str(path), "--device", "cpu", "--num_workers", "2",
                             "--output_dir", str(tmp_path / "workers")],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert read_record_file(tmp_path / "workers/hypersim.jsonl")[1][0]["valid_percent_pixel"] == 100
