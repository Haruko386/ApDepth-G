"""Portable coverage records, aggregation, and plots. No torch/dataset imports."""

import csv
import json
from pathlib import Path
import re

FORMAT_VERSION = 1
COUNT_KEYS = ("pixels", "valid_pixels", "data_invalid_pixels", "crop_excluded_pixels")
LATENT_KEYS = ("latent_cells", "latent_valid_cells", "latent_invalid_cells",
               "latent_mixed_cells", "latent_border_pixels")
DATASET_LABELS = {
    "hypersim": ("Hypersim", "Training"),
    "vkitti": ("Virtual KITTI 2", "Training"),
    "nyu_v2": ("NYUv2", "Test"),
    "kitti": ("KITTI", "Test"),
    "eth3d": ("ETH3D", "Test"),
    "scannet": ("ScanNet", "Test"),
    "diode": ("DIODE", "Test"),
}


def image_inventory(rows):
    """Count each dataset once, never once per optional semantic region."""
    datasets = []
    for row in rows:
        if row["region"] != "all":
            continue
        label, usage = DATASET_LABELS.get(row["dataset_type"], (row["dataset"], "Other"))
        datasets.append({"dataset": row["dataset"], "display_name": label, "usage": usage,
                         "images": row["images"]})
    return {"total_images": sum(row["images"] for row in datasets),
            "training_images": sum(row["images"] for row in datasets if row["usage"] == "Training"),
            "test_images": sum(row["images"] for row in datasets if row["usage"] == "Test"),
            "other_images": sum(row["images"] for row in datasets if row["usage"] == "Other"),
            "datasets": datasets}


def record_filename(dataset):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", dataset):
        raise ValueError(f"disp_name must be a safe filename (letters/digits/_.-): {dataset!r}")
    return f"{dataset}.jsonl"


def write_line(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def aggregate(rows):
    totals = {key: sum(row[key] for row in rows) for key in COUNT_KEYS}
    nonempty = [row for row in rows if row["pixels"]]
    result = {"images": len(rows), "nonempty_region_images": len(nonempty), **totals}
    for prefix, key in (("valid", "valid_pixels"), ("data_invalid", "data_invalid_pixels"),
                        ("crop_excluded", "crop_excluded_pixels")):
        result[f"{prefix}_percent_pixel"] = 100 * totals[key] / totals["pixels"] if totals["pixels"] else None
        result[f"{prefix}_percent_image"] = (100 * sum(row[key] / row["pixels"] for row in nonempty)
                                              / len(nonempty) if nonempty else None)
    return result


def aggregate_latent(records):
    totals = {key: sum(row[key] for row in records) for key in LATENT_KEYS}
    nonempty = [row for row in records if row["latent_cells"]]
    for prefix in ("latent_valid", "latent_invalid", "latent_mixed"):
        totals[f"{prefix}_percent_pixel"] = (100 * totals[f"{prefix}_cells"] / totals["latent_cells"]
                                               if totals["latent_cells"] else None)
        totals[f"{prefix}_percent_image"] = (100 * sum(row[f"{prefix}_cells"] / row["latent_cells"]
                                                       for row in nonempty) / len(nonempty) if nonempty else None)
    return {"images": len(records), **totals}


def nonnegative_int(record, key):
    value = record.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"{key} must be a nonnegative integer")
    return value


def read_record_file(path):
    """Validate a complete file and aggregate it using counts, not rounded rates."""
    path = Path(path)
    regions, latent_records = {}, []
    metadata = None
    completed = False
    last_index = -1
    region_names = None
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or completed:
                    raise ValueError("Unexpected record after completion or invalid JSON object")
                kind = record.get("kind")
                if metadata is None:
                    if kind != "metadata" or record.get("format_version") != FORMAT_VERSION:
                        raise ValueError("Missing metadata or unsupported format_version")
                    record_filename(record["dataset"])
                    if not isinstance(record.get("dataset_type"), str):
                        raise ValueError("Missing dataset_type")
                    if nonnegative_int(record, "selected_images") < 1:
                        raise ValueError("Empty dataset selection")
                    if nonnegative_int(record, "available_images") < record["selected_images"]:
                        raise ValueError("selected_images exceeds available_images")
                    if nonnegative_int(record, "latent_scale") < 1:
                        raise ValueError("Invalid latent_scale")
                    metadata = record
                elif kind == "image":
                    index = nonnegative_int(record, "index")
                    if index <= last_index or index >= metadata["available_images"]:
                        raise ValueError("Image indices must follow split order without duplicates")
                    last_index = index
                    if not isinstance(record.get("rgb"), str):
                        raise ValueError("Missing RGB split identifier")
                    height, width = nonnegative_int(record, "height"), nonnegative_int(record, "width")
                    if min(height, width) < 1:
                        raise ValueError("Invalid image shape")
                    rows = record["regions"]
                    names = [row["region"] for row in rows]
                    if names not in (["all"], ["all", "sky", "non_sky"]):
                        raise ValueError("Expected all, optionally followed by sky and non_sky")
                    if region_names is not None and region_names != names:
                        raise ValueError("Region annotations changed within one dataset")
                    region_names = names
                    for row in rows:
                        for key in COUNT_KEYS:
                            nonnegative_int(row, key)
                        if sum(row[key] for key in COUNT_KEYS[1:]) != row["pixels"]:
                            raise ValueError("Valid, invalid, and crop-excluded counts do not sum to pixels")
                        regions.setdefault(row["region"], []).append(row)
                    if rows[0]["pixels"] != height * width:
                        raise ValueError("Pixel count does not match image shape")
                    if len(rows) == 3 and any(rows[1][key] + rows[2][key] != rows[0][key] for key in COUNT_KEYS):
                        raise ValueError("Sky/non-sky counts do not partition the image")
                    latent = record["latent"]
                    for key in LATENT_KEYS:
                        nonnegative_int(latent, key)
                    if sum(latent[key] for key in LATENT_KEYS[1:4]) != latent["latent_cells"]:
                        raise ValueError("Latent block counts are inconsistent")
                    scale = metadata["latent_scale"]
                    if latent["latent_cells"] != (height // scale) * (width // scale):
                        raise ValueError("Latent count does not match shape and scale")
                    if latent["latent_cells"] * scale * scale + latent["latent_border_pixels"] != height * width:
                        raise ValueError("Latent border pixel count is inconsistent")
                    latent_records.append(latent)
                elif kind == "complete":
                    if nonnegative_int(record, "images") != len(latent_records):
                        raise ValueError("Completion count does not match image records")
                    if len(latent_records) != metadata["selected_images"]:
                        raise ValueError("Incomplete dataset selection")
                    completed = True
                else:
                    raise ValueError(f"Unexpected record kind: {kind}")
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid coverage record {path}:{number}: {error}") from error
    if not completed:
        raise ValueError(f"Incomplete coverage record (no completion marker): {path}")
    summaries = [{"dataset": metadata["dataset"], "dataset_type": metadata["dataset_type"],
                  "region": region, "available_images": metadata["available_images"], **aggregate(rows)}
                 for region, rows in regions.items()]
    latent = {"dataset": metadata["dataset"], "dataset_type": metadata["dataset_type"],
              **aggregate_latent(latent_records)}
    return metadata, summaries, latent


def collect_records(paths):
    """Preserve caller-specified file order; never double-count a dataset."""
    metadata, summaries, latent = [], [], []
    names, scales = set(), set()
    for path in paths:
        meta, rows, blocks = read_record_file(path)
        if meta["dataset"] in names:
            raise ValueError(f"Duplicate dataset record: {meta['dataset']}")
        names.add(meta["dataset"])
        scales.add(meta["latent_scale"])
        metadata.append(meta)
        summaries.extend(rows)
        latent.append(blocks)
    if len(scales) != 1:
        raise ValueError("All records must use the same latent_scale to compare blocks")
    return metadata, summaries, latent


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_charts(rows, latent_rows, destination, average):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def plot(group, columns, legends, colors, filename, ylabel, title, note):
        labels = []
        usages = []
        for row in group:
            name, usage = DATASET_LABELS.get(row.get("dataset_type"), (row["dataset"], "Other"))
            labels.append(f"{name}\nn = {row['images']:,}")
            usages.append(usage)
        fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(labels) + 2), 6.4), dpi=200)
        fig.subplots_adjust(left=.085, right=.985, top=.74, bottom=.29)
        fig.text(.085, .955, title, fontsize=15, fontweight="bold")
        count_note = "  |  ".join((f"Total: {sum(row['images'] for row in group):,} images",
                                    f"Training: {sum(row['images'] for row, usage in zip(group, usages) if usage == 'Training'):,}",
                                    f"Test: {sum(row['images'] for row, usage in zip(group, usages) if usage == 'Test'):,}"))
        fig.text(.085, .908, count_note, fontsize=10, color="#444444")
        start = 0
        while start < len(group):
            end = start + 1
            while end < len(group) and usages[end] == usages[start]:
                end += 1
            usage = usages[start]
            ax.axvspan(start - .5, end - .5, color="#EDF2F7" if usage == "Training" else "#F4F7F3", zorder=0)
            ax.text((start + end - 1) / 2, 1.025, f"{usage} datasets", transform=ax.get_xaxis_transform(),
                    ha="center", fontsize=10, fontweight="bold", color="#444444")
            if start:
                ax.axvline(start - .5, color="#AAAAAA", linestyle="--", linewidth=.8)
            start = end
        bottom = np.zeros(len(labels))
        for values, label, color in zip(columns, legends, colors):
            values = np.asarray(values, dtype=float)
            ax.bar(np.arange(len(labels)), values, bottom=bottom, color=color, label=label, width=.65)
            for index, value in enumerate(values):
                if value >= 4:
                    ax.text(index, bottom[index] + value / 2, f"{value:.1f}%", ha="center", va="center", fontsize=8)
            bottom += values
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=9, linespacing=1.7)
        ax.set_xlim(-.5, len(group) - .5)
        ax.set_ylim(0, 100)
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        ax.yaxis.grid(alpha=.25)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.13), ncol=3, frameon=False, fontsize=8.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        role_note = "Hypersim / Virtual KITTI 2: training data. NYUv2 / KITTI / ETH3D / ScanNet / DIODE: test data."
        fig.text(.085, .12, role_note, fontsize=8, color="#444444")
        fig.text(.085, .035, note, fontsize=8, color="#555555", linespacing=1.7)
        for extension in ("png", "pdf"):
            fig.savefig(destination / f"{filename}.{extension}", bbox_inches="tight")
        plt.close(fig)

    for region in ("all", "sky", "non_sky"):
        group = [row for row in rows if row["region"] == region and row["pixels"]]
        if not group:
            continue
        plot(group,
             [[row[f"{key}_percent_{average}"] for row in group]
              for key in ("valid", "data_invalid", "crop_excluded")],
             ("Valid support", "Missing / rejected depth", "Excluded by evaluation crop"),
             ("#4C78A8", "#E45756", "#BAB0AC"), f"coverage_{region}",
             "Pixel coverage (%)", "Depth supervision coverage" + (f" - {region.replace('_', ' ')}" if region != "all" else ""),
             f"Average: {average}. Denominator: image support after configured spatial crop/resize.\n"
             "Depth invalidity is not a semantic sky label. Evaluation masks measure evaluation support.")
    group = [row for row in latent_rows if row["latent_cells"]]
    if group:
        plot(group,
             [[row[f"{key}_percent_{average}"] for row in group]
              for key in ("latent_valid", "latent_invalid", "latent_mixed")],
             ("Fully valid blocks", "Fully invalid blocks", "Mixed blocks"),
             ("#4C78A8", "#E45756", "#F2A541"), "coverage_latent",
             "Latent-block support (%)", "Validity support of latent blocks",
             f"Average: {average}. Same block-mask rule as VGC; this is not the full VAE receptive field.\n"
             "Fully invalid blocks are candidate support, before VGC sample gates or DA2 far-field selection.")


def save_combined_chart(rows, latent_rows, destination, average, block_scale=8):
    """An additional 3.5-inch, single-column figure; leave original figures alone."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as effects

    pixels = {row["dataset"]: row for row in rows if row["region"] == "all"}
    group = [row for row in latent_rows if row["latent_cells"] and row["dataset"] in pixels]
    if not group:
        return
    labels, usages = [], []
    for row in group:
        name, usage = DATASET_LABELS.get(row.get("dataset_type"), (row["dataset"], "Other"))
        name = "VKITTI2" if row.get("dataset_type") == "vkitti" else name
        labels.append(f"{name}\nn={row['images']:,}")
        usages.append(usage)
    fig, ax = plt.subplots(figsize=(3.5, 2.55), dpi=600)
    fig.subplots_adjust(left=.14, right=.985, bottom=.20, top=.76)
    start = 0
    while start < len(group):
        end = start + 1
        while end < len(group) and usages[end] == usages[start]:
            end += 1
        ax.axvspan(start - .5, end - .5, color="#EDF2F7" if usages[start] == "Training" else "#F4F7F3", zorder=0)
        ax.text((start + end - 1) / 2, 1.035, "Train" if usages[start] == "Training" else usages[start],
                transform=ax.get_xaxis_transform(), ha="center", fontsize=7, fontweight="bold", color="#555555")
        if start:
            ax.axvline(start - .5, color="#999999", linestyle="--", linewidth=.6)
        start = end
    x = np.arange(len(group))
    bar_width = .23
    for offset, key, label, color in ((-bar_width, "latent_valid", "Fully valid blocks", "#4C78A8"),
                                      (0, "latent_invalid", "Fully invalid blocks", "#E45756"),
                                      (bar_width, "latent_mixed", "Mixed blocks", "#F2A541")):
        values = np.asarray([row[f"{key}_percent_{average}"] for row in group])
        ax.bar(x + offset, values, width=bar_width, color=color, edgecolor="white", linewidth=.25, label=label, zorder=2)
    coverage = [pixels[row["dataset"]][f"valid_percent_{average}"] for row in group]
    ax.plot(x, coverage, color="#202020", linewidth=1, marker="o", markersize=3,
            markerfacecolor="white", markeredgewidth=.8, label="Pixel coverage", zorder=5)
    for index, value in enumerate(coverage):
        text = ax.annotate(f"{value:.1f}", (index, value), xytext=(0, -9 if value > 94 else 5),
                           textcoords="offset points", ha="center", fontsize=5.5, color="#202020", zorder=6)
        text.set_path_effects([effects.withStroke(linewidth=1.5, foreground="white")])
    ax.set_xlim(-.5, len(group) - .5)
    ax.set_ylim(0, 104)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Share (%)", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.8, linespacing=1.7)
    ax.tick_params(axis="y", labelsize=6.5, width=.6, length=2.5)
    ax.tick_params(axis="x", width=.6, length=2.5)
    ax.set_axisbelow(True)
    ax.yaxis.grid(alpha=.25, linewidth=.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(.6)
    handles, legend_labels = ax.get_legend_handles_labels()
    order = [legend_labels.index(label) for label in ("Fully valid blocks", "Fully invalid blocks", "Mixed blocks", "Pixel coverage")]
    fig.legend([handles[i] for i in order], [legend_labels[i] for i in order],
               loc="center", bbox_to_anchor=(.55, .91), ncol=2, frameon=False, fontsize=6.2,
               handlelength=1.4, columnspacing=1.1, handletextpad=.5, labelspacing=.6)
    for extension in ("png", "pdf"):
        # Keep the requested physical width in the PDF for single-column use.
        fig.savefig(Path(destination) / f"coverage_combined_single_column.{extension}")
    plt.close(fig)
