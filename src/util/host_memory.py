"""Best-effort Linux host/container RAM diagnostics, without extra dependencies."""

from pathlib import Path


def host_memory_snapshot(proc_root="/proc", cgroup_root="/sys/fs/cgroup"):
    record = {}
    proc_root, cgroup_root = Path(proc_root), Path(cgroup_root)
    for filename, fields in (
        ("self/status", {"VmRSS": "process_rss_gib", "VmHWM": "process_peak_rss_gib"}),
        ("meminfo", {"MemAvailable": "host_available_gib", "MemTotal": "host_total_gib",
                     "SwapFree": "host_swap_free_gib"}),
    ):
        try:
            for line in (proc_root / filename).read_text().splitlines():
                key, _, value = line.partition(":")
                if key in fields:
                    record[fields[key]] = int(value.split()[0]) / 1024 ** 2
        except (OSError, ValueError, IndexError):
            pass
    # Docker's cgroup namespace normally exposes the container at this root.
    # v1 fallback is useful on older training hosts. "max" means no v2 limit.
    for candidates, key in (
        (("memory.current", "memory/memory.usage_in_bytes"), "container_used_gib"),
        (("memory.max", "memory/memory.limit_in_bytes"), "container_limit_gib"),
    ):
        for filename in candidates:
            try:
                value = (cgroup_root / filename).read_text().strip()
                if value != "max" and int(value) < 2 ** 60:
                    record[key] = int(value) / 1024 ** 3
                break
            except (OSError, ValueError):
                continue
    return record
