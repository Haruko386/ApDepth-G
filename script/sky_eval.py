"""SCOE command-line entry point; no model loading or network access."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU offline Sky-Depth Consistency and Ordering Evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "prepare", "evaluate", "report"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--run-id", help="New run ID for audit; same ID for following stages")
        command.add_argument("--dry-run", action="store_true", help="Validate configuration and print planned output; write nothing")
        if name == "audit":
            command.add_argument("--limit-samples", type=int, help="Maximum inspected images per dataset; diagnostic only")
        if name == "evaluate":
            command.add_argument("--model", required=True)
        if name == "report":
            command.add_argument("--formal", action="store_true", help="Require all data, leakage and representation gates")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        from src.sky_eval.workflow import audit, evaluate, load_config, prepare, report
        config, protocol, run = load_config(args.config, args.run_id)
        limit = getattr(args, "limit_samples", None)
        if limit is not None and limit < 1:
            raise ValueError("--limit-samples must be positive")
        if args.dry_run:
            logging.info("Dry run: %s -> %s; no data read, no files written", args.command, run)
            return 0
        if args.command == "audit":
            return audit(config, protocol, run, limit)
        if args.command == "prepare":
            return prepare(config, protocol, run)
        if args.command == "evaluate":
            return evaluate(config, protocol, run, args.model)
        return report(config, protocol, run, args.formal)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        logging.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
