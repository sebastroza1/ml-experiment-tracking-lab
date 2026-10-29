from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_data
from .tracking import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local MLflow experiment search.")
    parser.add_argument("--config", default="configs/quick.json")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output-dir", default="reports/quick-search")
    parser.add_argument("--database", default="mlflow.db")
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as stream:
        config = json.load(stream)
    splits, source = load_data(args.cache_dir, int(config["sample_size"]), int(config["seed"]))
    result = run_experiments(splits, config, args.output_dir, args.database, source)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
