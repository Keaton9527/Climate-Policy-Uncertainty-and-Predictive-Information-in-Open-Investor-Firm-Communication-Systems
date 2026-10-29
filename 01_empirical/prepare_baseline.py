#!/usr/bin/env python3
"""Optional: merge existing feature and sentiment panels to create baseline.parquet."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from nev_forecasting.baseline import build_baseline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--sentiment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(build_baseline(args.features, args.sentiment, args.output))
