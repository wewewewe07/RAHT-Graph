"""
run_pipeline.py
===============
End-to-end pipeline runner for RAHT-Graph.

Steps:
  1. Fetch raw OHLCV data from Yahoo Finance
  2. Feature engineering + market regime detection
  3. Build heterogeneous graph (HeteroData)
  4. Train model with walk-forward validation

Usage:
  python run_pipeline.py              # Full pipeline
  python run_pipeline.py --skip-data  # Skip data fetch (use cached)
  python run_pipeline.py --train-only # Skip to training
  python run_pipeline.py --tune       # Hyperparameter tuning instead of training
"""
import sys
import os
import argparse


def run_step(script_path, desc):
    print(f"\n{'='*60}")
    print(f">>> {desc}")
    print(f"{'='*60}")
    ret = os.system(f"python {script_path}")
    if ret != 0:
        print(f"\n[ERROR] {script_path} failed with exit code {ret}. Stopping.")
        sys.exit(ret)


def main():
    parser = argparse.ArgumentParser(description="RAHT-Graph Pipeline Runner")
    parser.add_argument("--skip-data",  action="store_true",
                        help="Skip data fetching (use cached raw data)")
    parser.add_argument("--train-only", action="store_true",
                        help="Skip to training (use cached graph)")
    parser.add_argument("--tune",       action="store_true",
                        help="Run hyperparameter tuning instead of training")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))

    if not args.train_only:
        if not args.skip_data:
            run_step(os.path.join(base, "data", "01_fetch_data.py"),
                     "Step 1: Fetch raw market data")
        run_step(os.path.join(base, "data", "02_feature_engineering.py"),
                 "Step 2: Feature engineering + regime detection")
        run_step(os.path.join(base, "data", "03_build_graph.py"),
                 "Step 3: Build heterogeneous graph")

    if args.tune:
        run_step(os.path.join(base, "raht", "tune.py"),
                 "Step 4: Hyperparameter tuning (Optuna)")
    else:
        run_step(os.path.join(base, "raht", "train.py"),
                 "Step 4: Walk-forward training")

    print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
