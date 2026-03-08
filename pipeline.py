"""
pipeline.py — Single entry point for the full FMCG supply-chain pipeline.

Run:
    python pipeline.py                   # full pipeline
    python pipeline.py --from stage2     # skip XGBoost, load checkpoint
    python pipeline.py --from stage3     # skip both, load inventory checkpoint
    python pipeline.py --no-checkpoints  # run without saving intermediate files

Stages:
    Stage 1 — XGBoost sales forecasting
    Stage 2 — Inventory transfer optimisation
    Stage 3 — Route optimisation
"""

import argparse
import os
import sys
import time
import pandas as pd

import config
import stage1_xgboost  as s1
import stage2_inventory as s2
import stage3_routes    as s3


# ── Helpers ────────────────────────────────────────────────────────────────

def _ensure_output_dir():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def _banner(title: str):
    print("\n" + "█" * 70)
    print(f"  {title}")
    print("█" * 70)


def _elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"


# ── Stage runners ──────────────────────────────────────────────────────────

def run_stage1(save_checkpoint: bool) -> pd.DataFrame:
    t = time.time()
    if not os.path.exists(config.INPUT_CSV):
        sys.exit(f"❌ Input CSV not found: {config.INPUT_CSV}\n"
                 f"   Update INPUT_CSV in config.py")

    raw = pd.read_csv(config.INPUT_CSV)
    print(f"   Loaded {len(raw):,} rows from {os.path.basename(config.INPUT_CSV)}")

    matrix = s1.run(raw)

    if save_checkpoint and config.CHECKPOINT_XGBOOST:
        s1.save_checkpoint(matrix, config.CHECKPOINT_XGBOOST)

    print(f"   ⏱  Stage 1 took {_elapsed(t)}")
    return matrix


def run_stage2(matrix: pd.DataFrame, save_checkpoint: bool) -> pd.DataFrame:
    t = time.time()
    transfers = s2.run(matrix)

    if save_checkpoint and config.CHECKPOINT_INVENTORY:
        s2.save_checkpoint(transfers, config.CHECKPOINT_INVENTORY)

    print(f"   ⏱  Stage 2 took {_elapsed(t)}")
    return transfers


def run_stage3(transfers: pd.DataFrame, save_output: bool) -> pd.DataFrame:
    t = time.time()
    routes_df, report = s3.run(transfers)

    if save_output:
        s3.save_checkpoint(routes_df, report, config.FINAL_OUTPUT)

    print(f"   ⏱  Stage 3 took {_elapsed(t)}")
    return routes_df


# ── Load from checkpoint ───────────────────────────────────────────────────

def _load_xgboost_checkpoint() -> pd.DataFrame:
    path = config.CHECKPOINT_XGBOOST
    if not path or not os.path.exists(path):
        sys.exit(f"❌ XGBoost checkpoint not found: {path}\n"
                 f"   Run without --from flag first to generate it.")
    print(f"   📂 Loading XGBoost checkpoint: {path}")
    return pd.read_excel(path)


def _load_inventory_checkpoint() -> pd.DataFrame:
    path = config.CHECKPOINT_INVENTORY
    if not path or not os.path.exists(path):
        sys.exit(f"❌ Inventory checkpoint not found: {path}\n"
                 f"   Run from stage2 or without --from flag first.")
    print(f"   📂 Loading inventory checkpoint: {path}")
    return pd.read_excel(path, sheet_name="Detailed_Transfers")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FMCG Supply-Chain Pipeline")
    parser.add_argument(
        "--from",
        dest="start_stage",
        choices=["stage1", "stage2", "stage3"],
        default="stage1",
        help="Start pipeline from a specific stage (uses saved checkpoints for earlier stages)",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Run pipeline without saving intermediate checkpoint files",
    )
    args = parser.parse_args()

    save_checkpoints = not args.no_checkpoints
    start            = args.start_stage

    _ensure_output_dir()
    total_start = time.time()

    _banner("FMCG SUPPLY-CHAIN PIPELINE")
    print(f"   Start stage  : {start}")
    print(f"   Checkpoints  : {'enabled' if save_checkpoints else 'disabled'}")
    print(f"   Output       : {config.FINAL_OUTPUT}")

    # ── Stage 1 ──────────────────────────────────────────────────────────
    if start == "stage1":
        matrix = run_stage1(save_checkpoints)
    else:
        matrix = _load_xgboost_checkpoint()

    # ── Stage 2 ──────────────────────────────────────────────────────────
    if start in ("stage1", "stage2"):
        transfers = run_stage2(matrix, save_checkpoints)
    else:
        transfers = _load_inventory_checkpoint()

    # ── Stage 3 ──────────────────────────────────────────────────────────
    routes_df = run_stage3(transfers, save_output=True)

    # ── Done ──────────────────────────────────────────────────────────────
    _banner("PIPELINE COMPLETE")
    print(f"   Total time   : {_elapsed(total_start)}")
    print(f"   Final output : {config.FINAL_OUTPUT}")
    print(f"   Routes       : {len(routes_df)}")
    print()


if __name__ == "__main__":
    main()
