#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="LightGBM benchmark (t3.micro-friendly)")
    parser.add_argument("--data", default="creditcard.csv", help="Path to creditcard.csv")
    parser.add_argument("--sample-size", type=int, default=50000, help="Rows sampled for quick benchmark")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--num-leaves", type=int, default=31, help="LightGBM num_leaves")
    parser.add_argument("--n-estimators", type=int, default=120, help="LightGBM number of trees")
    parser.add_argument(
        "--result-path", default="benchmark_result.json", help="Output metrics JSON file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Download creditcard.csv first."
        )

    t0 = time.perf_counter()
    df = pd.read_csv(data_path)
    load_data_seconds = time.perf_counter() - t0
    total_rows_loaded = len(df)

    # t3.micro-friendly: sample to keep memory and runtime manageable.
    sample_size = min(args.sample_size, len(df))
    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=args.random_state)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        random_state=args.random_state,
        n_jobs=1,  # keep CPU pressure low on t3.micro
    )

    t1 = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - t1

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc_roc = float(roc_auc_score(y_test, y_proba))
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))

    one_row = X_test.iloc[[0]]
    t2 = time.perf_counter()
    _ = model.predict_proba(one_row)
    inference_latency_ms = (time.perf_counter() - t2) * 1000

    batch = X_test.iloc[:1000] if len(X_test) >= 1000 else X_test
    t3 = time.perf_counter()
    _ = model.predict_proba(batch)
    batch_seconds = time.perf_counter() - t3
    throughput_rows_per_sec = float(len(batch) / batch_seconds) if batch_seconds > 0 else 0.0

    best_iteration = (
        int(model.best_iteration_) if getattr(model, "best_iteration_", None) else args.n_estimators
    )

    result = {
        "mode": "t3_micro_demo",
        "dataset_path": str(data_path),
        "rows_total_loaded": int(total_rows_loaded),
        "rows_used_for_benchmark": int(sample_size),
        "load_data_seconds": round(load_data_seconds, 4),
        "training_seconds": round(training_seconds, 4),
        "best_iteration": best_iteration,
        "auc_roc": round(auc_roc, 6),
        "accuracy": round(accuracy, 6),
        "f1_score": round(f1, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "inference_latency_ms_1_row": round(inference_latency_ms, 4),
        "inference_throughput_rows_per_sec_batch": round(throughput_rows_per_sec, 2),
        "batch_size_for_throughput": int(len(batch)),
        "n_estimators": args.n_estimators,
        "num_leaves": args.num_leaves,
        "note": "Demo benchmark for constrained instance; not equivalent to r5.2xlarge full benchmark.",
    }

    output_path = Path(args.result_path)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nSaved metrics to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
