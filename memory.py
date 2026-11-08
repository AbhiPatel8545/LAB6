import csv
from pathlib import Path

def predicted_memory(total_parameters, trainable_parameters):
    return {
        "weights_bytes": 4 * total_parameters,
        "gradients_bytes": 4 * trainable_parameters,
        "adam_state_bytes": 8 * trainable_parameters,
        "total_bytes": 4 * total_parameters + 12 * trainable_parameters
    }

def compare_results(csv_path="results/experiment_results.csv"):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        pred = predicted_memory(int(row["total"]), int(row["trainable"]))
        print(row["mode"], row["rank"], row["loss"], "seed", row["seed"])
        for key, value in pred.items():
            print(f"  predicted {key}: {value:,}")
        for key in ("weight_bytes", "grad_bytes", "optimizer_bytes", "peak_cuda_bytes"):
            print(f"  measured {key}: {row.get(key)}")

if __name__ == "__main__":
    compare_results()
