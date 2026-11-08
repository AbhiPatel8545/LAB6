import argparse
import csv
import json
import os
import platform
import random
import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt

import lab06_starter as S
from answer_only_loss import answer_only_labels
from lora_layer import replace_qv_with_lora, LoRALinear, expected_lora_parameters
from verify_lora import run_checks

OUT = Path("results")
OUT.mkdir(exist_ok=True)

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_custom(model, tok, train_examples, steps, lr, batch_size, seed, labels_fn):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    if S.DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    losses = []
    start = time.time()
    for step in range(1, steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr * min(1.0, step / 10)
        batch = rng.sample(train_examples, batch_size)
        ids, mask, prompt_lens = S.pad_batch(batch, tok.pad_token_id)
        labels = labels_fn(ids, mask, prompt_lens)
        output = model(input_ids=ids.to(S.DEVICE),
                       attention_mask=mask.to(S.DEVICE),
                       labels=labels.to(S.DEVICE))
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        losses.append(float(output.loss.item()))
    elapsed = time.time() - start
    weights, grads, opt_state = S.training_state_bytes(model, optimizer)
    trainable, total = S.count_params(model)
    peak = torch.cuda.max_memory_allocated() if S.DEVICE == "cuda" else None
    return {
        "first_loss": sum(losses[:5]) / min(5, len(losses)),
        "last_loss": sum(losses[-5:]) / min(5, len(losses)),
        "seconds": elapsed, "seconds_per_step": elapsed / steps,
        "trainable": trainable, "total": total,
        "weight_bytes": weights, "grad_bytes": grads,
        "optimizer_bytes": opt_state, "peak_cuda_bytes": peak
    }

def predict_memory(total_params, trainable_params):
    return {
        "pred_weight_bytes": 4 * total_params,
        "pred_grad_bytes": 4 * trainable_params,
        "pred_optimizer_bytes": 8 * trainable_params,
        "pred_total_bytes": 4 * total_params + 12 * trainable_params
    }

def save_csv(rows, path):
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def main(args):
    print("Device:", S.DEVICE)
    print("PyTorch:", torch.__version__)
    print("Platform:", platform.platform())
    run_checks()

    model0, tok = S.load_model()
    train_data, test_data, label_ids = S.load_task(
        "emotion", n_train=args.n_train, n_test=args.n_test,
        data_seed=args.data_seed, tokenizer=tok
    )
    zero = S.evaluate(model0, tok, test_data, label_ids)
    print(f"Zero-shot accuracy: {zero:.4f}; chance: {1/6:.4f}")

    ids, mask, lens = S.pad_batch(train_data[:args.batch_size], tok.pad_token_id)
    labels = answer_only_labels(ids, mask, lens)
    assert torch.all((labels != -100).sum(dim=1) == 1)
    assert torch.all(labels[torch.arange(len(lens)), lens] == ids[torch.arange(len(lens)), lens])
    print("PASS: exactly one answer token is scored per example.")

    count_rows = []
    for rank in (1, 4, 16):
        model, _ = S.load_model()
        paths = replace_qv_with_lora(model, rank, alpha=2 * rank)
        trainable, total = S.count_params(model)
        formula = sum(rank * (m.base.in_features + m.base.out_features)
                      for m in model.modules() if isinstance(m, LoRALinear))
        assert trainable == formula
        count_rows.append({
            "rank": rank, "targeted_layers": len(paths),
            "trainable_parameters": trainable, "total_parameters": total,
            "trainable_percent": 100 * trainable / total,
            "formula_parameters": formula
        })
        del model
    save_csv(count_rows, OUT / "lora_parameter_counts.csv")

    rows = []
    methods = [("full", None), ("lora", 1), ("lora", 4), ("lora", 16)]
    for mode, rank in methods:
        for seed in (0, 1, 2):
            seed_everything(seed)
            model, tokenizer = S.load_model()
            if mode == "lora":
                replace_qv_with_lora(model, rank, alpha=2 * rank)
                lr = args.lora_lr
            else:
                lr = args.full_lr
            measured = train_custom(
                model, tokenizer, train_data, args.steps, lr,
                args.batch_size, seed, answer_only_labels
            )
            accuracy = S.evaluate(model, tokenizer, test_data, label_ids)
            row = {
                "mode": mode, "rank": rank if rank is not None else "full",
                "seed": seed, "loss": "answer_only", "learning_rate": lr,
                "accuracy": accuracy, "zero_shot_accuracy": zero,
                "chance_accuracy": 1/6, **measured
            }
            row.update(predict_memory(measured["total"], measured["trainable"]))
            rows.append(row)
            print("RUN:", row["mode"], row["rank"], "seed", seed,
                  "accuracy", round(accuracy, 4))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for seed in (0, 1, 2):
        seed_everything(seed)
        model, tokenizer = S.load_model()
        replace_qv_with_lora(model, 16, alpha=32)
        measured = train_custom(
            model, tokenizer, train_data, args.steps, args.lora_lr,
            args.batch_size, seed, S.all_tokens_labels
        )
        accuracy = S.evaluate(model, tokenizer, test_data, label_ids)
        row = {
            "mode": "lora", "rank": 16, "seed": seed,
            "loss": "all_tokens", "learning_rate": args.lora_lr,
            "accuracy": accuracy, "zero_shot_accuracy": zero,
            "chance_accuracy": 1/6, **measured
        }
        row.update(predict_memory(measured["total"], measured["trainable"]))
        rows.append(row)
        print("RUN: lora 16 all_tokens seed", seed, "accuracy", round(accuracy, 4))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_csv(rows, OUT / "experiment_results.csv")
    with open(OUT / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "device": S.DEVICE, "torch_version": torch.__version__,
            "platform": platform.platform(), "zero_shot_accuracy": zero,
            "chance_accuracy": 1/6, "n_train": args.n_train,
            "n_test": args.n_test, "steps": args.steps,
            "batch_size": args.batch_size, "data_seed": args.data_seed,
            "full_learning_rate": args.full_lr, "lora_learning_rate": args.lora_lr,
            "seeds": [0, 1, 2]
        }, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        if row["loss"] != "answer_only":
            continue
        label = "Full fine-tuning" if row["mode"] == "full" else f"LoRA r={row['rank']}"
        ax.scatter(row["trainable"], row["accuracy"], label=label)
    ax.axhline(1/6, linestyle="--", label="Chance (1/6)")
    ax.axhline(zero, linestyle=":", label="Zero-shot")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("AIGC 5500 Lab 06: Accuracy vs Trainable Parameters")
    ax.grid(True, alpha=0.25)
    handles, labels_text = ax.get_legend_handles_labels()
    unique = dict(zip(labels_text, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_trainable_parameters.png", dpi=180)
    plt.close(fig)
    print("Completed. Results and figure saved in ./results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=600)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--full-lr", type=float, default=5e-5)
    parser.add_argument("--lora-lr", type=float, default=1e-3)
    main(parser.parse_args())
