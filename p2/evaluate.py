"""
evaluate.py
-----------
Task-2 : Quantitative Evaluation
Task-3 : Qualitative Analysis

Loads saved checkpoints and for each model computes:
  Novelty Rate  = % of generated names NOT in the training set
  Diversity     = unique generated names / total generated names

Configurations are defined directly in the script.
"""

import os
import time

import torch

from logger               import init_logging, get_logger, get_log_path
from data_utils           import load_names, Vocabulary
from model_rnn            import VanillaRNN
from model_blstm          import BidirectionalLSTM
from model_attention_rnn  import AttentionRNN
import model_rnn           as rnn_mod
import model_blstm         as blstm_mod
import model_attention_rnn as attn_mod


# ═══════════════════════════════════════════════════════════
#  Evaluation Configuration
# ═══════════════════════════════════════════════════════════

CONFIG = {
    "models_to_eval":    ["rnn", "blstm", "attention"], 
    "checkpoint_dir":    "checkpoints",                 
    "data_path":         "TrainingNames.txt",           
    "n_generate":        200,                          
    "temperature":       1.0,                          
    "n_samples_show":    15,                           
    "output_names_file": "generated_names.txt",        
    "log_dir":           "logs"
}


# ═══════════════════════════════════════════════════════════
#  Load model from checkpoint
# ═══════════════════════════════════════════════════════════

def load_checkpoint(ckpt_path: str, device: torch.device, log):
    log.info("Loading checkpoint: %s", ckpt_path)
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    vocab      = ckpt["vocab"]
    cfg        = ckpt["cfg"]
    model_name = ckpt["model_name"]

    log.info(
        "Checkpoint metadata — model=%s  epoch=%d  best_loss=%.4f",
        model_name, ckpt["epoch"], ckpt["best_loss"],
    )
    log.debug("Saved cfg: %s", cfg)

    kwargs = dict(
        vocab_size  = vocab.vocab_size,
        embed_dim   = cfg["embed_dim"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        dropout     = cfg["dropout"],
    )

    if model_name == "rnn":
        model = VanillaRNN(**kwargs)
    elif model_name == "blstm":
        model = BidirectionalLSTM(**kwargs)
    elif model_name == "attention":
        model = AttentionRNN(**{**kwargs, "attn_dim": cfg.get("attn_dim", 64)})
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model loaded and moved to %s  |  trainable params: %s", device, f"{n_params:,}")
    
    return model, vocab, cfg


# ═══════════════════════════════════════════════════════════
#  Bulk generation
# ═══════════════════════════════════════════════════════════

GENERATE_FN = {
    "rnn":       rnn_mod.generate_name,
    "blstm":     blstm_mod.generate_name,
    "attention": attn_mod.generate_name,
}

def bulk_generate(
    model,
    vocab:      Vocabulary,
    model_name: str,
    n:          int,
    temperature: float,
    device:     torch.device,
    log,
) -> list[str]:
    log.info("Generating %d names with temperature=%.2f …", n, temperature)
    gen_fn = GENERATE_FN[model_name]
    names  = []
    t0     = time.time()

    for i in range(n):
        name = gen_fn(model, vocab, max_len=40, temperature=temperature, device=device)
        if name.strip():
            names.append(name.strip())

        # Progress log every 100 names
        if (i + 1) % 100 == 0:
            log.info("  Generated %d / %d names …", i + 1, n)

    elapsed = time.time() - t0
    log.info(
        "Generation complete — %d names in %.1fs  (%.2f names/s)",
        len(names), elapsed, len(names) / elapsed,
    )
    return names


# ═══════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════

def compute_novelty(generated: list[str], training_set: set[str], log) -> float:
    novel     = [n for n in generated if n.lower() not in training_set]
    not_novel = [n for n in generated if n.lower() in training_set]
    rate      = len(novel) / len(generated) if generated else 0.0

    log.debug("Novelty — novel=%d  memorised=%d  rate=%.4f", len(novel), len(not_novel), rate)
    if not_novel:
        log.debug("Memorised samples (first 5): %s", not_novel[:5])
    return rate

def compute_diversity(generated: list[str], log) -> float:
    unique = set(generated)
    score  = len(unique) / len(generated) if generated else 0.0
    log.debug("Diversity — unique=%d  total=%d  score=%.4f", len(unique), len(generated), score)
    return score

def compute_avg_length(generated: list[str]) -> float:
    return sum(len(n) for n in generated) / len(generated) if generated else 0.0

def compute_metrics(generated: list[str], training_set: set[str], log) -> dict:
    log.info("Computing metrics on %d generated names …", len(generated))
    novelty   = compute_novelty(generated, training_set, log)
    diversity = compute_diversity(generated, log)
    avg_len   = compute_avg_length(generated)

    metrics = {
        "total_generated": len(generated),
        "unique_count":    len(set(generated)),
        "novelty_rate":    novelty,
        "diversity":       diversity,
        "avg_length":      avg_len,
    }

    log.info(
        "Metrics → novelty=%.1f%%  diversity=%.4f  unique=%d  avg_len=%.2f",
        novelty * 100, diversity, metrics["unique_count"], avg_len,
    )
    return metrics


# ═══════════════════════════════════════════════════════════
#  Pretty printers & Savers
# ═══════════════════════════════════════════════════════════

def log_metrics_table(results: dict[str, dict], log):
    sep = "─" * 68
    log.info(sep)
    log.info("  TASK-2 : QUANTITATIVE EVALUATION RESULTS")
    log.info(sep)
    log.info(
        "  %-14s  %9s  %10s  %8s  %8s  %8s",
        "Model", "Novelty%", "Diversity", "Unique", "Total", "AvgLen",
    )
    log.info(sep)
    for model_name, m in results.items():
        log.info(
            "  %-14s  %8.1f%%  %10.4f  %8d  %8d  %8.2f",
            model_name.upper(),
            m["novelty_rate"] * 100,
            m["diversity"],
            m["unique_count"],
            m["total_generated"],
            m["avg_length"],
        )
    log.info(sep)

def log_samples(model_name: str, samples: list[str], n_show: int, log):
    log.info("── %s : representative generated names ──────────────", model_name.upper())
    for i, name in enumerate(samples[:n_show], 1):
        log.info("  %3d.  %s", i, name)

def save_generated_names(all_samples: dict[str, list[str]], output_path: str, log):
    with open(output_path, "w", encoding="utf-8") as f:
        for model_name, names in all_samples.items():
            f.write(f"# {model_name.upper()}\n")
            for name in names:
                f.write(name + "\n")
            f.write("\n")
    log.info("Generated names saved to: %s", output_path)

# ═══════════════════════════════════════════════════════════
#  Main Evaluation Loop
# ═══════════════════════════════════════════════════════════

def main():
    # Initialize logging
    log_path = init_logging(run_name="evaluate", log_dir=CONFIG["log_dir"])
    log = get_logger("evaluate.main")
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    log.info("Starting Evaluation on device: %s", device)
    log.info("Configuration: %s", CONFIG)

    # Load training set for novelty check
    raw_names    = load_names(CONFIG["data_path"])
    training_set = {n.lower() for n in raw_names}
    log.info("Training set size : %d  (lower-cased for comparison)", len(training_set))
    
    all_results = {}
    all_samples = {}

    for model_name in CONFIG["models_to_eval"]:
        ckpt_path = os.path.join(CONFIG["checkpoint_dir"], f"{model_name}_best.pt")
        log.info("")
        log.info("─" * 55)
        
        if not os.path.exists(ckpt_path):
            log.warning("Checkpoint not found, skipping: %s", ckpt_path)
            continue

        # Load
        model, vocab, _ = load_checkpoint(ckpt_path, device, log)

        # Generate
        generated = bulk_generate(
            model, vocab, model_name,
            n           = CONFIG["n_generate"],
            temperature = CONFIG["temperature"],
            device      = device,
            log         = log,
        )

        # Metrics
        metrics = compute_metrics(generated, training_set, log)
        all_results[model_name] = metrics
        all_samples[model_name] = generated

    # ── Save all generated names to file ──────────────────
    if all_samples:
        save_generated_names(all_samples, CONFIG["output_names_file"], log)

    # ── Quantitative table ────────────────────────────────
    log.info("")
    if all_results:
        log_metrics_table(all_results, log)
    else:
        log.warning("No results to display — did any models train successfully?")

    # ── Qualitative samples ───────────────────────────────
    log.info("")
    log.info("=" * 55)
    log.info("  TASK-3 : QUALITATIVE SAMPLES")
    log.info("=" * 55)
    for model_name, samples in all_samples.items():
        log_samples(model_name, samples, n_show=CONFIG["n_samples_show"], log=log)
        log.info("")

    log.info("Evaluation complete. Logs saved to: %s", log_path)


if __name__ == "__main__":
    main()