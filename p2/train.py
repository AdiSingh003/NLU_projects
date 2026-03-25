"""
Unified training script for all three character-level name-generation models.
Configurations are now defined per-model directly in the script.
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from logger               import init_logging, get_logger, get_log_path
from data_utils           import build_dataloader, Vocabulary
from model_rnn            import VanillaRNN
from model_blstm          import BidirectionalLSTM
from model_attention_rnn  import AttentionRNN


# ═══════════════════════════════════════════════════════════
#  Per-Model Configurations
# ═══════════════════════════════════════════════════════════

CONFIGS = {
    "rnn": {
        "data_path":      "TrainingNames.txt",
        "epochs":         60,
        "batch_size":     64,
        "lr":             1e-3,
        "embed_dim":      32,
        "hidden_size":    128,
        "num_layers":     2,
        "dropout":        0.3,
        "grad_clip":      5.0,
        "patience":       15,
        "checkpoint_dir": "checkpoints",
        "log_dir":        "logs",
        "seed":           42,
    },
    "blstm": {
        "data_path":      "TrainingNames.txt",
        "epochs":         60,         
        "batch_size":     32,         
        "lr":             3e-3,       
        "embed_dim":      64,
        "hidden_size":    128,        
        "num_layers":     2,
        "dropout":        0.4,
        "grad_clip":      5.0,
        "patience":       20,
        "checkpoint_dir": "checkpoints",
        "log_dir":        "logs",
        "seed":           42,
    },
    "attention": {
        "data_path":      "TrainingNames.txt",
        "epochs":         60,
        "batch_size":     64,
        "lr":             1e-3,
        "embed_dim":      64,
        "hidden_size":    128,
        "num_layers":     2,
        "attn_dim":       128,         # Specific to AttentionRNN
        "dropout":        0.3,
        "grad_clip":      5.0,
        "patience":       15,
        "checkpoint_dir": "checkpoints",
        "log_dir":        "logs",
        "seed":           42,
    }
}


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def set_seed(seed: int, log):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.info("Random seed set to %d", seed)


def get_device(log) -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    log.info("Compute device: %s", dev)
    return dev


def build_model(model_name: str, vocab: Vocabulary, cfg: dict, log) -> nn.Module:
    kwargs = dict(
        vocab_size  = vocab.vocab_size,
        embed_dim   = cfg["embed_dim"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        dropout     = cfg["dropout"],
        pad_idx     = vocab.PAD_IDX,
    )
    log.info("Instantiating model '%s' …", model_name)
    if model_name == "rnn":
        return VanillaRNN(**kwargs)
    elif model_name == "blstm":
        return BidirectionalLSTM(**kwargs)
    elif model_name == "attention":
        return AttentionRNN(**{**kwargs, "attn_dim": cfg.get("attn_dim", 64)})
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ═══════════════════════════════════════════════════════════
#  Loss helpers
# ═══════════════════════════════════════════════════════════

def compute_loss_rnn(model, batch, criterion, device):
    inp, tgt = batch
    inp, tgt = inp.to(device), tgt.to(device)
    logits, _ = model(inp)
    B, T, V   = logits.shape
    return criterion(logits.reshape(B * T, V), tgt.reshape(B * T))


def compute_loss_blstm(model, batch, criterion, device):
    inp, tgt = batch
    inp, tgt = inp.to(device), tgt.to(device)
    logits   = model(inp)
    B, T, V  = logits.shape
    return criterion(logits.reshape(B * T, V), tgt.reshape(B * T))


def compute_loss_attention(model, batch, criterion, device):
    inp, tgt = batch
    inp, tgt = inp.to(device), tgt.to(device)
    logits   = model(src=inp, tgt=inp)
    B, T, V  = logits.shape
    return criterion(logits.reshape(B * T, V), tgt.reshape(B * T))


LOSS_FN = {
    "rnn":       compute_loss_rnn,
    "blstm":     compute_loss_blstm,
    "attention": compute_loss_attention,
}


# ═══════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════

def train_one_model(model_name: str, cfg: dict):
    # ── Logging ───────────────────────────────────────────
    log_path = init_logging(
        run_name = f"train_{model_name}",
        log_dir  = cfg["log_dir"],
    )
    log = get_logger(f"train.{model_name}")

    log.info("=" * 60)
    log.info("  TRAINING : %s", model_name.upper())
    log.info("=" * 60)
    log.info("Log file  : %s", log_path)

    # ── Config dump ───────────────────────────────────────
    log.info("Hyperparameters:")
    for k, v in cfg.items():
        log.info("  %-20s = %s", k, v)

    set_seed(cfg["seed"], log)
    device = get_device(log)

    # ── Data ──────────────────────────────────────────────
    log.info("Loading dataset …")
    loader, vocab, _ = build_dataloader(
        cfg["data_path"], batch_size=cfg["batch_size"]
    )
    log.info("Vocab size: %d  |  Batches per epoch: %d", vocab.vocab_size, len(loader))

    # ── Model ─────────────────────────────────────────────
    model    = build_model(model_name, vocab, cfg, log).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model moved to %s — trainable params: %s", device, f"{n_params:,}")

    # ── Optimiser & loss ──────────────────────────────────
    optimiser = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=8
    )
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_IDX)
    log.info(
        "Optimiser: Adam(lr=%.4f)  |  Scheduler: ReduceLROnPlateau  |  "
        "Loss: CrossEntropy(ignore_pad=%d)",
        cfg["lr"], vocab.PAD_IDX,
    )

    loss_fn = LOSS_FN[model_name]

    # ── Checkpoint dir ────────────────────────────────────
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["checkpoint_dir"], f"{model_name}_best.pt")
    log.info("Checkpoint path: %s", ckpt_path)

    # ── Training loop ─────────────────────────────────────
    best_loss  = float("inf")
    no_improve = 0
    history    = []
    train_start = time.time()

    log.info("Starting training for up to %d epochs …", cfg["epochs"])

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(loader):
            optimiser.zero_grad()
            loss = loss_fn(model, batch, criterion, device)
            loss.backward()

            # Gradient norm before clipping (logged at DEBUG level)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            log.debug(
                "epoch=%d  batch=%d/%d  batch_loss=%.4f  grad_norm=%.4f",
                epoch, batch_idx + 1, len(loader), loss.item(), grad_norm,
            )

            optimiser.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - epoch_start
        history.append(avg_loss)

        # LR after scheduler step
        old_lr = optimiser.param_groups[0]["lr"]
        scheduler.step(avg_loss)
        new_lr = optimiser.param_groups[0]["lr"]

        # ── Log every epoch at INFO; detail at DEBUG ───────
        log.info(
            "Epoch %4d/%d  |  loss=%.4f  |  best=%.4f  |  "
            "no_improve=%d  |  lr=%.2e  |  %.1fs",
            epoch, cfg["epochs"], avg_loss, best_loss,
            no_improve, new_lr, elapsed,
        )

        if new_lr < old_lr:
            log.info("  LR reduced: %.2e → %.2e", old_lr, new_lr)

        # ── Checkpoint ─────────────────────────────────────
        if avg_loss < best_loss:
            prev_best  = best_loss
            best_loss  = avg_loss
            no_improve = 0
            torch.save({
                "epoch":      epoch,
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "vocab":      vocab,
                "cfg":        cfg,
                "best_loss":  best_loss,
            }, ckpt_path)
            log.info(
                "  ✔ New best! loss improved %.4f → %.4f  |  checkpoint saved",
                prev_best, best_loss,
            )
        else:
            no_improve += 1
            log.debug("  No improvement (%d/%d patience)", no_improve, cfg["patience"])

        # ── Early stopping ─────────────────────────────────
        if no_improve >= cfg["patience"]:
            log.info(
                "Early stopping triggered at epoch %d  "
                "(no improvement for %d epochs)",
                epoch, cfg["patience"],
            )
            break

    total_time = time.time() - train_start
    log.info("-" * 60)
    log.info("Training complete in %.1fs  (%.1f min)", total_time, total_time / 60)
    log.info("Best loss    : %.4f", best_loss)
    log.info("Total epochs : %d",   len(history))
    log.info("Checkpoint   : %s",   ckpt_path)
    log.info("Log file     : %s",   get_log_path())
    log.info("=" * 60)

    return ckpt_path, history


def main():
    # Define which models to train. 
    # You can comment out a model here if you want to skip it during a run.
    models_to_train = ["rnn", "blstm", "attention"]

    results = {}
    for m in models_to_train:
        cfg = CONFIGS[m]
        ckpt, hist = train_one_model(m, cfg)
        results[m] = {
            "checkpoint":  ckpt,
            "final_loss":  hist[-1],
            "best_loss":   min(hist),
            "epochs_ran":  len(hist),
        }

    # ── Final summary (uses a fresh root logger) ──────────
    log = get_logger("train.summary")
    log.info("")
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║            TRAINING RUN SUMMARY                 ║")
    log.info("╠══════════════════════════════════════════════════╣")
    log.info("║  %-10s  %-10s  %-10s  %-8s  ║",
             "Model", "BestLoss", "FinalLoss", "Epochs")
    log.info("╠══════════════════════════════════════════════════╣")
    for m, r in results.items():
        log.info("║  %-10s  %-10.4f  %-10.4f  %-8d  ║",
                 m.upper(), r["best_loss"], r["final_loss"], r["epochs_ran"])
    log.info("╠══════════════════════════════════════════════════╣")
    log.info("╚══════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()