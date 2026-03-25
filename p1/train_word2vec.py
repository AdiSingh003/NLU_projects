"""
Task 2: Train Word2Vec models — both from-scratch (PyTorch) and library (Gensim).

Scratch models
--------------
  - CBOW  (PyTorch, implemented in models_scratch.py)
  - SGNS  (Skip-gram + Negative Sampling, PyTorch)

Library models (for comparison)
--------------------------------
  - Gensim CBOW  (sg=0, hs=0)
  - Gensim SGNS  (sg=1, negative=5, hs=0)

Hyperparameter experiments (embedding dim, window size, neg samples) are
run on both model types and reported in a formal table.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from gensim.models import Word2Vec as GensimWord2Vec

# ── Import our scratch implementations ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from models_scratch import (
    Word2VecVocab,
    CBOWDataset, SGNSDataset,
    CBOW, SkipGramNS,
    ScratchWordVectors,
)

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/train.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Global Hyperparameters ────────────────────────────────────────────────────
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIM  = 100   # default for primary model; varied in experiments
WINDOW_SIZE    = 8     # default; varied in experiments
NEG_SAMPLES    = 5     # default; varied in experiments
BATCH_SIZE     = 512
EPOCHS_SCRATCH = 15    # scratch is slower (Python vs C); 10 epochs is impactful
EPOCHS_GENSIM  = 50    # Gensim is fast; run more epochs for fairness
LEARNING_RATE  = 0.005
MIN_COUNT      = 1     # keep all words given limited academic corpus

# ── Hyperparameter experiment grid ────────────────────────────────────────────
EXPERIMENT_CONFIGS = [
    # ── Primary models (used for full semantic analysis + visualization) ───────
    {"id": "cbow_d100_w5_n5",  "model": "cbow", "dim": 100, "window": 8,  "neg": 5,  "tag": "PRIMARY"},
    {"id": "sgns_d100_w5_n5",  "model": "sgns", "dim": 200, "window": 8,  "neg": 5,  "tag": "PRIMARY"},

    # ── Embedding dimension experiments — CBOW ────────────────────────────────
    {"id": "cbow_d50_w5_n5",   "model": "cbow", "dim": 150,  "window": 8,  "neg": 5,  "tag": "DIM_EXP"},
    {"id": "cbow_d200_w5_n5",  "model": "cbow", "dim": 200, "window": 8,  "neg": 5,  "tag": "DIM_EXP"},
    # ── Embedding dimension experiments — SGNS ────────────────────────────────
    {"id": "sgns_d50_w5_n5",   "model": "sgns", "dim": 250,  "window": 8,  "neg": 5,  "tag": "DIM_EXP"},
    {"id": "sgns_d200_w5_n5",  "model": "sgns", "dim": 300, "window": 8,  "neg": 5,  "tag": "DIM_EXP"},

    # ── Context window experiments — CBOW ─────────────────────────────────────
    {"id": "cbow_d100_w3_n5",  "model": "cbow", "dim": 100, "window": 5,  "neg": 5,  "tag": "WIN_EXP"},
    {"id": "cbow_d100_w10_n5", "model": "cbow", "dim": 200, "window": 10, "neg": 5,  "tag": "WIN_EXP"},
    # ── Context window experiments — SGNS ─────────────────────────────────────
    {"id": "sgns_d100_w3_n5",  "model": "sgns", "dim": 100, "window": 5,  "neg": 5,  "tag": "WIN_EXP"},
    {"id": "sgns_d100_w10_n5", "model": "sgns", "dim": 200, "window": 10, "neg": 5,  "tag": "WIN_EXP"},

    # ── Negative samples experiments — CBOW ───────────────────────────────────
    {"id": "cbow_d100_w5_n1",  "model": "cbow", "dim": 100, "window": 8,  "neg": 10,  "tag": "NEG_EXP"},
    {"id": "cbow_d100_w5_n15", "model": "cbow", "dim": 200, "window": 8,  "neg": 15, "tag": "NEG_EXP"},
    # ── Negative samples experiments — SGNS ───────────────────────────────────
    {"id": "sgns_d100_w5_n1",  "model": "sgns", "dim": 100, "window": 8,  "neg": 10,  "tag": "NEG_EXP"},
    {"id": "sgns_d100_w5_n15", "model": "sgns", "dim": 200, "window": 8,  "neg": 15, "tag": "NEG_EXP"},
]


# ══════════════════════════════════════════════════════════════════════════════
# Corpus Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_corpus(path: str = "cleaned_corpus.txt") -> list[list[str]]:
    """Load cleaned corpus (one sentence per line, space-separated tokens)."""
    if not os.path.exists(path):
        log.error(f"Corpus not found at '{path}'. Run preprocess.py first.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        sentences = [line.split() for line in f if line.strip()]
    log.info(f"Loaded {len(sentences):,} sentences from '{path}'")
    return sentences


# ══════════════════════════════════════════════════════════════════════════════
# From-Scratch Training (PyTorch)
# ══════════════════════════════════════════════════════════════════════════════

def train_scratch_cbow(sentences: list[list[str]], vocab: Word2VecVocab,
                       dim: int, window: int) -> ScratchWordVectors:
    """
    Train CBOW from scratch.

    Returns a ScratchWordVectors wrapper around the learned input embeddings.
    """
    log.info(f"  [Scratch CBOW] dim={dim}  window={window}  epochs={EPOCHS_SCRATCH}")

    dataset = CBOWDataset(sentences, vocab, window_size=window)
    log.info(f"    Dataset size: {len(dataset):,} (target, context) pairs")

    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=0, drop_last=False)

    model     = CBOW(vocab.vocab_size, dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()

    model.train()
    for epoch in range(1, EPOCHS_SCRATCH + 1):
        total_loss = 0.0
        for context_batch, target_batch in loader:
            context_batch = context_batch.to(DEVICE)
            target_batch  = target_batch.to(DEVICE)

            optimizer.zero_grad()
            log_probs = model(context_batch)
            loss = criterion(log_probs, target_batch)
            loss.backward()
            # Gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        if epoch % 2 == 0 or epoch == 1:
            log.info(f"    Epoch {epoch:>3}/{EPOCHS_SCRATCH}  loss={avg_loss:.4f}")

    return ScratchWordVectors(model.get_embeddings(), vocab)


def train_scratch_sgns(sentences: list[list[str]], vocab: Word2VecVocab,
                       dim: int, window: int, neg: int) -> ScratchWordVectors:
    """
    Train Skip-gram with Negative Sampling from scratch.

    Returns a ScratchWordVectors wrapper around the learned target embeddings.
    """
    log.info(f"  [Scratch SGNS] dim={dim}  window={window}  neg={neg}  epochs={EPOCHS_SCRATCH}")

    dataset = SGNSDataset(sentences, vocab, window_size=window, num_neg=neg)
    log.info(f"    Dataset size: {len(dataset):,} (target, positive) pairs")

    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=0, drop_last=False)

    model     = SkipGramNS(vocab.vocab_size, dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(1, EPOCHS_SCRATCH + 1):
        total_loss = 0.0
        for target_batch, positive_batch, neg_batch in loader:
            target_batch   = target_batch.to(DEVICE)
            positive_batch = positive_batch.to(DEVICE)
            neg_batch      = neg_batch.to(DEVICE)

            optimizer.zero_grad()
            loss = model(target_batch, positive_batch, neg_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        if epoch % 2 == 0 or epoch == 1:
            log.info(f"    Epoch {epoch:>3}/{EPOCHS_SCRATCH}  loss={avg_loss:.4f}")

    return ScratchWordVectors(model.get_embeddings(), vocab)


# ══════════════════════════════════════════════════════════════════════════════
# Gensim Training (Library)
# ══════════════════════════════════════════════════════════════════════════════

def train_gensim(sentences: list[list[str]], sg: int,
                 dim: int, window: int, neg: int, model_id: str) -> GensimWord2Vec:
    """Train a Word2Vec model using the Gensim library for comparison."""
    model_type = "Skip-gram+NS" if sg == 1 else "CBOW"
    log.info(f"  [Gensim {model_type}] dim={dim}  window={window}  neg={neg}")

    model = GensimWord2Vec(
        sentences   = sentences,
        vector_size = dim,
        window      = window,
        min_count   = MIN_COUNT,
        sg          = sg,
        negative    = neg,
        hs          = 0,          # use Negative Sampling (not hierarchical softmax)
        workers     = 4,
        epochs      = EPOCHS_GENSIM,
        seed        = 42,
    )
    out_path = f"models/{model_id}.model"
    model.save(out_path)
    log.info(f"    Vocab: {len(model.wv.index_to_key):,}  → saved {out_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Main Training Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def train_all():
    log.info("=" * 70)
    log.info("PROBLEM 1 — Word2Vec Training Pipeline")
    log.info(f"Device: {DEVICE}")
    log.info("=" * 70)

    sentences = load_corpus()
    if not sentences:
        return

    os.makedirs("models",         exist_ok=True)
    os.makedirs("models_scratch", exist_ok=True)

    # Build a shared vocabulary (same for both scratch and Gensim for fairness)
    log.info("\nBuilding vocabulary …")
    vocab = Word2VecVocab(sentences, min_count=MIN_COUNT)
    log.info(f"  Vocabulary size : {vocab.vocab_size:,}")

    # ── Experiment results storage ─────────────────────────────────────────────
    results = []

    # ── Hyperparameter Experiments ─────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("HYPERPARAMETER EXPERIMENTS")
    log.info("=" * 70)

    for cfg in EXPERIMENT_CONFIGS:
        log.info(f"\n[CONFIG] {cfg['id']}  (tag={cfg['tag']})")

        if cfg["model"] == "cbow":
            # Scratch CBOW
            wv_scratch = train_scratch_cbow(sentences, vocab, cfg["dim"], cfg["window"])
            scratch_data = {
                "type": "cbow",
                "embeddings": wv_scratch.vectors,
                "word2idx":   vocab.word2idx,
            }
            with open(f"models_scratch/{cfg['id']}_scratch.pkl", "wb") as f:
                pickle.dump(scratch_data, f)

            # Gensim CBOW
            gensim_model = train_gensim(sentences, sg=0,
                                        dim=cfg["dim"], window=cfg["window"],
                                        neg=cfg["neg"], model_id=f"{cfg['id']}_lib")
        else:
            # Scratch SGNS
            wv_scratch = train_scratch_sgns(sentences, vocab,
                                            cfg["dim"], cfg["window"], cfg["neg"])
            scratch_data = {
                "type": "sgns",
                "embeddings": wv_scratch.vectors,
                "word2idx":   vocab.word2idx,
            }
            with open(f"models_scratch/{cfg['id']}_scratch.pkl", "wb") as f:
                pickle.dump(scratch_data, f)

            # Gensim SGNS
            gensim_model = train_gensim(sentences, sg=1,
                                        dim=cfg["dim"], window=cfg["window"],
                                        neg=cfg["neg"], model_id=f"{cfg['id']}_lib")

        results.append({**cfg, "vocab_size": vocab.vocab_size})

    # ── Print Experiment Table ─────────────────────────────────────────────────
    log.info("\n" + "=" * 80)
    log.info("HYPERPARAMETER EXPERIMENT RESULTS TABLE")
    log.info("=" * 80)
    header = f"{'Config ID':<25} | {'Model':<5} | {'Dim':>5} | {'Window':>6} | {'NegSamp':>7} | {'Vocab':>6} | Tag"
    log.info(header)
    log.info("-" * 80)
    for r in results:
        log.info(
            f"{r['id']:<25} | {r['model'].upper():<5} | {r['dim']:>5} | "
            f"{r['window']:>6} | {r['neg']:>7} | {r['vocab_size']:>6} | {r['tag']}"
        )
    log.info("=" * 80)

    # Save experiment summary as JSON for report generation
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("Experiment results saved → experiment_results.json")
    log.info("\nTraining complete.")


if __name__ == "__main__":
    train_all()