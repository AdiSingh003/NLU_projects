"""
Tasks 3 & 4: Semantic analysis and visualization for all trained models.

Compares FOUR model variants:
  1. Scratch CBOW       (PyTorch, implemented from scratch)
  2. Scratch SGNS       (PyTorch, implemented from scratch)
  3. Library CBOW       (Gensim)
  4. Library SGNS       (Gensim)

Task 3 — Semantic Analysis (cosine similarity):
  a. Top-5 nearest neighbours for: research, student, phd, exam
  b. Analogy experiments (≥3): A:B::C:?  via vector arithmetic

Task 4 — Visualization:
  a. PCA  2D projection with cluster coloring
  b. t-SNE 2D projection with cluster coloring
  c. Cluster interpretation logged for each model
"""

import os
import sys
import pickle
import logging
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gensim.models import Word2Vec as GensimWord2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(__file__))
from models_scratch import Word2VecVocab, ScratchWordVectors

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/analyze.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Target words for neighbour analysis ───────────────────────────────────────
TARGET_WORDS = ["research", "student", "phd", "exam"]

# ── Analogy triples: (positive_a, positive_b, negative_c)
# Semantics: b - c + a ≈ ?   →   "a is to c as b is to ?"
# All three formats conform to:  most_similar(positive=[a, b], negative=[c])
ANALOGIES = [
    {
        "label":    "UG:BTech :: PG:?  (degree progression)",
        "positive": ["pg", "btech"],
        "negative": ["ug"],
    },
    {
        "label":    "faculty:research :: student:?  (role-activity)",
        "positive": ["student", "research"],
        "negative": ["faculty"],
    },
    {
        "label":    "summer:may :: winter:?  (season-month)",
        "positive": ["winter", "may"],
        "negative": ["summer"],
    },
    {
        "label":    "academic:semester :: institute:?  (temporal-institutional)",
        "positive": ["institute", "semester"],
        "negative": ["academic"],
    },
    {
        "label":    "admission:ug :: fellowship:?  (program-stage)",
        "positive": ["fellowship", "ug"],
        "negative": ["admission"],
    },
]

# ── Word clusters for visualization ───────────────────────────────────────────
CLUSTERS = {
    "Degrees":   ["phd", "mtech", "mba", "ug", "pg", "btech", "msc"],
    "People":    ["student", "faculty", "professor", "researcher", "candidate"],
    "Academics": ["research", "coursework", "thesis", "exam", "semester", "grade"],
    "Admin":     ["admission", "registration", "institute", "department", "committee"],
    "Projects":  ["project", "proposal", "presentation", "defense", "abstract"],
}

CLUSTER_COLORS = {
    "Degrees":   "#e74c3c",
    "People":    "#3498db",
    "Academics": "#2ecc71",
    "Admin":     "#9b59b6",
    "Projects":  "#f39c12",
    "Other":     "#95a5a6",
}


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_gensim_model(model_id: str) -> GensimWord2Vec | None:
    """Load a primary Gensim model by its id string."""
    path = f"models/{model_id}.model"
    if not os.path.exists(path):
        log.error(f"Gensim model not found: {path}")
        return None
    return GensimWord2Vec.load(path)


def load_scratch_model(pkl_id: str) -> ScratchWordVectors | None:
    """Load a scratch model from its saved .pkl file."""
    path = f"models_scratch/{pkl_id}_scratch.pkl"
    if not os.path.exists(path):
        log.error(f"Scratch model not found: {path}")
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Reconstruct ScratchWordVectors from saved embeddings + word2idx
    vocab = Word2VecVocab.__new__(Word2VecVocab)
    vocab.word2idx  = data["word2idx"]
    vocab.idx2word  = {i: w for w, i in data["word2idx"].items()}
    vocab.vocab_size = len(data["word2idx"])
    # Dummy unigram dist (not needed for inference)
    vocab.unigram_dist = np.ones(vocab.vocab_size) / vocab.vocab_size

    return ScratchWordVectors(data["embeddings"], vocab)


# ══════════════════════════════════════════════════════════════════════════════
# Task 3a — Nearest Neighbours
# ══════════════════════════════════════════════════════════════════════════════

def report_neighbours(wv, model_label: str):
    """
    Print the top-5 nearest neighbours (by cosine similarity) for each
    word in TARGET_WORDS.

    Works for both Gensim KeyedVectors and ScratchWordVectors (unified API).
    """
    log.info(f"\n{'='*60}")
    log.info(f"  Top-5 Nearest Neighbours — {model_label}")
    log.info(f"{'='*60}")

    for word in TARGET_WORDS:
        if word not in wv:
            log.warning(f"  '{word}' not in vocabulary of {model_label} — skipping.")
            continue

        neighbours = wv.most_similar(word, topn=5)
        log.info(f"\n  '{word}':")
        for rank, (nbr, sim) in enumerate(neighbours, 1):
            log.info(f"    {rank}. {nbr:<22} cosine={sim:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Task 3b — Analogy Experiments
# ══════════════════════════════════════════════════════════════════════════════

def run_analogies(wv, model_label: str):
    """
    Run the defined analogy experiments using vector arithmetic:
        vector(positive_a) + vector(positive_b) − vector(negative_c) ≈ answer

    Semantically evaluates whether the analogy result is meaningful.
    """
    log.info(f"\n{'='*60}")
    log.info(f"  Analogy Experiments — {model_label}")
    log.info(f"{'='*60}")

    for i, exp in enumerate(ANALOGIES, 1):
        log.info(f"\n  [{i}] {exp['label']}")

        # Check all required words are in vocabulary
        all_words = exp["positive"] + exp["negative"]
        missing = [w for w in all_words if w not in wv]
        if missing:
            log.warning(f"      OOV words: {missing} — skipping.")
            continue

        results = wv.most_similar(
            positive=exp["positive"],
            negative=exp["negative"],
            topn=3,
        )
        pairs = [(w, round(s, 4)) for w, s in results]
        log.info(f"      Result: {pairs}")

# ══════════════════════════════════════════════════════════════════════════════
# Task 4 — Visualization Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _select_words(wv) -> tuple[list[str], dict[str, str]]:
    """
    Choose words to visualize:
      1. Cluster words present in vocabulary
      2. Topped up with the 60 most frequent vocabulary words
    Returns (word_list, cluster_label_map).
    """
    chosen = []
    label_map: dict[str, str] = {}

    for cluster, words in CLUSTERS.items():
        for w in words:
            if w in wv and w not in chosen:
                chosen.append(w)
                label_map[w] = cluster

    # Top-frequency words as context
    all_keys = wv.index_to_key if hasattr(wv, "index_to_key") else list(wv.key_to_index.keys())
    for w in all_keys[:80]:
        if w not in chosen:
            chosen.append(w)
            label_map[w] = "Other"

    return chosen, label_map


def _get_vector(wv, word: str) -> np.ndarray:
    """Retrieve a word vector regardless of model type."""
    return wv[word]


def _draw_projection(title: str, coords: np.ndarray,
                     words: list[str], label_map: dict[str, str],
                     save_path: str):
    """
    Scatter plot of 2D word projections, colour-coded by semantic cluster.
    Each word is annotated with its text label.
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    placed_labels: set[str] = set()

    for i, word in enumerate(words):
        cluster = label_map.get(word, "Other")
        color   = CLUSTER_COLORS.get(cluster, "#95a5a6")
        legend_label = cluster if cluster not in placed_labels else "_nolegend_"

        ax.scatter(
            coords[i, 0], coords[i, 1],
            c=color, s=55, edgecolors="k", linewidths=0.35,
            label=legend_label, zorder=3,
        )
        ax.annotate(
            word, (coords[i, 0], coords[i, 1]),
            fontsize=7.5, alpha=0.88,
            xytext=(4, 4), textcoords="offset points",
        )
        placed_labels.add(cluster)

    ax.legend(title="Cluster", loc="upper right", fontsize=8, framealpha=0.7)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"  Saved → {save_path}")


def visualize_pca(wv, model_label: str, file_id: str):
    """PCA 2D projection of selected word embeddings."""
    words, label_map = _select_words(wv)
    if len(words) < 5:
        log.warning(f"Too few words for PCA in {model_label}")
        return

    vecs = np.array([_get_vector(wv, w) for w in words])
    pca  = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vecs)

    ev = pca.explained_variance_ratio_
    log.info(f"  PCA explained variance: PC1={ev[0]:.3f}, PC2={ev[1]:.3f}")

    _draw_projection(
        title     = f"PCA — {model_label}  (EV: {ev[0]:.1%} + {ev[1]:.1%})",
        coords    = coords,
        words     = words,
        label_map = label_map,
        save_path = f"visualization_pca_{file_id}.png",
    )


def visualize_tsne(wv, model_label: str, file_id: str):
    """t-SNE 2D projection of selected word embeddings."""
    words, label_map = _select_words(wv)
    if len(words) < 10:
        log.warning(f"Too few words for t-SNE in {model_label}")
        return

    vecs       = np.array([_get_vector(wv, w) for w in words])
    perplexity = min(30, max(5, len(words) // 4))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(vecs)
    log.info(f"  t-SNE done  (perplexity={perplexity})")

    _draw_projection(
        title     = f"t-SNE — {model_label}",
        coords    = coords,
        words     = words,
        label_map = label_map,
        save_path = f"visualization_tsne_{file_id}.png",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main Orchestration
# ══════════════════════════════════════════════════════════════════════════════

def analyze():
    log.info("=" * 70)
    log.info("PROBLEM 1 — Semantic Analysis & Visualization")
    log.info("=" * 70)

    # Define all four models: (display_label, file_id, wv_object_or_None)
    models: list[tuple[str, str]] = [
        ("Scratch CBOW",       "cbow_d100_w5_n5"),
        ("Scratch SGNS",       "sgns_d100_w5_n5"),
        ("Library CBOW (Gensim)", "cbow_d100_w5_n5_lib"),
        ("Library SGNS (Gensim)", "sgns_d100_w5_n5_lib"),
    ]

    loaded: list[tuple[str, str, object]] = []

    for label, file_id in models:
        if "Scratch" in label:
            # Strip the _lib suffix for scratch pkl name
            pkl_id = file_id  # e.g. "cbow_d100_w5_n5"
            wv = load_scratch_model(pkl_id)
        else:
            gensim_model = load_gensim_model(file_id)
            wv = gensim_model.wv if gensim_model else None

        if wv is None:
            log.warning(f"  Cannot load {label} — skipping.")
            continue

        loaded.append((label, file_id, wv))

    # ── Run analysis for each loaded model ────────────────────────────────────
    for label, file_id, wv in loaded:
        log.info(f"\n{'#'*70}")
        log.info(f"# MODEL: {label}")
        log.info(f"{'#'*70}")

        # ── Task 3a: Nearest Neighbours ───────────────────────────────────────
        report_neighbours(wv, label)

        # ── Task 3b: Analogy Experiments ──────────────────────────────────────
        run_analogies(wv, label)

        # ── Task 4a: PCA ─────────────────────────────────────────────────────
        log.info(f"\n  Generating PCA visualization …")
        visualize_pca(wv, label, file_id)

        # ── Task 4b: t-SNE ────────────────────────────────────────────────────
        log.info(f"\n  Generating t-SNE visualization …")
        visualize_tsne(wv, label, file_id)

    log.info("\n=== Analysis complete. All visualizations saved to  ===")


if __name__ == "__main__":
    analyze()