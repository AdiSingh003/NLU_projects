"""
Word2Vec models implemented from scratch using PyTorch.

Classes
-------
Word2VecVocab         - Vocabulary builder with unigram^0.75 negative sampling table
CBOWDataset           - PyTorch Dataset for CBOW training pairs
SGNSDataset           - PyTorch Dataset for Skip-gram + Negative Sampling pairs
CBOW                  - CBOW neural network module
SkipGramNS            - Skip-gram with Negative Sampling neural network module
ScratchWordVectors    - Post-training wrapper that mirrors Gensim's wv API

Mathematical Background
-----------------------
CBOW objective:
  Maximize  log P(w_t | context) where
  P(w_t | context) = softmax( W_out · mean(W_in[context]) )
  Loss = NLLLoss ( log_softmax( W_out · mean(W_in[context]) ) )

Skip-gram + Negative Sampling (SGNS) objective:
  Maximize  log σ(v_c · v_w)  +  Σ_k E[log σ(-v_k · v_w)]
  where σ is sigmoid, v_w is target embedding, v_c is positive context embedding,
  and v_k are K sampled negative context embeddings.
  Negative samples drawn from unigram^0.75 distribution (Mikolov et al., 2013).
"""

import math
import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# Vocabulary & Sampling Distribution
# ══════════════════════════════════════════════════════════════════════════════

class Word2VecVocab:
    """
    Builds vocabulary from a list of tokenised sentences and constructs the
    unigram^0.75 negative sampling table (Mikolov et al., 2013).

    Attributes
    ----------
    word2idx : dict[str, int]
    idx2word : dict[int, str]
    vocab_size : int
    unigram_dist : np.ndarray  — probability array for negative sampling
    """

    def __init__(self, sentences: list[list[str]], min_count: int = 1):
        # Count all token frequencies
        freq = Counter(tok for sent in sentences for tok in sent)

        # Filter by min_count and sort by frequency (most common first)
        vocab_items = [(w, c) for w, c in freq.most_common() if c >= min_count]

        self.word2idx: dict[str, int] = {w: i for i, (w, _) in enumerate(vocab_items)}
        self.idx2word: dict[int, str] = {i: w for w, i in self.word2idx.items()}
        self.vocab_size: int = len(self.word2idx)

        # Build unigram^0.75 distribution for negative sampling
        counts = np.array([c for _, c in vocab_items], dtype=np.float64)
        dist = counts ** 0.75                   # raise each count to the 0.75 power
        self.unigram_dist = dist / dist.sum()   # normalize to probability

    def encode(self, word: str) -> int | None:
        return self.word2idx.get(word)

    def sample_negatives(self, n: int, exclude: set) -> list[int]:
        """Sample n negative indices, avoiding indices in 'exclude'."""
        candidates = np.random.choice(self.vocab_size, size=n * 3, p=self.unigram_dist)
        result = [int(idx) for idx in candidates if int(idx) not in exclude]
        return result[:n]


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch Datasets
# ══════════════════════════════════════════════════════════════════════════════

class CBOWDataset(Dataset):
    """
    For each target word w_t in a sentence, constructs a (context, target) pair.
    Context words within window_size are averaged in the model forward pass.

    Pads context to a fixed length (2 * window_size) with index 0 so that
    all samples in a batch have the same shape.
    """

    def __init__(self, sentences: list[list[str]], vocab: Word2VecVocab,
                 window_size: int = 5):
        self.window_size = window_size
        self.vocab = vocab
        self.pairs = []   # list of (context_tensor, target_idx)

        fixed_ctx_len = 2 * window_size

        for sent in sentences:
            indices = [vocab.encode(w) for w in sent]
            indices = [i for i in indices if i is not None]

            for center_pos, target in enumerate(indices):
                # Collect context indices within the window
                ctx = [
                    indices[j]
                    for j in range(
                        max(0, center_pos - window_size),
                        min(len(indices), center_pos + window_size + 1)
                    )
                    if j != center_pos
                ]
                if not ctx:
                    continue

                # Truncate or pad context to fixed_ctx_len
                ctx = ctx[:fixed_ctx_len]
                ctx += [0] * (fixed_ctx_len - len(ctx))

                self.pairs.append((torch.tensor(ctx, dtype=torch.long), target))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class SGNSDataset(Dataset):
    """
    For each (target, positive_context) pair, also pre-samples 'num_neg' negatives
    from the unigram^0.75 distribution.

    Returns (target_idx, positive_idx, neg_tensor) per sample.
    """

    def __init__(self, sentences: list[list[str]], vocab: Word2VecVocab,
                 window_size: int = 5, num_neg: int = 5):
        self.vocab = vocab
        self.num_neg = num_neg
        self.pairs = []   # list of (target, positive_context)

        for sent in sentences:
            indices = [vocab.encode(w) for w in sent]
            indices = [i for i in indices if i is not None]

            for center_pos, target in enumerate(indices):
                for j in range(
                    max(0, center_pos - window_size),
                    min(len(indices), center_pos + window_size + 1)
                ):
                    if j == center_pos:
                        continue
                    self.pairs.append((target, indices[j]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        target, positive = self.pairs[idx]
        # Sample negatives at retrieval time so they vary across epochs
        negatives = np.random.choice(
            self.vocab.vocab_size,
            size=self.num_neg,
            p=self.vocab.unigram_dist
        )
        return (
            torch.tensor(target,    dtype=torch.long),
            torch.tensor(positive,  dtype=torch.long),
            torch.tensor(negatives, dtype=torch.long),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Neural Network Modules
# ══════════════════════════════════════════════════════════════════════════════

class CBOW(nn.Module):
    """
    CBOW: Predict the center word from the mean of its context embeddings.

    Architecture
    ------------
    Input  : context word indices  (batch, 2*window)
    Embed  : nn.Embedding          (vocab_size, embedding_dim)
    Pool   : mean over context     (batch, embedding_dim)
    Linear : projection to vocab   (batch, vocab_size)
    Output : log_softmax scores    (batch, vocab_size)

    Loss used externally: nn.NLLLoss
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

        # Xavier uniform init for stable early training
        nn.init.xavier_uniform_(self.embeddings.weight.data.unsqueeze(0)).squeeze(0)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, context_indices: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        context_indices : (batch, context_len)  — indices of context words

        Returns
        -------
        log_probs : (batch, vocab_size)
        """
        # Embed each context word: (batch, context_len, dim)
        embeds = self.embeddings(context_indices)
        # Average pooling over context positions: (batch, dim)
        pooled = embeds.mean(dim=1)
        # Project to vocabulary: (batch, vocab_size)
        logits = self.linear(pooled)
        return F.log_softmax(logits, dim=1)

    def get_embeddings(self) -> np.ndarray:
        """Return input embeddings as a NumPy array."""
        return self.embeddings.weight.detach().cpu().numpy()


class SkipGramNS(nn.Module):
    """
    Skip-gram with Negative Sampling (SGNS).

    Uses two separate embedding tables:
      - target_embeddings  : for the center/target word
      - context_embeddings : for context words (positive and negative)

    The SGNS loss (maximized) is:
      log σ(v_target · v_positive)  +  Σ_k log σ(−v_target · v_negative_k)

    We return the negated value as a minimization loss.

    Parameters
    ----------
    vocab_size    : size of vocabulary
    embedding_dim : dimensionality of word vectors
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.target_embeddings  = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Standard SGNS init: target uniform(-0.5/d, 0.5/d), context zeros
        init_range = 0.5 / embedding_dim
        nn.init.uniform_(self.target_embeddings.weight,  -init_range, init_range)
        nn.init.constant_(self.context_embeddings.weight, 0.0)

    def forward(self,
                target:   torch.Tensor,   # (batch,)
                positive: torch.Tensor,   # (batch,)
                negatives: torch.Tensor   # (batch, num_neg)
                ) -> torch.Tensor:
        """
        Computes the SGNS loss for a batch.

        Positive score  : σ(v_t · v_c)         — should be high
        Negative scores : σ(−v_t · v_k)  ∀k   — should also be high
        Loss = −[log σ(v_t·v_c) + Σ log σ(−v_t·v_k)]
        """
        # Embed target: (batch, 1, dim)
        v_target = self.target_embeddings(target).unsqueeze(1)

        # Embed positive context: (batch, 1, dim)
        v_positive = self.context_embeddings(positive).unsqueeze(1)

        # Embed negatives: (batch, num_neg, dim)
        v_negatives = self.context_embeddings(negatives)

        # Positive score: dot product (batch, 1, 1) → squeeze → (batch,)
        pos_score = torch.bmm(v_target, v_positive.transpose(1, 2)).squeeze()
        pos_loss  = F.logsigmoid(pos_score).mean()

        # Negative scores: (batch, 1, num_neg) → squeeze → (batch, num_neg)
        neg_scores = torch.bmm(v_target, v_negatives.transpose(1, 2)).squeeze(1)
        neg_loss   = F.logsigmoid(-neg_scores).sum(dim=1).mean()

        # SGNS loss to minimize
        return -(pos_loss + neg_loss)

    def get_embeddings(self) -> np.ndarray:
        """Return target (center-word) embeddings as a NumPy array."""
        return self.target_embeddings.weight.detach().cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Post-training: Gensim-compatible wrapper for scratch embeddings
# ══════════════════════════════════════════════════════════════════════════════

class ScratchWordVectors:
    """
    Wraps the raw NumPy embedding matrix from a scratch-trained model to
    provide the same API as Gensim's KeyedVectors:

        wv[word]                → embedding vector
        word in wv              → membership test
        wv.most_similar(...)    → nearest-neighbor / analogy queries
        wv.index_to_key         → list of words ordered by vocabulary index
    """

    def __init__(self, embeddings: np.ndarray, vocab: Word2VecVocab):
        self.vectors  = embeddings.astype(np.float32)
        self.vocab    = vocab
        self.index_to_key = [vocab.idx2word[i] for i in range(vocab.vocab_size)]

        # Pre-normalise vectors for efficient cosine similarity (dot product)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)  # avoid division by zero
        self.vectors_norm = self.vectors / norms

    # ── Membership and indexing ────────────────────────────────────────────────

    def __contains__(self, word: str) -> bool:
        return word in self.vocab.word2idx

    def __getitem__(self, word: str) -> np.ndarray:
        idx = self.vocab.word2idx[word]
        return self.vectors[idx]

    # ── Neighbor / analogy search ──────────────────────────────────────────────

    def most_similar(self,
                     word: str | None = None,
                     positive: list[str] | None = None,
                     negative: list[str] | None = None,
                     topn: int = 5
                     ) -> list[tuple[str, float]]:
        """
        Find the topn most similar words.

        Single-word query  : most_similar("research", topn=5)
        Analogy query      : most_similar(positive=["A", "B"], negative=["C"], topn=3)

        The query vector is computed as:
            q = Σ norm(pos_i)  −  Σ norm(neg_j)
        then re-normalized before computing cosine similarity against all vectors.
        """
        # Normalise arguments
        if word is not None:
            positive = [word]
        if positive is None:
            positive = []
        if negative is None:
            negative = []
        if isinstance(positive, str):
            positive = [positive]
        if isinstance(negative, str):
            negative = [negative]

        # Build query vector
        query = np.zeros(self.vectors.shape[1], dtype=np.float32)
        exclude: set[int] = set()

        for w in positive:
            if w in self.vocab.word2idx:
                idx = self.vocab.word2idx[w]
                query += self.vectors_norm[idx]
                exclude.add(idx)
            # silently skip OOV words

        for w in negative:
            if w in self.vocab.word2idx:
                idx = self.vocab.word2idx[w]
                query -= self.vectors_norm[idx]
                exclude.add(idx)

        # Re-normalize query
        q_norm = np.linalg.norm(query)
        if q_norm > 0:
            query /= q_norm

        # Cosine similarity against all vocabulary vectors (fast: single matmul)
        similarities = self.vectors_norm @ query   # (vocab_size,)

        # Rank and filter
        ranked = np.argsort(similarities)[::-1]
        results = []
        for idx in ranked:
            if int(idx) in exclude:
                continue
            results.append((self.index_to_key[idx], float(similarities[idx])))
            if len(results) >= topn:
                break
        return results