"""
Task-1 · Model 1 : Vanilla Recurrent Neural Network (from scratch)

Architecture
────────────
Embedding  →  RNN (stacked, optionally multi-layer)  →  Linear projection  →  Softmax

The RNN cell is the classic Elman cell:

    h_t = tanh( x_t · W_ih + b_ih + h_{t-1} · W_hh + b_hh )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────
#  Custom single Elman-RNN cell  (from scratch)
# ─────────────────────────────────────────────

class VanillaRNNCell(nn.Module):
    """
    Single-step Elman RNN cell.

    Parameters
    ----------
    input_size  : dimensionality of the input vector x_t
    hidden_size : dimensionality of the hidden state h_t
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Input-to-hidden weight matrix + bias
        self.W_ih = nn.Linear(input_size,  hidden_size, bias=True)
        # Hidden-to-hidden weight matrix (no separate bias to avoid double-counting)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, input_size)
        h : (batch, hidden_size)
        returns h_next : (batch, hidden_size)
        """
        return torch.tanh(self.W_ih(x) + self.W_hh(h))


# ─────────────────────────────────────────────
#  Full model
# ─────────────────────────────────────────────

class VanillaRNN(nn.Module):
    """
    Character-level name generator using a stacked Vanilla RNN.
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int = 64,
        hidden_size: int = 256,
        num_layers:  int = 2,
        dropout:     float = 0.3,
        pad_idx:     int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # ── Embedding ──────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # ── Stacked RNN cells ──────────────────────────────
        # Layer 0 takes the embedding; subsequent layers take hidden from prev layer
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_size = embed_dim if i == 0 else hidden_size
            self.cells.append(VanillaRNNCell(in_size, hidden_size))

        # ── Dropout ────────────────────────────────────────
        self.drop = nn.Dropout(dropout)

        # ── Output projection ──────────────────────────────
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        n_params = self.count_parameters()
        log.info(
            "VanillaRNN built — vocab=%d  embed=%d  hidden=%d  layers=%d  "
            "dropout=%.2f  trainable_params=%s",
            vocab_size, embed_dim, hidden_size, num_layers, dropout, f"{n_params:,}",
        )

    # ── Initialise hidden states ───────────────────────────

    def init_hidden(self, batch_size: int, device: torch.device) -> list[torch.Tensor]:
        """Returns a list of zero-initialised hidden states, one per layer."""
        return [
            torch.zeros(batch_size, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]

    # ── Forward pass ───────────────────────────────────────

    def forward(
        self,
        x:      torch.Tensor,                     # (batch, seq_len)
        hidden: list[torch.Tensor] | None = None, # list of (batch, hidden_size)
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Returns
        -------
        logits : (batch, seq_len, vocab_size)
        hidden : updated list of hidden states
        """
        batch_size, seq_len = x.shape
        device = x.device

        if hidden is None:
            hidden = self.init_hidden(batch_size, device)

        embedded = self.embedding(x)      # (batch, seq_len, embed_dim)
        embedded = self.drop(embedded)

        all_logits = []
        for t in range(seq_len):
            xt = embedded[:, t, :]        # (batch, embed_dim)

            # propagate through each layer
            new_hidden = []
            for layer_idx, cell in enumerate(self.cells):
                ht = cell(xt, hidden[layer_idx])
                if layer_idx < self.num_layers - 1:
                    ht = self.drop(ht)
                new_hidden.append(ht)
                xt = ht                   # next layer input

            hidden = new_hidden
            all_logits.append(self.fc_out(hidden[-1]))   # (batch, vocab_size)

        logits = torch.stack(all_logits, dim=1)           # (batch, seq_len, vocab_size)
        return logits, hidden

    # ── Parameter count ────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
#  Greedy / temperature-sampled generation
# ─────────────────────────────────────────────

@torch.no_grad()
def generate_name(
    model:       VanillaRNN,
    vocab,                          # Vocabulary object from data_utils
    max_len:     int   = 30,
    temperature: float = 1.0,
    device:      torch.device = torch.device("cpu"),
) -> str:
    """
    Auto-regressively sample one name from the RNN.

    Sampling stops when <EOS> is produced or max_len is reached.
    """
    model.eval()
    hidden = model.init_hidden(1, device)
    log.debug("generate_name (RNN) — temperature=%.2f  max_len=%d", temperature, max_len)

    # Seed with <SOS>
    token = torch.tensor([[vocab.SOS_IDX]], device=device)   # (1, 1)
    generated = []

    for _ in range(max_len):
        logits, hidden = model(token, hidden)         # logits: (1,1,vocab_size)
        logits = logits[:, -1, :] / temperature       # (1, vocab_size)
        probs  = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        if next_token == vocab.EOS_IDX:
            break
        generated.append(next_token)
        token = torch.tensor([[next_token]], device=device)

    result = vocab.decode(generated, strip_special=True)
    log.debug("RNN generated: '%s'", result)
    return result


# ─────────────────────────────────────────────
#  Quick architecture summary (standalone)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    VOCAB_SIZE  = 70    # approximate for Indian names
    EMBED_DIM   = 64
    HIDDEN_SIZE = 256
    NUM_LAYERS  = 2
    DROPOUT     = 0.3

    model = VanillaRNN(
        vocab_size  = VOCAB_SIZE,
        embed_dim   = EMBED_DIM,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
    )

    print("=" * 55)
    print("  VANILLA RNN — ARCHITECTURE SUMMARY")
    print("=" * 55)
    print(model)
    print("-" * 55)
    print(f"  Trainable parameters : {model.count_parameters():,}")
    print(f"  Vocab size           : {VOCAB_SIZE}")
    print(f"  Embedding dim        : {EMBED_DIM}")
    print(f"  Hidden size          : {HIDDEN_SIZE}")
    print(f"  Num layers           : {NUM_LAYERS}")
    print(f"  Dropout              : {DROPOUT}")
    print("=" * 55)