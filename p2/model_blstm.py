"""
Task-1 · Model 2 : Bidirectional LSTM (from scratch)

Architecture
────────────
Embedding
  → BiLSTM layers  (forward + backward passes concatenated)
  → Linear projection that maps 2·hidden_size  →  hidden_size
  → Linear output head (hidden_size → vocab_size)

LSTM equations (implemented manually)
──────────────────────────────────────
  i_t = σ(W_ii·x_t + b_ii + W_hi·h_{t-1} + b_hi)   input  gate
  f_t = σ(W_if·x_t + b_if + W_hf·h_{t-1} + b_hf)   forget gate
  g_t = tanh(W_ig·x_t + b_ig + W_hg·h_{t-1} + b_hg) cell  gate
  o_t = σ(W_io·x_t + b_io + W_ho·h_{t-1} + b_ho)   output gate
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
  h_t = o_t ⊙ tanh(c_t)

For a Bidirectional LSTM we run one forward-LSTM left→right and one
backward-LSTM right→left, then concatenate their hidden states at each step
to get a 2·hidden_size representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────
#  1.  Single LSTMCell (from scratch)
# ─────────────────────────────────────────────

class LSTMCell(nn.Module):
    """
    A single LSTM step implementing all four gates manually.

    Parameters
    ----------
    input_size  : dimension of input vector x_t
    hidden_size : dimension of hidden / cell state
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Combined gate matrix: maps [x_t | h_{t-1}] → 4·hidden_size
        # This is equivalent to the four separate weight matrices but batched
        # for efficiency: [W_ii, W_if, W_ig, W_io] & [W_hi, W_hf, W_hg, W_ho]
        self.W_gates = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=True)

    def forward(
        self,
        x:  torch.Tensor,   # (batch, input_size)
        hc: tuple[torch.Tensor, torch.Tensor],  # ((batch, hidden), (batch, hidden))
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        h_next : (batch, hidden_size)
        c_next : (batch, hidden_size)
        """
        h, c = hc
        combined = torch.cat([x, h], dim=-1)          # (batch, input+hidden)
        gates    = self.W_gates(combined)              # (batch, 4·hidden)

        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


# ─────────────────────────────────────────────
#  2.  Single-layer Bidirectional LSTM
# ─────────────────────────────────────────────

class BiLSTMLayer(nn.Module):
    """
    One Bidirectional LSTM layer.

    Runs a forward LSTMCell left→right and a backward LSTMCell right→left,
    concatenates their outputs at each time step.

    Output shape: (batch, seq_len, 2·hidden_size)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.fwd_cell = LSTMCell(input_size, hidden_size)
        self.bwd_cell = LSTMCell(input_size, hidden_size)

    def init_state(self, batch_size: int, device: torch.device):
        z = torch.zeros(batch_size, self.hidden_size, device=device)
        return (z.clone(), z.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, input_size)
        returns : (batch, seq_len, 2·hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        hc_fwd = self.init_state(batch_size, device)
        hc_bwd = self.init_state(batch_size, device)

        fwd_outputs, bwd_outputs = [], []

        # ── Forward pass ──────────────────────
        for t in range(seq_len):
            hc_fwd = self.fwd_cell(x[:, t, :], hc_fwd)
            fwd_outputs.append(hc_fwd[0])           # keep h, not c

        # ── Backward pass (Causal Look-back) ──
        for t in range(seq_len):
            hc_bwd_t = self.init_state(batch_size, device)
            # Read prefix x[:, :t+1] in reverse
            for rev_t in range(t, -1, -1):
                hc_bwd_t = self.bwd_cell(x[:, rev_t, :], hc_bwd_t)
            bwd_outputs.append(hc_bwd_t[0])

        # ── Concatenate ───────────────────────
        fwd_stack = torch.stack(fwd_outputs, dim=1)  # (batch, seq_len, hidden)
        bwd_stack = torch.stack(bwd_outputs, dim=1)
        return torch.cat([fwd_stack, bwd_stack], dim=-1)  # (batch, seq, 2·hidden)


# ─────────────────────────────────────────────
#  3.  Full BiLSTM model
# ─────────────────────────────────────────────

class BidirectionalLSTM(nn.Module):
    """
    Character-level name generator using stacked BiLSTM layers.
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int   = 32,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
        pad_idx:     int   = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # ── Embedding ──────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # ── Stacked BiLSTM + projection layers ────────────
        # Layer 0: embed_dim → 2·hidden (BiLSTM) → hidden (projection)
        # Layer k: hidden   → 2·hidden (BiLSTM) → hidden (projection)
        self.bilstm_layers  = nn.ModuleList()
        self.proj_layers    = nn.ModuleList()

        for i in range(num_layers):
            in_size = embed_dim if i == 0 else hidden_size
            self.bilstm_layers.append(BiLSTMLayer(in_size, hidden_size))
            self.proj_layers.append(nn.Linear(2 * hidden_size, hidden_size))

        # ── Dropout ────────────────────────────────────────
        self.drop = nn.Dropout(dropout)

        # ── Output head ────────────────────────────────────
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        n_params = self.count_parameters()
        log.info(
            "BidirectionalLSTM built — vocab=%d  embed=%d  hidden=%d (per dir)  "
            "layers=%d  dropout=%.2f  trainable_params=%s",
            vocab_size, embed_dim, hidden_size, num_layers, dropout, f"{n_params:,}",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (batch, seq_len)   – token indices
        returns: (batch, seq_len, vocab_size)   logits
        """
        out = self.drop(self.embedding(x))         # (batch, seq_len, embed_dim)

        for bilstm, proj in zip(self.bilstm_layers, self.proj_layers):
            out = bilstm(out)                       # (batch, seq, 2·hidden)
            out = torch.tanh(proj(out))             # (batch, seq, hidden)
            out = self.drop(out)

        logits = self.fc_out(out)                   # (batch, seq, vocab_size)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
#  4.  Greedy / temperature-sampled generation
# ─────────────────────────────────────────────

@torch.no_grad()
def generate_name(
    model:       BidirectionalLSTM,
    vocab,
    max_len:     int   = 30,
    temperature: float = 1.0,
    device:      torch.device = torch.device("cpu"),
) -> str:
    """
    Left-to-right sampling from the BiLSTM model.

    At each step the model processes the full prefix generated so far and
    predicts the next character from the final time step's logits.
    """
    model.eval()
    tokens = [vocab.SOS_IDX]
    log.debug("generate_name (BiLSTM) — temperature=%.2f  max_len=%d", temperature, max_len)

    for _ in range(max_len):
        x      = torch.tensor([tokens], device=device)
        logits = model(x)
        last   = logits[:, -1, :] / temperature
        probs  = F.softmax(last, dim=-1)
        nxt    = torch.multinomial(probs, 1).item()

        if nxt == vocab.EOS_IDX:
            break
        tokens.append(nxt)

    result = vocab.decode(tokens[1:], strip_special=True)
    log.debug("BiLSTM generated: '%s'", result)
    return result


# ─────────────────────────────────────────────
#  Quick architecture summary (standalone)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    VOCAB_SIZE  = 70
    EMBED_DIM   = 64
    HIDDEN_SIZE = 256
    NUM_LAYERS  = 2
    DROPOUT     = 0.3

    model = BidirectionalLSTM(
        vocab_size  = VOCAB_SIZE,
        embed_dim   = EMBED_DIM,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
    )

    print("=" * 60)
    print("  BIDIRECTIONAL LSTM — ARCHITECTURE SUMMARY")
    print("=" * 60)
    print(model)
    print("-" * 60)
    print(f"  Trainable parameters : {model.count_parameters():,}")
    print(f"  Vocab size           : {VOCAB_SIZE}")
    print(f"  Embedding dim        : {EMBED_DIM}")
    print(f"  Hidden size          : {HIDDEN_SIZE}  (per direction)")
    print(f"  Bidirectional output : {2 * HIDDEN_SIZE}  (before projection)")
    print(f"  Num layers           : {NUM_LAYERS}")
    print(f"  Dropout              : {DROPOUT}")
    print("=" * 60)