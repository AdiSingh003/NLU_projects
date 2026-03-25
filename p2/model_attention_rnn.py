"""
Task-1 · Model 3 : Vanilla RNN + Basic Additive (Bahdanau-style) Attention

Architecture
────────────
Embedding
  → Encoder RNN  (stacked Elman cells, produces context vectors for every step)
  → Decoder RNN  (generates one character at a time)
      at each step:
        1. attention scores  e_ti = v · tanh(W_a · h_encoder_i + U_a · h_decoder_{t-1})
        2. attention weights α_ti = softmax(e_ti)
        3. context vector    c_t  = Σ α_ti · h_encoder_i
        4. decoder input     = concat(embed(y_{t-1}), c_t)   → Elman cell → h_decoder_t
  → Linear head  h_decoder_t → vocab_size logits

This is a simplified seq2seq model where encoder and decoder share the same
sequence (self-attention style) for name generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────
#  1.  Elman RNN cell (reused from model_rnn)
# ─────────────────────────────────────────────

class ElmanCell(nn.Module):
    """Single-step Elman RNN cell: h_t = tanh(W_ih·x_t + W_hh·h_{t-1} + b)"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_ih = nn.Linear(input_size,  hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.W_ih(x) + self.W_hh(h))


# ─────────────────────────────────────────────
#  2.  Additive (Bahdanau) Attention
# ─────────────────────────────────────────────

class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention.

    Energy function:
        e_i = v · tanh( W_a · encoder_h_i  +  U_a · decoder_h )

    Parameters
    ----------
    encoder_hidden : dimensionality of encoder hidden state
    decoder_hidden : dimensionality of decoder hidden state
    attn_dim       : internal attention projection dimension
    """

    def __init__(self, encoder_hidden: int, decoder_hidden: int, attn_dim: int = 128):
        super().__init__()
        self.W_a = nn.Linear(encoder_hidden, attn_dim, bias=False)
        self.U_a = nn.Linear(decoder_hidden, attn_dim, bias=True)
        self.v   = nn.Linear(attn_dim,       1,        bias=False)

    def forward(
        self,
        encoder_outputs: torch.Tensor,   # (batch, src_len, encoder_hidden)
        decoder_hidden:  torch.Tensor,   # (batch, decoder_hidden)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        context : (batch, encoder_hidden)          – weighted sum of encoder states
        weights : (batch, src_len)                 – attention distribution (for viz)
        """
        # (batch, src_len, attn_dim)
        proj_enc = self.W_a(encoder_outputs)
        # (batch, 1, attn_dim)  →  broadcast over src_len
        proj_dec = self.U_a(decoder_hidden).unsqueeze(1)

        energy   = self.v(torch.tanh(proj_enc + proj_dec)).squeeze(-1)  # (batch, src_len)
        weights  = F.softmax(energy, dim=-1)                             # (batch, src_len)
        context  = (weights.unsqueeze(-1) * encoder_outputs).sum(dim=1) # (batch, enc_h)
        return context, weights


# ─────────────────────────────────────────────
#  3.  Encoder
# ─────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Multi-layer Elman RNN encoder.
    Produces a hidden state at every position: (batch, seq_len, hidden_size).
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int,
        hidden_size: int,
        num_layers:  int,
        dropout:     float,
        pad_idx:     int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_size = embed_dim if i == 0 else hidden_size
            self.cells.append(ElmanCell(in_size, hidden_size))
        self.drop = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

    def forward(self, x: torch.Tensor):
        """
        x : (batch, seq_len)
        returns
            all_h       : (batch, seq_len, hidden_size)   – topmost encoder states
            final_h     : list of (batch, hidden_size)    – last h per layer
        """
        batch_size, seq_len = x.shape
        device  = x.device
        embedded = self.drop(self.embedding(x))              # (batch, seq, embed)

        hidden = [torch.zeros(batch_size, self.hidden_size, device=device)
                  for _ in range(self.num_layers)]

        top_outputs = []
        for t in range(seq_len):
            xt = embedded[:, t, :]
            new_h = []
            for i, cell in enumerate(self.cells):
                ht = cell(xt, hidden[i])
                if i < self.num_layers - 1:
                    ht = self.drop(ht)
                new_h.append(ht)
                xt = ht
            hidden = new_h
            top_outputs.append(hidden[-1])

        all_h = torch.stack(top_outputs, dim=1)              # (batch, seq, hidden)
        return all_h, hidden


# ─────────────────────────────────────────────
#  4.  Decoder with Attention
# ─────────────────────────────────────────────

class AttentionDecoder(nn.Module):
    """
    Single-step decoder that attends to all encoder outputs before updating
    its own hidden state.

    Input to decoder Elman cell = concat(embedding, context_vector).
    """

    def __init__(
        self,
        vocab_size:     int,
        embed_dim:      int,
        hidden_size:    int,
        attn_dim:       int,
        dropout:        float,
        pad_idx:        int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = AdditiveAttention(hidden_size, hidden_size, attn_dim)
        # Input = embed_dim + context (hidden_size)
        self.cell      = ElmanCell(embed_dim + hidden_size, hidden_size)
        self.drop      = nn.Dropout(dropout)
        self.fc_out    = nn.Linear(hidden_size, vocab_size)

    def step(
        self,
        token:           torch.Tensor,   # (batch,)
        decoder_hidden:  torch.Tensor,   # (batch, hidden)
        encoder_outputs: torch.Tensor,   # (batch, src_len, hidden)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One decoder step.

        Returns
        -------
        logits         : (batch, vocab_size)
        decoder_hidden : (batch, hidden_size)   updated
        attn_weights   : (batch, src_len)
        """
        embed = self.drop(self.embedding(token))                     # (batch, embed)
        context, attn_w = self.attention(encoder_outputs, decoder_hidden)
        rnn_input    = torch.cat([embed, context], dim=-1)           # (batch, e+h)
        decoder_hidden = self.cell(rnn_input, decoder_hidden)
        logits         = self.fc_out(decoder_hidden)                 # (batch, V)
        return logits, decoder_hidden, attn_w


# ─────────────────────────────────────────────
#  5.  Full model
# ─────────────────────────────────────────────

class AttentionRNN(nn.Module):
    """
    Character-level name generator: RNN Encoder + Attention + RNN Decoder.
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int   = 32,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        attn_dim:    int   = 64,
        dropout:     float = 0.3,
        pad_idx:     int   = 0,
    ):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_size, num_layers, dropout, pad_idx)
        self.decoder = AttentionDecoder(vocab_size, embed_dim, hidden_size, attn_dim, dropout, pad_idx)
        self.hidden_size = hidden_size

        n_params = self.count_parameters()
        log.info(
            "AttentionRNN built — vocab=%d  embed=%d  hidden=%d  layers=%d  "
            "attn_dim=%d  dropout=%.2f  trainable_params=%s",
            vocab_size, embed_dim, hidden_size, num_layers, attn_dim, dropout, f"{n_params:,}",
        )

    def forward(
        self,
        src:   torch.Tensor,   # (batch, seq_len)   – encoder input
        tgt:   torch.Tensor,   # (batch, seq_len)   – decoder input (teacher forcing)
    ) -> torch.Tensor:
        """
        Returns
        -------
        logits : (batch, seq_len, vocab_size)
        """
        encoder_outputs, encoder_hidden = self.encoder(src)    # (batch, src, hidden)
        decoder_hidden = encoder_hidden[-1]                    # top encoder layer

        batch_size, tgt_len = tgt.shape
        all_logits = []

        for t in range(tgt_len):
            # Causal mask: decoder at step t can only attend to encoder_outputs up to t
            curr_enc_out = encoder_outputs[:, :t+1, :]
            logits, decoder_hidden, _ = self.decoder.step(
                tgt[:, t], decoder_hidden, curr_enc_out
            )
            all_logits.append(logits)

        return torch.stack(all_logits, dim=1)                  # (batch, tgt_len, V)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
#  6.  Greedy / temperature-sampled generation
# ─────────────────────────────────────────────

@torch.no_grad()
def generate_name(
    model:       AttentionRNN,
    vocab,
    max_len:     int   = 30,
    temperature: float = 1.0,
    device:      torch.device = torch.device("cpu"),
) -> str:
    """
    Auto-regressively generate one name.

    The encoder first processes a seed [SOS] token (or a short seed sequence)
    to obtain context vectors, then the decoder samples character by character.
    """
    model.eval()
    log.debug("generate_name (AttentionRNN) — temperature=%.2f  max_len=%d", temperature, max_len)

    # ── Seed encoder with just <SOS> ──────────────────────────
    seed   = torch.tensor([[vocab.SOS_IDX]], device=device)
    enc_out, enc_hidden = model.encoder(seed)
    dec_h  = enc_hidden[-1]

    token  = torch.tensor([vocab.SOS_IDX], device=device)
    generated = []

    for step in range(max_len):
        logits, dec_h, attn_w = model.decoder.step(token, dec_h, enc_out)
        logits = logits / temperature
        probs  = F.softmax(logits, dim=-1)
        nxt    = torch.multinomial(probs, 1).item()

        log.debug("  step=%02d  sampled_idx=%d  attn_max_pos=%d",
                  step, nxt, attn_w.argmax().item())

        if nxt == vocab.EOS_IDX:
            log.debug("  EOS reached at step %d", step)
            break
        generated.append(nxt)
        token = torch.tensor([nxt], device=device)

        # ── Re-encode the full prefix for richer context ──────
        full_prefix = torch.tensor([[vocab.SOS_IDX] + generated], device=device)
        enc_out, _  = model.encoder(full_prefix)

    result = vocab.decode(generated, strip_special=True)
    log.debug("AttentionRNN generated: '%s'", result)
    return result


# ─────────────────────────────────────────────
#  Quick architecture summary (standalone)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    VOCAB_SIZE  = 70
    EMBED_DIM   = 64
    HIDDEN_SIZE = 256
    NUM_LAYERS  = 2
    ATTN_DIM    = 128
    DROPOUT     = 0.3

    model = AttentionRNN(
        vocab_size  = VOCAB_SIZE,
        embed_dim   = EMBED_DIM,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        attn_dim    = ATTN_DIM,
        dropout     = DROPOUT,
    )

    print("=" * 65)
    print("  RNN + ADDITIVE ATTENTION — ARCHITECTURE SUMMARY")
    print("=" * 65)
    print(model)
    print("-" * 65)
    print(f"  Trainable parameters : {model.count_parameters():,}")
    print(f"  Vocab size           : {VOCAB_SIZE}")
    print(f"  Embedding dim        : {EMBED_DIM}")
    print(f"  Encoder hidden size  : {HIDDEN_SIZE}")
    print(f"  Decoder hidden size  : {HIDDEN_SIZE}")
    print(f"  Attention dim        : {ATTN_DIM}")
    print(f"  Encoder num layers   : {NUM_LAYERS}")
    print(f"  Dropout              : {DROPOUT}")
    print("=" * 65)