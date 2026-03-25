"""
Handles loading, preprocessing, and encoding of the TrainingNames.txt dataset
for character-level name generation.
"""

import torch
from torch.utils.data import Dataset, DataLoader

from logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────
#  1.  Load raw names
# ─────────────────────────────────────────────

def load_names(filepath: str) -> list[str]:
    """Read one name per line; strip whitespace; drop empty lines."""
    log.info("Loading names from '%s'", filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    log.info("Loaded %d names", len(names))
    log.debug("First 5 names: %s", names[:5])
    log.debug("Last  5 names: %s", names[-5:])
    return names


# ─────────────────────────────────────────────
#  2.  Vocabulary
# ─────────────────────────────────────────────

class Vocabulary:
    """
    Builds a character-level vocabulary from a list of names.

    Special tokens
    --------------
    <PAD> : 0   – used for padding sequences in a batch
    <SOS> : 1   – start-of-sequence marker
    <EOS> : 2   – end-of-sequence marker
    """

    PAD, SOS, EOS = "<PAD>", "<SOS>", "<EOS>"
    PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2

    def __init__(self, names: list[str]):
        log.info("Building character vocabulary …")
        chars      = sorted(set("".join(names)))
        specials   = [self.PAD, self.SOS, self.EOS]
        all_tokens = specials + chars

        self.char2idx: dict[str, int] = {c: i for i, c in enumerate(all_tokens)}
        self.idx2char: dict[int, str] = {i: c for c, i in self.char2idx.items()}
        self.vocab_size: int = len(all_tokens)

        log.info("Vocabulary built: size=%d  (3 special + %d chars)", self.vocab_size, len(chars))
        log.debug("All characters : %s", chars)

    def encode(self, name: str) -> list[int]:
        """Name → [SOS, c1, c2, …, EOS]"""
        encoded = (
            [self.SOS_IDX]
            + [self.char2idx[c] for c in name if c in self.char2idx]
            + [self.EOS_IDX]
        )
        log.debug("encode('%s') → length %d", name, len(encoded))
        return encoded

    def decode(self, indices: list[int], strip_special: bool = True) -> str:
        """Integer indices → string"""
        skip = {self.PAD_IDX, self.SOS_IDX, self.EOS_IDX} if strip_special else set()
        result = "".join(
            self.idx2char[i] for i in indices
            if i in self.idx2char and i not in skip
        )
        log.debug("decode(%s…) → '%s'", indices[:6], result)
        return result


# ─────────────────────────────────────────────
#  3.  PyTorch Dataset
# ─────────────────────────────────────────────

class NamesDataset(Dataset):
    """
    Returns (input_seq, target_seq) pairs for teacher-forcing.
      input  = [SOS, c1, c2, …, cN]
      target = [c1,  c2, …, cN, EOS]
    """

    def __init__(self, names: list[str], vocab: Vocabulary):
        log.info("Building NamesDataset from %d names …", len(names))
        self.vocab   = vocab
        self.encoded = [vocab.encode(n) for n in names]
        lengths      = [len(e) for e in self.encoded]
        log.info(
            "NamesDataset ready — seq lengths: min=%d  max=%d  avg=%.1f",
            min(lengths), max(lengths), sum(lengths) / len(lengths),
        )

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int):
        seq = self.encoded[idx]
        return (
            torch.tensor(seq[:-1], dtype=torch.long),
            torch.tensor(seq[1:],  dtype=torch.long),
        )


# ─────────────────────────────────────────────
#  4.  Collate: variable-length → padded batch
# ─────────────────────────────────────────────

def collate_fn(batch, pad_idx: int = Vocabulary.PAD_IDX):
    """Pad sequences in a batch to the same length."""
    inputs, targets = zip(*batch)
    inputs  = torch.nn.utils.rnn.pad_sequence(inputs,  batch_first=True, padding_value=pad_idx)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_idx)
    return inputs, targets


# ─────────────────────────────────────────────
#  5.  Convenience builder
# ─────────────────────────────────────────────

def build_dataloader(
    filepath:   str,
    batch_size: int  = 64,
    shuffle:    bool = True,
) -> tuple[DataLoader, Vocabulary, list[str]]:
    """
    One-stop function used by training scripts.

    Returns
    -------
    loader  : DataLoader
    vocab   : Vocabulary
    names   : raw name list (needed for Novelty computation)
    """
    log.info(
        "build_dataloader — filepath='%s'  batch_size=%d  shuffle=%s",
        filepath, batch_size, shuffle,
    )
    names   = load_names(filepath)
    vocab   = Vocabulary(names)
    dataset = NamesDataset(names, vocab)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, pad_idx=vocab.PAD_IDX),
    )
    log.info("DataLoader ready: %d batches of size %d", len(loader), batch_size)
    return loader, vocab, names


# ─────────────────────────────────────────────
#  6.  Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from logger import init_logging
    init_logging("data_utils_check")

    loader, vocab, names = build_dataloader("TrainingNames.txt", batch_size=4)
    inp, tgt = next(iter(loader))
    log.info("Sample batch — input shape=%s  target shape=%s", tuple(inp.shape), tuple(tgt.shape))

    sample = names[0]
    enc    = vocab.encode(sample)
    dec    = vocab.decode(enc)
    log.info("Round-trip: original='%s'  decoded='%s'  match=%s", sample, dec, sample == dec)