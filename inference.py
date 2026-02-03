"""
Embedding-only inference example using ProstT5.
Loads a PDB structure, derives an AA sequence, and produces embeddings.
"""

import re
from pathlib import Path

import torch
from transformers import T5EncoderModel, T5Tokenizer

# Select CUDA if available, otherwise fall back to CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)

# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)

# Only GPUs support half-precision; on CPU keep full precision.
model.float() if device.type == "cpu" else model.half()

# Standard three-letter residue codes to one-letter sequence.
THREE_TO_ONE = {
  "ALA": "A",
  "ARG": "R",
  "ASN": "N",
  "ASP": "D",
  "CYS": "C",
  "GLN": "Q",
  "GLU": "E",
  "GLY": "G",
  "HIS": "H",
  "ILE": "I",
  "LEU": "L",
  "LYS": "K",
  "MET": "M",
  "PHE": "F",
  "PRO": "P",
  "SER": "S",
  "THR": "T",
  "TRP": "W",
  "TYR": "Y",
  "VAL": "V",
  "SEC": "U",
  "PYL": "O",
}


def _seq_from_seqres(lines):
  # Use SEQRES if present, since it represents the intended full sequence.
  residues = []
  for line in lines:
    if not line.startswith("SEQRES"):
      continue
    parts = line.split()
    for res in parts[4:]:
      residues.append(THREE_TO_ONE.get(res.upper(), "X"))
  return "".join(residues)


def _seq_from_atom(lines):
  # Fall back to ATOM records, using CA atoms to infer residue order.
  residues = []
  seen = set()
  for line in lines:
    if not line.startswith("ATOM"):
      continue
    atom_name = line[12:16].strip()
    if atom_name != "CA":
      continue
    resname = line[17:20].strip().upper()
    chain_id = line[21].strip()
    resseq = line[22:26].strip()
    icode = line[26].strip()
    key = (chain_id, resseq, icode)
    if key in seen:
      continue
    seen.add(key)
    residues.append(THREE_TO_ONE.get(resname, "X"))
  return "".join(residues)


def load_sequence_from_pdb(pdb_path: Path) -> str:
  # Read the PDB and extract sequence from SEQRES or ATOM.
  lines = pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines()
  seq = _seq_from_seqres(lines)
  if not seq:
    seq = _seq_from_atom(lines)
  if not seq:
    raise ValueError(f"Unable to derive sequence from PDB: {pdb_path}")
  return seq


def preprocess_sequence(seq: str) -> str:
  # ProstT5 expects AA tokens separated by spaces and prefixed with <AA2fold>.
  seq = re.sub(r"[UZOB]", "X", seq.upper())
  spaced = " ".join(list(seq))
  return "<AA2fold> " + spaced

pdb_path = Path("examples/P62593_Beta_lactamase.pdb")
# Load example PDB and derive its AA sequence.
sequence = load_sequence_from_pdb(pdb_path)
sequence_examples = [preprocess_sequence(sequence)]

# Tokenize sequences and pad up to the longest sequence in the batch.
ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest", return_tensors="pt").to(device)

# Generate embeddings with the frozen encoder.
with torch.no_grad():
  embedding_repr = model(ids.input_ids, attention_mask=ids.attention_mask)

# Extract residue embeddings for the first sequence in the batch and remove prefix tokens.
emb_0 = embedding_repr.last_hidden_state[0, 1:]  # shape (L x 1024)
# If you want a single per-protein embedding, mean-pool residues.
emb_0_per_protein = emb_0.mean(dim=0)  # shape (1024)
