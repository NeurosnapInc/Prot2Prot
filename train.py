"""
Single-task training script for a frozen ProstT5 encoder with a lightweight adapter
and classifier head. This is the baseline before multi-task extensions.
"""

import re
from collections import Counter

import torch
import torch.nn as nn
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer, get_linear_schedule_with_warmup

# =====================
# Config
# =====================
MODEL_NAME = "Rostlab/ProstT5"  # ProstT5 encoder
AGGREGATED_DATA_DIR = "data/aggregated"
TASK_NAME = "solubility"
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 10  # was 3
MAX_LENGTH = 1024 * 2  # was 1024
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu") #TODO TMP


# =====================
# Adapter module
# =====================
class Adapter(nn.Module):
  def __init__(self, input_dim, adapter_dim=64, dropout_prob=0.1):
    super().__init__()
    self.norm = nn.LayerNorm(input_dim)
    self.down_project = nn.Linear(input_dim, adapter_dim)
    self.activation = nn.GELU()
    self.up_project = nn.Linear(adapter_dim, input_dim)
    self.dropout = nn.Dropout(dropout_prob)
    self.scale = nn.Parameter(torch.tensor(1e-3))
    nn.init.normal_(self.down_project.weight, std=1e-3)
    nn.init.normal_(self.up_project.weight, std=1e-3)
    nn.init.zeros_(self.up_project.bias)

  def forward(self, x):
    x_norm = self.norm(x)
    down = self.down_project(x_norm)
    activated = self.activation(down)
    up = self.up_project(activated)
    dropped = self.dropout(up)
    return self.scale * dropped


class AttnPool(nn.Module):
  def __init__(self, d_model, hidden=256, dropout=0.1):
    super().__init__()
    self.proj = nn.Sequential(
      nn.Linear(d_model, hidden),
      nn.GELU(),
      nn.Dropout(dropout),
    )
    self.context = nn.Linear(hidden, 1, bias=False)

  def forward(self, x, mask):
    # x: [B,L,d], mask: [B,L] (1 for tokens to keep)
    h = self.proj(x)  # [B,L,h]
    scores = self.context(h).squeeze(-1)  # [B,L]
    scores = scores.masked_fill(mask == 0, -1e9)
    attn = torch.softmax(scores, dim=1)  # [B,L]
    pooled = torch.bmm(attn.unsqueeze(1), x).squeeze(1)  # [B,d]
    return pooled


# =====================
# Solubility prediction model with ProstT5 + Adapter
# =====================
class SolubilityAdapterModel(nn.Module):
  def __init__(self, base_model, embed_dim, num_classes=2, adapter_dim=64, dropout=0.1):
    super().__init__()
    self.base = base_model
    for p in self.base.parameters():
      p.requires_grad = False
    self.adapter = Adapter(embed_dim, adapter_dim, dropout_prob=dropout)
    self.pool = AttnPool(embed_dim, hidden=256, dropout=dropout)
    self.classifier = nn.Sequential(
      nn.LayerNorm(embed_dim),
      nn.Linear(embed_dim, num_classes),
    )

  def forward(self, input_ids, attention_mask):
    if input_ids.dtype != torch.long:
      input_ids = input_ids.long()
    if attention_mask.dtype not in (torch.long, torch.int64, torch.bool):
      attention_mask = attention_mask.long()

    # Frozen ProstT5 in fp16; AMP with new API
    autocast_enabled = input_ids.is_cuda
    with torch.amp.autocast("cuda", enabled=autocast_enabled):
      out = self.base(input_ids=input_ids, attention_mask=attention_mask)
      token_repr = out.last_hidden_state  # [B,L,d] (fp16)

    token_repr = token_repr.float()
    adapted = token_repr + self.adapter(token_repr)
    pooled = self.pool(adapted, attention_mask)
    logits = self.classifier(pooled)
    return logits


# =====================
# Preprocess function
# =====================
def preprocess_sequence(seq):
  seq = re.sub(r"[UZOB]", "X", seq.upper())
  spaced = " ".join(list(seq))
  return "<AA2fold> " + spaced  # <-- ProstT5 expects this for AA inputs


def collate_fn(batch, tokenizer):
  sequences = [preprocess_sequence(item["sequence"]) for item in batch]
  labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
  enc = tokenizer(sequences, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
  return enc["input_ids"], enc["attention_mask"], labels


# =====================
# Load data
# =====================
print("Loading data")
dataset = load_from_disk(AGGREGATED_DATA_DIR)
dataset = dataset.filter(lambda x: x["task"] == TASK_NAME)
dataset = {
  "train": dataset.filter(lambda x: x["split"] == "train"),
  "validation": dataset.filter(lambda x: x["split"] == "validation"),
  "test": dataset.filter(lambda x: x["split"] == "test"),
}
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
base_model = T5EncoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
if DEVICE.type == "cuda":
  base_model.half()

# In the HuggingFace dataset, assume: train, validation, test splits
train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer))
val_loader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer))

# class label balancing
labels_train = [int(x) for x in dataset["train"]["label"]]
counts = Counter(labels_train)
num0, num1 = counts.get(0, 0), counts.get(1, 0)
total = num0 + num1
# inverse-frequency weights
w0 = total / (2.0 * max(1, num0))
w1 = total / (2.0 * max(1, num1))
class_weights = torch.tensor([w0, w1], dtype=torch.float, device=DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# =====================
# Init model
# =====================
print("Initializing model")
embed_dim = base_model.config.d_model
# keep adapter+head in fp32 for stability
model = SolubilityAdapterModel(base_model, embed_dim, num_classes=2, adapter_dim=64).to(DEVICE)
model.float()  # ensures adapter/classifier are fp32


optimizer = torch.optim.AdamW([{"params": model.adapter.parameters()}, {"params": model.classifier.parameters()}], lr=LR, weight_decay=1e-2)

num_training_steps = len(train_loader) * EPOCHS
num_warmup_steps = int(0.05 * num_training_steps)  # 5% warmup
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

best_f1, patience, best_state = 0.0, 3, None
stale = 0

for epoch in range(EPOCHS):
  model.train()
  total_loss = 0.0
  for input_ids, attn_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
    input_ids, attn_mask, labels = input_ids.to(DEVICE), attn_mask.to(DEVICE), labels.to(DEVICE)

    logits = model(input_ids, attn_mask)
    loss = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    total_loss += loss.item()

  # validation
  model.eval()
  all_preds, all_labels = [], []
  with torch.no_grad():
    for input_ids, attn_mask, labels in val_loader:
      input_ids, attn_mask, labels = input_ids.to(DEVICE), attn_mask.to(DEVICE), labels.to(DEVICE)
      preds = model(input_ids, attn_mask).argmax(dim=1)
      all_preds.extend(preds.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

  acc = accuracy_score(all_labels, all_preds)
  f1 = f1_score(all_labels, all_preds)
  print(f"Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f} F1: {f1:.4f}")

  # early stopping on F1
  if f1 > best_f1:
    best_f1 = f1
    stale = 0
    best_state = {
      "adapter": {k: v.cpu() for k, v in model.adapter.state_dict().items()},
      "classifier": {k: v.cpu() for k, v in model.classifier.state_dict().items()},
    }
  else:
    stale += 1
    if stale >= patience:
      print("Early stopping.")
      break

# =====================
# Save model
# =====================
# restore best and save
if best_state is not None:
  model.adapter.load_state_dict(best_state["adapter"])
  model.classifier.load_state_dict(best_state["classifier"])

torch.save(
  {
    "adapter_state_dict": model.adapter.state_dict(),
    "classifier_state_dict": model.classifier.state_dict(),
    "config": {
      "embed_dim": embed_dim,
      "num_classes": 2,
      "adapter_dim": 64,
      "model_name": MODEL_NAME,
    },
  },
  "./deepsol_prostt5_adapter_best.pt",
)
print("Saved best adapter+head → ./deepsol_prostt5_adapter_best.pt")
