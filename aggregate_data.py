"""
Aggregate multiple protein-property datasets into a unified multi-task corpus.

The output is a normalized dataset on disk with a consistent schema:
sequence, structure, structure_format, task, label, label_type, source, split.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import concatenate_datasets, load_dataset


@dataclass(frozen=True)
class TaskSpec:
  # Dataset configuration for one property task.
  name: str
  dataset: str
  split: Iterable[str]
  sequence_col: str
  label_col: Optional[str]
  label_type: str
  structure_col: Optional[str] = None
  structure_format: Optional[str] = None
  subset: Optional[str] = None


TASKS: List[TaskSpec] = [
  # Example task. Extend this list with additional properties/datasets.
  TaskSpec(
    name="solubility",
    dataset="AI4Protein/DeepSol",
    split=("train", "validation", "test"),
    sequence_col="aa_seq",
    label_col="label",
    label_type="binary",
    structure_col=None,
    structure_format=None,
    subset=None,
  ),
]


def _normalize_example(example: Dict[str, Any], task: TaskSpec, split: str) -> Dict[str, Any]:
  # Map a dataset-specific record into the unified schema.
  sequence = example.get(task.sequence_col)
  label = example.get(task.label_col) if task.label_col else None
  structure = example.get(task.structure_col) if task.structure_col else None
  return {
    "sequence": sequence,
    "structure": structure,
    "structure_format": task.structure_format,
    "task": task.name,
    "label": label,
    "label_type": task.label_type,
    "source": task.dataset if task.subset is None else f"{task.dataset}:{task.subset}",
    "split": split,
  }


def _load_and_map(task: TaskSpec, split: str, cache_dir: Optional[str]):
  # Load one split and normalize columns to the unified schema.
  ds = load_dataset(task.dataset, task.subset, split=split, cache_dir=cache_dir)
  return ds.map(
    lambda ex: _normalize_example(ex, task, split),
    remove_columns=ds.column_names,
    desc=f"Normalize {task.name}:{split}",
  )


def aggregate(tasks: List[TaskSpec], out_dir: Path, cache_dir: Optional[str]):
  # Build a single concatenated dataset from all tasks/splits.
  out_dir.mkdir(parents=True, exist_ok=True)

  mapped_parts = []
  for task in tasks:
    for split in task.split:
      # Load + normalize per-task split before concatenation.
      mapped_parts.append(_load_and_map(task, split, cache_dir))

  if not mapped_parts:
    raise ValueError("No tasks/splits to aggregate.")

  combined = concatenate_datasets(mapped_parts)
  combined.save_to_disk(out_dir.as_posix())

  # Save metadata for reproducibility and downstream inspection.
  manifest = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "num_rows": len(combined),
    "tasks": [task.__dict__ for task in tasks],
  }
  (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _parse_args():
  # CLI arguments to set output and cache paths.
  parser = argparse.ArgumentParser(description="Aggregate multiple datasets into a unified multi-task corpus.")
  parser.add_argument("--out-dir", default="data/aggregated", help="Output directory for the combined dataset.")
  parser.add_argument("--cache-dir", default=None, help="Optional HuggingFace datasets cache directory.")
  return parser.parse_args()


def main():
  # Entrypoint for CLI usage.
  args = _parse_args()
  aggregate(TASKS, Path(args.out_dir), args.cache_dir)


if __name__ == "__main__":
  main()
