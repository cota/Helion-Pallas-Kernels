#!/usr/bin/env python3
"""Filters benchmark results from stdin to show the best configuration per prefix.

This script reads benchmark output (e.g., from run_variants.py) from standard
input.
For each unique benchmark prefix (e.g., "v7_bfloat16_1024x1024x1024"), it
selects the
best (fastest) baseline execution time and the best (fastest) execution time for
each
benchmark variant across all tiling/subblocking configurations.

It then recomputes the speedup and error margins for each variant against the
selected
canonical baseline and formats a consolidated summary table.

Usage:
  ./run_benchmarks.sh ... | python3 filter_best_speedups.py
"""

from __future__ import annotations
from __future__ import generator_stop

import collections
import collections.abc
import dataclasses
import itertools
import math
import re
import sys

COLUMN_SPLIT_RE = re.compile(r"\s{2,}")
CELL_METRIC_RE = re.compile(
    r"(?P<mean>[\d.]+)\s*±\s*(?P<std>[\d.]+)\s*(?P<unit>[a-zA-Z%]+)?"
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ParsedCell:
  """A parsed table cell containing performance data or status.

  Attributes:
      raw: The raw cell string.
      mean: The extracted float mean, or None if FAILED/invalid.
      std: The extracted float standard deviation, or None if invalid.
      unit: The extracted measurement unit (e.g., 'us' or 'x').
  """
  raw: str
  mean: float | None
  std: float | None
  unit: str


@dataclasses.dataclass(frozen=True, kw_only=True)
class ParsedRow:
  """A parsed benchmark data row.

  Attributes:
      line: The raw input line.
      prefix: The problem prefix extracted from the label.
      cells: Sequence of parsed table cells for this row.
  """
  line: str
  prefix: str
  cells: collections.abc.Sequence[ParsedCell]

  @property
  def baseline(self) -> ParsedCell:
    """The baseline cell (column index 1)."""
    return self.cells[1]

  def get_variant(self, col_idx: int) -> ParsedCell:
    """Returns the variant performance cell at the given column index."""
    return self.cells[col_idx]


def parse_cell(cell_str: str) -> ParsedCell:
  """Parses a benchmark metric string like '126.85 ± 4.24 us' into ParsedCell."""
  cell_str = cell_str.strip()
  match = CELL_METRIC_RE.fullmatch(cell_str)
  if match:
    return ParsedCell(
        raw=cell_str,
        mean=float(match["mean"]),
        std=float(match["std"]),
        unit=match["unit"] or "",
    )
  return ParsedCell(raw=cell_str, mean=None, std=None, unit="")


def format_variant_summary(
    group: collections.abc.Sequence[ParsedRow],
    variant_col_idx: int,
    best_baseline: ParsedCell | None,
) -> tuple[str, str]:
  """Calculates peak variant performance and speedup across a benchmark group.

  Args:
      group: Sequence of rows sharing a common problem prefix.
      variant_col_idx: Column index containing variant performance measurements.
      best_baseline: The canonical baseline cell selected for this group.

  Returns:
      A tuple of (formatted_perf_string, formatted_speedup_string).
  """
  best_var = min(
      (
          r.get_variant(variant_col_idx)
          for r in group
          if len(r.cells) > variant_col_idx
          and r.get_variant(variant_col_idx).mean is not None
      ),
      key=lambda x: x.mean if x.mean is not None else math.inf,
      default=None,
  )
  perf_str = (
      f"{best_var.mean:.2f} ± {best_var.std:.2f} {best_var.unit}".strip()
      if best_var is not None and best_var.mean is not None
      else "FAILED"
  )

  if (
      best_baseline is not None
      and best_baseline.mean is not None
      and best_baseline.mean > 0
      and best_var is not None
      and best_var.mean is not None
      and best_var.mean > 0
  ):
    s = best_baseline.mean / best_var.mean
    s_std = s * math.sqrt(
        (best_baseline.std / best_baseline.mean) ** 2
        + (best_var.std / best_var.mean) ** 2
    )
    speedup_str = f"{s:.2f} ± {s_std:.2f}x"
  else:
    speedup_str = "N/A"

  return perf_str, speedup_str


def main(argv: collections.abc.Sequence[str]) -> None:
  """Consolidates benchmark results from standard input by prefix.

  Reads lines from standard input, groups benchmark data rows by problem prefix,
  selects the absolute minimum execution time for each column variant, and
  recomputes speedups against the canonical baseline.

  Args:
      argv: Command-line arguments passed to the script (including script name).
  """
  if len(argv) > 1:
    sys.exit(
        "Usage: filter_best_speedups.py\n"
        "This script takes no command-line arguments."
    )

  header = []
  footer = []
  parsed_rows: list[ParsedRow] = []

  for line in sys.stdin:
    if not line.strip().startswith(("v7_", "v6_", "v5_")):
      if not parsed_rows:
        header.append(line)
      else:
        footer.append(line)
      continue

    cols = COLUMN_SPLIT_RE.split(line.strip())
    label, *_ = cols

    label_parts = label.split("_")
    prefix = "_".join(label_parts[:-1]) if len(label_parts) >= 4 else label

    cells = [parse_cell(c) for c in cols]
    parsed_rows.append(ParsedRow(line=line, prefix=prefix, cells=cells))

  sep_line = next(
      (h.strip() for h in reversed(header) if h.strip().startswith("---")),
      None,
  )
  col_widths = (
      [len(p) for p in COLUMN_SPLIT_RE.split(sep_line)] if sep_line else []
  )

  rows_by_prefix = collections.defaultdict(list)
  for pr in parsed_rows:
    rows_by_prefix[pr.prefix].append(pr)

  summary_lines = []
  for prefix, group in sorted(rows_by_prefix.items()):
    best_baseline = min(
        (
            r.baseline
            for r in group
            if len(r.cells) > 1 and r.baseline.mean is not None
        ),
        key=lambda x: x.mean if x.mean is not None else math.inf,
        default=None,
    )

    if best_baseline is not None and best_baseline.mean is not None:
      baseline_perf = (
          f"{best_baseline.mean:.2f} ± {best_baseline.std:.2f}"
          f" {best_baseline.unit}"
      ).strip()
    else:
      baseline_perf = "FAILED"

    max_cols = max((len(r.cells) for r in group), default=0)
    out_cols = [prefix, baseline_perf] + list(
        itertools.chain.from_iterable(
            format_variant_summary(group, variant_col_idx, best_baseline)
            for variant_col_idx in range(2, max_cols, 2)
        )
    )

    # Align columns to match table formatting using zip
    # to avoid magic index lookups.
    padded_widths = col_widths + [len(c) for c in out_cols[len(col_widths) :]]
    summary_lines.append(
        "  ".join(
            col_str.ljust(width)
            for col_str, width in zip(out_cols, padded_widths)
        )
        + "\n"
    )

  for h in header:
    print(h, end="")

  for sl in summary_lines:
    print(sl, end="")

  for f in footer:
    print(f, end="")


if __name__ == "__main__":
  main(sys.argv)
