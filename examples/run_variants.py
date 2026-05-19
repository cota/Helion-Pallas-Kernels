#!/usr/bin/env python3
"""Aggregates and compares benchmarks sequentially."""

from __future__ import annotations
from __future__ import generator_stop

import argparse
import collections
import collections.abc
import dataclasses
import math
import os
import re
import subprocess
import sys

import tabulate

DEFAULT_DEVICE = "v7"
RESULT_RE = re.compile(r"RESULT:(\w+):([\d.]+):([\d.]+)")


@dataclasses.dataclass(frozen=True)
class PerfResult:
  """A parsed performance metric result.

  Attributes:
      perf_str: Formatted string representing performance (e.g., '1.23 ± 0.04
        us').
      mean: Unpacked float mean execution time.
      std: Unpacked float standard deviation.
  """

  perf_str: str
  mean: float | None
  std: float | None


def _get_script_path(benchmark_name: str) -> str:
  """Gets the absolute path of the benchmark script, handling .py extensions."""
  base_script = (
      benchmark_name
      if benchmark_name.endswith(".py")
      else f"{benchmark_name}.py"
  )
  if os.path.exists(base_script):
    return base_script
  return os.path.join(os.path.dirname(os.path.abspath(__file__)), base_script)


def run_benchmark(benchmark_name: str) -> str:
  """Runs a single benchmark script sequentially and returns stdout + stderr.

  Args:
      benchmark_name: Name of the benchmark to execute (e.g., 'matmul_jax').

  Returns:
      The combined stdout and stderr output string of the benchmark script.
  """
  script_path = _get_script_path(benchmark_name)
  cmd = [sys.executable, script_path]
  print(f"  Running {benchmark_name}...")

  try:
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return f"{completed.stdout}\n{completed.stderr}"
  except subprocess.CalledProcessError as e:
    print(
        f"ERROR: Benchmark {benchmark_name} failed with exit code"
        f" {e.returncode}",
        file=sys.stderr,
    )
    print(f"Stdout:\n{e.stdout}", file=sys.stderr)
    print(f"Stderr:\n{e.stderr}", file=sys.stderr)
    return f"{e.stdout}\n{e.stderr}"


def parse_results(
    *,
    benchmark_name: str,
    stdout_content: str,
    result_by_label_by_benchmark: collections.abc.MutableMapping[
        str, collections.abc.MutableMapping[str, tuple[float, float]]
    ],
) -> None:
  """Parses RESULT lines from a benchmark run and populates results database.

  Args:
      benchmark_name: Name of the benchmark variant.
      stdout_content: Standard output string containing RESULT records.
      result_by_label_by_benchmark: Database mapping labels to benchmarks.
  """
  for line in stdout_content.splitlines():
    match = RESULT_RE.search(line)
    if match:
      label, mean, std = match.groups()
      result_by_label_by_benchmark[f"{DEFAULT_DEVICE}_{label}"][
          benchmark_name
      ] = (
          float(mean),
          float(std),
      )


def extract_perf(res: tuple[float, float] | None) -> PerfResult:
  """Unpacks the mean and std deviation tuple into a PerfResult object."""
  if res is not None:
    mean, std = res
    return PerfResult(perf_str=f"{mean:.2f} ± {std:.2f} us", mean=mean, std=std)
  return PerfResult(perf_str="FAILED", mean=None, std=None)


def _calculate_speedup(
    *,
    b_mean: float | None,
    b_std: float | None,
    o_mean: float | None,
    o_std: float | None,
) -> str:
  """Computes the relative speedup and combined error margin between results.

  Args:
      b_mean: Baseline mean execution time.
      b_std: Baseline standard deviation.
      o_mean: Variant mean execution time.
      o_std: Variant standard deviation.

  Returns:
      Formatted speedup string (e.g. '1.23 ± 0.04x') or 'N/A'.
  """
  if (
      b_mean is not None
      and b_std is not None
      and o_mean is not None
      and o_std is not None
      and o_mean > 0
      and b_mean > 0
  ):
    s = b_mean / o_mean
    s_std = s * math.sqrt((b_std / b_mean) ** 2 + (o_std / o_mean) ** 2)
    return f"{s:.2f} ± {s_std:.2f}x"
  return "N/A"


def _build_row(
    label: str,
    baseline: str,
    remaining_benchmarks: collections.abc.Sequence[str],
    result_db: collections.abc.MutableMapping[
        str, collections.abc.Mapping[str, tuple[float, float]]
    ],
) -> list[str]:
  """Builds a tabular comparison row for a specific label prefix."""
  b_perf = extract_perf(result_db[label].get(baseline))
  row = [label, b_perf.perf_str]

  for b in remaining_benchmarks:
    o_perf = extract_perf(result_db[label].get(b))
    speedup_str = _calculate_speedup(
        b_mean=b_perf.mean,
        b_std=b_perf.std,
        o_mean=o_perf.mean,
        o_std=o_perf.std,
    )
    row.extend([o_perf.perf_str, speedup_str])

  return row


def main() -> None:
  """Consolidates and compares benchmark results executed sequentially."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "benchmarks",
      nargs="+",
      help=(
          "Names of the benchmarks to run (e.g., 'matmul_jax',"
          " 'matmul_pallas')."
      ),
  )
  args = parser.parse_args()

  benchmark_names = args.benchmarks

  print("Launching sequential benchmarks...")
  result_by_label_by_benchmark = collections.defaultdict(dict)

  for name in benchmark_names:
    output = run_benchmark(name)
    parse_results(
        benchmark_name=name,
        stdout_content=output,
        result_by_label_by_benchmark=result_by_label_by_benchmark,
    )

  total_labels = result_by_label_by_benchmark.keys()
  if not total_labels:
    print("!!! NO RESULTS FOUND !!!", file=sys.stderr)
    sys.exit(1)

  labels = sorted(total_labels)
  baseline, *remaining_benchmarks = benchmark_names

  remaining_headers = ["Label", f"{baseline} (baseline)"]
  for b in remaining_benchmarks:
    remaining_headers.extend([f"{b} Perf", "Speedup"])
  header = remaining_headers

  table = [
      _build_row(
          label, baseline, remaining_benchmarks, result_by_label_by_benchmark
      )
      for label in labels
  ]

  print("\nGenerating report...\n")
  print(tabulate.tabulate(table, headers=header))


if __name__ == "__main__":
  main()
