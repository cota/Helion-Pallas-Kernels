import os
import sys
import timeit

from absl import app
import numpy as np
import torch

from matmul_configs import BLOCK_CONFIGS, DTYPES, SHAPES

# Force full autotuning effort to explore alternative block sizes and compilation parameters
os.environ["HELION_AUTOTUNE_EFFORT"] = "full"

import helion
from helion.autotuner.benchmarking import synchronize_device
import helion.language as hl
from helion.runtime.settings import _get_backend, is_pallas_interpret

if _get_backend() == "pallas" and is_pallas_interpret():
  DEVICE = torch.device("cpu")
else:
  DEVICE = torch.device("tpu")


_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
dtypes = [_TORCH_DTYPES[d] for d in DTYPES]


@helion.kernel(backend="pallas", static_shapes=True)
def helion_matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  m, k = x.size()
  _, n = y.size()
  out = torch.empty(
      [m, n], device=x.device, dtype=torch.promote_types(x.dtype, y.dtype)
  )
  for tile_m, tile_n in hl.tile([m, n]):
    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
    for tile_k in hl.tile(k):
      acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
    out[tile_m, tile_n] = acc
  return out


def run_helion_benchmarks() -> None:

  for dtype in dtypes:
    dtype_name = "bfloat16" if dtype == torch.bfloat16 else "float32"
    for shape in SHAPES:
      m, k, n = shape

      # Deterministic input generation
      torch.manual_seed(0)
      x = torch.randn((m, k), dtype=dtype, device=DEVICE)
      torch.manual_seed(1)
      y = torch.randn((k, n), dtype=dtype, device=DEVICE)

      # Baseline reference computed in float32 for higher precision accumulation
      expected = (x.float() @ y.float()).cpu().numpy()

      # Hoist full autotuning search outside the block configs loop to prevent redundant evaluations
      try:
        bound = helion_matmul_kernel.bind((x, y))
        best_config = bound.autotune((x, y), force=True)
        print(
            f"Optimal autotuned config for {shape}: {best_config}",
            file=sys.stderr,
        )
        compiled_fn = bound.compile_config(best_config, allow_print=False)
        actual_t = compiled_fn(x, y)
        synchronize_device(actual_t)
        actual = actual_t.float().cpu().numpy()

        rtol = 5e-2 if dtype_name == "bfloat16" else (5e-2 if n == 1 else 1e-3)
        atol = 5e-2 if dtype_name == "bfloat16" else (5e-2 if n == 1 else 1e-3)
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        is_correct = True
        err_msg = None
      except Exception as e:
        is_correct = False
        err_msg = str(e)

      if not is_correct:
        # If accuracy or compilation fails for the shape, print failure records for all block suffixes
        for bm, bk, bn in BLOCK_CONFIGS:
          print(
              f"FAILED correctness or compilation for {shape} {bm}x{bk}x{bn}"
              f" {dtype_name}: {err_msg}",
              file=sys.stderr,
          )
        continue

      def _run():
        out = compiled_fn(x, y)
        synchronize_device(out)

      # Warm up once per shape
      _run()

      n_iter = 20
      n_repeats = 5
      samples = (
          np.array(timeit.repeat(_run, repeat=n_repeats, number=n_iter))
          / n_iter
      )
      mean = np.mean(samples) * 1e6
      std = np.std(samples) * 1e6

      # Report the exact same optimized performance metrics using expected block size suffixes
      for bm, bk, bn in BLOCK_CONFIGS:
        print(f"RESULT:{dtype_name}_{m}x{k}x{n}_{bm}x{bk}x{bn}:{mean}:{std}")


def main(argv: list[str]) -> None:
  del argv
  run_helion_benchmarks()


if __name__ == "__main__":
  app.run(main)
