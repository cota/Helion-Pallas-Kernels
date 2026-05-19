import sys
import timeit

import jax
import jax.numpy as jnp
import numpy as np

from matmul_configs import BLOCK_CONFIGS, DTYPES, SHAPES

_JAX_DTYPES = {
    "bfloat16": jnp.bfloat16,
    "float32": jnp.float32,
}
dtypes = [_JAX_DTYPES[d] for d in DTYPES]


def run(matmul_fn):
  for dtype in dtypes:
    for shape in SHAPES:
      m, k, n = shape

      x = jax.random.normal(jax.random.key(0), (m, k), dtype=dtype)
      y = jax.random.normal(jax.random.key(1), (k, n), dtype=dtype)
      expected = jnp.dot(x, y, preferred_element_type=jnp.float32)

      for bm, bk, bn in BLOCK_CONFIGS:

        # Correctness check
        try:
          actual = matmul_fn(x, y, bm=bm, bk=bk, bn=bn).block_until_ready()
          rtol = 5e-2 if dtype == jnp.bfloat16 else (5e-2 if n == 1 else 1e-3)
          atol = 5e-2 if dtype == jnp.bfloat16 else (5e-2 if n == 1 else 1e-3)
          np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        except Exception as e:
          print(
              f"FAILED correctness or compilation for {shape} {bm}x{bk}x{bn}"
              f" {dtype}: {e}",
              file=sys.stderr,
          )
          continue

        def _run():
          matmul_fn(x, y, bm=bm, bk=bk, bn=bn).block_until_ready()

        # Warm up.
        _run()

        n_iter = 20
        n_repeats = 5
        samples = (
            np.array(timeit.repeat(_run, repeat=n_repeats, number=n_iter))
            / n_iter
        )
        mean = np.mean(samples) * 1e6
        std = np.std(samples) * 1e6

        print(
            f"RESULT:{np.dtype(dtype).name}_{m}x{k}x{n}_{bm}x{bk}x{bn}:{mean}:{std}"
        )
