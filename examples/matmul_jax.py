import functools
import sys
from absl import app
import jax
import jax.numpy as jnp

import matmul_bench


@functools.partial(jax.jit, static_argnames=["bm", "bk", "bn"])
def jax_matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
) -> jax.Array:
  # Ignore blocking kwargs for pure JAX
  return jnp.matmul(x, y)


def main(argv: list[str]) -> None:
  matmul_bench.run(jax_matmul)


if __name__ == "__main__":
  app.run(main)
