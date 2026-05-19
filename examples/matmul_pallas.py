import functools
import sys
from typing import Any

from absl import app
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

import matmul_bench


def _mask_tile(
    val: jax.Array, rem: int, dim: int, is_last: jax.Array
) -> jax.Array:
  if rem > 0:
    mask = jax.lax.broadcasted_iota(jnp.int32, val.shape, dim) < rem
    return jnp.where(is_last, jnp.where(mask, val, 0.0), val)
  return val


def outer_kernel(
    x_ref: Any,
    y_ref: Any,
    z_ref: Any,
    *,
    bm: int,
    bn: int,
) -> None:
  x_val = x_ref[...]
  y_val = y_ref[...]
  # Outer product of vectors has just one multiplication per cell and
  # therefore bf16 is accurate here. Note that older TPUs might not support
  # bf16 VPU ops natively, and would need to upcast to f32.
  z_ref[...] = (x_val * y_val).astype(z_ref.dtype)


def matvec_kernel(
    x_ref: Any,
    y_ref: Any,
    z_ref: Any,
    acc_ref: Any,
    *,
    nsteps: int,
    k_rem: int,
    bm: int,
    bk: int,
) -> None:
  @pl.when(pl.program_id(1) == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  x_val = x_ref[...]
  y_val = y_ref[...]

  is_last_k = pl.program_id(1) == nsteps - 1

  x_val = _mask_tile(x_val, k_rem, 1, is_last_k)
  y_val = _mask_tile(y_val, k_rem, 0, is_last_k)

  # Use f32 to ensure accumulation precision.
  prod = x_val.astype(jnp.float32) * y_val.T.astype(jnp.float32)
  acc_ref[...] += jnp.sum(prod, axis=1, keepdims=True)

  @pl.when(is_last_k)
  def _():
    z_ref[...] = acc_ref[...].astype(z_ref.dtype)


def vecmat_kernel(
    x_ref: Any,
    y_ref: Any,
    z_ref: Any,
    acc_ref: Any,
    *,
    nsteps: int,
    k_rem: int,
    bn: int,
    bk: int,
) -> None:
  @pl.when(pl.program_id(1) == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  x_val = x_ref[...]
  y_val = y_ref[...]

  is_last_k = pl.program_id(1) == nsteps - 1

  x_val = _mask_tile(x_val, k_rem, 1, is_last_k)
  y_val = _mask_tile(y_val, k_rem, 0, is_last_k)

  # Use f32 to ensure accumulation precision.
  prod = x_val.T.astype(jnp.float32) * y_val.astype(jnp.float32)
  acc_ref[...] += jnp.sum(prod, axis=0, keepdims=True)

  @pl.when(is_last_k)
  def _():
    z_ref[...] = acc_ref[...].astype(z_ref.dtype)


def matmul_kernel(
    x_ref: Any,
    y_ref: Any,
    z_ref: Any,
    acc_ref: Any,
    *,
    nsteps: int,
    k_rem: int,
    bm: int,
    bn: int,
    bk: int,
) -> None:
  @pl.when(pl.program_id(2) == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  x_val = x_ref[...]
  y_val = y_ref[...]

  is_last_k = pl.program_id(2) == nsteps - 1

  x_val = _mask_tile(x_val, k_rem, 1, is_last_k)
  y_val = _mask_tile(y_val, k_rem, 0, is_last_k)

  # No f32 cast needed here; the MXU accumulates in f32.
  acc_ref[...] += pl.dot(x_val, y_val)

  @pl.when(is_last_k)
  def _():
    z_ref[...] = acc_ref[...].astype(z_ref.dtype)


@functools.partial(jax.jit, static_argnames=["bm", "bk", "bn"])
def pallas_matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
) -> jax.Array:
  m, k = x.shape
  _, n = y.shape

  bm = min(m, bm)
  bn = min(n, bn)
  bk = min(k, bk)

  if k == 1:
    grid_m = pl.cdiv(m, bm)
    grid_n = pl.cdiv(n, bn)
    return pl.pallas_call(
        functools.partial(outer_kernel, bm=bm, bn=bn),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, 1), lambda i, j: (i, 0)),
                pl.BlockSpec((1, bn), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            grid=(grid_m, grid_n),
        ),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, y)

  if n == 1:
    grid_m = pl.cdiv(m, bm)
    grid_k = pl.cdiv(k, bk)
    k_rem = k % bk
    return pl.pallas_call(
        functools.partial(
            matvec_kernel, nsteps=grid_k, k_rem=k_rem, bm=bm, bk=bk
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, k: (i, k)),
                pl.BlockSpec((bk, 1), lambda i, k: (k, 0)),
            ],
            out_specs=pl.BlockSpec((bm, 1), lambda i, k: (i, 0)),
            scratch_shapes=[pltpu.VMEM((bm, 1), jnp.float32)],
            grid=(grid_m, grid_k),
        ),
        out_shape=jax.ShapeDtypeStruct((m, 1), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
    )(x, y)

  if m == 1:
    grid_n = pl.cdiv(n, bn)
    grid_k = pl.cdiv(k, bk)
    k_rem = k % bk
    return pl.pallas_call(
        functools.partial(
            vecmat_kernel, nsteps=grid_k, k_rem=k_rem, bn=bn, bk=bk
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((1, bk), lambda j, k: (0, k)),
                pl.BlockSpec((bk, bn), lambda j, k: (k, j)),
            ],
            out_specs=pl.BlockSpec((1, bn), lambda j, k: (0, j)),
            scratch_shapes=[pltpu.VMEM((1, bn), jnp.float32)],
            grid=(grid_n, grid_k),
        ),
        out_shape=jax.ShapeDtypeStruct((1, n), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
    )(x, y)

  grid_m = pl.cdiv(m, bm)
  grid_n = pl.cdiv(n, bn)
  grid_k = pl.cdiv(k, bk)

  k_rem = k % bk

  return pl.pallas_call(
      functools.partial(
          matmul_kernel,
          nsteps=grid_k,
          k_rem=k_rem,
          bm=bm,
          bn=bn,
          bk=bk,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
              pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
          ],
          out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
          scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
          grid=(grid_m, grid_n, grid_k),
      ),
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary")
      ),
  )(x, y)


def main(argv: list[str]) -> None:
  del argv
  matmul_bench.run(pallas_matmul)


if __name__ == "__main__":
  app.run(main)
