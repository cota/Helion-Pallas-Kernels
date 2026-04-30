from torch_tpu import api as tpu_api  # type: ignore[import-not-found]
tpu_api.tpu_device()  # initialize TPU runtime

import functools
import math
import os
import time

from absl import app
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

if jax.devices()[0].platform == "cpu":
  from jax._src.pallas.mosaic import tpu_info
  tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
      tpu_info.ChipVersion.TPU_V4, 1
  )

MIN_BLOCK_SIZE = 128
TRANS_B_DIM_NUMBERS = (((1,), (1,)), ((), ()))
LOG2E = float(jnp.log2(jnp.e))


# ===========================================================================
# Kernel
# ===========================================================================

def _flash_attn_kernel(
    q_ref, k_hbm_ref, v_hbm_ref, o_ref,
    m_scratch_ref, l_scratch_ref, acc_scratch_ref,
    *, sm_scale, block_q, block_k, kv_seq_len, num_q_per_kv, causal,
    padded_dim,
):
  batch_idx = pl.program_id(0)
  kv_head_idx = pl.program_id(1)
  q_tile_idx = pl.program_id(2)
  head_dim_repeats = padded_dim // MIN_BLOCK_SIZE
  block_k_repeats = block_k // MIN_BLOCK_SIZE
  sm_scale_log2 = sm_scale * LOG2E

  def l_broadcast(x):
    return jnp.tile(x, (1, head_dim_repeats))

  for h in range(num_q_per_kv):
    m_scratch_ref[h] = jnp.full(m_scratch_ref.shape[1:], -jnp.inf, jnp.float32)
    l_scratch_ref[h] = jnp.zeros(l_scratch_ref.shape[1:], jnp.float32)
    acc_scratch_ref[h] = jnp.zeros(acc_scratch_ref.shape[1:], jnp.float32)

  if causal:
    def kv_body(kv_tile_indices, k_tile_ref, v_tile_ref):
      kv_tile_idx = kv_tile_indices[0]
      k = k_tile_ref[0, 0]
      v = v_tile_ref[0, 0]
      for h in range(num_q_per_kv):
        q = q_ref[0, h]
        s = jax.lax.dot_general(
            q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)

        q_start = q_tile_idx * block_q
        k_start = kv_tile_idx * block_k
        q_idxs = q_start + jax.lax.broadcasted_iota(jnp.int32, s.shape, 0)
        k_idxs = k_start + jax.lax.broadcasted_iota(jnp.int32, s.shape, 1)
        s = jnp.where(q_idxs >= k_idxs, s, jnp.finfo(jnp.float32).min)

        m_prev = m_scratch_ref[h]
        l_prev = l_scratch_ref[h]
        m_curr = jnp.max(s, axis=1)[:, None]
        m_next = jnp.maximum(m_prev, m_curr)
        p = jnp.exp2(sm_scale_log2 * (s - jnp.tile(m_next, (1, block_k_repeats))))
        exp_m_diff = jnp.exp2(sm_scale_log2 * (m_prev - m_next))
        l_scratch_ref[h] = jnp.sum(p, axis=1)[:, None] + exp_m_diff * l_prev
        m_scratch_ref[h] = m_next
        acc = acc_scratch_ref[h]
        o_curr = jax.lax.dot(
            p.astype(v.dtype), v, preferred_element_type=jnp.float32)
        acc_scratch_ref[h] = l_broadcast(exp_m_diff) * acc + o_curr

    pltpu.emit_pipeline(
        kv_body, grid=(kv_seq_len // block_k,),
        in_specs=[
            pl.BlockSpec((1, 1, block_k, padded_dim),
                         lambda i: (batch_idx, kv_head_idx, i, 0),
                         pipeline_mode=pl.Buffered(buffer_count=2)),
            pl.BlockSpec((1, 1, block_k, padded_dim),
                         lambda i: (batch_idx, kv_head_idx, i, 0),
                         pipeline_mode=pl.Buffered(buffer_count=2)),
        ],
        _explicit_indices=True,
    )(k_hbm_ref, v_hbm_ref)
  else:
    def kv_body(k_tile_ref, v_tile_ref):
      k = k_tile_ref[0, 0]
      v = v_tile_ref[0, 0]
      for h in range(num_q_per_kv):
        q = q_ref[0, h]
        s = jax.lax.dot_general(
            q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)

        m_prev = m_scratch_ref[h]
        l_prev = l_scratch_ref[h]
        m_curr = jnp.max(s, axis=1)[:, None]
        m_next = jnp.maximum(m_prev, m_curr)
        p = jnp.exp2(sm_scale_log2 * (s - jnp.tile(m_next, (1, block_k_repeats))))
        exp_m_diff = jnp.exp2(sm_scale_log2 * (m_prev - m_next))
        l_scratch_ref[h] = jnp.sum(p, axis=1)[:, None] + exp_m_diff * l_prev
        m_scratch_ref[h] = m_next
        acc = acc_scratch_ref[h]
        o_curr = jax.lax.dot(
            p.astype(v.dtype), v, preferred_element_type=jnp.float32)
        acc_scratch_ref[h] = l_broadcast(exp_m_diff) * acc + o_curr

    pltpu.emit_pipeline(
        kv_body, grid=(kv_seq_len // block_k,),
        in_specs=[
            pl.BlockSpec((1, 1, block_k, padded_dim),
                         lambda i: (batch_idx, kv_head_idx, i, 0),
                         pipeline_mode=pl.Buffered(buffer_count=2)),
            pl.BlockSpec((1, 1, block_k, padded_dim),
                         lambda i: (batch_idx, kv_head_idx, i, 0),
                         pipeline_mode=pl.Buffered(buffer_count=2)),
        ],
    )(k_hbm_ref, v_hbm_ref)

  for h in range(num_q_per_kv):
    acc = acc_scratch_ref[h]
    l_val = l_scratch_ref[h]
    o_ref[0, h] = (acc * l_broadcast(1.0 / l_val)).astype(o_ref.dtype)


def flash_attention_oob(q, k, v, *, sm_scale, block_q=1024, block_k=1024,
                        causal=False):
  """OOB flash attention. Works for any head_dim multiple of 128."""
  batch_size, num_q_heads, seq_q, head_dim = q.shape
  _, num_kv_heads, seq_kv, _ = k.shape
  num_q_per_kv = num_q_heads // num_kv_heads
  padded_dim = max(head_dim, MIN_BLOCK_SIZE)

  # Pad to padded_dim if head_dim < MIN_BLOCK_SIZE
  pad_amount = padded_dim - head_dim
  if pad_amount > 0:
    pad_width = ((0, 0), (0, 0), (0, 0), (0, pad_amount))
    q = jnp.pad(q, pad_width, constant_values=0.0)
    k = jnp.pad(k, pad_width, constant_values=0.0)
    v = jnp.pad(v, pad_width, constant_values=0.0)

  out = pl.pallas_call(
      functools.partial(
          _flash_attn_kernel, sm_scale=sm_scale, block_q=block_q,
          block_k=block_k, kv_seq_len=seq_kv, num_q_per_kv=num_q_per_kv,
          causal=causal, padded_dim=padded_dim,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(batch_size, num_kv_heads, seq_q // block_q),
          in_specs=[
              pl.BlockSpec((1, num_q_per_kv, block_q, padded_dim),
                           lambda b, h, qt: (b, h, qt, 0)),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
          ],
          out_specs=pl.BlockSpec((1, num_q_per_kv, block_q, padded_dim),
                                lambda b, h, qt: (b, h, qt, 0)),
          scratch_shapes=[
              pltpu.VMEM((num_q_per_kv, block_q, MIN_BLOCK_SIZE), jnp.float32),
              pltpu.VMEM((num_q_per_kv, block_q, MIN_BLOCK_SIZE), jnp.float32),
              pltpu.VMEM((num_q_per_kv, block_q, padded_dim), jnp.float32),
          ],
      ),
      out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel"),
          disable_bounds_checks=True,
          disable_semaphore_checks=True,
      ),
  )(q, k, v)

  if pad_amount > 0:
    out = out[..., :head_dim]
  return out

B, H, S, D = 8, 32, 8192, 128
sm_scale = 1.0 / math.sqrt(D)
dtype = jnp.bfloat16

key = jax.random.PRNGKey(42)
k1, k2, k3 = jax.random.split(key, 3)
q_test = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
k_test = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
v_test = jax.random.normal(k3, (B, H, S, D), dtype=dtype)


attn_flops = 4 * B * H * S * S * D


jit_fn = jax.jit(functools.partial(flash_attention_oob, sm_scale=sm_scale))
jit_fn(q_test, k_test, v_test).block_until_ready()
jit_fn(q_test, k_test, v_test).block_until_ready()
jit_fn(q_test, k_test, v_test).block_until_ready()
times = []
for _ in range(100):
    t0 = time.perf_counter()
    jit_fn(q_test, k_test, v_test).block_until_ready()
    times.append(time.perf_counter() - t0)
median_s = sorted(times)[50]
ms = median_s * 1000
tflops = attn_flops / median_s / 1e12

print('tflops',tflops)
