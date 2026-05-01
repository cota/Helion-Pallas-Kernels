import math
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from jax import random
from jax._src import test_util as jtu
from jax.experimental.pallas.ops.tpu import flash_attention
import jax.numpy as jnp
import numpy as np

from attention import flash_attention_oob as markus_attention


class AttentionTest(jtu.JaxTestCase):

  def _test_attention(
      self,
      causal: bool = False,
      seq_len: int = 512,
      batch_size: int = 2,
      head_dim: int = 128,
      n_heads: int = 2,
      dtype: jnp.dtype = jnp.bfloat16,
      block_q: int = 128,
      block_k: int = 128,
  ):
    if dtype is jnp.bfloat16 and not jtu.is_device_tpu_at_least(version=4):
      self.skipTest("bf16 support requires at least TPU v4")

    q_shape = (batch_size, n_heads, seq_len, head_dim)
    kv_shape = (batch_size, n_heads, seq_len, head_dim)

    q_key, k_key, v_key = random.split(random.key(0), 3)
    q = random.normal(q_key, q_shape, dtype=dtype)
    k = random.normal(k_key, kv_shape, dtype=dtype)
    v = random.normal(v_key, kv_shape, dtype=dtype)

    sm_scale = 1.0 / math.sqrt(head_dim)

    out = markus_attention(
        q,
        k,
        v,
        sm_scale=sm_scale,
        block_q=block_q,
        block_k=block_k,
        causal=causal,
    )

    q32, k32, v32 = map(lambda x: x.astype(jnp.float32), (q, k, v))
    out_ref = flash_attention.mha_reference(
        q32, k32, v32, None, None, causal=causal, sm_scale=sm_scale
    )

    np.testing.assert_allclose(out, out_ref, atol=0.05, rtol=0.05)

  @parameterized.product(
      causal=(True, False),
      dtype=(jnp.float32, jnp.bfloat16),
  )
  def test_attention_basic(self, causal, dtype):
    self._test_attention(causal=causal, dtype=dtype)

  @parameterized.parameters(
    (8192, 256),
    (8192, 512),
  )
  def test_attention_block(self, block_q, block_k):
    self._test_attention(block_q=block_q, block_k=block_k)

  @parameterized.product(
      seq_len=(300,),
      block_q=(128,),
      block_k=(128,),
      dtype=(jnp.bfloat16,),
  )
  @unittest.expectedFailure
  def test_attention_non_divisible(self, seq_len, block_q, block_k, dtype):
    self._test_attention(seq_len=seq_len, block_q=block_q, block_k=block_k, dtype=dtype)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
