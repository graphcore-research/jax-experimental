# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from unittest import SkipTest
from functools import partial

import os
import numpy as np
import jax
import jax.numpy as jnp

from jax import lax
from jax.config import config


class IpuBasicsTest(jtu.JaxTestCase):

  def test_device_count(self):
    expected = os.getenv('XLA_IPU_PLATFORM_DEVICE_COUNT')
    expected = int(expected) if expected else 1

    assert jax.device_count(backend='ipu') == expected
    assert len(jax.devices("ipu")) == expected

  def test_default_backend(self):
    config.FLAGS.jax_platform_name = 'ipu'
    assert jax.default_backend() == 'ipu'

  def test_compile_mhlo(self):
    mhlo_code = """
        #loc0 = loc(unknown)
        module @jit_func.0 {
          func.func public @main(%arg0: tensor<2x3xf32> loc(unknown), %arg1: tensor<3x2xf32> loc(unknown), %arg2: tensor<3x3xf32> loc(unknown)) -> tensor<3x3xf32> {
            %0 = "mhlo.dot_general"(%arg1, %arg0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<3x2xf32>, tensor<2x3xf32>) -> tensor<3x3xf32> loc(#loc1)
            %1 = mhlo.add %0, %arg2 : tensor<3x3xf32> loc(#loc2)
            return %1 : tensor<3x3xf32> loc(#loc0)
          } loc(#loc0)
        } loc(#loc0)
        #loc1 = loc("jit(func)/jit(main)/dot_general[dimension_numbers=(((1,), (0,)), ((), ())) precision=None preferred_element_type=None]"("./tests/ipu/basics_test.py":53:1))
        #loc2 = loc("jit(func)/jit(main)/add"("./tests/ipu/basics_test.py":53:1))"""

    d = jax.devices("ipu")[0]
    raise SkipTest("IPU PJRT client not supported MHLO")
    # Crashing Python interpreter at the moment.
    d.client.compile(mhlo_code)

  def test_single_op_add(self):
    def add(a,b):
      return lax.add(a,b)

    jit_add = jax.jit(fun=add, backend='ipu')
    a = np.array([0, 1, 2, 3], dtype=np.float32)
    b = np.array([4, 5, 6, 7], dtype=np.float32)
    c = jit_add(a, b)
    self.assertAllClose(c, a + b)


  def test_lax_argmin_argmax(self):
    @partial(jax.jit, backend="ipu")
    def argminmax(v):
      return (lax.argmin(v, axis=0, index_dtype=np.int32),
              lax.argmax(v, axis=0, index_dtype=np.int32))

    a = np.array([1, 0, 5, 3], dtype=np.float32)
    amin, amax = argminmax(a)
    assert amin == 1
    assert amax == 2


  @parameterized.parameters(
      [
          (np.uint32, 1),
          (np.uint32, 32),
          (np.int32, 32),
          (np.uint8, 8),
          # (np.uint16, 8), UINT16 not supported.
          (np.uint64, 1),
          (np.uint64, 64),
      ]
  )
  def test_lax_shift_ops(self, dtype, shift):
    shift = dtype(shift)
    @partial(jax.jit, backend="ipu")
    def shift_op(v):
      return (lax.shift_left(v, shift),
              lax.shift_right_logical(v, shift))

    a = np.array([0, 1, 3, 5], dtype=dtype)
    vleft, vright_log = shift_op(a)
    self.assertAllClose(vleft, np.left_shift(a, shift))
    self.assertAllClose(vright_log, np.right_shift(a, shift))


  def test_linear_layer(self):
    @partial(jax.jit, backend="ipu")
    def func(x, w, b):
      return jnp.matmul(w, x) + b

    x = np.random.normal(size=[2, 3])
    w = np.random.normal(size=[3, 2])
    b = np.random.normal(size=[3, 3])
    r = func(x, w, b)
    self.assertAllClose(r, w @ x + b)


  def test_lax_sort(self):
    @partial(jax.jit, backend="ipu")
    def sort_fn(v):
      return lax.sort(v)

    a = np.array([1, 0, 5, -1], dtype=np.float32)
    asorted = sort_fn(a)
    self.assertAllClose(asorted, np.sort(a))


  def test_lax_sort_key_val(self):
    @partial(jax.jit, backend="ipu")
    def sort_key_val_fn(k, v):
      return lax.sort_key_val(k, v)

    keys = np.array([1, 0, 5, -1], dtype=np.float32)
    values = np.array([0, 1, 2, 3], dtype=np.int32)

    ks, vs = sort_key_val_fn(keys, values)
    self.assertAllClose(ks, np.sort(np.asarray(keys)))
    self.assertAllClose(vs, np.array([3, 1, 0, 2]))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
