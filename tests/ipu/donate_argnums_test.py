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

from functools import partial

from absl.testing import absltest
from unittest import SkipTest
from jax._src import test_util as jtu

import jax
import numpy as np

class IpuDonateArgnumsTest(jtu.JaxTestCase):

  def testSingleDonateBufferFirstArgument(self):

    @partial(jax.jit, backend='ipu', donate_argnums=[0])
    def f(x, y):
      return x + 1, y + 2

    x = np.float32(0)
    y = np.ones((2, 2), dtype=np.float32)
    raise SkipTest("IPU XLA not supporting donate argnums with first parameter.")
    self.assertAllClose(f(x, y), (1., y + 2))

  def testSingleDonateBufferLastArgument(self):

    @partial(jax.jit, backend='ipu', donate_argnums=[1])
    def f(x, y):
      return x + 1, y + 2

    x = np.float32(0)
    y = np.ones((2, 2), dtype=np.float32)
    self.assertAllClose(f(x, y), (1., y + 2))

  def testSingleDonateBufferRevertOrder(self):

    @partial(jax.jit, backend='ipu', donate_argnums=[1])
    def f(x, y):
      return y + 2, x + 1

    x = np.float32(0)
    y = np.ones((2, 2), dtype=np.float32)
    raise SkipTest("IPU XLA not supporting donate argnums revert order.")
    self.assertAllClose(f(x, y), (y + 2, 1.))

  def testMultiDonateBuffers(self):

    @partial(jax.jit, backend='ipu', donate_argnums=[0, 1])
    def f(x, y):
      return x + 1, y + 2

    x = np.float32(1)
    y = np.ones((3), dtype=np.float32)
    self.assertAllClose(f(x, y), (x + 1, np.ones((3), dtype=np.float32) + 2))

  def testInterleavedDonateBuffers(self):

    @partial(jax.jit, backend='ipu', donate_argnums=[0, 2])
    def f(x, y, z):
      x = x + 1
      y = y + 2
      z = z + 3
      return x, y, z

    x = np.float32(1)
    y = np.reshape(np.arange(3, dtype=np.float32), (3, 1))
    z = np.reshape(np.arange(4, dtype=np.float32), (2, 2))
    raise SkipTest("IPU XLA not supporting interleaved donate argnums.")
    self.assertAllClose(f(x, y, z), (2., y + 2, z + 3))

  def testInterleavedDonateBuffersRevertOrder(self):

    @partial(jax.jit, backend='ipu', donate_argnums=[0, 2])
    def f(x, y, z):
      x = x + 1
      y = y + 2
      z = z + 3
      return z, y, x

    x = np.float32(1)
    y = np.reshape(np.arange(3, dtype=np.float32), (3, 1))
    z = np.float32(5)
    raise SkipTest("IPU XLA not supporting interleaved donate argnums.")
    self.assertAllClose(f(x, y, z), (z + 3, y + 2, x + 1))

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
