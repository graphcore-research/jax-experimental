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

from absl.testing import absltest
from jax._src import test_util as jtu
from unittest import SkipTest

import jax
from jax import jit
import jax.numpy as jnp
from jax.config import config

import numpy as np


# Set XLA_IPU_PLATFORM_DEVICE_COUNT=N os env to attach multiple IPUs
class MultiDeviceTest(jtu.JaxTestCase):
  def test_jit_with_multi_devices(self):
    config.FLAGS.jax_platform_name = 'ipu'
    self.assertEqual(jax.default_backend(), 'ipu')

    devices = jax.devices()
    if len(jax.devices()) < 2:
      raise SkipTest("IPU test requires multiple devices")

    def func(x, w, b):
      return jnp.matmul(w, x) + b

    x = np.random.normal(size=[2, 3])
    w = np.random.normal(size=[3, 2])
    b = np.random.normal(size=[3, 3])

    for i in range(len(devices)):
      jit_func = jit(func, device=devices[i])
      r = jit_func(x, w, b)
      self.assertAllClose(r, w @ x + b)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
