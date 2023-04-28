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

import unittest
from absl.testing import absltest
from jax._src import test_util as jtu
from functools import partial

import jax
from jax import jit
import jax.numpy as jnp
from jaxlib.ipu_xla_client import IpuTargetType

import numpy as np

ipu_num_devices = len(jax.devices("ipu"))
is_ipu_model = len(jax.devices("ipu")
                  ) > 0 and jax.devices("ipu")[0].target_type == IpuTargetType.IPU_MODEL


# Set JAX_IPU_DEVICE_COUNT=N os env to attach multiple IPUs
class IpuMultiDeviceTest(jtu.JaxTestCase):

  @unittest.skipIf(ipu_num_devices < 2, "Requires multiple IPU devices")
  def test_jit_on_multi_devices(self):
    self.assertEqual(jax.default_backend(), 'ipu')
    ipu_devices = jax.devices()

    def func(x, w, b):
      return jnp.matmul(w, x) + b

    x = np.random.normal(size=[2, 3])
    w = np.random.normal(size=[3, 2])
    b = np.random.normal(size=[3, 3])

    # Just testing on 2 IPU devices, to reduce compilation time.
    # TODO: support portable executable?
    for i in range(2):
      jit_func = jit(func, device=ipu_devices[i])
      r = jit_func(x, w, b)
      self.assertAllClose(r, w @ x + b)

  @unittest.skipIf(
      ipu_num_devices < 2 or is_ipu_model, "Requires multiple IPU hardware devices"
  )
  def test_pmap_simple_reduce(self):
    N = 3
    data = np.arange(2 * N, dtype=np.float32).reshape((-1, N))

    @partial(jax.pmap, axis_name='i', donate_argnums=(1,), backend="ipu")
    def parallel_fn(x, y):
      z = x + jax.lax.psum(y, 'i')
      return z

    output = parallel_fn(data ** 2, data)
    self.assertAllClose(output, data ** 2 + np.sum(data, axis=0))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
