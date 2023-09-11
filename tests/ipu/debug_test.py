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

from jax._src import test_util as jtu
from functools import partial

import numpy as np
import jax

from jax.config import config
from jaxlib.ipu_xla_client import IpuPjRtDevice

# Skipping tests on legacy IPU backend.
is_ipu_legacy_backend = not isinstance(jax.devices("ipu")[0], IpuPjRtDevice)


class IpuDebuggingTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.is_ipu_model = config.FLAGS.jax_ipu_use_model

  def test_host_debug_callback(self):
    host_arrays = []

    def host_callback(data):
      host_arrays.append(data)

    @partial(jax.jit, backend="ipu")
    def fn(x):
      x = 2 * x
      jax.debug.callback(host_callback, x + 1)
      return x + 2

    data = np.array([1, 2, 3, 4], dtype=np.float32)

    # FIXME: IPU PjRt client blocking if not using returned value.
    out = fn(data)
    out = fn(out)
    out.block_until_ready()

    assert len(host_arrays) == 2
    assert all([isinstance(v, np.ndarray) for v in host_arrays])
    self.assertAllClose(out, 4 * data + 6)
    self.assertAllClose(host_arrays[0], 2 * data + 1)
    self.assertAllClose(host_arrays[1], 4 * data + 5)
