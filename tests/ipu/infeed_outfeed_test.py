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

import threading
import unittest

from absl.testing import absltest
from jax._src import test_util as jtu

import jax
from jax import lax, numpy as jnp
from jax import config
from jax.experimental import host_callback as hcb
from jax.lib import xla_client
import numpy as np

from jaxlib.ipu_xla_client import IpuPjRtDevice

config.parse_flags_with_absl()
FLAGS = config.FLAGS

# Skipping tests on new IPU backend.
is_ipu_legacy_backend = not isinstance(jax.devices("ipu")[0], IpuPjRtDevice)


class IpuInfeedTest(jtu.JaxTestCase):

  @unittest.skipUnless(
      is_ipu_legacy_backend, "Infeed/outfeed not yet supported on IPU."
  )
  def testInfeed(self):

    @jax.jit
    def f(x):
      token = lax.create_token(x)
      (y,), token = lax.infeed(token, shape=(jax.ShapedArray((3, 4), jnp.float32),))
      return x + y

    x = np.float32(1.5)
    y = np.reshape(
        np.arange(12, dtype=np.float32), (3, 4)
    )  # np.random.randn(3, 4).astype(np.float32)
    device = jax.local_devices()[0]
    device.transfer_to_infeed((y,))
    self.assertAllClose(f(x), x + y)

  @unittest.skipUnless(
      is_ipu_legacy_backend, "Infeed/outfeed not yet supported on IPU."
  )
  def testInfeedPytree(self):

    x = np.float32(1.5)
    y = np.reshape(np.arange(12, dtype=np.int16), (3, 4))
    to_infeed = dict(a=x, b=y)
    to_infeed_shape = dict(
        a=jax.ShapedArray((), dtype=np.float32),
        b=jax.ShapedArray((3, 4), dtype=np.int16)
    )

    @jax.jit
    def f(x):
      token = lax.create_token(x)
      res, token = lax.infeed(token, shape=to_infeed_shape)
      return res

    device = jax.local_devices()[0]
    # We must transfer the flattened data, as a tuple.
    flat_to_infeed, _ = jax.tree_util.tree_flatten(to_infeed)
    device.transfer_to_infeed(tuple(flat_to_infeed))
    self.assertAllClose(f(x), to_infeed)

  @unittest.skipUnless(
      is_ipu_legacy_backend, "Infeed/outfeed not yet supported on IPU."
  )
  def testOutfeed(self):
    hcb.stop_outfeed_receiver()

    @jax.jit
    def f(x):
      token = lax.create_token(x)
      token = lax.outfeed(token, x + np.float32(1))
      return x - 1

    x = np.float32(7.5)
    execution = threading.Thread(target=lambda: f(x))
    execution.start()
    device = jax.local_devices()[0]
    out, = device.transfer_from_outfeed(
        xla_client.shape_from_pyval((x,)).with_major_to_minor_layout_if_absent()
    )
    execution.join()
    self.assertAllClose(f(x) + 2, out)

    # Remove data queued by f(x) in assertAllClose
    device.transfer_from_outfeed(
        xla_client.shape_from_pyval((x,)).with_major_to_minor_layout_if_absent()
    )

  @unittest.skipUnless(
      is_ipu_legacy_backend, "Infeed/outfeed not yet supported on IPU."
  )
  def testOutfeedPytree(self):
    hcb.stop_outfeed_receiver()

    @jax.jit
    def f(x, y):
      token = lax.create_token(x)
      token = lax.outfeed(token, (x + 1, y))
      return x - 1

    x = np.float32(7.5)
    y = np.reshape(np.arange(12, dtype=np.int16), (3, 4))
    execution = threading.Thread(target=lambda: f(x, y))
    execution.start()
    device = jax.local_devices()[0]
    out = device.transfer_from_outfeed(
        xla_client.shape_from_pyval((x, y)).with_major_to_minor_layout_if_absent()
    )
    execution.join()
    self.assertAllClose((f(x, y) + 2, y), out)

    # Remove data queued by f(x) in assertAllClose
    device.transfer_from_outfeed(
        xla_client.shape_from_pyval((x, y)).with_major_to_minor_layout_if_absent()
    )

  @unittest.skipUnless(
      is_ipu_legacy_backend, "Infeed/outfeed not yet supported on IPU."
  )
  def testInfeedThenOutfeed(self):
    hcb.stop_outfeed_receiver()

    @jax.jit
    def f(x):
      token = lax.create_token(x)
      y, token = lax.infeed(token, shape=jax.ShapedArray((3, 4), jnp.float32))
      token = lax.outfeed(token, y + np.float32(1))
      return x - 1

    x = np.float32(7.5)
    y = np.random.randn(3, 4).astype(np.float32)
    execution = threading.Thread(target=lambda: f(x))
    execution.start()
    device = jax.local_devices()[0]
    device.transfer_to_infeed((y,))
    out, = device.transfer_from_outfeed(
        xla_client.shape_from_pyval((y,)).with_major_to_minor_layout_if_absent()
    )
    execution.join()
    self.assertAllClose(out, y + np.float32(1))

  @unittest.skipUnless(
      is_ipu_legacy_backend, "Infeed/outfeed not yet supported on IPU."
  )
  def testInfeedThenOutfeedInALoop(self):
    hcb.stop_outfeed_receiver()

    def doubler(_, token):
      y, token = lax.infeed(token, shape=jax.ShapedArray((3, 4), jnp.float32))
      return lax.outfeed(token, y * np.float32(2))

    @jax.jit
    def f(n):
      token = lax.create_token(n)
      token = lax.fori_loop(0, n, doubler, token)
      return n

    device = jax.local_devices()[0]
    n = 10
    execution = threading.Thread(target=lambda: f(n))
    execution.start()
    for _ in range(n):
      x = np.random.randn(3, 4).astype(np.float32)
      device.transfer_to_infeed((x,))
      y, = device.transfer_from_outfeed(
          xla_client.shape_from_pyval((x,)).with_major_to_minor_layout_if_absent()
      )
      self.assertAllClose(y, x * np.float32(2))
    execution.join()


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
