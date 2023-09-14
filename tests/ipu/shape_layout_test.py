# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
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
from typing import Any
import ctypes

import numpy as np
import numpy.testing as npt
import jax
from absl.testing import parameterized

from jax.lib import xla_extension as xe


def make_xla_shape(shape: Any, layout) -> xe.Shape:
  """Build XLA shape with layout.
  """
  return xe.Shape.array_shape(xe.PrimitiveType.F32, shape, layout)


def make_array_with_layout(data: np.ndarray, layout: Any, device: Any):
  """Build a JAX array with a specific layout, data and device.
  """
  from jax._src.device_array import make_device_array

  data = np.asarray(data)
  assert data.dtype == np.float32
  xla_shape = xe.Shape.array_shape(xe.PrimitiveType.F32, data.shape, layout)
  # Create empty buffer with XLA shape + layout.
  client = device.client
  buffer = client.create_uninitialized_buffer(xla_shape)
  # Read-write buffer view.
  buffer_ptr = np.asarray(buffer).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  buffer_view = np.ctypeslib.as_array(buffer_ptr, shape=data.shape)
  # Fix the strides of the array!
  buffer_view = np.lib.stride_tricks.as_strided(
      buffer_view, buffer_view.shape,
      np.asarray(buffer).strides
  )
  # Copy data into the buffer.
  buffer_view[:] = data[:]
  # Build final JAX array.
  aval = jax.ShapedArray(data.shape, dtype=data.dtype)
  array = make_device_array(aval, device, buffer)
  return array


class IpuXlaShapeLayoutTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.seed = 42
    np.random.seed(self.seed)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name": b,
          "backend": b
      } for b in ["cpu", "ipu"])
  )
  def test__make_array_with_layout__proper_data_layout(self, backend):
    device = jax.devices(backend)[0]
    data = np.random.rand(2, 3, 4).astype(np.float32)
    layout = (1, 2, 0)
    arr = make_array_with_layout(data, layout, device)
    expected_shape = make_xla_shape(data.shape, layout)

    # Make sure we are getting everything right! => different layout from standard C
    self.assertEqual(arr.device_buffer.xla_shape(), expected_shape)
    self.assertNotEqual(np.asarray(arr).strides, data.strides)
    self.assertEqual(arr.device(), device)
    npt.assert_array_equal(arr, data)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name": str(layout),
          "layout": layout
      } for layout in [(1, 2, 0), (1, 0, 2), (0, 1, 2)])
  )
  def test__fn_change_xla_layout__proper_result(self, layout):
    # Not passing on CPU backend. Maybe because you can never normally
    # have a non-standard layout on CPU?
    backend = "ipu"
    device = jax.devices("cpu")[0]

    data0 = np.random.rand(2, 3, 4).astype(np.float32)
    data1 = np.random.rand(2, 3, 4).astype(np.float32)

    arr0 = jax.device_put(data0, device)
    arr1a = jax.device_put(data1, device)
    arr1b = make_array_with_layout(data1, layout, device)

    def fn(x, y):
      return x + y

    # Same jitted function should be compatible with different layouts.
    fn = jax.jit(fn, backend=backend)
    # Major to minor layout.
    out1a = fn(arr0, arr1a)
    # Custom buffer layout.
    out1b = fn(arr0, arr1b)

    npt.assert_array_equal(arr1a, arr1b)
    npt.assert_array_equal(out1a, out1b)
    npt.assert_array_equal(out1b, data0 + data1)
