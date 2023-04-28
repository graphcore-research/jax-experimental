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
import sys
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

import numpy as np

import jax
from jax.lib import xla_extension
from jax.ipu.primitive.cppimport_utils import patch_function
from jax.ipu.primitive.xla_utils import dtype_to_tf_datatype_enum, xla_shape_to_aval


def simple_function(x):
  return x * 2


class PatchFunctionTests(jtu.JaxTestCase):

  def test__patch_function__simple_numeric_function(self):

    @patch_function(simple_function, [sys.modules[simple_function.__module__]])
    def simple_function_patch(orig_fn, x):
      return -orig_fn(x)

    self.assertEqual(simple_function(3), -6)


class XLAUtilsTests(jtu.JaxTestCase):

  @parameterized.parameters([
      (np.int8, 6),
      (np.int16, 5),
      (np.int32, 3),
  ])
  def test__dtype_to_tf_datatype_enum__proper_int_enum(self, dtype, expected_value):
    dtype = np.dtype(dtype)
    datatype_enum = dtype_to_tf_datatype_enum(dtype)
    assert isinstance(datatype_enum, int)
    assert datatype_enum == expected_value

  def test__xla_shape_to_aval__proper_jax_shaped_array(self):
    xla_shape = xla_extension.Shape.array_shape(
        xla_extension.PrimitiveType.F16, (3, 4, 5)
    )
    aval = xla_shape_to_aval(xla_shape)
    self.assertIsInstance(aval, jax.ShapedArray)
    self.assertEqual(aval.shape, (3, 4, 5))
    self.assertEqual(aval.dtype, np.float16)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
