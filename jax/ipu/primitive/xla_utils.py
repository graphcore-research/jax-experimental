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
import numpy as np

from jax.lib import xla_extension
from jax.interpreters.xla import ShapedArray, dtype_to_primitive_type, aval_to_xla_shapes
"""Numpy to TF datatype enum, necessary for IPU custom primitives.

This dict is hardcoded to avoid bringing TensorFlow as a dependency.
For some reason, this TF enum seems to be different from XLA primitive types?

These values can be generated using:
```python
from tensorflow.python.eager.execute import make_type

enum_val = make_type(dtype, "dt")
```
"""
_dtype_to_tf_datatype_enum = {
    np.dtype('bool'): 10,
    np.dtype('int8'): 6,
    np.dtype('int16'): 5,
    np.dtype('int32'): 3,
    np.dtype('int64'): 9,
    np.dtype('uint8'): 4,
    np.dtype('uint16'): 17,
    np.dtype('uint32'): 22,
    np.dtype('uint64'): 23,
    np.dtype('float16'): 19,
    np.dtype('float32'): 1,
    np.dtype('float64'): 2,
    np.dtype('complex64'): 8,
    np.dtype('complex128'): 18,
}


def dtype_to_tf_datatype_enum(dtype: np.dtype) -> int:
  """Convert a Numpy dtype to TensorFlow dtype enum.
  """
  assert isinstance(dtype, np.dtype), type(dtype)
  try:
    return _dtype_to_tf_datatype_enum[dtype]
  except KeyError as err:
    raise TypeError(f"No TF datatype enum lowering for NumPy dtype: {dtype}") from err


def xla_shape_to_aval(xla_shape: xla_extension.Shape) -> ShapedArray:
  """Convert an XLA shape into a JAX shaped array.
  """
  assert xla_shape.is_array()
  return ShapedArray(xla_shape.dimensions(), xla_shape.numpy_dtype())


def aval_to_xla_shape(aval: ShapedArray) -> xla_extension.Shape:
  """Convert a JAX shaped array into an XLA shape.
  """
  return aval_to_xla_shapes(aval)[0]
