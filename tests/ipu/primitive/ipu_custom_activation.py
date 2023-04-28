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

from typing import Sequence
import numpy as np
import cppimport

from jax.core import Primitive, ShapedArray
from jax.interpreters import mlir
from jax.interpreters.mlir import ir, mhlo

from jax.ipu.primitive import ipu_mlir_lowering_custom_primitive

# Import pybind11 extension with IPU primitive implementation.
# This import will automatically trigger the compilation thanks to cppimport.
ipu_custom_activation_impl = cppimport.imp("ipu_custom_activation_impl")

CustomActivationPrimitive = ipu_custom_activation_impl.CustomActivationPrimitive
custom_activation_p = Primitive("custom_activation")
"""Declaring a custom primitive in JAX, with IPU specific MLIR translation.

Following the JAX official guide: https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html
"""


def custom_activation(x, y):
  return custom_activation_p.bind(x, y)


def custom_activation_numpy_impl(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """Custom activation Numpy implementation.
  """
  return np.abs(x) * y


def custom_activation_abstract_eval(xs: ShapedArray, ys: ShapedArray) -> ShapedArray:
  """Custom activation abstract eval: no change in shape & dtype.
  """
  return xs


def custom_activation_default_lowering(
    ctx: mlir.LoweringRuleContext, xc: ir.Value, yc: ir.Value
) -> Sequence[ir.Value]:
  """`custom_activation` default MLIR lowering, for CPU/GPU/TPU backends.
  """
  return mhlo.MulOp(mhlo.AbsOp(xc), yc).results


def custom_activation_ipu_lowering(
    ctx: mlir.LoweringRuleContext, xc: ir.Value, yc: ir.Value
) -> Sequence[ir.Value]:
  """`custom_activation` IPU backend MLIR lowering, as a custom (optimized) Poplar primitive.
  """
  outputs = ipu_mlir_lowering_custom_primitive(
      ipu_custom_activation_impl.CustomActivationPrimitive, ctx, [xc, yc]
  )
  return outputs


# Register the primal implementation with JAX
custom_activation_p.def_impl(custom_activation_numpy_impl)
# Register the abstract evaluation with JAX
custom_activation_p.def_abstract_eval(custom_activation_abstract_eval)
# Register MLIR default lowering for CPU/GPU/TPU.
mlir.register_lowering(custom_activation_p, custom_activation_default_lowering)
# Register specific MLIR lowering for IPU.
mlir.register_lowering(
    custom_activation_p, custom_activation_ipu_lowering, platform="ipu"
)
