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
import cppimport
import os
from typing import Sequence

from jax.interpreters import mlir
from jax.interpreters.mlir import ir

from jax.ipu.primitive import ipu_mlir_lowering_custom_primitive

# Pybind11 extension import (and compilation if necessary).
# Explicit path is more robust to different `pip install` usages.
ext_filename = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "ipu_random_primitive_impl.cpp")
)
ipu_random_primitive_impl = cppimport.imp_from_filepath(
    ext_filename, "jax.ipu.random.ipu_random_primitive_impl"
)


def ipu_threefry2x32_vertex_filename() -> str:
  filename = "ipu_random_vertex.cpp"
  return os.path.abspath(os.path.join(os.path.dirname(__file__), filename))


def ipu_threefry2x32_lowering(
    ctx: mlir.LoweringRuleContext, key0: ir.Value, key1: ir.Value, data0: ir.Value,
    data1: ir.Value
) -> Sequence[ir.Value]:
  """`threefry2x32` algorithm IPU backend MLIR lowering, as a custom (optimized) Poplar primitive.

  Args:
    ctx: MLIR lowering context.
    key0, key1: Key uint32 arrays.
    data0, data1: Data uint32 arrays.
  Returns:
    ThreeFry 2x uint32 resulting arrays.
  """
  avals_in = ctx.avals_in
  avals_out = ctx.avals_out
  assert len(avals_in) == 4, avals_in
  assert len(avals_out) == 2, avals_out

  inputs = [key0, key1, data0, data1]
  outputs = ipu_mlir_lowering_custom_primitive(
      ipu_random_primitive_impl.IpuThreeFry2x32Primitive,
      ctx,
      inputs,
      ipu_gp_filename=ipu_threefry2x32_vertex_filename()
  )
  return outputs
