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
import json

from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jax._src.lib.mlir.dialects import mhlo

from jax.ipu.primitive.ipu_custom_primitive_utils import make_custom_primitive_attributes

from jaxlib.ipu_xla_client import _ipu_xla

# Pybind11 extension import (and compilation if necessary).
# Explicit path is more robust to different `pip install` usages.
ext_filename = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "ipu_python_callback_impl.cpp")
)
ipu_debug_callback_impl = cppimport.imp_from_filepath(
    ext_filename, "jax.ipu.debug.ipu_python_callback_impl"
)


def ipu_debug_callback_custom_call(
    ctx, result, operands_, *, has_side_effect: bool, callback_descriptor: int
):
  """IPU backend debug callback custom call.

    This function is called inside `emit_python_callback` in JAX MLIR interpreter,
    when the IPU XLA backend is used.
    """
  # Only supporting some specific configs!
  if len(ctx.avals_out) > 0:
    raise NotImplementedError(
        "IPU backend only does not support outputs in host Python callbacks."
    )
  # Raw attributes: callback ptr + IPU XLA library filename.
  opaque_attributes = ";".join([str(callback_descriptor), _ipu_xla.__file__])
  # Custom op/primitive attributes.
  opattributes = make_custom_primitive_attributes(
      ipu_debug_callback_impl.IpuPythonCallbackPrimitive,
      ctx.avals_out,
      opaque_attributes=opaque_attributes,
      ipu_gp_filename=None
  )
  # On HOST function => specific IPU XLA option.
  opattributes["is_user_read_write"] = True
  # IPU XLA backend custom op static name.
  call_target_name = "UserOp"
  # IPU XLA custom op expecting backend attributes encoded as json.
  backend_config = json.dumps(opattributes)
  result = mhlo.CustomCallOp(
      result,
      operands_,
      call_target_name=ir.StringAttr.get(call_target_name),
      has_side_effect=ir.BoolAttr.get(has_side_effect),
      api_version=mlir.i32_attr(0),  # original CustomCall API.
      called_computations=None,
      backend_config=ir.StringAttr.get(backend_config),
      operand_layouts=None,
      result_layouts=None
  )
  return result
