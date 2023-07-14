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
from .ipu_random_primitive import ipu_threefry2x32_lowering

from jax.interpreters import mlir
from jax._src.lib.mlir.dialects import mhlo
from jax._src.prng import threefry2x32_p


def _ipu_threefry2x32_broadcast_lowering(ctx, k1, k2, x1, x2):
  aval_out, _ = ctx.avals_out
  k1_aval, k2_aval, x1_aval, x2_aval = ctx.avals_in
  rank = len(aval_out.shape)
  if 0 in aval_out.shape:
    zeros = mlir.full_like_aval(0, aval_out)
    return [zeros, zeros]

  def _broadcast(x, aval):
    return mhlo.BroadcastInDimOp(
        mlir.aval_to_ir_type(aval_out), x,
        mlir.dense_int_elements(range(rank - len(aval.shape), rank))
    ).result

  ctx = ctx.replace(avals_in=[aval_out] * 4)
  # TODO: optimize in the case of scalar keys.
  return ipu_threefry2x32_lowering(
      ctx, _broadcast(k1, k1_aval), _broadcast(k2, k2_aval), _broadcast(x1, x1_aval),
      _broadcast(x2, x2_aval)
  )


mlir.register_lowering(
    threefry2x32_p, _ipu_threefry2x32_broadcast_lowering, platform="ipu"
)
