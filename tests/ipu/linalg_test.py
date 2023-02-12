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
from typing import Any

import numpy as np
import jax


def make_symmetric_matrix(rng, N: int, dtype: Any = np.float32) -> np.ndarray:
    a = rng.rand(N, N).astype(dtype)
    a = (a + a.T) * 0.5
    return a


class IpuLinalgTest(jtu.JaxTestCase):
  """Coverage test of JAX LAX linear algebra operators on IPU.
  """
  def setUp(self):
    super().setUp()
    self.rng = np.random.RandomState(42)

  def test_linalg_cholesky(self):
    a = np.diag(self.rng.rand(4)).astype(np.float32)
    cpu_cholesky = jax.jit(jax.lax.linalg.cholesky, backend="cpu")
    ipu_cholesky = jax.jit(jax.lax.linalg.cholesky, backend="ipu")
    cpu_res = cpu_cholesky(a)
    ipu_res = ipu_cholesky(a)
    self.assertAllClose(ipu_res, cpu_res, rtol=1e-5, atol=1e-5)

  def test_linalg_eigh(self):
    a = make_symmetric_matrix(self.rng, 4)
    cpu_eigh = jax.jit(jax.lax.linalg.eigh, backend="cpu")
    ipu_eigh = jax.jit(jax.lax.linalg.eigh, backend="ipu")
    cpu_eigvecs, cpu_eigvals = cpu_eigh(a)
    ipu_eigvecs, ipu_eigvals = ipu_eigh(a)
    self.assertAllClose(np.abs(ipu_eigvecs), np.abs(cpu_eigvecs), rtol=1e-5, atol=1e-5)
    self.assertAllClose(ipu_eigvals, cpu_eigvals, rtol=1e-5, atol=1e-5)

  def test_linalg_lu(self):
    a = make_symmetric_matrix(self.rng, 4)
    ipu_lu = jax.jit(jax.lax.linalg.lu, backend="ipu")
    with self.assertRaises(Exception):
        # XLA custom call not yet implemented on IPU
        ipu_lu(a)

  def test_linalg_qr(self):
    a = make_symmetric_matrix(self.rng, 4)
    cpu_qr = jax.jit(jax.lax.linalg.qr, backend="cpu")
    ipu_qr = jax.jit(jax.lax.linalg.qr, backend="ipu")
    cpu_q, cpu_r = cpu_qr(a)
    ipu_q, ipu_r = ipu_qr(a)
    self.assertAllClose(ipu_q, cpu_q)
    self.assertAllClose(ipu_r, cpu_r)

  def test_linalg_svd(self):
    a = make_symmetric_matrix(self.rng, 4)
    cpu_svd = jax.jit(jax.lax.linalg.svd, backend="cpu")
    ipu_svd = jax.jit(jax.lax.linalg.svd, backend="ipu")
    cpu_vals, *_ = cpu_svd(a)
    ipu_vals, *_ = ipu_svd(a)
    self.assertAllClose(np.abs(ipu_vals), np.abs(cpu_vals), rtol=1e-4, atol=1e-4)
