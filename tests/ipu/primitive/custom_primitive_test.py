# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

import numpy as np
import numpy.testing as npt

import jax
from ipu_custom_activation import custom_activation, CustomActivationPrimitive


class IPUCustomPrimitiveTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    # Check we have at least one IPU device to run these tests.
    assert len(jax.devices("ipu")) >= 1

  def test__custom_activation__metadata(self):
    # Proper metadata for the custom JAX IPU primitive.
    metadata = CustomActivationPrimitive.metadata(num_inputs=2)
    assert metadata.num_inputs == 2
    assert metadata.is_elementwise
    assert metadata.is_stateless
    assert metadata.is_hashable
    assert metadata.input_to_output_tensor_aliasing == {0: 0}
    assert len(metadata.allocating_indices) == 0
    assert len(metadata.replica_identical_output_indices) == 0

  def test__custom_activation__numpy_array_implementation(self):
    in0 = np.random.rand(2, 3, 4).astype(np.float32)
    in1 = np.random.rand(2, 3, 4).astype(np.float32)
    output = custom_activation(in0, in1)
    npt.assert_array_equal(output, np.abs(in0) * in1)

  @parameterized.parameters(["cpu", "ipu"])
  def test__custom_activation__multi_backends_jitting(self, backend):
    in0 = np.random.rand(2, 3, 4).astype(np.float32)
    in1 = np.random.rand(2, 3, 4).astype(np.float32)

    custom_activation_jit = jax.jit(custom_activation, backend=backend)
    output = custom_activation_jit(in0, in1)
    npt.assert_array_equal(output, np.abs(in0) * in1)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
