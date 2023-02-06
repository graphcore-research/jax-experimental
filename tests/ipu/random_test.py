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

import numpy as np
import jax
from jax import random, prng
from jax.config import config


class IpuRandomTest(jtu.JaxTestCase):
  """Basic random test coverage for IPU. Should be fast to run.

  See the main JAX `random_test.py` for full coverage.
  """

  def testThreefry2x32(self):
    # We test the hash by comparing to known values provided in the test code of
    # the original reference implementation of Threefry. For the values, see
    # https://github.com/DEShawResearch/Random123-Boost/blob/65e3d874b67aa7b3e02d5ad8306462f52d2079c0/libs/random/test/test_threefry.cpp#L30-L32
    def result_to_hex(result):
      return tuple(hex(x.copy()).rstrip("L") for x in result)

    expected = ("0x6b200159", "0x99ba4efe")
    result = prng.threefry_2x32(np.uint32([0, 0]), np.uint32([0, 0]))

    self.assertEqual(expected, result_to_hex(result))

    expected = ("0x1cb996fc", "0xbb002be7")
    result = prng.threefry_2x32(np.uint32([-1, -1]), np.uint32([-1, -1]))
    self.assertEqual(expected, result_to_hex(result))

    expected = ("0xc4923a9c", "0x483df7a0")
    result = prng.threefry_2x32(
        np.uint32([0x13198a2e, 0x03707344]),
        np.uint32([0x243f6a88, 0x85a308d3]))
    self.assertEqual(expected, result_to_hex(result))

  def testRngRandomBits(self):
    # Test specific outputs to ensure consistent random values between JAX versions.
    key = random.PRNGKey(1701)

    bits8 = jax._src.random._random_bits(key, 8, (3,))
    expected8 = np.array([216, 115,  43], dtype=np.uint8)
    self.assertArraysEqual(bits8, expected8)

    # U16 not supported at the moment on IPU XLA backend.
    # bits16 = jax._src.random._random_bits(key, 16, (3,))
    # expected16 = np.array([41682,  1300, 55017], dtype=np.uint16)
    # self.assertArraysEqual(bits16, expected16)

    bits32 = jax._src.random._random_bits(key, 32, (3,))
    expected32 = np.array([56197195, 4200222568, 961309823], dtype=np.uint32)
    self.assertArraysEqual(bits32, expected32)

    with jtu.ignore_warning(category=UserWarning, message="Explicitly requested dtype.*"):
      bits64 = jax._src.random._random_bits(key, 64, (3,))
    if config.x64_enabled:
      expected64 = np.array([3982329540505020460, 16822122385914693683,
                             7882654074788531506], dtype=np.uint64)
    else:
      expected64 = np.array([676898860, 3164047411, 4010691890], dtype=np.uint32)
    self.assertArraysEqual(bits64, expected64)
