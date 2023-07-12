/* Copyright (c) 2023 Graphcore Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

/**
 * @brief ThreeFry algorithm IPU implementation.
 *
 * The interface is following the default XLA implementation in JAX.
 */
class ThreeFry2x32Vertex : public MultiVertex {
 public:
  using T = std::uint32_t;
  // Pair of 32bit arrays used as keys.
  Input<Vector<T, poplar::VectorLayout::ONE_PTR>> key0;
  Input<Vector<T, poplar::VectorLayout::ONE_PTR>> key1;

  // Pair of input data arrays.
  Input<Vector<T, poplar::VectorLayout::SPAN>> data0;
  Input<Vector<T, poplar::VectorLayout::SPAN>> data1;

  // Output arrays, same shape as input data ones.
  Output<Vector<T, poplar::VectorLayout::ONE_PTR>> out0;
  Output<Vector<T, poplar::VectorLayout::ONE_PTR>> out1;

  bool compute(unsigned wid) {
    constexpr int num_workers = 6;
    const int N = data0.size();
    // Rotation distances specified by the Threefry2x32 algorithm.
    std::uint32_t rotations[8] = {13, 15, 26, 6, 17, 29, 16, 24};

    for (int idx = wid; idx < N; idx += num_workers) {
      std::uint32_t x[2];
      std::uint32_t ks[3];

      // 0x1BD11BDA is a parity constant specified by the ThreeFry2x32
      // algorithm.
      ks[2] = 0x1BD11BDA;

      ks[0] = key0[idx];
      x[0] = data0[idx];
      ks[2] = ks[2] ^ key0[idx];

      ks[1] = key1[idx];
      x[1] = data1[idx];
      ks[2] = ks[2] ^ key1[idx];

      auto rotate_left = [](std::uint32_t v, std::uint32_t distance) {
        return (v << distance) | (v >> (32 - distance));
      };

      // Performs a single round of the Threefry2x32 algorithm, with a rotation
      // amount 'rotation'.
      auto round = [&](std::uint32_t* v, std::uint32_t rotation) {
        v[0] += v[1];
        v[1] = rotate_left(v[1], rotation);
        v[1] ^= v[0];
      };

      // There are no known statistical flaws with 13 rounds of Threefry2x32.
      // We are conservative and use 20 rounds.
      x[0] = x[0] + ks[0];
      x[1] = x[1] + ks[1];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        round(x, rotations[i]);
      }

      x[0] = x[0] + ks[1];
      x[1] = x[1] + ks[2] + 1u;
#pragma unroll
      for (int i = 4; i < 8; ++i) {
        round(x, rotations[i]);
      }

      x[0] = x[0] + ks[2];
      x[1] = x[1] + ks[0] + 2u;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        round(x, rotations[i]);
      }

      x[0] = x[0] + ks[0];
      x[1] = x[1] + ks[1] + 3u;
#pragma unroll
      for (int i = 4; i < 8; ++i) {
        round(x, rotations[i]);
      }

      x[0] = x[0] + ks[1];
      x[1] = x[1] + ks[2] + 4u;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        round(x, rotations[i]);
      }

      out0[idx] = x[0] + ks[2];
      out1[idx] = x[1] + ks[0] + 5u;
    }
    return true;
  }
};
