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
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poputil/exceptions.hpp>
#include <type_traits>

namespace jax {
namespace ipu {

/**
 * @brief JAX IPU primitive metadata.
 *
 * The metadata structure is gathering proper information for the XLA backend on
 * how to handle the op, and if possible, perform some optimizations on the XLA
 * graph.
 *
 * Please see the IPU TensorFlow documentation for a full description of
 * metadata fields:
 *   https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/tensorflow/custom_codelet.html#metadata
 */
struct PrimitiveMetadata {
  /** Number of inputs. */
  std::uint32_t num_inputs = 1;
  /** output[0].shape == input[0].shape. */
  bool is_elementwise = false;
  /** Is the primitive op stateless. Should always be true in JAX. */
  bool is_stateless = true;
  /** Is the primitive op hashable (for graph caching). */
  bool is_hashable = false;
  /** Inputs/outputs aliasing map. */
  std::map<std::int64_t, std::int64_t> input_to_output_tensor_aliasing;
  /** Which input tensors have a custom allocation. */
  std::vector<std::int64_t> allocating_indices;
  /** Experimental replica information. */
  std::vector<std::int64_t> replica_identical_output_indices;
};

/**
 * @brief JAX IPU base primitive interface.
 *
 * Standard interface to follow for implementing a JAX IPU custom primitive.
 * This API is an adaptation of the IPU XLA custom op API to JAX:
 *   https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/tensorflow/custom_codelet.html
 */
class PrimitiveInterface {
 public:
  /**
   * @brief Build the metadata info of the primitive, for a given number of
   * inputs.
   * @param num_inputs Number of inputs.
   * @return Primitive metadata.
   */
  static PrimitiveMetadata metadata(std::uint32_t num_inputs) {
    return PrimitiveMetadata{.num_inputs = num_inputs};
  }

  /**
   * @brief Build the Poplar program corresponding to the JAX primitive.
   * @param graph Poplar graph.
   * @param inputs Vector of input tensors.
   * @param outputs Vector of output tensors (to fill).
   * @param attributes Raw attributes, encoded as a string.
   * @param debug_prefix Debugging prefix to use.
   */
  static poplar::program::Program program(
      poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
      std::vector<poplar::Tensor>& outputs, const std::string& raw_attributes,
      const std::string& debug_prefix);

  /**
   * @brief Custom allocator method for primitive inputs.
   *
   * If some inputs have not been created & allocated by Poplar, this method
   * provides the option to customize the tensor tile mapping on the IPU.
   *
   * @param graph Poplar graph.
   * @param operand Input index.
   * @param shape Shape of the input tensor.
   * @param type (Poplar) type of the input tensor.
   * @param attributes Custom raw attributes.
   * @return Allocated & mapped Poplar tensor.
   */
  static poplar::Tensor allocator(poplar::Graph& graph, std::uint32_t operand,
                                  const std::vector<size_t>& shape,
                                  poplar::Type type,
                                  const std::string& attributes,
                                  const std::string& debug_prefix) {
    // By default: no custom allocation of inputs.
    throw poputil::poplibs_error(
        "IPU JAX primitive allocator not implemented.");
  }
};

}  // namespace ipu
}  // namespace jax

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b

// Generate the metadata C exported function `#opclsExport_metadata`
#define IPU_JAX_PRIMITIVE_METADATA(opcls)                                     \
  extern "C" __attribute__((visibility("default"))) void CONCAT(              \
      opcls, Export_metadata)(                                                \
      std::vector<std::int64_t> & allocating_indices,                         \
      std::vector<std::int64_t> & replica_identical_output_indices,           \
      std::map<std::int64_t, std::int64_t> & input_to_output_tensor_aliasing, \
      bool& is_elementwise, bool& is_stateless, bool& is_hashable,            \
      std::uint32_t num_inputs) {                                             \
    auto metadata = opcls::metadata(num_inputs);                              \
    allocating_indices = std::move(metadata.allocating_indices);              \
    replica_identical_output_indices =                                        \
        std::move(metadata.replica_identical_output_indices);                 \
    input_to_output_tensor_aliasing =                                         \
        std::move(metadata.input_to_output_tensor_aliasing);                  \
    is_elementwise = metadata.is_elementwise;                                 \
    is_stateless = metadata.is_stateless;                                     \
    is_hashable = metadata.is_hashable;                                       \
  }
// Generate the main program C exported function `#opclsExport`.
#define IPU_JAX_PRIMITIVE_PROGRAM(opcls)                                     \
  extern "C" __attribute__((visibility("default"))) poplar::program::Program \
  CONCAT(opcls, Export)(                                                     \
      poplar::Graph & graph, const std::vector<poplar::Tensor>& inputs,      \
      std::vector<poplar::Tensor>& outputs, const std::string& attributes,   \
      const std::string& debug_prefix) {                                     \
    return opcls::program(graph, inputs, outputs, attributes, debug_prefix); \
  }
// Generate the allocator C exported function `#opclsExport_allocator`.
#define IPU_JAX_PRIMITIVE_ALLOCATOR(opcls)                                 \
  extern "C" __attribute__((visibility("default"))) poplar::Tensor CONCAT( \
      opcls, Export_allocator)(                                            \
      poplar::Graph & graph, std::uint32_t operand,                        \
      const std::vector<size_t>& shape, poplar::Type type,                 \
      const std::string& attributes, const std::string& debug_prefix) {    \
    return opcls::allocator(graph, operand, shape, type, attributes,       \
                            debug_prefix);                                 \
  }
// Export IPU JAX primitive as C functions in shared library.
#define EXPORT_IPU_JAX_PRIMITIVE(opcls) \
  IPU_JAX_PRIMITIVE_METADATA(opcls);    \
  IPU_JAX_PRIMITIVE_PROGRAM(opcls);     \
  IPU_JAX_PRIMITIVE_ALLOCATOR(opcls);

// Custom op API level default value.
// Should not required to be changed for ops, only when upgrading Poplar SDK.
extern "C" {
// pybind11 using -fvisibility=hidden. Need to explicit export symbols.
__attribute__((visibility("default"))) int32_t custom_op_api_level = 5;
}
