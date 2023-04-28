// cppimport
// NOTE: comment necessary for automatic JIT compilation of the module.

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
#include <ipu_custom_primitive.hpp>
#include <popops/ElementWise.hpp>

namespace pe = popops::expr;
/**
 * @brief Custom elementwise primitive (for unit testing).
 */
class CustomActivationPrimitive : public jax::ipu::PrimitiveInterface {
 public:
  static jax::ipu::PrimitiveMetadata metadata(std::uint32_t num_inputs) {
    return jax::ipu::PrimitiveMetadata{
        .num_inputs = num_inputs,
        .is_elementwise = true,
        .is_stateless = true,
        .is_hashable = true,
        .input_to_output_tensor_aliasing = {{0, 0}}};
  }

  static poplar::program::Program program(
      poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
      std::vector<poplar::Tensor>& outputs, const std::string& attributes,
      const std::string& debug_prefix) {
    if (inputs.size() != 2) {
      throw poputil::poplibs_error(
          "IPU custom mul primitive expecting two input tensors.");
    }
    // Map inplace: x = abs(x) * y
    auto expr = pe::Mul(pe::Abs(pe::_1), pe::_2);
    poplar::program::Sequence prog;
    popops::mapInPlace(graph, expr, {inputs[0], inputs[1]}, prog,
                       poplar::DebugContext(debug_prefix),
                       poplar::OptionFlags());
    outputs.push_back(inputs[0]);
    return prog;
  }
};

// Export the IPU JAX primitive in the shared library.
EXPORT_IPU_JAX_PRIMITIVE(CustomActivationPrimitive);

// Declare a pybind11, to provide easy compilation & import from Python.
PYBIND11_MODULE(ipu_custom_activation_impl, m) {
  pybind11::class_<CustomActivationPrimitive>(m, "CustomActivationPrimitive")
      .def_static("metadata", &CustomActivationPrimitive::metadata,
                  pybind11::arg("num_inputs"));
}

// cppimport configuration for compiling the pybind11 module.
// clang-format off
/*
<%
cfg['extra_compile_args'] = ['-std=c++17', '-fPIC', '-O2', '-Wall']
cfg['libraries'] = ['poplar', 'poputil', 'poprand', 'popops']
cfg['include_dirs'] = []
setup_pybind11(cfg)
%>
*/
