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

#include "ipu_custom_primitive.hpp"

PYBIND11_MODULE(ipu_custom_primitive_impl, m) {
  // Primitive metadata bindings.
  using PrimitiveMetadata = jax::ipu::PrimitiveMetadata;
  namespace py = pybind11;
  py::class_<PrimitiveMetadata>(m, "PrimitiveMetadata")
      .def(py::init<>())
      .def_readwrite("num_inputs", &PrimitiveMetadata::num_inputs)
      .def_readwrite("is_elementwise", &PrimitiveMetadata::is_elementwise)
      .def_readwrite("is_stateless", &PrimitiveMetadata::is_stateless)
      .def_readwrite("is_hashable", &PrimitiveMetadata::is_hashable)
      .def_readwrite("input_to_output_tensor_aliasing",
                     &PrimitiveMetadata::input_to_output_tensor_aliasing)
      .def_readwrite("allocating_indices",
                     &PrimitiveMetadata::allocating_indices)
      .def_readwrite("replica_identical_output_indices",
                     &PrimitiveMetadata::replica_identical_output_indices);
}

// cppimport configuration for compiling the pybind11 module.
// clang-format off
/*
<%
cfg['extra_compile_args'] = ['-std=c++17', '-fPIC', '-O2', '-Wall']
cfg['libraries'] = ['poplar', 'poputil', 'poprand']
cfg['include_dirs'] = []
setup_pybind11(cfg)
%>
*/
