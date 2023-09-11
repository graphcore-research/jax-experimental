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
#include <dlfcn.h>

#include <cstdlib>
#include <iostream>
#include <ipu_custom_primitive.hpp>
#include <sstream>

struct XlaCustomCallStatus;
/**
 * Typedefs of XLA host callback functions exported in IPU XLA shared library.
 */
typedef void (*XlaPythonCpuCallbackType)(uint64_t callback_ptr, void* output,
                                         void** inputs,
                                         XlaCustomCallStatus* status);
typedef XlaCustomCallStatus* (*XlaAllocateCustomCallStatusType)();
typedef void (*XlaFreeCustomCallStatusType)(XlaCustomCallStatus* status);
typedef const char* (*XlaGetErrorCustomCallStatusType)(
    XlaCustomCallStatus* status);

/**
 * @brief IPU Python callback primitive implementation.
 *
 * Using host custom op mechanism in IPU XLA backend.
 */
class IpuPythonCallbackPrimitive : public jax::ipu::PrimitiveInterface {
 public:
  static jax::ipu::PrimitiveMetadata metadata(std::uint32_t num_inputs) {
    return jax::ipu::PrimitiveMetadata{.num_inputs = num_inputs,
                                       .is_elementwise = false,
                                       .is_stateless = false,
                                       .is_hashable = false,
                                       .input_to_output_tensor_aliasing = {}};
  }

  static void hostCallback(const std::vector<const void*>& data,
                           const std::vector<std::uint32_t>& number_of_elements,
                           const std::vector<void*>& outputs,
                           const std::string& attributes,
                           const std::string& debugPrefix) {
    static_assert(sizeof(uintptr_t) == sizeof(uint64_t),
                  "Expected 64-bit pointers");
    // Parse host callback metadata: callback pointer + library filename.
    std::istringstream insstream(attributes);
    std::string callback_ptr_str, library_filename;
    std::getline(insstream, callback_ptr_str, ';');
    std::getline(insstream, library_filename, ';');
    const std::uint64_t callback_ptr = std::atol(callback_ptr_str.c_str());

    // Opening IPU XLA extension library
    // TODO: loading only once? Proper closing?
    void* ipu_xla_extension_lib = dlopen(library_filename.c_str(), RTLD_LAZY);
    // Get IPU callbacks methods.
    const auto xla_allocate_status =
        reinterpret_cast<XlaAllocateCustomCallStatusType>(
            dlsym(ipu_xla_extension_lib, "IpuXlaAllocateCustomCallStatus"));
    const auto xla_free_status = reinterpret_cast<XlaFreeCustomCallStatusType>(
        dlsym(ipu_xla_extension_lib, "IpuXlaFreeCustomCallStatus"));
    const auto xla_cpu_callback = reinterpret_cast<XlaPythonCpuCallbackType>(
        dlsym(ipu_xla_extension_lib, "IpuXlaPythonCpuCallback"));
    const auto xla_error_msg_status =
        reinterpret_cast<XlaGetErrorCustomCallStatusType>(
            dlsym(ipu_xla_extension_lib, "IpuXlaGetErrorCustomCallStatus"));

    // Call XLA backend CPU callback, passing input data pointers.
    auto status = xla_allocate_status();
    void** inputs_ptr = (void**)(data.data());
    xla_cpu_callback(callback_ptr, nullptr, inputs_ptr, status);
    const char* call_error_msg = xla_error_msg_status(status);
    if (call_error_msg != nullptr) {
      std::cerr << "IPU host callback error. " << call_error_msg << std::endl;
      throw pybind11::value_error(call_error_msg);
    }
    xla_free_status(status);
  }
};

EXPORT_IPU_JAX_HOST_CALLBACK(IpuPythonCallbackPrimitive);

// Declare a pybind11, to provide easy compilation & import from Python.
PYBIND11_MODULE(ipu_python_callback_impl, m) {
  pybind11::class_<IpuPythonCallbackPrimitive>(m, "IpuPythonCallbackPrimitive")
      .def_static("metadata", &IpuPythonCallbackPrimitive::metadata,
                  pybind11::arg("num_inputs"));
}

// cppimport configuration for compiling the pybind11 module.
// clang-format off
/*
<%
cfg['extra_compile_args'] = ['-std=c++17', '-fPIC', '-O2', '-Wall']
cfg['libraries'] = ['poplar', 'poputil']
cfg['include_dirs'] = []
setup_pybind11(cfg)
%>
*/
