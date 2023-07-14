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
#include <iostream>
#include <ipu_custom_primitive.hpp>
#include <poplar/Graph.hpp>
#include <poputil/TileMapping.hpp>

/**
 * @brief IPU ThreeFry2x32 primitive implementation.
 */
class IpuThreeFry2x32Primitive : public jax::ipu::PrimitiveInterface {
 public:
  static jax::ipu::PrimitiveMetadata metadata(std::uint32_t num_inputs) {
    return jax::ipu::PrimitiveMetadata{.num_inputs = num_inputs,
                                       .is_elementwise = false,
                                       .is_stateless = true,
                                       .is_hashable = true,
                                       .input_to_output_tensor_aliasing = {}};
  }

  static poplar::program::Program program(
      poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
      std::vector<poplar::Tensor>& outputs, const std::string& attributes,
      const std::string& debug_prefix) {
    if (inputs.size() != 4) {
      throw poputil::poplibs_error(
          "IPU ThreeFry primitive expecting 4 input tensors.");
    }
    const auto& target = graph.getTarget();
    const auto num_tiles = target.getNumTiles();
    const auto outshape = inputs[2].shape();
    // Linear mapping for all inputs, to optimally fit ThreeFry sampling (and
    // following unary ops).
    const unsigned grainsize = 16;
    const auto linear_mapping =
        poputil::calcLinearTileMapping(graph, outshape, 1, grainsize);
    graph.setTileMapping(inputs[0], linear_mapping);
    graph.setTileMapping(inputs[1], linear_mapping);
    graph.setTileMapping(inputs[2], linear_mapping);
    graph.setTileMapping(inputs[3], linear_mapping);

    // Flattening all inputs.
    const auto& key0 = inputs[0].flatten();
    const auto& key1 = inputs[1].flatten();
    const auto& data0 = inputs[2].flatten();
    const auto& data1 = inputs[3].flatten();

    const auto debugContext = poplar::DebugContext("ipu_threefry2x32");
    // Output tensors, per tile (following same tile mapping).
    std::vector<poplar::Tensor> out0_list;
    std::vector<poplar::Tensor> out1_list;
    // Using data0 tile mapping as reference. May mean comms on key arrays.
    const auto mapping0 = graph.getTileMapping(data0);
    const auto cs = graph.addComputeSet(debugContext);

    for (unsigned tile = 0; tile < num_tiles; tile++) {
      // no data on this tile.
      if (mapping0[tile].empty()) {
        continue;
      }
      // Get contiguous regions we can pass to vertex.
      const auto tileContiguousRegions =
          graph.getSortedContiguousRegions(data0, mapping0[tile]);
      if (tileContiguousRegions.size() == 0) {
        continue;
      }

      for (const auto& regions : tileContiguousRegions) {
        for (const auto& r : regions) {
          const auto region_size = r.size();
          // Corresponding output regions
          auto out0_region = graph.addVariable(poplar::UNSIGNED_INT,
                                               {region_size}, debugContext);
          auto out1_region = graph.addVariable(poplar::UNSIGNED_INT,
                                               {region_size}, debugContext);
          graph.setTileMapping(out0_region, tile);
          graph.setTileMapping(out1_region, tile);
          out0_list.push_back(out0_region);
          out1_list.push_back(out1_region);

          // Call threefry custom vertex on the tile.
          auto v = graph.addVertex(cs, "ThreeFry2x32Vertex",
                                   {{"key0", key0.slice(r)},
                                    {"key1", key1.slice(r)},
                                    {"data0", data0.slice(r)},
                                    {"data1", data1.slice(r)},
                                    {"out0", out0_region},
                                    {"out1", out1_region}});
          graph.setTileMapping(v, tile);
          // Empirical perf. estimate, measured on IPU hardware.
          graph.setPerfEstimate(v, region_size * 130);
        }
      }
    }
    auto out0 = poplar::concat(out0_list).reshape(outshape);
    auto out1 = poplar::concat(out1_list).reshape(outshape);
    outputs.push_back(out0);
    outputs.push_back(out1);
    return poplar::program::Execute(cs, debugContext);
  }
};

// Export the IPU JAX primitive in the shared library.
EXPORT_IPU_JAX_PRIMITIVE(IpuThreeFry2x32Primitive);

// Declare a pybind11, to provide easy compilation & import from Python.
PYBIND11_MODULE(ipu_random_primitive_impl, m) {
  pybind11::class_<IpuThreeFry2x32Primitive>(m, "IpuThreeFry2x32Primitive")
      .def_static("metadata", &IpuThreeFry2x32Primitive::metadata,
                  pybind11::arg("num_inputs"));
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
