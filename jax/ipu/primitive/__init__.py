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
import os.path

from .cppimport_utils import cppimport_append_include_dirs
from .ipu_custom_primitive_utils import ipu_mlir_lowering_custom_primitive, PrimitiveMetadata
from .xla_utils import dtype_to_primitive_type, dtype_to_tf_datatype_enum, xla_shape_to_aval

# Update default `cppimport` library dirs to include headers in this directory.
cppimport_append_include_dirs([os.path.dirname(__file__)])
