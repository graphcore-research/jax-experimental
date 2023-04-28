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

from functools import wraps
from typing import Any, Callable, List
import os.path

import cppimport
import cppimport.build_module
import cppimport.importer


def patch_function(orig_fn: Callable, modules_to_patch: List[Any]):
  """Decorator util helping patching any Python function.
  A simple example of patching a Numpy function (or JAX):
  @patch_function(numpy.sin, [numpy])
  def noisy_sin(orig_sin, x):
    print('sining!')
    return orig_sin(x)
  Note that the first argument has to be the original function being patched, to
  avoid some kind of circular dependency in the implementation.
  Args:
      orig_fn: Original function to patch.
      modules_to_patch: Python modules to update with the patched function.
  Returns:
      Patching decorator, taking as first argument the original function.
  """

  def decorator_patch_fn(patched_fn: Callable):

    @wraps(orig_fn)
    def patch_wrapper(*args, **kwargs):
      return patched_fn(orig_fn, *args, **kwargs)

    fn_name = orig_fn.__name__
    for m in modules_to_patch:
      setattr(m, fn_name, patch_wrapper)
    return patch_wrapper

  return decorator_patch_fn


def cppimport_append_include_dirs(include_dirs: List[str]):
  """Append C++ include directories to the default cppimport configuration.
  Args:
    include_dirs: List of additional include directories.
  """
  include_dirs = [os.path.abspath(v) for v in include_dirs]

  # A bit patching of the cppimport building function...
  @patch_function(
      cppimport.build_module.build_module, [cppimport.build_module, cppimport.importer]
  )
  def build_module(orig_fn, module_data):
    cfg = module_data["cfg"]
    cfg["include_dirs"] = cfg.get("include_dirs", []) + include_dirs
    return orig_fn(module_data)
