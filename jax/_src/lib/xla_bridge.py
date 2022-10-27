# Copyright 2018 Google LLC
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

"""Interface and utility functions to XLA.

This module wraps the XLA client(s) and builders to standardize their interfaces
and provide some automatic type mapping logic for converting between Numpy and
XLA. There are also a handful of related casting utilities.
"""

from functools import partial, lru_cache
import os
import platform as py_platform
import threading
from typing import Any, Dict, List, Optional, Union
import warnings

from absl import logging
# Disable "WARNING: Logging before flag parsing goes to stderr." message
logging._warn_preinit_stderr = 0

import jax._src.lib
from jax._src.config import flags, bool_env, int_env
from jax._src import distributed
from jax._src.lib import tpu_driver_client
from jax._src.lib import xla_client
from jax._src import util, traceback_util
from jax.config import config
import numpy as np

from jaxlib import ipu_xla_client

iree: Optional[Any]

try:
  import jax._src.iree as iree  # type: ignore
except (ModuleNotFoundError, ImportError):
  iree = None

traceback_util.register_exclusion(__file__)


XlaBackend = xla_client._xla.Client

FLAGS = flags.FLAGS

# TODO(phawkins): Remove jax_xla_backend.
flags.DEFINE_string(
    'jax_xla_backend', '',
    'Deprecated, please use --jax_platforms instead.')
flags.DEFINE_string(
    'jax_backend_target',
    os.getenv('JAX_BACKEND_TARGET', '').lower(),
    'Either "local" or "rpc:address" to connect to a remote service target.')
# TODO(skye): warn when this is used once we test out --jax_platforms a bit
flags.DEFINE_string(
    'jax_platform_name',
    os.getenv('JAX_PLATFORM_NAME', '').lower(),
    'Deprecated, please use --jax_platforms instead.')
flags.DEFINE_bool(
    'jax_disable_most_optimizations',
    bool_env('JAX_DISABLE_MOST_OPTIMIZATIONS', False),
    'Try not to do much optimization work. This can be useful if the cost of '
    'optimization is greater than that of running a less-optimized program.')
flags.DEFINE_integer(
    'jax_xla_profile_version', int_env('JAX_XLA_PROFILE_VERSION', 0),
    'Optional profile version for XLA compilation. '
    'This is meaningful only when XLA is configured to '
    'support the remote compilation profile feature.')

def get_compile_options(
    num_replicas: int,
    num_partitions: int,
    device_assignment=None,
    use_spmd_partitioning: bool = True,
    use_auto_spmd_partitioning: bool = False,
    auto_spmd_partitioning_mesh_shape=[],
    auto_spmd_partitioning_mesh_ids=[]) -> xla_client.CompileOptions:
  """Returns the compile options to use, as derived from flag values.

  Args:
    num_replicas: Number of replicas for which to compile.
    num_partitions: Number of partitions for which to compile.
    device_assignment: Optional ndarray of jax devices indicating the assignment
      of logical replicas to physical devices (default inherited from
      xla_client.CompileOptions). Must be consistent with `num_replicas` and
      `num_partitions`.
    use_spmd_partitioning: boolean indicating whether to enable SPMD or MPMD
      partitioning in XLA.
    use_auto_spmd_partitioning: boolean indicating whether to automatically
      generate XLA shardings for SPMD partitioner.
    auto_spmd_partitioning_mesh_shape: device mesh shape used to create
      auto_spmd_partitioning search space.
    auto_spmd_partitioning_mesh_ids: device ids used to create
      auto_spmd_partitioning search space.
  """
  compile_options = xla_client.CompileOptions()
  compile_options.num_replicas = num_replicas
  compile_options.num_partitions = num_partitions
  build_options = compile_options.executable_build_options
  build_options.use_spmd_partitioning = use_spmd_partitioning
  build_options.use_auto_spmd_partitioning = use_auto_spmd_partitioning
  if use_auto_spmd_partitioning:
    build_options.auto_spmd_partitioning_mesh_shape = auto_spmd_partitioning_mesh_shape
    build_options.auto_spmd_partitioning_mesh_ids = auto_spmd_partitioning_mesh_ids
  if device_assignment is not None:
    logging.vlog(
        2,
        'get_compile_options: num_replicas=%s num_partitions=%s device_assignment=%s',
        num_replicas, num_partitions, device_assignment)
    device_assignment = np.array(device_assignment)

    # Allow 1D device assignment if num_partitions is 1.
    if (device_assignment.ndim == 1) and (num_partitions == 1):
      device_assignment = device_assignment[:, None]

    if num_replicas != device_assignment.shape[0]:
      msg = 'device_assignment does not match num_replicas: {} vs {}.'
      raise ValueError(msg.format(device_assignment, num_replicas))

    if num_partitions != device_assignment.shape[1]:
      msg = 'device_assignment does not match num_partitions: {} vs {}.'
      raise ValueError(msg.format(device_assignment, num_partitions))

    if device_assignment.dtype == object:
      device_assignment = np.vectorize(lambda d: d.id, otypes=[int])(
          device_assignment)
    device_assignment = xla_client.DeviceAssignment.create(device_assignment)
    assert device_assignment.replica_count() == num_replicas
    assert device_assignment.computation_count() == num_partitions
    compile_options.device_assignment = device_assignment

  debug_options = compile_options.executable_build_options.debug_options
  if jax._src.lib.cuda_path is not None:
    debug_options.xla_gpu_cuda_data_dir = jax._src.lib.cuda_path

  if FLAGS.jax_disable_most_optimizations:

    debug_options.xla_backend_optimization_level = 0
    debug_options.xla_llvm_disable_expensive_passes = True
    debug_options.xla_test_all_input_layouts = False

  if jax._src.lib.xla_extension_version >= 68:
    compile_options.profile_version = FLAGS.jax_xla_profile_version
  return compile_options


# Backends

def _make_tpu_driver_client():
  if tpu_driver_client is None:
    logging.info("Remote TPU is not linked into jax; skipping remote TPU.")
    return None
  if FLAGS.jax_backend_target is None:
    logging.info("No --jax_backend_target was provided; skipping remote TPU.")
    return None
  return tpu_driver_client.TpuBackend.create(worker=FLAGS.jax_backend_target)


def tpu_client_timer_callback(timer_secs: float):
  def _log_warning():
    warnings.warn(
      f'TPU backend initialization is taking more than {timer_secs} seconds. '
      'Did you run your code on all TPU hosts? '
      'See https://jax.readthedocs.io/en/latest/multi_process.html '
      'for more information.')

  # Will log a warning after `timer_secs`.
  t = threading.Timer(timer_secs, _log_warning)
  t.start()

  try:
    client = xla_client.make_tpu_client()
  finally:
    t.cancel()

  return client


# Backends, in increasing order of preference.
# We have no particular opinion about how "backends" relate to "devices". For
# example, there could be multiple backends that provide the same kind of
# device.
_backend_factories = {}
_default_backend = None
_backends : Dict[str, Any] = {}
_backends_errors : Dict[str, str] = {}
_backend_lock = threading.Lock()

def register_backend_factory(name, factory, *, priority=0):
  with _backend_lock:
    if name in _backends:
      raise RuntimeError(f"Backend {name} already initialized")
  _backend_factories[name] = (factory, priority)


register_backend_factory('interpreter', xla_client.make_interpreter_client,
                         priority=-100)
register_backend_factory('cpu',
                         partial(xla_client.make_cpu_client, use_tfrt=True),
                         priority=0)
register_backend_factory('tpu_driver', _make_tpu_driver_client,
                         priority=100)
# Registering IPU client in the same way as GPU/TPU.
register_backend_factory('ipu', ipu_xla_client.make_ipu_client,
                         priority=100)


def make_gpu_client(platform_name=None):
  return xla_client.make_gpu_client(
    distributed_client=distributed.global_state.client,
    node_id=distributed.global_state.process_id,
    platform_name=platform_name)

if hasattr(xla_client, "make_gpu_client"):
  register_backend_factory(
      'cuda', partial(make_gpu_client, platform_name='cuda'),
      priority=200)
  register_backend_factory(
      'rocm', partial(make_gpu_client, platform_name='rocm'),
      priority=200)

if hasattr(xla_client, "make_tpu_client"):
  register_backend_factory(
    'tpu', partial(tpu_client_timer_callback, timer_secs=60.0), priority=300)

if iree is not None:
  register_backend_factory("iree", iree.iree_client_factory, priority=-100)

_platform_aliases = {
  "cuda": "gpu",
  "rocm": "gpu",
}

_alias_to_platforms: Dict[str, List[str]] = {}
for _platform, _alias in _platform_aliases.items():
  _alias_to_platforms.setdefault(_alias, []).append(_platform)


def is_known_platform(platform: str):
  # A platform is valid if there is a registered factory for it. It does not
  # matter if we were unable to initialize that platform; we only care that
  # we've heard of it and it isn't, e.g., a typo.
  return (platform in _backend_factories.keys() or
          platform in _platform_aliases.keys())


def canonicalize_platform(platform: str) -> str:
  """Replaces platform aliases with their concrete equivalent.

  In particular, replaces "gpu" with either "cuda" or "rocm", depending on which
  hardware is actually present. We want to distinguish "cuda" and "rocm" for
  purposes such as MHLO lowering rules, but in many cases we don't want to
  force users to care.
  """
  platforms = _alias_to_platforms.get(platform, None)
  if platforms is None:
    return platform

  b = backends()
  for p in platforms:
    if p in b.keys():
      return p
  raise RuntimeError(f"Unknown backend: '{platform}' requested, but no "
                     f"platforms that are instances of {platform} are present. "
                     "Platforms present are: " + ",".join(b.keys()))


def expand_platform_alias(platform: str) -> List[str]:
  """Expands, e.g., "gpu" to ["cuda", "rocm"].

  This is used for convenience reasons: we expect cuda and rocm to act similarly
  in many respects since they share most of the same code.
  """
  return _alias_to_platforms.get(platform, [platform])

def is_gpu(platform):
  return platform in ("cuda", "rocm")

def backends():
  global _backends
  global _backends_errors
  global _default_backend

  with _backend_lock:
    if _backends:
      return _backends
    if config.jax_platforms:
      jax_platforms = config.jax_platforms.split(",")
      platforms = []
      # Allow platform aliases in the list of platforms.
      for platform in jax_platforms:
        platforms.extend(expand_platform_alias(platform))
      priorities = range(len(platforms), 0, -1)
      platforms_and_priorites = zip(platforms, priorities)
    else:
      platforms_and_priorites = (
          (platform, priority) for platform, (_, priority)
          in _backend_factories.items())
    default_priority = -1000
    for platform, priority in platforms_and_priorites:
      try:
        backend = _init_backend(platform)
        _backends[platform] = backend

        if priority > default_priority:
          _default_backend = backend
          default_priority = priority
      except Exception as err:
        if platform in ('cpu', 'interpreter'):
          # We always expect the CPU and interpreter backends to initialize
          # successfully.
          raise
        else:
          # If the backend isn't built into the binary, or if it has no devices,
          # we expect a RuntimeError.
          err_msg = f"Unable to initialize backend '{platform}': {err}"
          if config.jax_platforms:
            raise RuntimeError(err_msg)
          else:
            _backends_errors[platform] = str(err)
            logging.info(err_msg)
            continue
    # We don't warn about falling back to CPU on Mac OS, because we don't
    # support anything else there at the moment and warning would be pointless.
    if (py_platform.system() != "Darwin" and
        _default_backend.platform == "cpu" and
        FLAGS.jax_platform_name != 'cpu'):
      logging.warning('No GPU/TPU found, falling back to CPU. '
                      '(Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)')
    return _backends


def _clear_backends():
  global _backends
  global _backends_errors
  global _default_backend

  logging.info("Clearing JAX backend caches.")
  with _backend_lock:
    _backends = {}
    _backends_errors = {}
    _default_backend = None

  get_backend.cache_clear()


def _init_backend(platform):
  factory, unused_priority = _backend_factories.get(platform, (None, None))
  if factory is None:
    raise RuntimeError(f"Unknown backend '{platform}'")

  logging.vlog(1, "Initializing backend '%s'" % platform)
  backend = factory()
  # TODO(skye): consider raising more descriptive errors directly from backend
  # factories instead of returning None.
  if backend is None:
    raise RuntimeError(f"Could not initialize backend '{platform}'")
  if backend.device_count() == 0:
    raise RuntimeError(f"Backend '{platform}' provides no devices.")
  util.distributed_debug_log(("Initialized backend", backend.platform),
                             ("process_index", backend.process_index()),
                             ("device_count", backend.device_count()),
                             ("local_devices", backend.local_devices()))
  logging.vlog(1, "Backend '%s' initialized" % platform)
  return backend


def _get_backend_uncached(platform=None):
  # TODO(mattjj,skyewm): remove this input polymorphism after we clean up how
  # 'backend' values are handled
  if not isinstance(platform, (type(None), str)):
    return platform

  platform = (platform or FLAGS.jax_xla_backend or FLAGS.jax_platform_name
              or None)

  bs = backends()
  if platform is not None:
    platform = canonicalize_platform(platform)
    backend = bs.get(platform, None)
    if backend is None:
      if platform in _backends_errors:
        raise RuntimeError(f"Backend '{platform}' failed to initialize: "
                           f"{_backends_errors[platform]}")
      raise RuntimeError(f"Unknown backend {platform}")
    return backend
  else:
    return _default_backend


@lru_cache(maxsize=None)  # don't use util.memoize because there is no X64 dependence.
def get_backend(platform=None):
  return _get_backend_uncached(platform)


def get_device_backend(device=None):
  """Returns the Backend associated with `device`, or the default Backend."""
  if device is not None:
    return device.client
  return get_backend()


def device_count(backend: Optional[Union[str, XlaBackend]] = None) -> int:
  """Returns the total number of devices.

  On most platforms, this is the same as :py:func:`jax.local_device_count`.
  However, on multi-process platforms where different devices are associated
  with different processes, this will return the total number of devices across
  all processes.

  Args:
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    Number of devices.

  """
  return int(get_backend(backend).device_count())


def local_device_count(backend: Optional[Union[str, XlaBackend]] = None) -> int:
  """Returns the number of devices addressable by this process."""
  return int(get_backend(backend).local_device_count())


def devices(backend: Optional[Union[str, XlaBackend]] = None) -> List[xla_client.Device]:
  """Returns a list of all devices for a given backend.

  .. currentmodule:: jaxlib.xla_extension

  Each device is represented by a subclass of :class:`Device` (e.g.
  :class:`CpuDevice`, :class:`GpuDevice`). The length of the returned list is
  equal to ``device_count(backend)``. Local devices can be identified by
  comparing :attr:`Device.process_index` to the value returned by
  :py:func:`jax.process_index`.

  If ``backend`` is ``None``, returns all the devices from the default backend.
  The default backend is generally ``'gpu'`` or ``'tpu'`` if available,
  otherwise ``'cpu'``.

  Args:
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    List of Device subclasses.
  """
  return get_backend(backend).devices()


def default_backend() -> str:
  """Returns the platform name of the default XLA backend."""
  return get_backend(None).platform


def local_devices(process_index: Optional[int] = None,
                  backend: Optional[Union[str, XlaBackend]] = None,
                  host_id: Optional[int] = None) -> List[xla_client.Device]:
  """Like :py:func:`jax.devices`, but only returns devices local to a given process.

  If ``process_index`` is ``None``, returns devices local to this process.

  Args:
    process_index: the integer index of the process. Process indices can be
      retrieved via ``len(jax.process_count())``.
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    List of Device subclasses.
  """
  if host_id is not None:
    warnings.warn(
        "The argument to jax.local_devices has been renamed from `host_id` to "
        "`process_index`. This alias will eventually be removed; please update "
        "your code.")
    process_index = host_id
  if process_index is None:
    process_index = get_backend(backend).process_index()
  if not (0 <= process_index < process_count()):
    raise ValueError(f"Unknown process_index {process_index}")
  return [d for d in devices(backend) if d.process_index == process_index]


def process_index(backend: Optional[Union[str, XlaBackend]] = None) -> int:
  """Returns the integer process index of this process.

  On most platforms, this will always be 0. This will vary on multi-process
  platforms though.

  Args:
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    Integer process index.
  """
  return get_backend(backend).process_index()


# TODO: remove this sometime after jax 0.2.13 is released
def host_id(backend=None):
  warnings.warn(
      "jax.host_id has been renamed to jax.process_index. This alias "
      "will eventually be removed; please update your code.")
  return process_index(backend)


def process_count(backend: Optional[Union[str, XlaBackend]] = None) -> int:
  """Returns the number of JAX processes associated with the backend."""
  return max(d.process_index for d in devices(backend)) + 1


# TODO: remove this sometime after jax 0.2.13 is released
def host_count(backend=None):
  warnings.warn(
      "jax.host_count has been renamed to jax.process_count. This alias "
      "will eventually be removed; please update your code.")
  return process_count(backend)


# TODO: remove this sometime after jax 0.2.13 is released
def host_ids(backend=None):
  warnings.warn(
      "jax.host_ids has been deprecated; please use range(jax.process_count()) "
      "instead. jax.host_ids will eventually be removed; please update your "
      "code.")
  return list(range(process_count(backend)))
