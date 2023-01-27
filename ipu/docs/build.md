# How to build experimental JAX on IPU Mk2

Building and testing JAX on IPU requires the following base configuration: Ubuntu 20.04, [Graphcore Poplar SDK 3.1](https://www.graphcore.ai/posts/poplar-sdk-3.1-now-available) and [Bazel 5.2.0](https://docs.bazel.build/versions/5.2.0/install.html). Other OS or Python versions have not been tested.

Additionally, one needs to install to the following packages in your Python environment:
```bash
pip install numpy==1.21.6 scipy cython pytest
```
Note: the build process will work with more recent versions of Numpy, but that will then limit Numpy backward compatibility.

## Build IPU JAXLIB

Building `jaxlib` currently requires the branch `jax-v0.3.15-ipu`. Once the branch checked out, the build process is similar to other backends:
```bash
export TF_POPLAR_BASE=#...poplar install directory
python build/build.py --bazel_options=--override_repository=org_tensorflow=PATH/tensorflow-jax-mk2-experimental
```
The `override_repository` config is optional. By default, the build process will pull the experimental IPU TensorFlow XLA code from the repository https://github.com/graphcore-research/tensorflow-jax-mk2-experimental.

If the build is successful, a binary `jaxlib` Python wheel will be produced in the `dist/` directory.


For testing purposes, it is also possible to produce a build with debug info:
```bash
python build/build.py --bazel_options='--copt=-g3' --bazel_options='--strip=never' --bazel_options='--linkopt' --bazel_options='-Wl,--gdb-index'
```

## Package IPU JAX

Packaging `jax` Python wheel is fairly straightforward, using the default branch `jax-v0.3.16-ipu`:
```bash
python setup.py bdist_wheel
```
Similarly to the previous section, the `jax` Python wheel will be produced in `dist/`.

## Run IPU unit tests

The branch `jax-v0.3.16-ipu` contains a collection of IPU specific unit tests in `tests/ipu`, mostly covering bug fixes improving feature coverage. The later can be run as following using the IPU model:
```bash
XLA_IPU_PLATFORM_DEVICE_COUNT=2 TF_POPLAR_FLAGS='--use_ipu_model' pytest -vvv ./tests/ipu/
```
