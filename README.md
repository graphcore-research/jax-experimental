<div align="center">
<img src="https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png" alt="logo"></img>
</div>


# :red_circle: **Non-official experimental** :red_circle: JAX on Graphcore IPU

:red_circle: :warning: **Non-official experimental** :warning: :red_circle: It's a very thin fork of http://github.com/google/jax for Graphcore IPU. This package is provided by Graphcore research team for **experimentation purposes only**, not production (inference or training).

## Features and limitations of experimental JAX on IPUs

Experimental JAX on IPUs supports the following **features**:

* Vanilla JAX API: no additional IPU specific API, any code written for IPUs is backward compatible with other backends (CPU/GPU/TPU); 
* Large coverage of [JAX lax operators](https://jax.readthedocs.io/en/latest/jax.lax.html#operators);
* Support of JAX [buffer donation](https://jax.readthedocs.io/en/latest/faq.html#buffer-donation) to keep parameters on IPU SRAM;
* Support of infeeds and outfeeds for high performance on-device training loop;

Known **limitations** of the project:

* Single IPU support (hence no `pjit`/`pmap`);
* No eager mode (every JAX call has to be compiled, loaded, and finally executed on IPU device);
* IPU code generated can be much larger than official Graphcore [TensorFlow](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/tensorflow/intro.html) or [PopTorch](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/index.html) (limiting batch size or model size);
* Missing support of [linear algebra operators](https://jax.readthedocs.io/en/latest/jax.lax.html#module-jax.lax.linalg);
* Incomplete support of JAX random numbers generation on IPU device;

There is no at the moment **no plan** to tackle these issues. Use at your own risk!

## Installation

The experimental JAX wheels require Ubuntu 20.04, Graphcore **Poplar SDK 3.1** and **Python 3.8**, and can be installed as following:
```bash
pip install TODO: public url jaxlib.whl
pip install TODO: public url jax.whl
```

Alternatively, download the `zip` archive of the latest release.

## Minimal example

The following example can be run on [Graphcore IPU Paperspace](https://www.paperspace.com/graphcore) (or locally using the IPU model):

```python
from functools import partial
import jax
import numpy as np

@partial(jax.jit, backend="ipu")
def ipu_function(data):
    return data**2 + 1

data = np.array([1, -2, 3], np.float32)
output = ipu_function(data)
print(output, output.device())
```

**Additional JAX on IPU examples:**

* [JAX on IPU quickstart notebook](ipu/examples/ipu_quickstart.ipynb);
* [MNIST classifier training on IPU](ipu/examples/mnist_classifier.py)
* [MNIST classifier training on IPU, with infeeds](ipu/examples/mnist_classifier_with_infeed.py)


**Useful environment variables:**

* Use IPU model: `TF_POPLAR_FLAGS='--use_ipu_model --ipu_model_tiles=8'`
* Configure the default JAX backend: `JAX_PLATFORM_NAME=cpu|gpu|tpu|ipu`
* Generate PopVision Graph analyser profile: `POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "debug.allowOutOfMemory":"true"}'`
* Generate PopVision system analyser profile: `PVTI_OPTIONS='{"enable":"true", "directory":"./reports"}'`

## Documentation

* [Performance tips for JAX on IPUs;](ipu/docs/performance.md)
* [How to build experimental JAX Python wheels for IPUs;](ipu/docs/build.md)
* [Original JAX readme;](README_ORIGINAL.md)
