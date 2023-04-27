<div align="center">
<img src="https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png" alt="logo"></img>
</div>


# :red_circle: **Non-official experimental** :red_circle: JAX on Graphcore IPU

[![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/graphcore-research/jax-experimental?container=graphcore%2Fpytorch-jupyter%3A3.1.0-ubuntu-20.04&machine=Free-IPU-POD4&file=%2Fipu%2Fexamples%2Fjax_ipu_quickstart.ipynb)
![Continuous integration](https://github.com/graphcore-research/jax-experimental/actions/workflows/jax-ci-ipu-public.yaml/badge.svg)

[**Install guide**](#installation)
| [**Quickstart**](#minimal-example)
| [**IPU JAX on Paperspace**](#jax-on-ipu-paperspace-notebooks)
| [**Documentation**](#documentation)

:red_circle: :warning: **Non-official experimental** :warning: :red_circle:

This is a very thin fork of http://github.com/google/jax for Graphcore IPU.
This package is provided by Graphcore Research for **experimentation purposes only**, not production (inference or training).


## Features and limitations of experimental JAX on IPUs

The following **features** are supported:

* Vanilla JAX API: no additional IPU specific API, any code written for IPUs is backward compatible with other backends (CPU/GPU/TPU);
* [JAX asynchronous dispatch](https://jax.readthedocs.io/en/latest/async_dispatch.html) on IPU backend;
* Multiple IPUs with collectives using `pmap` and (experimental) `pjit`;
* Large coverage of [JAX lax operators](https://jax.readthedocs.io/en/latest/jax.lax.html#operators);
* Support of JAX [buffer donation](https://jax.readthedocs.io/en/latest/faq.html#buffer-donation) to keep parameters on IPU SRAM;

Known **limitations** of the project:

* No eager mode (every JAX call has to be compiled, loaded, and finally executed on IPU device);
* IPU code generated can be larger than official Graphcore [TensorFlow](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/tensorflow/intro.html) or [PopTorch](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/index.html) (limiting batch size or model size);
* Multi-IPUs collective have topology restrictions (following Graphcore GCL API);
* Missing [linear algebra operators](https://jax.readthedocs.io/en/latest/jax.lax.html#module-jax.lax.linalg);
* Incomplete support of JAX random number generation on IPU device;
* Deactivated support of JAX infeeds and outfeeds;

This is a **research project**, not an official Graphcore product. Expect bugs and sharp edges. Please help by trying it out, [reporting bugs](https://github.com/graphcore-research/jax-experimental/issues), and letting us know what you think!

## Installation

The experimental JAX wheels require **Ubuntu 20.04**, [**Graphcore Poplar SDK 3.1 or 3.2**](https://www.graphcore.ai/) and **Python 3.8**, and can be installed as following:
```bash
pip install jax==0.3.16+ipu jaxlib==0.3.15+ipu.sdk310 -f https://graphcore-research.github.io/jax-experimental/wheels.html
```
For SDK 3.2, please change `jaxlib` version to `jaxlib==0.3.15+ipu.sdk320`.

## Minimal example

The following example can be run on [Graphcore IPU Paperspace](https://www.paperspace.com/graphcore) (or on a non-IPU machine using the IPU emulator):

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

## JAX on IPU Paperspace notebooks

* [JAX on IPU quickstart](ipu/examples/jax_ipu_quickstart.ipynb) [![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/graphcore-research/jax-experimental?container=graphcore%2Fpytorch-jupyter%3A3.1.0-ubuntu-20.04&machine=Free-IPU-POD4&file=%2Fipu%2Fexamples%2Fjax_ipu_quickstart.ipynb)
* [JAX `pmap` on IPUs quickstart](ipu/examples/jax_ipu_pmap_quickstart.ipynb) [![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/graphcore-research/jax-experimental?container=graphcore%2Fpytorch-jupyter%3A3.1.0-ubuntu-20.04&machine=Free-IPU-POD4&file=%2Fipu%2Fexamples%2Fjax_ipu_pmap_quickstart.ipynb)
* [Stateful linear regression on IPU](ipu/examples/jax_ipu_linear_regression.ipynb) [![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/graphcore-research/jax-experimental?container=graphcore%2Fpytorch-jupyter%3A3.1.0-ubuntu-20.04&machine=Free-IPU-POD4&file=%2Fipu%2Fexamples%2Fjax_ipu_linear_regression.ipynb)
* [MNIST neural net training on IPU](ipu/examples/jax_ipu_mnist_neural_net_training.ipynb) [![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/graphcore-research/jax-experimental?container=graphcore%2Fpytorch-jupyter%3A3.1.0-ubuntu-20.04&machine=Free-IPU-POD4&file=%2Fipu%2Fexamples%2Fjax_ipu_mnist_neural_net_training.ipynb)
* [GNN training on IPU](ipu/examples/jax_ipu_gnn_training.ipynb) [![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/graphcore-research/jax-experimental?container=graphcore%2Fpytorch-jupyter%3A3.1.0-ubuntu-20.04&machine=Free-IPU-POD4&file=%2Fipu%2Fexamples%2Fjax_ipu_gnn_training.ipynb)
* [JAX `pjit` on IPUs quickstart](ipu/examples/jax_ipu_pjit_quickstart.ipynb) [![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/graphcore-research/jax-experimental?container=graphcore%2Fpytorch-jupyter%3A3.1.0-ubuntu-20.04&machine=Free-IPU-POD4&file=%2Fipu%2Fexamples%2Fjax_ipu_pjit_quickstart.ipynb)


**Additional JAX on IPU examples:**

* [JAX on IPU quickstart notebook](ipu/examples/ipu_quickstart.ipynb);
* [MNIST classifier training on IPU](ipu/examples/mnist_classifier.py);
* [MNIST classifier training on multiple IPUs using `pmap`](ipu/examples/mnist_classifier_multi_ipus.py);


**Useful JAX backend flags:**

As standard in JAX, these flags can be set using `from jax.config import config` import.


| Flag | Description |
| ---- | --- |
| `config.FLAGS.jax_platform_name ='ipu'/'cpu'` | Configure default JAX backend. Useful for CPU initialization. |
| `config.FLAGS.jax_ipu_use_model = True`       | Use IPU model emulator. |
| `config.FLAGS.jax_ipu_model_num_tiles = 8`    | Set the number of tiles in the IPU model. |
| `config.FLAGS.jax_ipu_device_count = 2`       | Set the number of IPUs visible in JAX. Can be any local IPU available. |
| `config.FLAGS.jax_ipu_visible_devices = '0,1'`  | Set the specific collection of local IPUs to be visible in JAX. |

Alternatively, like other JAX flags, these can be set using environment variables (e.g. `JAX_IPU_USE_MODEL`, `JAX_IPU_MODEL_NUM_TILES`,...).

**Useful [PopVision](https://www.graphcore.ai/developer/popvision-tools) environment variables:**

* Generate PopVision Graph analyser profile: `POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "debug.allowOutOfMemory":"true"}'`
* Generate PopVision system analyser profile: `PVTI_OPTIONS='{"enable":"true", "directory":"./reports"}'`

## Documentation

* [Performance tips for JAX on IPUs;](ipu/docs/performance.md)
* [How to build experimental JAX Python wheels for IPUs;](ipu/docs/build.md)
* [Original JAX readme;](README_ORIGINAL.md)

## License

The project remains licensed under the **Apache License 2.0**, with the following files unchanged:
* [JAX Apache license 2.0](./LICENSE);
* [JAXLIB and compiled dependencies licenses;](build/LICENSE.txt)

The additional dependencies introduced for Graphcore IPU support are:
* [Poplar: licensed under the Graphcore EULA](https://docs.graphcore.ai/projects/eula/en/latest/);
* [PopLibs: licensed under MIT license](https://docs.graphcore.ai/en/latest/licenses.html);
* [TensorFlow XLA for IPU: licensed under Apache license 2.0](https://docs.graphcore.ai/en/latest/licenses.html);
