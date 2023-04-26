# Performance tips for JAX on Graphcore IPUs

The experimental JAX on IPUs supports most of the standard JAX API, and should be compatible with existing single device JAX code. Nevertheless, a few modifications may be required to take full advantage of IPU hardware acceleration. Note that any of this modification is (backend) backward compatible: the same code will keep running in the same way on CPU/GPU/TPU backends.

## Use CPU for random numbers & short running functions

JAX on IPU supports being the default JAX backend. Nevertheless, in this setting, every JAX call will result into a series of compilation/engine load/engine call, where the first two steps are costly from a performance perspective. Hence, the common advice is to use `cpu` as the default backend for short running functions such as model state initialization and data pre-processing, and then use the `ipu` backend to accelerate the core function, where the bulk of computation happens, e.g. the training loop.

The default backend can be set as following in Python:
```python
from jax.config import config
config.FLAGS.jax_platform_name = 'cpu'
```
or using an environment variable: `JAX_PLATFORM_NAME=cpu`.

Then, the typical `update` function of a neural network will have the following signature:
```python
@partial(jax.jit, backend="ipu")
def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)
```
where we indicate to JAX that  `update` needs to be jitted and run using the `ipu` backend.

The training loop remains a classic Python for loop:
```python
for idx, batch in enumerate(batches):
    opt_state = update(idx, opt_state, batch)
```

## Use `donate_argnums` to keep training parameters/weights on IPU SRAM

In the previous example, at every call of `update`, both arguments `opt_state` and `batch` will be transfered from host to device (and `opt_state` transfered back to host after update). Whereas this is the expected behaviour for the input data `batch`, it is clearly highly inefficient to transfer the state at every micro batch iteration.

In order to keep the state on the IPU SRAM (and avoid unnecessary transfers), JAX on IPUs supports the [buffer donation functionality](https://jax.readthedocs.io/en/latest/faq.html#buffer-donation). Hence, slightly modifying the `update` function into:
```python
@partial(jax.jit, backend="ipu", donate_argnums=(1,))
def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(grad(loss)(params, batch), opt_state)
```
will result into an IPU jitted function where only the `batch` is transfered at every call from host to device, and the `opt_state` remains on the IPU SRAM (after being transfered at the first call). The training loop does not require any additional modification.

Please refer to the [MNIST example](../examples/mnist_classifier.py) for a full example of buffer donation on the IPU.
