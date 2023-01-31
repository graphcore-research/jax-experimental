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

## Use infeed/outfeed to speed-up host data transfer

Using JAX buffer donation functionality allows to unlock 80% of the performance on IPUs. Nevertheless, in the cases where one wants to optimize as much as possible a training loop (or any similar loop), there is the possibility to modify more heavily the former using JAX infeeds (and outfeeds).

Let's start by showing a skeleton of training code using infeeds:
```python
# Batch was an argument, now an infeed
batch_shape = (jax.ShapedArray((batch_size, 784), dtype=jnp.float32),
               jax.ShapedArray((batch_size, 10), dtype=jnp.float32))

def update_with_infeed(i, opt_state):
    # Using infeed to transfer batch.
    batch_token = lax.create_token()
    batch, _ = lax.infeed(batch_token, shape=batch_shape)
    # Use the same `update` function.
    return update(i, opt_state, batch)

@partial(jit, backend='ipu', donate_argnums=(2,))
def train_steps(start, end, opt_state):
    # Running epoch loop on device.
    opt_state = lax.fori_loop(start, end, update_with_infeed, opt_state)
    return opt_state

device = jax.devices(backend='ipu')[0]
# Pre-fill the infeed queue with batches.
for batch in batches:
    device.transfer_to_infeed(batch)
# Run first epoch.
opt_state = train_steps(0, len(batches), opt_state)
```

The main idea behind the use of JAX infeed/outfeed is to be able to directly run the main training loop on the device, using a JAX `fori_loop`, instead of the host Python for loop. The main benefit is then to remove the latency cost of calling a jitted function on the IPU (a single call at every epoch instead of `num_batches`). 

As previously said, the use of infeed/outfeed is only beneficial to users who wants to obtain the maximal performance. In most cases, buffer donation with `donate_argnums` would be enough, and has the benefit of keeping the training loop much simpler.

Please refer to the [MNIST with infeeds example](../examples/mnist_classifier_with_infeed.py) for a full example of how to use infeeds on IPUs.
