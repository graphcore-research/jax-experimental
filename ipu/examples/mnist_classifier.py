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
# This file has been modified by Graphcore Ltd

"""A basic MNIST example using JAX with the mini-libraries stax and optimizers.

The mini-library jax.example_libraries.stax is for neural network building, and
the mini-library jax.example_libraries.optimizers is for first-order stochastic
optimization.
"""


import time
import itertools
from functools import partial

import numpy.random as npr

import jax
import jax.numpy as jnp
from jax import jit, grad, random
from jax.tree_util import tree_map
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax
from jax.config import config

import datasets


def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return -jnp.mean(jnp.sum(preds * targets, axis=1))

@partial(jit, backend='cpu')
def accuracy(params, batch):
  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(predict(params, inputs), axis=1)
  return jnp.mean(predicted_class == target_class)

init_random_params, predict = stax.serial(
    Dense(1024), Relu,
    Dense(1024), Relu,
    Dense(10), LogSoftmax)

if __name__ == "__main__":
  # Using CPU backend by default for state initialization, randomized data pipeline, ...
  # Not necessary, but speeding up the setup phase of the training.
  config.FLAGS.jax_platform_name = 'cpu'
  rng = random.PRNGKey(0)

  step_size = 0.001
  num_epochs = 10
  batch_size = 128
  momentum_mass = 0.9

  train_images, train_labels, test_images, test_labels = datasets.mnist()
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        # Pad the leftover batch to avoid engine change
        if len(batch_idx) < batch_size:
          batch_idx = perm[-batch_size:]
        yield train_images[batch_idx], train_labels[batch_idx]
  batches = data_stream()

  opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

  # JIT 'update' method for IPU.  We use `donate_argnums` to keep params on IPU SRAM.
  @partial(jit, backend='ipu', donate_argnums=[1])
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

  _, init_params = init_random_params(rng, (-1, 28 * 28))
  opt_state = opt_init(init_params)
  itercount = itertools.count()

  print("Number of IPU visible devices:", len(jax.devices("ipu")))
  print("Starting training...")
  for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
      # State is kept on SRAM. Only transferred once.
      opt_state = update(next(itercount), opt_state, next(batches))

    # Block to get accurate timing of the epoch!
    tree_map(lambda x: x.block_until_ready(), opt_state)
    epoch_time = time.time() - start_time

    # Getting back the state on HOST for accuracy.
    params = get_params(opt_state)
    train_acc = accuracy(params, (train_images, train_labels))
    test_acc = accuracy(params, (test_images, test_labels))
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))
