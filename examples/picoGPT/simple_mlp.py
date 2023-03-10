from functools import partial

import jax
import numpy as onp
from jax.config import config
from tqdm import tqdm
from typing import List
from utils import load_encoder_hparams_and_params

import jax.numpy as np

config.FLAGS.jax_platform_name = 'ipu'


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")
x = np.array(onp.random.randint(hparams["n_vocab"], size=(64))  )
print(hparams)


n_hid = 128
w1 = onp.random.random((hparams["n_embd"], n_hid))
b1 = onp.random.random((n_hid))
w2 = onp.random.random((n_hid, n_hid))
b2 = onp.random.random((n_hid)) 
w3 = onp.random.random((n_hid, hparams["n_vocab"]))
b3 = onp.random.random((hparams["n_vocab"])) 

model_params = dict(
        l1 = dict(w = w1, b = b1),
        l2 = dict(w = w2, b = b2),
        l3 = dict(w = w3, b = b3),
    )

def model(x, wte, l1, l2, l3, num_reps):
    x = wte[x, :] 
    x = linear(x, **l1)
    for _ in num_reps:
        x = linear(x, **l2)
    out = linear(x, **l3)
    out /= np.sum(out)
    return out

num_to_generate = np.arange(10)

@jax.jit
def generate(x, wte, num_layers, model_params, num_to_generate):
    for rep in tqdm(num_to_generate, desc="generating"):   
        fake_logs = model(x, wte, **model_params, num_reps=num_layers)
        next_id = np.argmax(fake_logs[-1])
        x = np.concatenate((x, next_id.reshape(1)), axis=0)
        
    print(x[0])
    return x

wte = params["wte"]
num_layers = 5
num_layers = np.arange(num_layers)
out = generate(x, wte, num_layers, model_params, num_to_generate)
print(out, out.shape)

output_ids = onp.array(out)
# TODO: This needs checking for dimmensions.
output_text = encoder.decode(output_ids)
print(output_text)

