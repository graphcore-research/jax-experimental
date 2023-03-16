from functools import partial

import jax
import numpy as onp
from jax.config import config
from tqdm import tqdm
from typing import List


import jax.numpy as np

config.FLAGS.jax_platform_name = "ipu"


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b

def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = linear(x, **c_fc)  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def attention(
    q, k, v, mask
):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


# encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")
hparams = {"n_vocab": 50257, "n_ctx": 1024, "n_embd": 768, "n_head": 12, "n_layer": 12}
n_seq = 25
x = np.array(onp.random.randint(hparams["n_vocab"], size=(n_seq)))
print(hparams)


n_hid = 768
w1 = onp.random.random((hparams["n_embd"], n_hid))  # [768, 768] = 589,000
b1 = onp.random.random((n_hid))  # [768]
w3 = onp.random.random(
    (n_hid, hparams["n_vocab"])
)  # [768, 50257] = 38,000,000 <---- ~ 144 Mb in fp32, 72 Mb in fp16
b3 = onp.random.random((hparams["n_vocab"]))  # [50257]

model_params = dict(
    l1=dict(w=w1, b=b1),
    l3=dict(w=w3, b=b3),
)


def model(x, wte, blocks, l1, l3):
    x = wte[x, :]
    x = linear(x, **l1)
    for block in blocks:
        x = ffn(x, **block) # [n_seq, n_embd] -> [n_seq, n_embd]
    
    out = linear(x, **l3)
    out /= np.sum(out)
    return out


num_to_generate = np.arange(10)


@jax.jit
def generate(x, wte, blocks, model_params, num_to_generate):
    for rep in tqdm(num_to_generate, desc="generating"):
        fake_logs = model(
            x,
            wte,
            blocks,
            **model_params,
        )
        next_id = np.argmax(fake_logs[-1])
        x = np.concatenate((x, next_id.reshape(1)), axis=0)

    return x

# word token embeddings
wte = onp.random.random((hparams["n_vocab"], n_hid)).astype(
    np.float16
)  # [768, 50257] = 38,000,000  --- 72 Mb in fp16


"""
Keeping the shapes and sizes 
"""
num_blocks = 5

# ~ 5,000,000 parameters / block --> ~ 10 Mb each 
fnn_blocks = [{
            "c_fc": {
                "w": onp.random.random([768, 3072]).astype(onp.float16),
                "b": onp.random.random([3072]).astype(onp.float16),
            },
            "c_proj": {
                "w": onp.random.random([3072, 768]).astype(onp.float16),
                "b": onp.random.random([768]).astype(onp.float16),
            },} for _ in range(num_blocks)]

out = generate(x, wte, fnn_blocks, model_params, num_to_generate)
print(out, out.shape)

output_ids = onp.array(out)

# output_text = encoder.decode(output_ids)
# print(output_text)
