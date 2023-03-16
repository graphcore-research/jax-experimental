from functools import partial

import jax
import numpy as onp
from jax.config import config
from tqdm import tqdm
from typing import List
# from utils import load_encoder_hparams_and_params
from gpt2_ipu import transformer_block

import jax.numpy as np

config.FLAGS.jax_platform_name = "ipu"

print("WARNING: Converted all variables loaded into fp16!")


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


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
        x = transformer_block(
            x, **block, n_head=12
        )  # [n_seq, n_embd] -> [n_seq, n_embd]
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


# wte = params["wte"]  # [50257, 768]
# import ipdb; ipdb.set_trace()
wte = onp.random.random((hparams["n_vocab"], n_hid)).astype(
    np.float16
)  # [768, 50257] = 38,000,000  --- 72 Mb in fp16

"""
'attn' :
    'c_attn':
        'w': [768, 2304] = 1,790,000
        'b': [2304]
    'c_proj':
        'w': [768, 768] = 589,000
        'b': [768]
'ln_1':
    'b': [768]
    'g': [768]
'ln_2':
    'b': [768]
    'g': [768]
'mlp':
    'c_fc':
        'w': [768, 3072] = 2,359,000
        'b': [3072, ]
    'c_proj':
        'w': [3072, 768] = 2,359,000
        'b': [768]
Approximate: ~7,100,000 parameters. * 2 bytes for fp16 = 14,200,000 bytes / 1,048,576 = 13 Mb
"""
num_blocks = 2
blocks = [
    {
        "attn": {
            "c_attn": {
                "w": onp.random.random([768, 2304]).astype(onp.float16),
                "b": onp.random.random([2304]).astype(onp.float16),
            },
            "c_proj": {
                "w": onp.random.random([768, 768]).astype(onp.float16),
                "b": onp.random.random([768]).astype(onp.float16),
            },
        },
        "ln_1": {
            "b": onp.random.random([768]).astype(onp.float16),
            "g": onp.random.random([768]).astype(onp.float16),
        },
        "ln_2": {
            "b": onp.random.random([768]).astype(onp.float16),
            "g": onp.random.random([768]).astype(onp.float16),
        },
        "mlp": {
            "c_fc": {
                "w": onp.random.random([768, 3072]).astype(onp.float16),
                "b": onp.random.random(
                    [
                        3072,
                    ]
                ).astype(onp.float16),
            },
            "c_proj": {
                "w": onp.random.random([3072, 768]).astype(onp.float16),
                "b": onp.random.random([768]).astype(onp.float16),
            },
        },
    }
    for _ in range(num_blocks)
]


out = generate(x, wte, blocks, model_params, num_to_generate)
print(out, out.shape)

output_ids = onp.array(out)

# output_text = encoder.decode(output_ids)
# print(output_text)
