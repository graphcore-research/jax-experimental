from functools import partial

import jax
import numpy as onp
from jax.config import config
from tqdm import tqdm
from typing import List

import jax.numpy as np
from utils import load_encoder_hparams_and_params

config.FLAGS.jax_platform_name = 'cpu'

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b

def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    # w = c_fc["w"]
    # b = c_fc["b"]
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v

def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # split into heads
    # qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]
    qkv_heads = [np.split(x, n_head, axis=-1) for x in qkv]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

    # merge heads
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x

# @partial(jax.jit)
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]
    # import ipdb; ipdb.set_trace()
    # print(x)
    # print(x.shape)
    # x =  np.add(x, y)

    return x


# @partial(jax.jit)
def gpt2(inputs, wte, wpe, blocks, ln_f):  # [n_seq] -> [n_seq, n_vocab]
    n_head = 12
    # token + positional embeddings
    # print(inputs, tuple(inputs), wte.shape)
    x = wte[inputs, :] 
    x+= wpe[np.arange((len(inputs)))]  # [n_seq] -> [n_seq, n_embd]
    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    out = x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]
    # import ipdb; ipdb.set_trace()
    return out


# @partial(jax.jit, static_argnames=["start", "n_head", "tokens_to_generate"])
def generate(inputs, start, params, n_head, n_tokens_to_generate):
    for _ in tqdm(n_tokens_to_generate):  # auto-regressive decode loop
        # import ipdb; ipdb.set_trace()

        # import ipdb; ipdb.set_trace()
        logits = gpt2(inputs, **params)  # model forward pass
        print(logits[-1], logits[-1].shape, np.argmax(logits[-1]))
        next_id = np.argmax(logits[-1])  # greedy sampling
        # mask = mask.at[start + i].set(1)
        # import ipdb; ipdb.set_trace()
        # inputs = inputs.at[start + i].set(next_id)  # append prediction to input
        inputs = np.concatenate((inputs, next_id.reshape((1))), axis=0)
        print(inputs)
        # inputs.append(next_id)
    # return 1
    return inputs  


def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    print("IPU devices:", jax.devices("ipu"))

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt) 
    mask = [1] * len(input_ids)
    print(mask)
    mask += [0]*(n_tokens_to_generate - len(input_ids))
    # input_ids += [-100]*(n_tokens_to_generate - len(input_ids))
    print(mask)
    print(f"Num inputs: {len(input_ids)}")
    
    # assert len(mask) == len(input_ids), f"{len(mask)} != {len(input_ids)}"
    
    

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    input_ids = np.array(input_ids)
    print(input_ids)
    start = mask.index(0)
    mask = np.array(mask)
    n_tokens_to_generate = np.arange(n_tokens_to_generate)
    output_ids = generate(input_ids, start, params, hparams["n_head"], n_tokens_to_generate)
    # generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # decode the ids back into a string
    print(output_ids)
    print(output_ids.device())
    output_ids = onp.array(output_ids)
    # TODO: This needs checking for dimmensions.
    output_text = encoder.decode(output_ids)
    print(f"how many generated: {len(output_ids)}")
    print(f"The generated_output is {output_text}")

    return output_text

if __name__ == "__main__":
    import fire

    fire.Fire(main)



# @partial(jax.jit, backend="ipu")
# def ipu_function(data, pow, bias):
#     return data**pow + bias

# data = np.array([1, -2, 3], np.float32)
# output = ipu_function(data, 2, 1)
# print(output, output.device())