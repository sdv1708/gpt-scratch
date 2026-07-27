# GPT from Scratch

Building a working GPT one primitive at a time — starting from a hand-derived gradient descent step and ending at a transformer that generates text. No autograd shortcuts in the foundations layer: every gradient in `foundations/` is derived and implemented by hand in NumPy before PyTorch is allowed anywhere near it.

The goal isn't to produce another wrapper around `nn.Transformer`. It's to be able to explain, at every layer, exactly what the math is doing and why.

**Author:** Sanjay DV

---

## Status

The build runs bottom-up. Each layer is finished and verified before the next one starts.

| Stage | Progress |
| --- | --- |
| Math foundations | ✅ Complete |
| Neural network from scratch | ✅ Complete |
| PyTorch & normalization | ✅ Complete |
| Attention & transformer blocks | 🔜 In progress |
| Data pipeline & tokenization | 🔜 Planned |
| Training loop & generation | 🔜 Planned |

---

## What's built

### `foundations/` — the math, by hand

Pure NumPy. Every derivative written out explicitly, no `.backward()`.

| Module | What it implements |
| --- | --- |
| `gradient_descent.py` | Gradient descent on a scalar objective — the update rule in isolation |
| `activations.py` | Sigmoid and ReLU |
| `softmax.py` | Softmax with max-subtraction for numerical stability |
| `loss.py` | Binary and categorical cross-entropy, with probability clipping |
| `linear_regression.py` | Forward pass and mean squared error |
| `linear_regression_training.py` | Analytic MSE gradient and the full training loop |
| `neuron.py` | A single neuron: weighted sum, bias, activation |
| `backprop.py` | Backprop through one sigmoid neuron — the chain rule, spelled out |
| `multi_layer_backprop.py` | Full forward and backward pass through a two-layer network, including the ReLU gradient mask |
| `mlp.py` | Arbitrary-depth MLP forward pass |
| `weight_init.py` | Xavier/Glorot and Kaiming/He init, plus an activation-variance probe across depth |
| `pytorch_basics.py` | Tensor reshaping, reduction, concatenation, and loss in PyTorch |

### `model/` — normalization

| Module | What it implements |
| --- | --- |
| `normalization.py` | Layer normalization with learnable scale and shift |
| `batch_normalization.py` | Batch normalization with running statistics and separate train/inference paths |
| `rms_normalization.py` | RMS normalization — no mean centering, no shift (the LLaMA/modern-LLM variant) |

---

## Notes on the implementation

A few decisions worth calling out, since they're the parts that actually matter:

**Numerical stability is handled, not assumed.** Softmax subtracts the row max before exponentiating so large logits don't overflow. Cross-entropy clips predictions into `[ε, 1-ε]` rather than nudging them with an additive epsilon — clipping bounds the loss from both directions, which matters when a confident model is wrong.

**Backprop is derived, not called.** `multi_layer_backprop.py` propagates the error through the output layer, back across `W2`, through the ReLU gradient mask, and into `W1` — each step written as an explicit matrix expression. This is the piece that makes the rest of the stack legible.

**Batch norm's two modes are treated as genuinely different.** Training normalizes against batch statistics and updates the running estimates; inference uses the running estimates only and touches nothing. Conflating the two is the classic batch norm bug, so the branch is explicit.

**Weight initialization is verified empirically.** `check_activations` forwards a random input through a configurable stack and reports the activation standard deviation at each layer — making signal decay under bad init something you can observe rather than take on faith.

---

## Roadmap

Next up, in order:

1. **Attention** — scaled dot-product self-attention, then multi-head
2. **Transformer block** — attention + feed-forward + residual connections + normalization
3. **GPT** — embeddings, positional encoding, stacked blocks, output head
4. **Data pipeline** — character-level vocabulary, BPE tokenizer, batched loader
5. **Training & generation** — training loop and autoregressive sampling
6. **Inference optimization** — KV cache, grouped-query attention

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. Dependencies are NumPy, PyTorch, and torchtyping.
