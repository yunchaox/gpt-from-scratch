# GPT-from-scratch — a character-level GPT built line by line

A decoder-only transformer (GPT) implemented **from scratch in PyTorch** and trained to generate Shakespeare-style text at the character level. Every component — attention, LayerNorm, the transformer block, the training loop, and autoregressive generation — is written by hand to understand *why* the architecture is shaped the way it is, not just how to call it.

> Trained on the ~1MB tiny-Shakespeare corpus. Character-level tokenizer (vocab ≈ 65). No external transformer libraries used for the model itself.

## Results

Loss starts at **4.33** at initialization — slightly above the uniform-guessing
baseline of `ln(vocab_size)` ≈ 4.17, since default PyTorch init produces small
non-zero logits rather than a flat distribution — and falls to **1.24 train /
1.53 val** over 3k steps. Both splits are measured the same way, averaged over
200 batches with dropout disabled (`model.eval()`), so the train/val gap is a
real generalization signal and not a dropout artifact:

| step | 0 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|---|
| **train** | 4.330 | 1.874 | 1.520 | 1.376 | 1.306 | 1.264 | 1.244 |
| **val**   | 4.321 | 1.995 | 1.719 | 1.601 | 1.555 | 1.536 | 1.527 |

An earlier run used a flat `lr = 1e-4` and reached 1.44 / 1.65. Replacing it
with 100-step linear warmup into cosine decay (peak `3e-4`, floor `3e-5`) plus
gradient clipping at norm 1.0 cut val loss by 0.12 nats — an 11.6% perplexity
reduction — for the same 3k-step budget.

Val loss is nearly flat over the final 500 steps (−0.009), so this run is
converged for this configuration rather than compute-limited. The train/val gap
of 0.283 nats — wider than the 0.21 nats of the slower run — is the expected
cost of better optimization, and points at regularization, not more steps, as
the next lever: dropout is currently applied in the MLP only, not on attention
probabilities or the residual path.

Sample generation (untrained → gibberish; trained → learns the *form* of a play):

```
APTISTA:
A am yet like you, mark your live; you presied.

MOPSA:
This upon is youngly and he pardon my lord be
And house pramised use surped commands.

CATESBY:
Which is is! what thou Cousin!
A shall has sack.
Who Patil! it me in a proclaiment:
Gea, sir!

Clongsentymeant.
Which must wear,
To plot Capule the craise soul to their arder
The tank that, what, Has sack'd in my living
Flow's muse with the set it for thurswer'd.

Luch:
O, heaven, now not that is ower but of hence.
That is contrumnious 
```

Semantically it's nonsense — it's a tiny model — but starting from characters alone it induced words, capitalization, apostrophes, line breaks, and the `CHARACTER:` dialogue format.

## Architecture

Standard decoder-only GPT:
<p align="center">
  <img src="architecture.svg" alt="GPT architecture diagram" width="680">
</p>

- **Token + learned positional embeddings**, summed → `(B, T, C)`.
- **N stacked transformer blocks**, each = causal multi-head self-attention + position-wise MLP, **pre-norm** with residual connections.
- **Final LayerNorm** → **LM head** (linear → vocab logits).
- Loss: cross-entropy on next-token prediction at every position.

Default config: `d_model=384`, `6` heads, `6` layers, `block_size=256`, `batch=64`.

## Design notes (the *why* behind the choices)

- **Character-level tokenizer.** Tiny vocab (~65) → a clean sanity check that initial loss ≈ `ln(65) ≈ 4.17`, and the measured 4.33 is the small excess from default PyTorch init producing non-zero rather than flat logits. At scale you'd switch to BPE — attention is O(T²), so shorter subword sequences win — but for a learning project char-level's simplicity dominates.
- **Causal masked attention.** Self-attention is permutation-invariant, so positional embeddings inject order. The causal mask blocks each position from attending to future tokens — which is what makes training on *all* positions at once valid (no position can peek at its own target). Remove the mask and the next-token loss collapses to ~0 while learning nothing.
- **Pre-norm, not post-norm.** LayerNorm is applied to each sublayer's *input* (`x = x + sublayer(LN(x))`), leaving the residual stream an unnormalized identity path from input to output. That clean path keeps gradients stable at depth (no per-layer LayerNorm rescaling compounding), so training tolerates a far shorter warmup than post-norm transformers require. The one final `ln_f` before the head normalizes the accumulated residual stream once.
- **Numerically stable softmax.** Subtract the row max before exponentiating; with `-inf` masking, the causal mask guarantees every row keeps at least its diagonal entry, so there's never a fully-masked row to produce NaNs.
- **Generation crops context to `block_size`.** The positional-embedding table has exactly `block_size` rows, so the generated sequence must be cropped to the last `block_size` tokens each step — otherwise the position lookup overflows the table. Sampling uses `multinomial` (not greedy `argmax`) for diversity; temperature would sit between the two.
- **AdamW.** Per-parameter adaptive rates (via the second moment) suit the varied gradient scales in a transformer; decoupled weight decay keeps regularization consistent instead of being distorted by the adaptive scaling.
- **Warmup + cosine LR decay.** 100 steps of linear warmup into cosine decay
  (peak `3e-4`, floor `3e-5`), with gradients clipped to norm 1.0. Warmup exists
  because Adam's second-moment estimate is unreliable in the first few dozen
  steps; the decay exists because a fixed high LR can't settle into a minimum.
  Replacing a flat `1e-4` with this cut val loss from 1.65 to 1.53 on the same
  3k-step budget — an 11.6% perplexity reduction.

## What I learned / debugging notes

First generation came out as pure noise despite a trained model. Rather than tweak the generation code, I isolated model-vs-code with a one-line loss check, confirmed the generate function was correct, and traced it to a re-initialized model instance — a notebook cell-order bug. Fixing execution order fixed the output.

## Run

```bash
pip install -r requirements.txt
python train.py      # trains, prints loss, saves the model
python generate.py   # samples text from the trained model
```

## Roadmap

Next: extend toward post-training — a reward model, then DPO, then PPO/GRPO — to explore RLHF end-to-end.

---
*Built by Yunchao Xu as a from-scratch study of transformer internals.*
