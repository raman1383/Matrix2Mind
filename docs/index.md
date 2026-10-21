---
layout: default
title: Matrix2Mind Blog
description: Exploring transformers, data pipelines, large-scale training & inference.
theme: jekyll-theme-hacker
---

# Matrix2Mind: From Matrix to Mind

Exploring transformers, data pipelines, large-scale training & inference.

I built a small(~10M params) transformer model with some interesting configs:

* with just two numbers(q_heads, kv_heads) you can configure the attention sub_blocks to multiple variants: single-head, multi-query, grouped query and finally multi-head attention.

* for the FFN sub_block you can choose the config from simple MLP, gated linear unit, or mixture of experts.

* for inference I've implemented a ring-buffer kv cache with continuous batching, it adapts to the kv configs you set in attention sub_blocks above.

* NNs can fail silently, I've written tests that reveal such silent killers of performance before going on an expensive training run.

* did studies comparing architecture variants and their compute/performance characteristic


## Articles

1. [Transformer Model Overview](1_transformer_model_overview.md)
2. [Data Pipeline](2_data_pipeline.md)
3. [Transformer Block Internals](3_transformer_block_internals.md)
4. [Scaling Up](4_scaling_up.md)
