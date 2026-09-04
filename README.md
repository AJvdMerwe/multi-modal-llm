
# Multi Model LLM
Journey to build a multi-modal LLM from scratch. This is done to build a deeper intuition about LLM architecture.

This will mainly be based off of Andrej Karpathy's building GPT-2 from scratch and then extending that to multi-modal models (e.g. text, vision, audio).

## Source material
Andrej Karpathy's Build GPT-2 from Scratch
https://youtu.be/l8pRSuU81PU?si=LWjxpxLxHL0JuKhH

## Architecture Updates (Current Branch)
This branch introduces several structural and optimization upgrades to the base GPT-2 architecture:

### 🔹 Rotary Positional Embeddings (RoPE)
- Absolute positional encodings have been replaced with **Rotary Positional Embeddings** to enhance sequence modeling and extrapolation capabilities [1].
- Queries and Keys are rotated using precomputed sinusoidal caches before attention computation.
- The standard positional embedding layer (`wpe`) is omitted; positional information is now implicit in the attention mechanism.

### 🔹 Soft Mixture of Experts (Soft-MoE)
- The standard Feed-Forward Network (MLP) in each Transformer block is replaced with a **Soft-MoE** layer, building on foundational MoE research [2] and soft-routing methodologies [3].
- **Slot-Based Soft Routing:** Tokens are dynamically dispatched across expert slots using a learnable 3D gate parameter. Dispatch and combination weights are computed via dual softmax operations.
- **Shared Expert Pathway:** A parallel shared expert preserves general knowledge and stabilizes gradient flow, a pattern widely adopted in modern MoE scaling efforts [4].
- **Efficient Computation:** Expert weights are parameterized as 3D tensors and computed in parallel via batch matrix multiplications (`bmm`), significantly improving throughput.

### 🔹 Configuration & Routing Parameters
The `GPTConfig` dataclass now includes MoE-specific hyperparameters:
- `n_experts`: Total number of experts per block (default: 8)
- `top_k`: Experts activated per token (used in the alternative hard-routing `MoE_Expert` variant)
- `slots_per_expert`: Granularity of the soft dispatch mechanism (default: 1)

### 🔹 Training & Optimization Stack
- **Model Compilation:** `torch.compile()` is applied to accelerate execution.
- **Mixed Precision:** Training utilizes `bfloat16` autocasting for memory efficiency and faster convergence.
- **Distributed Training:** Full DDP support (NCCL backend), gradient accumulation, and synchronized loss averaging across processes.
- **Optimizer & Scheduling:** Custom AdamW configuration with weight decay (`0.1`), gradient clipping (`max_norm=1.0`), and a cosine decay learning rate schedule with warmup.

## References
[1] Sun, Y., et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864.  
[2] Shazeer, N., et al. (2017). *Outrageously Large Neural Networks: The Sparsed-Gated Mixture-of-Experts Layer*. arXiv:1701.06538.  
[3] Lewis, P., et al. (2020). *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*. arXiv:2006.16668.  
[4] Jiang, A. Q., et al. (2024). *Mixtral of Experts*. arXiv:2401.04088. (Shared expert & routing stabilization patterns)