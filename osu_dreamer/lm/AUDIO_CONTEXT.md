# Audio Context: Hierarchical Multi-Scale Windows

### Core Idea
- Let each scale s have its own radius parameters: $r_{\text{past}}^{(s)}$ and $r_{\text{future}}^{(s)}$.
- Scale 0 (current frame): window length 1, represents the current audio frame only.
- Scale s ≥ 1: window length $r_{\text{past}}^{(s)} + r_{\text{future}}^{(s)}$, each vector represents $W_{s-1}$ frames where $W_0 = 1$ and $W_s = 1 + r_{\text{past}}^{(s)} + r_{\text{future}}^{(s)}$ for s ≥ 1.
- Continue stacking scales until the desired total temporal coverage is achieved.

With S additional scales (beyond scale 0):
- Vectors per scale: $N_0 = 1$, and $N_s = r_{\text{past}}^{(s)} + r_{\text{future}}^{(s)}$ for s ≥ 1.
- Total context vectors: $N_{\text{total}} = 1 + \sum_{s=1}^{S} \left(r_{\text{past}}^{(s)} + r_{\text{future}}^{(s)}\right)$.
- Total temporal coverage (in frames): $\prod_{s=1}^{S} \left(1 + r_{\text{past}}^{(s)} + r_{\text{future}}^{(s)}\right)$.

### Context Size and Coverage

Forward-biased (r_past=1, r_future=3):
| S | 1 | 2 | 3 | 4 | 5 | 6 |
|---|-----|-----|-----|-----|-----|-----|
| Context Size | 5 | 9 | 13 | 17 | 21 | 25 |
| Temporal Coverage | 5 | 25 | 125 | 625 | 3125 | 15625 |

Symmetric (r_past=2, r_future=2):
| S | 1 | 2 | 3 | 4 | 5 |
|---|-----|-----|-----|-----|-----|
| Context Size | 5 | 9 | 13 | 17 | 21 |
| Temporal Coverage | 5 | 25 | 125 | 625 | 3125 |

Strong forward bias (r_past=1, r_future=4):
| S | 1 | 2 | 3 | 4 |
|---|-----|-----|-----|-----|
| Context Size | 6 | 11 | 16 | 21 |
| Temporal Coverage | 6 | 36 | 216 | 1296 |

### Layout at Decode Time t
For each scale s:
- Center at audio time t.
- Scale 0 (finest): indices $\{t - R, \dots, t, \dots, t + R\}$ (stride 1 frame).
- Scales $s \ge 1$: place $2R$ vectors symmetrically around the finest window, each summarizing a block of $W_{s-1} = 1 + 2R$ frames. Block centers are offset by multiples of $W_{s-1}$.

### Feature Construction
- Frame-level embeddings at hop H (e.g., 10 ms) form the base per-scale features.
- Scale pooling:
  - Scale 0: identity (frame embeddings).
  - Scale s≥1: summarize contiguous blocks of $W_{s-1}$ frames (e.g., mean pooling).

### Temporal Alignment and Edges
- Center windows at t; clamp near boundaries.
- If insufficient frames exist for a block, pad with zeros and mark with an attention mask.
- Provide an attention mask for padded vectors (optional but recommended).

### Training and Inference Behavior
- Training:
  - The audio encoder is unfrozen and trained end-to-end with the model.
- Inference:
  - Audio is encoded once for the entire track at every scale (precompute frame embeddings per scale).
  - The hierarchical context is updated whenever a `TIME_SHIFT` token is emitted by slicing the precomputed per-scale embeddings at the current time index t for each scale.

### Cross-Attention Shapes
- Let B be batch size, L be token sequence length, C = (1 + 2RS) be context vectors per step, and D be feature dim.
- Cross-attention Keys/Values must support per-(B, L) context:
  - Shape: [B, L, C, D]. Each token position L_i for each batch B_i receives its own context slice at time t_i.

### Recommended Defaults (tunable)
- $R = 2$, $S = 3$ → 13 vectors cover $(1+2R)^S = 125$ frames.
- Frame hop H = 10 ms (encoder output).
- D (post-projection) = 256–512.
- Pooling: mean pooling + per-scale LayerNorm before projection.
- Scale embeddings: learned vector per scale.

### Implementation Notes
- Indexing:
  - Scale 0 indices: $t + \Delta$ where $\Delta \in [-R, R]$.
  - Scale s≥1 block centers: $t + k \cdot W_{s-1}$, $k \in \{-R, \dots, -1, 1, \dots, R\}$ (skip 0 to avoid overlap with finer window's center).
- Caching: precompute frame embeddings; at runtime, gather by index and pool blocks.
- Context update: at runtime, derive per-(B, L) time indices from token timestamps and slice precomputed per-scale embeddings accordingly.
- Determinism: given $R$, $S$, H, and encoder, context composition is deterministic.

### Versioning
Record $(R, S, H, D)$, pooling choice, per-scale projections, and encoder checkpoint with the model/dataset. 