# HyperbolicHPE — v3 changes

Surgical changes targeting MPJPE reduction *and* compute efficiency, motivated
by the assessment in `../HYPERPOSE_assessment.md`.

## Files touched

| File | Change |
| --- | --- |
| `model/attention.py` | Tangent-flow API · drop `acosh` from HKPSA logit · fused QKV · hierarchical kinematic bias `γ₁A + γ₂A² + γ₃A³` · velocity reuses fused QKV |
| `model/network.py` | Tangent-flow data path between blocks · multi-scale temporal windows · per-joint output head · velocity stays at origin |
| `train.py` | `compute_kinematics` → central differences · pass `temporal_windows` and `num_joints` through to the model |
| `configs/hyperbolic_hpe.yaml` | New `temporal_windows: [3, 9, 27]` field (legacy scalar still honoured) |
| `tests/test_attention.py` | Rewritten for the v3 tangent-flow API |
| `scripts/smoke_test_v3.py` | New end-to-end smoke test |

## What each change buys

### MPJPE drivers

* **Multi-scale temporal windows `W ∈ {3, 9, 27}`** — effective receptive field
  jumps from ~19 frames to ~165 frames, covering H36M's gait cycle. This is
  the single highest-leverage change.
* **Hierarchical kinematic bias `γ₁A + γ₂A² + γ₃A³`** — captures
  parent / sibling / cousin relationships in the kinematic tree, not just
  immediate adjacency. `γ_k` are learnable, init γ₁=1, γ₂=γ₃=0 so behaviour
  matches v2 at start.
* **Per-joint output head** — replaces the shared `Linear(d, 3)` decoder
  with 17 small per-joint heads (vectorised via einsum, one fused GEMM).
  Decouples joints at decode time.
* **Central-difference velocity** — `(x_{t+1} − x_{t−1}) / 2` is unbiased to
  second order and has ~2× the SNR of backward differences. Zero new params.

### Compute / speed

* **Drop `acosh` from HKPSA logit** — softmax is invariant under monotone
  transforms of logits and `−<q,k>_L = cosh(d_L)` is monotone in `d_L`, so
  `(1 + <q,k>_L) / τ` gives the same attention weights up to a τ
  reparameterisation. Saves an `acosh`, a `clamp`, and a `**2` per pair — and
  removes a bf16-precision-sensitive operation.
* **Tangent-flow architecture** — hidden state lives in `T_oℍᵈ` between blocks.
  Manifold representations are constructed only inside HKPSA (`exp_o` on Q/K
  for the geodesic logit) and once at the optional `return_manifold` exit.
  Eliminates ~6 `log_o`/`exp_o` round-trips per forward pass.
* **Velocity at origin** — embedding no longer parallel-transports velocities
  to joint locations, and the spatial block no longer transports them back.
  Removes 2 PT calls per spatial block × 3 blocks = 6 PTs per forward.
* **Fused QKV linear** — one `Linear(d, 3d)` instead of three `Linear(d, d)`.
  Reused on velocity for the kinematic-coherence penalty.
* **Temporal block manifold-free** — since v2's temporal block already did
  Euclidean dot-product attention in the tangent space, the v3 version drops
  the `log_o` / `exp_o` boundary calls entirely.

## How to run

```bash
conda activate hpe
cd hyperbolic
python scripts/smoke_test_v3.py            # end-to-end shape / drift / grad check
pytest tests/test_attention.py -v          # unit tests
python train.py --config configs/hyperbolic_hpe.yaml --use-wandb
```

## Notes on resuming from v2 checkpoints

The architecture has changed (fused QKV, per-joint head, hierarchical γ_k,
removed in-block PTs). v2 checkpoints will not load directly into v3 —
`load_state_dict` will report missing/unexpected keys. Start a fresh run.

## What was *deliberately* not changed

* Embedding dim `d = 512` — kept identical so we can isolate the effect of
  the architectural changes from any capacity change.
* Number of spatial / temporal blocks (3 each).
* Loss suite — `L_mpjpe`, `L_vel`, `L_bone`, `L_drift` and
  `UncertaintyWeightedLoss` are unchanged. Drift will be *much* smaller in v3
  (single `exp_o` at exit instead of one per block per layer).
* Curvature `c = −1` — learnable curvature is on the wishlist but adds risk.
  Defer until v3 baseline is measured.
