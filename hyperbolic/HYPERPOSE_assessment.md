# Assessment of HYPERPOSE — Hyperbolic Kinematic Phase-Space Attention for 3D HPE

*Reviewer notes against both the manuscript and the codebase under `hyperbolic/` (model/, math_utils/, loss/).*

---

## 1. What the paper actually does

HYPERPOSE is the first 3D-HPE *lifting* network whose entire spatio-temporal reasoning runs on the Lorentz hyperboloid $\mathbb{H}^d$ (curvature $c=-1$). The pitch is geometric: a kinematic skeleton is a tree, and trees embed into hyperbolic space with $O(\log)$ distortion versus the exponential distortion in Euclidean space. The architecture is:

1. **Phase-space embedding** — a single linear layer maps each 2D keypoint to $\mathbb{R}^d$, then `project()` adds the time coordinate $x_0=\sqrt{1+\|x\|^2}$ to land on $\mathbb{H}^d$. Velocities $\Delta p_t$ are linearly embedded, padded with a zero time component, and parallel-transported from the origin to the joint location.
2. **Hyperbolic Kinematic Phase-Space Attention (HKPSA)** — spatial attention whose logits combine: (i) negative squared geodesic distance $-d_\mathcal{L}^2(q_i,k_j)/\tau$, (ii) a velocity-coherence penalty computed at the origin tangent space using the $\|a-b\|^2 = \|a\|^2+\|b\|^2-2\langle a,b\rangle$ trick, and (iii) an additive skeleton-adjacency bias $\gamma A_{ij}$. Aggregation is the Einstein midpoint (avoids iterative Fréchet mean).
3. **Windowed hyperbolic temporal attention** — for each joint, attend to $\pm W$ neighbours via `unfold`, giving $O(TW)$ instead of $O(T^2)$. Default $T=243$, $W=3$ → $\sim 35\times$ FLOP cut on the temporal path.
4. **Tangent-space residuals + LayerNorm + TangentFFN**, then `exp_o` back to $\mathbb{H}^d$.
5. **Riemannian loss suite** — MPJPE (Euclidean), geodesic velocity consistency, geodesic bone-length, and a *detached* manifold-drift diagnostic, balanced via Kendall et al. uncertainty weighting.

Reported result: **49.9 mm MPJPE on Human3.6M** (CPN 2D, $T=243$) at **17.6 M params**, but the authors are explicit that this is epoch 19 of 60 and is still improving.

---

## 2. Detailed pipeline

```mermaid
flowchart TD
  A[2D keypoints<br/>B×T×J×3 from CPN]:::input --> B[Linear embed Wp+b<br/>→ B·T×J×d]
  B --> C[project: prepend x0 = √(1+‖·‖²)<br/>→ B·T×J×d+1 on H^d]
  A --> D[Δp_t = p_t − p_t-1]
  D --> E[Linear embed → pad zero time-coord]
  E --> F[Parallel transport o→h_x<br/>velocity tangent at joint]
  C --> G[Spatial Block 1 - HKPSA]
  F --> G
  G --> H[Temporal Block 1 - Windowed]
  H --> I[Spatial Block 2 - HKPSA]
  I --> J[Temporal Block 2 - Windowed]
  J --> K[Spatial Block 3 - HKPSA]
  K --> L[Temporal Block 3 - Windowed]
  L --> M[log_o · spatial part only]
  M --> N[MLP head 2-layer<br/>→ B×T×J×3]:::output

  subgraph HKPSA [HKPSA — spatial block internals]
    direction TB
    g1[log_o x] --> g2[Q,K,V linear in tangent]
    g2 --> g3[exp_o for Q,K · keep V tangent]
    g3 --> g4[d_L² q_i,k_j  geodesic logits]
    F -.PT v→origin.-> g5[‖v_i−v_j‖² via dot-trick]
    g4 --> g6[+ λ·kin + γ·A_ij + softmax]
    g5 --> g6
    g6 --> g7[Einstein midpoint aggregate]
    g7 --> g8[exp_o → manifold]
  end

  subgraph TEMP [Windowed temporal block]
    direction TB
    t1[log_o all frames per joint] --> t2[Q,K,V linear]
    t2 --> t3[unfold K,V into ±W window]
    t3 --> t4[scaled dot-product over KW=2W+1]
    t4 --> t5[mask out-of-bounds frames]
    t5 --> t6[Einstein-midpoint aggregate]
    t6 --> t7[exp_o → manifold]
  end

  classDef input fill:#e3f2fd,stroke:#1976d2;
  classDef output fill:#e8f5e9,stroke:#388e3c;
```

**Per-block residual flow** (geometrically principled — direct addition on $\mathbb{H}^d$ is undefined):
1. `x_tan = log_o(x)[1:]`
2. `z_tan = log_o(attn(x))[1:]`
3. `h = LN(x_tan + dropout(z_tan))`
4. `h = LN(h + TangentFFN(h))`
5. `x_out = exp_o(pad0(h))`

---

## 3. Strengths — what the paper gets right

**Geometric motivation that holds up.** The kinematic-tree → hyperbolic-space argument is not hand-waved; the Lorentz model (rather than Poincaré) is the right choice for downstream gradient stability, and the paper explicitly acknowledges Nickel & Kiela's evidence on this. Peng et al. 2020 used hyperbolic embeddings only for action *classification*, so this work is the first end-to-end *regression* attempt.

**Lorentz-native architecture, not a band-aid.** Most hyperbolic deep-learning papers do an `exp_o` at the input, immediately `log_o` everything for the actual layers, then `exp_o` again at the end — i.e. they do Euclidean computation with hyperbolic packaging. HYPERPOSE actually computes geodesic distances $d_\mathcal{L}(q_i, k_j)$ on the manifold for the attention logits and uses parallel transport for velocity comparison, which is the geometrically faithful thing to do.

**Real engineering thought went into avoiding the obvious traps.**
- The $O(N^2 \cdot D)$ pairwise parallel transport is replaced by transport-once-to-origin then dot-product trick (`attention.py:65–69`). This is mathematically equivalent because the Lorentzian inner product is preserved by parallel transport, and it cuts the velocity penalty cost from quadratic to linear in $D$.
- Einstein midpoint instead of iterative Fréchet mean (the standard $\sim 5-10$ Newton iterations would make every layer differentiable but slow; Lou et al.'s differentiable midpoint is one-shot).
- Windowed temporal attention via `F.unfold` — practical, simple, and the $\sim 35\times$ FLOP claim is realistic for $T{=}243, W{=}3$.
- `exp_map0` and `log_map0` are fused, allocation-free closed forms (`lorentz.py:84–110, 133–160`) — they correctly observe that at the origin the formulas collapse to scalar `cosh/sinh/acosh` ops on the spatial part only.
- Tangent-norm clamping at `MAX_NORM = 15.0` and Lorentz inner-product clamping at $1+\varepsilon$ are well-chosen for bf16/fp32 mixed precision — `cosh(15)² ≈ 2.7·10^{12}` is the right ceiling.

**Loss design is principled and not over-engineered.** Geodesic velocity consistency (`L_vel`) and geodesic bone length (`L_bone`) genuinely live on the manifold. The drift loss is correctly *detached* in `UncertaintyWeightedLoss.forward` (line 145) — you treat it as a diagnostic, not a training signal, which is right because `exp_map0` already guarantees manifold membership by construction. Kendall uncertainty weighting removes the grid search over $\lambda$'s.

**Interleaved (not stacked) spatial-temporal design.** The 3×(Spatial→Temporal) interleaving lets per-frame joint structure inform temporal context and vice versa, mirroring MotionAGFormer's two-stream insight but inside a single stream.

---

## 4. Weaknesses — honest critiques

### Headline performance is the elephant in the room
At epoch 19/60 you sit at 49.9 mm — competitive with PoseFormer 2021 (T=81), but the same-era / same-T peers are well below: MixSTE 40.9, STCFormer 40.5, MotionBERT 39.2, MotionAGFormer **38.4**. With 17.6 M params you're in the same parameter band as MotionAGFormer (19.2 M), so geometry is not buying you compute back. The "we're the *only* method on a non-Euclidean manifold" narrative is true but not, on its own, a publishable contribution if the final number lands at 45+. Convergence beyond epoch 19 needs to be shown.

### The ablation table has a tell
`Replace geodesic attention with Euclidean dot-product → 54.2 mm` and `Poincaré ball instead of Lorentz → 55.7 mm`. So the geometric inductive bias is worth $\sim 4-5$ mm versus a fair Euclidean baseline of *the same architecture*. That's a real signal, but it should be reported with a *strong* Euclidean baseline (e.g. take the same backbone and train MotionAGFormer-style on 17.6 M params) — otherwise the comparison is against a self-handicapped baseline.

### Numerical and architectural over-conservatism
- **`MAX_NORM = 15.0`** clamps the tangent norm at every `exp_map`. In a $d{=}512$ embedding this *will* clip during early training — the Lorentz constraint will silently push back toward the origin, costing capacity. There's no logging of how often this fires.
- **Single curvature, hard-coded $c=-1$.** The future-work section acknowledges this but it's a free perf lever you're leaving on the table — Chami et al.'s HGCN already showed per-layer curvature helps.
- **LayerNorm in tangent space at origin is fine for residuals but isn't curvature-aware.** It treats $T_o\mathbb{H}^d$ as Euclidean, which is only locally isometric — small perturbations OK, large ones not. Combined with `MAX_NORM=15`, you're essentially forcing locality.

### Velocity branch is under-engineered
- Velocity uses `self.embed.fc` (the *same* linear layer as positions — `network.py:128`). That's good for parameter sharing but it ties the velocity representation to the position one without a separate gain. A dedicated tiny FC for velocities would cost <1% params and probably help phase-space discrimination.
- `Δp_t = p_t − p_{t-1}` is a backward difference. Central differences $(p_{t+1}-p_{t-1})/2$ are higher-SNR estimators and add no parameters.
- Velocity is parallel-transported to joint locations at embed time, then transported *back to origin* inside every spatial block (`attention.py:57–60`). Two PTs per layer. Either keep velocity at origin throughout (and only use `h_x` for position-side attention), or cache the origin-transported velocity once.

### Window $W=3$ is too small for $T=243$
With kernel width 7 the temporal block only sees a 7-frame context per layer. With 3 stacked temporal blocks the receptive field is at most 19 frames — but human gait cycles at H36M's 50 Hz are ~50 frames, and many actions have 60-100-frame periodicities. The "$\sim 35\times$ FLOP reduction" comes at the cost of throwing away exactly the long-range structure that motivates having $T=243$ in the first place. The ablation `Full temporal attention (W=T) instead of windowed → 50.3` actually *worsens* the model versus 49.9, which is surprising and might be telling you that at epoch 19 the full-attention variant is still under-trained, or that the long-range signal is being washed out by a lack of multi-scale design.

### Code-level issues
- **Triple-redundant `runs/run_2026*` snapshots** of `model/`, `loss/`, `utils/`, etc. These are training-time copies and bloat the repo by ~3×. Add to `.gitignore`.
- **Hard-coded `H36M_SKELETON` in `loss/hyperbolic_loss.py:16-22`** — fine for now, but makes 3DPW evaluation (mentioned as future work) harder. Move skeleton definitions to `data/skeletons.py`.
- **`UncertaintyWeightedLoss` only accepts positional args** — easy to silently drop a loss when refactoring. Take a `dict` instead.
- **`SpatialBlock.forward` re-embeds and re-PTs velocity each forward**; could be cached at sequence level.
- **Tests directory has both `test_lorentz.py` and `test_attention.py` but no test for end-to-end manifold drift** under realistic input ranges. Given how central drift is to your numerical-health story, a regression test that asserts `|<h,h>+1| < 1e-4` after every block on synthetic inputs would be cheap insurance.

### Marketing claims to tighten
- "First 3D HPE on Lorentz" — true.
- "Geometric guarantees that Euclidean architectures cannot offer" — vague. Be specific: which guarantee? Probably you mean "$L_{drift}$-bounded preservation of the manifold constraint by construction," which is real but is *numerical* not *generalisation*.

---

## 5. Suggestions: reduce compute / increase speed

These are listed roughly in expected ROI order.

**(A) Drop the `acosh` from the attention logits — biggest single win, zero accuracy cost.**
The HKPSA logit is $-d_\mathcal{L}^2(q,k)/\tau = -[\text{acosh}(-\langle q,k\rangle_\mathcal{L})]^2/\tau$. The softmax that follows is invariant under monotone transforms of the logit, so use $-(-\langle q,k\rangle_\mathcal{L} - 1)/\tau$ or simply $\langle q,k\rangle_\mathcal{L}/\tau$ as the *score* — $\text{acosh}$ is monotone increasing on $[1,\infty)$, so the rank order is preserved up to a temperature reparameterisation. This saves: an `acosh`, the `clamp(min=1+\epsilon)`, and the `**2` per pair, *per layer*. Shouldn't change MPJPE, will speed up the spatial block measurably.

**(B) Cache tangent-space representations between adjacent blocks.**
Right now every block does `log_o(x)` at entry and `exp_o(...)` at exit; the next block immediately calls `log_o` again. Since `log_o ∘ exp_o = id` on the tangent at origin, you can pass the tangent vector directly between blocks and only `exp_o` once at the very end before the loss / drift check. Estimated saving: 6 `log_o` + 6 `exp_o` calls per forward pass. The only places you genuinely need the manifold representation are (i) the geodesic-distance computation in HKPSA (and even that can be done from tangent reps after an `exp_o` of just $Q,K$, which you already do), and (ii) `L_vel`/`L_bone` evaluation.

**(C) Multi-head HKPSA at reduced per-head dim.**
You currently run a single "head" at $d=512$. Splitting into $H=8$ heads at $d_h=64$ keeps the parameter count identical but parallelises better on GPU and lets each head specialise (e.g. one for upper body, one for legs, one for head). This is a one-line change to the projection layer.

**(D) bf16-friendly Lorentz path.**
`bfloat16` is what you train in but `lorentz.py` uses `EPS = 1e-7` — that's below bf16 representability (bf16 epsilon ≈ $4\cdot 10^{-3}$). Either switch the Lorentz operations to fp32 explicitly (small overhead, negligible memory) or bump `EPS` to $1\cdot 10^{-3}$ when running in bf16. Right now the EPS is silently rounded, which may explain why the drift floor sits at $\log_{10}(\text{drift})\approx 5.3$ — that drift is consistent with bf16 round-off accumulating across 6 manifold round-trips.

**(E) Shared QKV projection.**
Currently `q_proj`, `k_proj`, `v_proj` are three separate `nn.Linear(d, d)`. Fuse to one `nn.Linear(d, 3d)` then split — same parameters, ~25% fewer kernel launches, better memory locality.

**(F) Replace `unfold` in temporal block with `F.scaled_dot_product_attention` + custom mask.**
`unfold` materialises a $[BJ, T, KW, d]$ tensor — for $B{=}16, J{=}17, T{=}243, KW{=}7, d{=}512$ that is $\sim 1.4$ GB in fp32. PyTorch ≥ 2.1's `scaled_dot_product_attention` with a banded mask runs the same math without the explicit unfolded tensor and dispatches to a fused kernel where available.

**(G) Sparse spatial attention via the topology bias.**
You already have $A_{ij}$ and $\gamma=1$. Instead of computing all $J^2 = 289$ pairs and adding $\gamma A_{ij}$, hard-mask attention to $A^k$ for $k=1$ in the first layer, $k=2$ in the second, $k=3$ in the third (i.e. progressively wider receptive field along the kinematic graph). $J=17$ is small so the saving is modest in raw FLOPs, but it gives a strong inductive bias and reduces noise in early-training attention.

**(H) Knowledge distillation from a Euclidean teacher.**
Use the public MotionAGFormer-base checkpoint (38.4 mm) as a teacher and add an MSE-on-tangent-output term. Costs nothing at inference and historically buys 2–4 mm on H36M lifters.

**(I) Reduce $d=512 \to d=384$.**
Hyperbolic space's exponential volume growth means the *effective* representational capacity per dimension is higher than Euclidean. Empirically, hyperbolic embeddings are competitive at half the Euclidean dim. Worth an ablation; probable params/FLOPs cut of ~25% for ≤ 1 mm cost.

**(J) Drop the redundant `exp_o(pad0(...))` in `SpatialBlock.forward` line 42.**
You build `h_full = F.pad(h, (1,0), value=0.0)` then `exp_map0(h_full)` only to `log_map0` it again at the start of the next block. See (B) — this is the same fix.

---

## 6. Suggestions: improve performance

**(α) Multi-scale temporal stack — likely the highest-impact single change.**
Your three temporal blocks all use $W=3$ giving an effective receptive field of 19 frames. Replace with $W \in \{3, 9, 27\}$ in the three layers (kernel widths 7, 19, 55). Total temporal cost rises from $3 \cdot 7 = 21$ to $3 \cdot (7+19+55)/3 \approx 27$ — about $1.3\times$ — but you now cover gait-cycle and action-cycle scales. This is a near-direct port of dilated convolutions / WaveNet's success to the hyperbolic temporal axis.

**(β) Learnable per-layer curvature.**
Replace `c = -1` with `c_l = -softplus(α_l)` per layer. Two extra scalars per layer, six total. Chami et al. report ~5% gain on graph tasks; pose lifting may see less but the cost is trivially small.

**(γ) Hierarchical kinematic bias using $A^k$.**
Instead of a single binary $A_{ij}$, use a learned mixture $\gamma_1 A + \gamma_2 A^2 + \gamma_3 A^3$. This explicitly captures parent / sibling / cousin relationships in the kinematic tree and is something Euclidean GCN-pose-lifters (SemGCN) already exploit.

**(δ) Acceleration channel in phase space.**
Phase space in physics is (position, momentum); your "phase space" is currently (position, velocity). Adding $\Delta^2 p_t = p_{t+1} - 2p_t + p_{t-1}$ as a third tangent channel and including an acceleration-coherence term in HKPSA would make the attention reason about *jerk-aware* joint coupling — useful for dynamic actions (sit-down, jump) where high-acceleration joints should attend to each other regardless of distance.

**(ε) Per-joint output head.**
A shared `nn.Linear(d, 3)` decoder couples all 17 joints through the same weights. Replace with `nn.Linear(d, 3*J)` reshaped, or 17 small per-joint heads. Costs ~2% more params, often gives 0.5–1 mm.

**(ζ) Curriculum on the Riemannian losses.**
$L_{vel}$ and $L_{bone}$ measure manifold geometry of *predictions*; they're noisy at the start of training when predictions are random. Warm them up (zero weight for first 10 epochs, ramp to full over the next 10) instead of relying entirely on Kendall uncertainty to do this. This alone is often worth 1–2 mm.

**(η) Hybrid temporal — Euclidean is not a sin if it's principled.**
The kinematic-tree argument applies to the *spatial* graph, not to the temporal axis (frames are linearly ordered, not tree-structured). Run spatial attention in Lorentz (where the inductive bias pays off) and temporal attention in Euclidean (a bog-standard transformer block, possibly pre-trained from MotionBERT). You keep the geometric story for spatial reasoning while halving the manifold round-trips.

**(θ) Pre-train on AMASS with the same architecture.**
You list this as future work, but it's likely the single largest performance lever. Self-supervised pretext on AMASS sequences (mask-and-recover frames or joints) followed by Human3.6M fine-tune is how MotionBERT got to 39.2 mm with comparable params.

**(ι) Train-time tangent-norm regularisation.**
Add an explicit term penalising the fraction of tangent vectors that hit `MAX_NORM=15`. Right now the clamp is silent; making it explicit pushes the network to stay in a numerically comfortable regime *and* gives you a real-time signal on whether to raise/lower `MAX_NORM`.

**(κ) Output residual from a tiny Euclidean refiner.**
Add a small ($\sim 100$k param) Euclidean MLP that takes $\hat y$ and outputs a residual $\delta\hat y$. Trains end-to-end with the manifold backbone. This handles the "last-mm" coordinate-system biases without giving up the geometric pipeline. Combine with (η) for a clean hybrid story.

---

## 7. Two concrete things I'd ship first

If you only do two things before the next epoch checkpoint, do these:

1. **Multi-scale temporal windows** ($W \in \{3, 9, 27\}$). One-line change in the model constructor; addresses the most plausible cause of your sub-MotionAGFormer MPJPE.
2. **Drop the `acosh` from the attention logit.** Free speedup, no accuracy risk, simplifies the numerical-stability story.

After that, fix the velocity-PT redundancy (B/J), add learnable curvature (β), and run a fair ablation against an architecturally-identical Euclidean baseline.

---

*Notes on positioning when finalising for submission:* the geometric guarantee story is real and worth keeping, but the headline result needs a fair-baseline win or a clear efficiency-at-iso-quality story. If 49.9 → 42 mm by epoch 60, you have a paper. If 49.9 → 46 mm, you'll need to lean harder on the efficiency angle (params + FLOPs at iso-MPJPE rather than MPJPE at iso-budget).
