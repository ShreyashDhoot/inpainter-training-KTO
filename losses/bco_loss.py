import torch
import torch.nn.functional as F


def _predict_z0(pred_noise, zt, t, scheduler):
    """Reconstruct denoised latent z0 from noise prediction using DDPM formula."""
    alpha_bar = scheduler.alphas_cumprod.to(device=zt.device, dtype=zt.dtype)
    a     = alpha_bar[t].view(-1, 1, 1, 1).sqrt()
    sigma = (1 - alpha_bar[t]).view(-1, 1, 1, 1).sqrt()
    return (zt - sigma * pred_noise) / a.clamp(min=1e-6)


def _to_per_sample_weight(weight, reference_tensor):
    """
    Coerce ``weight`` into a [B] float tensor on the same device/dtype as
    ``reference_tensor``.

    Accepts:
        float / int  → broadcast to all B samples
        [B] Tensor   → used as-is (must already match device)
    """
    if isinstance(weight, torch.Tensor):
        return weight.to(dtype=reference_tensor.dtype,
                         device=reference_tensor.device)
    # scalar path
    B = reference_tensor.shape[0]
    return torch.full((B,), float(weight),
                      dtype=reference_tensor.dtype,
                      device=reference_tensor.device)


def bco_loss(
    pred_train,
    pred_ref,
    noise,
    label,                  # [B, 3] One-Hot (Safe, Nudity, Violence)
                            #   OR [B] binary int/bool (1=safe, 0=unsafe)
    mask_l,
    z0,
    zt,
    t,
    scheduler,
    beta=7,
    recon_weight=200.0,     # float  → applied uniformly to all samples
                            # [B] tensor → per-sample weights (dynamic recon mode)
                            #   recommended: safe=200.0, unsafe=0.0–5.0
    identity_weight=30.0,   # float  → applied uniformly to all samples
                            # [B] tensor → per-sample weights (dynamic recon mode)
                            #   recommended: safe=30.0, unsafe=0.0
    reward_shift_ema=None,  # float | None — running δ passed in from trainer
    mask_weight=0.5,
    bco_coeffs=None,        # {'safe': 1.0, 'nudity': 5.0, 'violence': 12.0}
                            # ignored when label is binary [B]
):
    """
    Fidelity-Guided Binary Classifier Optimization (BCO) Loss
    for Safe/Unsafe Stable Diffusion Inpainting.

    Replaces the KTO loss with the BCO formulation from
    "Binary Classifier Optimization for LLM Alignment" (Jung et al., 2024).

    Core idea (adapted to diffusion):
      • Reward  r_i  = −MSE_drift_i  (scalar per sample, masked latent space).
        Higher r → train model stayed close to the original z0.
        For SAFE   images we WANT high r  (thumbs-up  → BCE positive term).
        For UNSAFE images we WANT low  r  (thumbs-down → BCE negative term).

      • Reward shift δ (Theorem 3 in paper):
            δ = (E[r | safe] + E[r | unsafe]) / 2
        Tightens the BCE upper-bound on the implicit DPO loss.
        Maintained as an EMA across batches by the caller.

      • BCE alignment loss (Eq. 11, paper):
            safe   branch : −log σ( r − δ)
            unsafe branch : −log σ(−(r − δ))
        Both weighted by per-class coefficients w_y.

      • Hinge cap on unsafe reward to prevent collapse.

      • Steel Anchor — per-sample dynamic recon (noise-space background MSE).
        Pass recon_weight as a [B] tensor to give safe samples full weight
        and drop unsafe samples to near-zero, breaking the paradox.

      • Identity Guardrail — per-sample dynamic (z0-space global drift guard).
        Pass identity_weight as a [B] tensor to protect safe identity only.

    Supports both label formats:
      • Tri-class  [B, 3] one-hot  (Safe | Nudity | Violence)
      • Binary     [B]    int/bool (1 = safe, 0 = unsafe)

    Dynamic recon usage in training loop
    ─────────────────────────────────────
        B = pred_train.shape[0]
        rw = torch.where(is_safe_mask,
                         torch.full((B,), 200.0, device=device),
                         torch.full((B,),   2.0, device=device))
        iw = torch.where(is_safe_mask,
                         torch.full((B,),  30.0, device=device),
                         torch.zeros(B, device=device))
        loss, debug, delta_raw = bco_loss(..., recon_weight=rw, identity_weight=iw)

    Args:
        pred_train      : [B, C, H, W]  — UNet noise prediction, trainable model
        pred_ref        : [B, C, H, W]  — UNet noise prediction, frozen reference
        noise           : [B, C, H, W]  — ground-truth noise added at step t
        label           : [B, 3] one-hot  OR  [B] binary
        mask_l          : [B, 1, H, W]  — inpainting mask (1 = masked region)
        z0              : [B, C, H, W]  — original clean latent
        zt              : [B, C, H, W]  — noised latent at timestep t
        t               : [B]           — diffusion timesteps
        scheduler                       — diffusion scheduler (has .alphas_cumprod)
        beta            : float         — logit scale for BCE sigmoid (default 7)
        recon_weight    : float | [B] Tensor
        identity_weight : float | [B] Tensor
        reward_shift_ema: float | None  — δ EMA from trainer; None = batch estimate
        mask_weight     : float         — extra weight on masked pixels in MSE
        bco_coeffs      : dict | None   — per-class loss weights (tri-class mode only)

    Returns:
        loss        : scalar Tensor
        debug       : dict of float scalars for logging
        delta_raw   : float — raw batch δ estimate for the caller's EMA update
    """

    # ── 0. Parse label format ──────────────────────────────────────────────
    if label.dim() == 1:
        # Binary [B]: 1=safe, 0=unsafe
        is_safe = label.bool()
        is_nudity   = ~is_safe   # treat all unsafe as one class
        is_violence = torch.zeros_like(is_safe)   # empty; only used in debug
        tri_class   = False
    else:
        # Tri-class [B, 3]
        is_safe     = label[:, 0].bool()
        is_nudity   = label[:, 1].bool()
        is_violence = label[:, 2].bool()
        tri_class   = True

    if bco_coeffs is None:
        bco_coeffs = {"safe": 1.0, "nudity": 5.0, "violence": 12.0}

    # ── 1. Reconstruct z0 predictions ─────────────────────────────────────
    with torch.no_grad():
        z0_pred_ref = _predict_z0(pred_ref.float(), zt.float(), t, scheduler)

    z0_pred_train = _predict_z0(pred_train.float(), zt.float(), t, scheduler)

    # ── 2. Per-sample masked MSE against original z0 (drift metric) ───────
    mask_expanded = mask_l.expand_as(z0).to(dtype=torch.float32)
    mask_pixels   = mask_expanded.sum(dim=[1, 2, 3]).clamp(min=1)

    mse_train_z0_sample = F.mse_loss(z0_pred_train.float(), z0.float(), reduction="none")
    mse_ref_z0_sample   = F.mse_loss(z0_pred_ref.float(),   z0.float(), reduction="none")

    if mask_weight > 0:
        w = 1.0 + mask_weight * mask_l.to(dtype=torch.float32)
        mse_train_z0_sample = mse_train_z0_sample * w.expand_as(mse_train_z0_sample)
        mse_ref_z0_sample   = mse_ref_z0_sample   * w.expand_as(mse_ref_z0_sample)

    mse_train_masked = (mse_train_z0_sample * mask_expanded).sum(dim=[1, 2, 3]) / mask_pixels
    mse_ref_masked   = (mse_ref_z0_sample   * mask_expanded).sum(dim=[1, 2, 3]) / mask_pixels

    # ── 3. Per-sample reward: r_i = −(drift of train vs ref) ──────────────
    # Positive reward  → train model stayed CLOSE to z0 (safe behavior).
    # Negative reward  → train model drifted AWAY from z0 (refusal behavior).
    #
    # Mathematically:  r_i = −(mse_train_i − mse_ref_i)
    #                      =   mse_ref_i  − mse_train_i
    #
    # This mirrors the LLM reward  r(x,y) = β log π_θ/π_ref  in that a
    # higher reward means the train policy is "more preferred" than reference.
    reward = mse_ref_masked - mse_train_masked   # [B]

    # ── 4. Hinge cap on unsafe reward (prevents runaway gradient) ─────────
    # For unsafe samples, reward becoming very negative is the training signal;
    # we cap how POSITIVE it can become (model should not "stay safe" on unsafe).
    ref_reward_mean = reward[~is_safe].mean().detach() if (~is_safe).any() else torch.tensor(0., device=reward.device)
    unsafe_reward_cap = ref_reward_mean.abs() * 1.5  # symmetric cap above zero

    reward = torch.where(
        is_safe,
        reward,
        reward.clamp(max=unsafe_reward_cap.item()),
    )

    # ── 5. Reward Shift δ (BCO Theorem 3) ─────────────────────────────────
    # δ = (E[r | safe] + E[r | unsafe]) / 2
    # Minimises the error term  e^{-(r_safe - δ)} + e^{(r_unsafe - δ)}
    # which tightens BCE as an upper bound on the implicit DPO loss.
    if is_safe.any():
        mean_safe_reward   = reward[is_safe].mean().detach()
    else:
        mean_safe_reward   = torch.tensor(0., device=reward.device)

    if (~is_safe).any():
        mean_unsafe_reward = reward[~is_safe].mean().detach()
    else:
        mean_unsafe_reward = torch.tensor(0., device=reward.device)

    delta_raw = ((mean_safe_reward + mean_unsafe_reward) / 2.0).item()

    # Use EMA delta from caller for training stability (like KL EMA in KTO)
    if reward_shift_ema is not None:
        delta = torch.tensor(reward_shift_ema, device=reward.device, dtype=reward.dtype)
    else:
        # Batch estimate; clamp for first-step stability
        delta = torch.tensor(delta_raw, device=reward.device, dtype=reward.dtype)
        delta = delta.clamp(min=-0.05, max=0.05)

    # ── 6. BCO BCE Loss (Eq. 11, Jung et al. 2024) ────────────────────────
    #
    # Safe   (thumbs-up)  : −log σ( β(r − δ))
    # Unsafe (thumbs-down): −log σ(−β(r − δ))
    #
    # Unified form using label_sgn ∈ {+1, −1}:
    #   loss_i = −log σ(label_sgn_i * β * (r_i − δ))
    #          = softplus(−label_sgn_i * β * (r_i − δ))
    #
    label_sgn = torch.where(is_safe,
                            torch.ones_like(reward),
                            -torch.ones_like(reward))

    r_shifted  = reward - delta                          # [B]
    bce_per_sample = F.softplus(-label_sgn * beta * r_shifted)   # numerically stable

    # ── 7. Per-class weighting w_y ────────────────────────────────────────
    w_y = torch.ones_like(bce_per_sample)
    if tri_class:
        w_y[is_safe]     = bco_coeffs["safe"]
        w_y[is_nudity]   = bco_coeffs["nudity"]
        w_y[is_violence] = bco_coeffs["violence"]
    else:
        # Binary mode: safe=1.0, unsafe weighted heavier (mirrors nudity default)
        w_y[is_safe]  = bco_coeffs["safe"]
        w_y[~is_safe] = bco_coeffs["nudity"]  # single unsafe class

    bco_loss_final = (w_y * bce_per_sample).mean()

    # ── 8. Reconstruction Loss — Dynamic Steel Anchor ─────────────────────
    # Noise-space MSE on the unmasked background, computed per-sample so that
    # per-sample recon_weight values (safe=200, unsafe≈0) can be applied.
    #
    # Shape walkthrough:
    #   unmask          : [B, C, H, W]
    #   sq_err          : [B, C, H, W]  element-wise squared error
    #   recon_per_sample: [B]           mean over (C, H, W)
    #   rw              : [B]           per-sample weight (or scalar broadcast)
    #   recon_loss      : scalar        weighted mean over batch
    #
    # When recon_weight is a plain float the behaviour is identical to before.
    unmask = (1.0 - mask_l).expand_as(pred_train)  # [B, C, H, W]
    sq_err_recon     = ((pred_train * unmask) - (noise * unmask)) ** 2  # [B, C, H, W]
    recon_per_sample = sq_err_recon.mean(dim=[1, 2, 3])                  # [B]

    rw = _to_per_sample_weight(recon_weight, recon_per_sample)           # [B]
    # Guard: if ALL unsafe weights are 0, avoid a zero-denominator mean.
    rw_sum = rw.sum().clamp(min=1e-6)
    recon_loss = (rw * recon_per_sample).sum() / rw_sum                  # scalar

    # ── 9. Identity Guardrail — Dynamic z0-space Drift Protection ─────────
    # Per-sample squared L2 drift between train and ref z0 predictions.
    # Applying identity_weight per-sample lets us protect safe image identity
    # while giving unsafe samples room to structurally change (paint clothing,
    # alter composition) without being penalised for z0 drift.
    #
    #   sq_err_id        : [B, C, H, W]
    #   identity_per_sample: [B]         mean over (C, H, W)
    #   drift_threshold  : scalar        hinge — only penalise drift above this
    #   iw               : [B]           per-sample weight
    #
    sq_err_id            = (z0_pred_train.float() - z0_pred_ref.float().detach()) ** 2
    identity_per_sample  = sq_err_id.mean(dim=[1, 2, 3])                 # [B]
    drift_threshold      = 0.02
    # Hinge: penalise only the excess above threshold, squared.
    hinged_per_sample    = (F.relu(identity_per_sample - drift_threshold)) ** 2  # [B]

    iw = _to_per_sample_weight(identity_weight, hinged_per_sample)       # [B]
    iw_sum = iw.sum().clamp(min=1e-6)
    identity_loss = (iw * hinged_per_sample).sum() / iw_sum              # scalar

    # Scalar gap for debug logging (unweighted mean, as before)
    identity_gap = identity_per_sample.mean().detach()

    # ── 10. Final Combined Loss ────────────────────────────────────────────
    # 4× on alignment term matches prior KTO convention.
    # recon and identity are now already weighted internally, so their
    # coefficients here are 1.0 — the magnitude lives in the weight tensors.
    # When scalar weights are passed the result is numerically identical to
    # the old formulation:  scalar * mean(per_sample) == weighted_mean.
    loss = 4.0 * bco_loss_final + recon_loss + identity_loss

    # ── Debug Metrics ──────────────────────────────────────────────────────
    # Satisfaction proxy: σ(label_sgn * β * r_shifted) ∈ (0,1).
    # 1.0 = perfectly aligned, 0.0 = completely misaligned.
    h = torch.sigmoid(label_sgn * beta * r_shifted).detach()

    debug = {
        # Reward stats
        "reward_mean":          reward.mean().item(),
        "reward_std":           reward.std().item(),
        "reward_safe":          reward[is_safe].mean().item()    if is_safe.any()    else float("nan"),
        "reward_unsafe":        reward[~is_safe].mean().item()   if (~is_safe).any() else float("nan"),
        # Reward shift
        "delta":                delta.item(),
        "delta_raw":            delta_raw,
        # MSE diagnostics
        "mse_train_z0":         mse_train_masked.mean().item(),
        "mse_ref_z0":           mse_ref_masked.mean().item(),
        # Satisfaction h per class
        "h_safe":               h[is_safe].mean().item()     if is_safe.any()     else float("nan"),
        "h_nudity":             h[is_nudity].mean().item()   if is_nudity.any()   else float("nan"),
        "h_violence":           h[is_violence].mean().item() if is_violence.any() else float("nan"),
        # Component losses
        "bco_loss":             bco_loss_final.item(),
        "recon_loss":           recon_loss.item(),
        "identity_loss":        identity_loss.item(),
        "identity_gap":         identity_gap.item(),
        # Per-class recon — monitor that safe stays anchored, unsafe is free.
        "recon_per_safe":       recon_per_sample[is_safe].mean().item()    if is_safe.any()    else float("nan"),
        "recon_per_unsafe":     recon_per_sample[~is_safe].mean().item()   if (~is_safe).any() else float("nan"),
        "identity_per_safe":    identity_per_sample[is_safe].mean().item()  if is_safe.any()    else float("nan"),
        "identity_per_unsafe":  identity_per_sample[~is_safe].mean().item() if (~is_safe).any() else float("nan"),
        # Effective weights seen this batch — sanity-check your weight tensors.
        "rw_mean_safe":         rw[is_safe].mean().item()    if is_safe.any()    else float("nan"),
        "rw_mean_unsafe":       rw[~is_safe].mean().item()   if (~is_safe).any() else float("nan"),
    }

    return loss, debug, delta_raw


# ─────────────────────────────────────────────────────────────────────────────
# Reward Shift EMA Helper
# ─────────────────────────────────────────────────────────────────────────────

class RewardShiftEMA:
    """
    Maintains an exponential moving average of the BCO reward shift δ.

    Usage in your training loop:
        ema = RewardShiftEMA(momentum=0.99)
        ...
        loss, debug, delta_raw = bco_loss(..., reward_shift_ema=ema.value)
        ema.update(delta_raw)
    """

    def __init__(self, momentum: float = 0.99, init: float = 0.0):
        self.momentum = momentum
        self.value    = init
        self._initialized = False

    def update(self, delta_raw: float):
        if not self._initialized:
            self.value = delta_raw
            self._initialized = True
        else:
            self.value = self.momentum * self.value + (1.0 - self.momentum) * delta_raw

    def state_dict(self):
        return {"value": self.value, "initialized": self._initialized}

    def load_state_dict(self, d):
        self.value        = d["value"]
        self._initialized = d["initialized"]
