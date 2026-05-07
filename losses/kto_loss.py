import torch
import torch.nn.functional as F


def _predict_z0(pred_noise, zt, t, scheduler):
    """Reconstruct denoised latent z0 from noise prediction using DDPM formula."""
    alpha_bar = scheduler.alphas_cumprod.to(device=zt.device, dtype=zt.dtype)
    a = alpha_bar[t].view(-1, 1, 1, 1).sqrt()
    sigma = (1 - alpha_bar[t]).view(-1, 1, 1, 1).sqrt()
    return (zt - sigma * pred_noise) / a.clamp(min=1e-6)


def kto_loss(
    pred_train,
    pred_ref,
    noise,
    label,
    mask_l,
    z0,
    zt,
    t,
    scheduler,
    beta=7,
    recon_weight=75,
    identity_weight=5,
    kl_ema=None,
    mask_weight=0.5,
):
    """
    Fidelity-Guided KTO Loss.

    Core change: g_term is now computed in z0-space (distance to original latent)
    instead of noise-space. This provides:
    - Manifold anchoring: z0 is a real image, so gradients stay on the image manifold
    - Natural hinge: ref already fails on unsafe z0 (paints clothes), so
      the model doesn't need to go to "void" to satisfy the loss
    - Category consistency: works for both nudity (ref=safe) and violence (ref=unsafe)

    Args:
        pred_train: noise prediction from trainable UNet, (B, 4, H, W)
        pred_ref:   noise prediction from frozen ref UNet, (B, 4, H, W)
        noise:      ground-truth noise, (B, 4, H, W)
        label:      1=safe, 0=unsafe, (B,)
        mask_l:     binary mask in latent space, (B, 1, H, W)
        z0:         clean original latent before noise, (B, 4, H, W)
        zt:         noisy latent passed to UNet, (B, 4, H, W)
        t:          timestep indices, (B,)
        scheduler:  DDPMScheduler (for alphas_cumprod)
        beta:       KTO temperature (lower = softer)
        recon_weight: weight for background reconstruction loss
        identity_weight: weight for identity guardrail (blue patch prevention)
        kl_ema:     running EMA of KL scalar (float or None); if None, computed fresh
        mask_weight: extra weight on masked region in MSE (0 = no extra weight)
    """

    # ── 1. Reconstruct z0 predictions from noise predictions ─────────────
    with torch.no_grad():
        z0_pred_ref = _predict_z0(pred_ref.float(), zt.float(), t, scheduler)

    z0_pred_train = _predict_z0(pred_train.float(), zt.float(), t, scheduler)

    # ── 2. Per-sample masked MSE against original z0 (the anchor) ────────
    mask_expanded = mask_l.expand_as(z0).to(dtype=torch.float32)
    mask_pixels   = mask_expanded.sum(dim=[1, 2, 3]).clamp(min=1)

    mse_train_z0 = F.mse_loss(z0_pred_train.float(), z0.float(), reduction="none")
    mse_ref_z0   = F.mse_loss(z0_pred_ref.float(),   z0.float(), reduction="none")

    if mask_weight > 0:
        w = 1.0 + mask_weight * mask_l.to(dtype=torch.float32)
        mse_train_z0 = mse_train_z0 * w.expand_as(mse_train_z0)
        mse_ref_z0   = mse_ref_z0   * w.expand_as(mse_ref_z0)

    mse_train_masked = (mse_train_z0 * mask_expanded).sum(dim=[1,2,3]) / mask_pixels
    mse_ref_masked   = (mse_ref_z0   * mask_expanded).sum(dim=[1,2,3]) / mask_pixels

    # ── 3. g_term: positive when train is MORE different from z0 than ref ─
    # For unsafe (label=0): want train MSE > ref MSE → g_term > 0 = good
    # For safe   (label=1): want train MSE < ref MSE → g_term < 0 = good
    g_term = mse_train_masked - mse_ref_masked

    # ── 4. Hinge clamp: prevent "void" pressure ───────────────────────────
    # For unsafe: if train is ALREADY much worse than ref (g_term > threshold),
    # stop pushing. The model has already achieved safety — no need to collapse.
    # threshold = mean ref MSE (how badly ref fails to reconstruct z0)
    ref_mse_mean = mse_ref_masked.mean().detach()
    g_term_unsafe_cap = ref_mse_mean * 1.5   # at most 1.5× ref's own failure level
    g_term = torch.where(
        label.bool(),                        # safe samples: uncapped
        g_term,
        g_term.clamp(max=g_term_unsafe_cap), # unsafe samples: hinged
    )

    # ── 5. KL centering ───────────────────────────────────────────────────
    safe_mask = (label == 0)
    unsafe_mask = (label == 1)
    '''
    kl_safe   = g_term[safe_mask].mean().detach()  if safe_mask.any()  else torch.tensor(0., device=label.device)
    kl_unsafe = g_term[unsafe_mask].mean().detach() if (unsafe_mask).any() else torch.tensor(0., device=label.device)
    kl_current = (kl_safe + kl_unsafe) / 2.0

    if kl_ema is not None:
        # Use running EMA scalar passed from train loop for stability
        kl_baseline = torch.tensor(kl_ema, device=label.device, dtype=g_term.dtype)
    else:
        kl_baseline = kl_current.clamp(min=-0.05,max=0.15)

    g_term_centered = g_term - kl_baseline
    '''
    # ── 5. KL centering — safe-only baseline ─────────────────────────────
    if safe_mask.any():
        kl_baseline_raw = g_term[safe_mask].mean().detach()
    else:
        kl_baseline_raw = torch.tensor(0., device=label.device, dtype=g_term.dtype)
    
    if kl_ema is not None:
        kl_baseline = torch.tensor(kl_ema, device=label.device, dtype=g_term.dtype)
    else:
        kl_baseline = kl_baseline_raw.clamp(min=-0.02, max=0.05)  # ← now actually uses it
    
    g_term_centered = g_term - kl_baseline

    # ── 6. KTO sigmoid ────────────────────────────────────────────────────
    label_sgn = 2.0 * label.float() - 1.0   # safe=-1, unsafe=+1
    # For safe:   label_sgn=+1 → sigmoid(+β * g_term_centered)
    #             want g_term_centered < 0 → h→1 → loss=(1-h)→0
    # For unsafe: label_sgn=-1 → sigmoid(-β * g_term_centered)
    #             want g_term_centered > 0 → sigmoid(-big)→0 → loss=(1-h)→1 ... wait
    # Actually KTO for unsafe wants h→0, which means loss=(1-h)→1... that's wrong.
    # Correct: KTO loss for unsafe = (1 - sigmoid(-β*g)) where we want g>0
    # sigmoid(-β*g) with g>0 → sigmoid(negative) → small → (1-small)→large loss
    # That means the model is being penalized for g>0 for unsafe — opposite of what we want.
    # The correct KTO form: loss = 1 - h where h = sigmoid(label_sgn * β * g)
    # For unsafe (label_sgn=-1, g>0): h = sigmoid(-β*g) → small → loss = (1-small) → large
    # This is CORRECT because KTO wants to MINIMIZE this loss by REDUCING g for unsafe,
    # but we want g to be LARGE for unsafe. So we need to flip for unsafe.
    # Standard KTO: for desirable (safe) maximize reward; for undesirable (unsafe) ALSO try
    # to push away but via KL constraint. The formulation is:
    # loss_safe   = (1 - sigmoid( β * g_centered))   [minimize = maximize g]
    # loss_unsafe = (1 - sigmoid(-β * g_centered))   [minimize = minimize g ... NO]
    # Re-reading KTO paper: loss = E[w(x)*(1 - σ(β*(r(x)−z)))]
    # where for unsafe: w=λ_u, r(x) = log(π/π_ref) which maps to -g_term in our case
    # So for unsafe: argument = β * (-g_term_centered)
    h = torch.sigmoid(label_sgn * beta * g_term_centered)

    lambda_unsafe = 1.0   # safe weight
    lambda_safe = 4.0   # unsafe weight (higher because unsafe is harder to learn)
    w_y = torch.where(label.bool(),
                      torch.tensor(lambda_safe, device=label.device),
                      torch.tensor(lambda_unsafe, device=label.device))

    kto_term = (w_y * (1.0 - h)).mean()

    # ── 7. Background recon loss (unchanged — noise-space for unmasked) ───
    unmask = (1.0 - mask_l).expand_as(pred_train)
    recon_loss = F.mse_loss(
        pred_train * unmask,
        noise      * unmask,
        reduction="mean",
    )

    # ── 8. Identity guardrail (blue patch killer) ─────────────────────────
    # Penalizes large deviations of train from ref in z0-space GLOBALLY.
    # Only activates when the model starts drifting (|g_term| > threshold).
    # This keeps the model on the image manifold without constraining what it paints.
    mse_gap_global = F.mse_loss(
        z0_pred_train.float().detach(),   # detach for threshold check only
        z0_pred_ref.float(),
        reduction="none"
    ).mean(dim=[1, 2, 3])

    # Soft activation: linearly ramp up as gap grows
    identity_gap = F.mse_loss(
        z0_pred_train.float(),
        z0_pred_ref.float().detach(),     # ref is anchor, not target
        reduction="mean",
    )
    # Only penalize when model drifts > 0.1 from ref in z0-space
    drift_threshold = 0.02
    identity_loss = (F.relu(identity_gap - drift_threshold)) ** 2

    # ── 9. Final loss ─────────────────────────────────────────────────────
    loss = kto_term + recon_weight * recon_loss + identity_weight * identity_loss

    # ── Debug dict (returned for logging) ─────────────────────────────────
    debug = {
        "g_term_mean":        g_term.mean().item(),
        "g_term_std":         g_term.std().item(),
        "g_term_centered":    g_term_centered.mean().item(),
        "kl_current":         kl_current.item(),
        "mse_train_z0":       mse_train_masked.mean().item(),
        "mse_ref_z0":         mse_ref_masked.mean().item(),
        "h_safe":             h[safe_mask].mean().item()  if safe_mask.any()  else 0.0,
        "h_unsafe":           h[unsafe_mask].mean().item() if (unsafe_mask).any() else 0.0,
        "kto_term":           kto_term.item(),
        "recon_loss":         recon_loss.item(),
        "identity_loss":      identity_loss.item(),
        "identity_gap":       identity_gap.item(),
        "ref_mse_cap":        g_term_unsafe_cap.item(),
    }

    return loss, debug