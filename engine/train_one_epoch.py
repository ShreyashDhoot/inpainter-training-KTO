import torch
import torch.nn.functional as F
import copy

from models.unet_wrapper import unet_forward
from models.diffusion_utils import q_sample
from losses.bco_loss import bco_loss, RewardShiftEMA   # ← was: kto_loss


def train_loop(
    unet,
    ref_unet,          # frozen static reference (pass from train.py)
    vae,
    text_enc,
    scheduler,
    optimizer,
    lr_sched,
    scaler,
    train_loader,
    pipe,
    val_vis_samples,
    wandb_log_fn,
    save_fn,
    visual_eval_fn,
    cfg,
    device="cuda",
):
    unet.train()
    ref_unet.eval()    # always frozen
    vae.eval()
    text_enc.eval()
    g_std_ema = None

    global_step   = 0
    micro_step    = 0
    accum_loss    = 0.0
    grad_norm     = torch.tensor(0.0, device=device)

    # ── BCO reward-shift EMA (replaces kl_ema_scalar) ────────────────────────
    # RewardShiftEMA handles first-step initialisation internally.
    # .value is the float passed to bco_loss(reward_shift_ema=...).
    reward_shift_ema = RewardShiftEMA(momentum=0.999)   # ← was: kl_ema_scalar = None

    max_epochs       = cfg["training"].get("max_epochs", 100)
    grad_accum_steps = cfg["training"]["grad_accum_steps"]
    log_every        = cfg["training"]["log_every"]
    beta             = cfg["training"]["beta"]
    recon_weight     = cfg["training"].get("recon_weight", 200.0)
    identity_weight  = cfg["training"].get("identity_weight", 30.0)
    # bco_coeffs reuses the same config key as kto_coeffs — no config change needed
    bco_coeffs = cfg["training"].get("kto_coeffs", {"safe": 1.0, "nudity": 5.0, "violence": 12.0})

    # Accumulators for logging
    label_pos_count = 0
    label_neg_count = 0
    mse_gap_sum     = 0.0
    mse_gap_count   = 0
    debug_accum     = {}

    for epoch in range(max_epochs):
        for batch in train_loader:
            z0            = batch["z0"].to(device, non_blocking=True)
            masked_latent = batch["masked_latent"].to(device, non_blocking=True)
            mask_l        = batch["mask_latent"].to(device, non_blocking=True)
            input_ids     = batch["input_ids"].to(device, non_blocking=True)
            label         = batch["label"].to(device, non_blocking=True)

            with torch.no_grad():
                enc_hidden = text_enc(input_ids).last_hidden_state

            t     = torch.randint(200, 900, (z0.shape[0],), device=device)
            noise = torch.randn_like(z0)
            zt    = q_sample(z0, t, noise, scheduler)

            # ── Build is_safe here so dynamic weights can use it ─────────────
            # Works for both label formats:
            #   [B, 3] one-hot → column 0 is Safe
            #   [B]    binary  → 1 = safe
            if label.dim() == 2:
                is_safe = label[:, 0].bool()
            else:
                is_safe = label.bool()

            # ── Dynamic per-sample recon / identity weights ───────────────────
            # Safe   samples: full anchor weight — keep background + identity intact.
            # Unsafe samples: near-zero recon, zero identity — let the model
            #                 structurally change to perform refusal.
            B = pred_train_placeholder = z0.shape[0]   # batch size; pred_train not yet computed

            recon_weight_t = torch.where(
                is_safe,
                torch.full((B,), float(recon_weight), device=device),
                torch.full((B,), 2.0,                 device=device),
            )
            identity_weight_t = torch.where(
                is_safe,
                torch.full((B,), float(identity_weight), device=device),
                torch.zeros(B,                           device=device),
            )

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                # Trainable forward pass
                pred_train = unet_forward(unet, zt, t, enc_hidden, mask_l, masked_latent)

                # Frozen reference forward pass
                with torch.no_grad():
                    pred_ref = unet_forward(ref_unet, zt, t, enc_hidden, mask_l, masked_latent)

                # ── BCO loss call ─────────────────────────────────────────────
                # Returns 3 values now (was 2 with kto_loss).
                # kto_coeffs  → bco_coeffs  (renamed kwarg).
                # kl_ema      → reward_shift_ema.value  (float, updated after step).
                loss, debug, delta_raw = bco_loss(       # ← was: loss, debug = kto_loss(
                    pred_train       = pred_train,
                    pred_ref         = pred_ref,
                    noise            = noise,
                    label            = label,
                    mask_l           = mask_l,
                    z0               = z0,
                    zt               = zt,
                    t                = t,
                    scheduler        = scheduler,
                    beta             = beta,
                    recon_weight     = recon_weight_t,   # ← now a [B] tensor
                    identity_weight  = identity_weight_t,# ← now a [B] tensor
                    reward_shift_ema = reward_shift_ema.value,  # ← was: kl_ema (broken var)
                    bco_coeffs       = bco_coeffs,       # ← was: kto_coeffs=kto_coeffs
                )
                loss = loss / grad_accum_steps

                # ── Beta probe via reward_std (was g_term_std) ────────────────
                g_std_val = debug["reward_std"]          # ← was: debug["g_term_std"]
                if g_std_ema is None:
                    g_std_ema = g_std_val
                else:
                    g_std_ema = 0.99 * g_std_ema + 0.01 * g_std_val

                beta = cfg["training"]["beta"]

            # ── BCO reward-shift EMA update ───────────────────────────────────
            # delta_raw is the raw batch estimate returned by bco_loss.
            # RewardShiftEMA handles first-step init automatically.
            with torch.no_grad():
                reward_shift_ema.update(delta_raw)       # ← replaces the manual kl_ema_scalar block

            # ── Logging accumulators ──────────────────────────────────────────
            with torch.no_grad():
                mse_gap = abs(debug["mse_train_z0"] - debug["mse_ref_z0"])
                mse_gap_sum   += mse_gap
                mse_gap_count += 1

                for k, v in debug.items():
                    if not (isinstance(v, float) and v != v):  # skip NaN entries
                        debug_accum[k] = debug_accum.get(k, 0.0) + v

            scaler.scale(loss).backward()
            accum_loss += loss.item() * grad_accum_steps
            micro_step += 1

            if micro_step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    unet.parameters(), cfg["training"]["grad_clip_norm"]
                )
                scaler.step(optimizer)
                scaler.update()
                lr_sched.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_every == 0:
                    avg_loss    = accum_loss / float(log_every)
                    avg_mse_gap = mse_gap_sum / float(max(1, mse_gap_count))
                    avg_debug   = {k: v / float(log_every) for k, v in debug_accum.items()}

                    # Satisfaction deltas (same logic, new key names)
                    h_s     = avg_debug.get("h_safe",     0.5)
                    delta_n = avg_debug.get("h_nudity",   0.5) - h_s
                    delta_v = avg_debug.get("h_violence",  0.5) - h_s

                    wandb_log_fn({
                        # ── Core training ─────────────────────────────────────
                        "train/loss":               avg_loss,
                        "train/reward_gap":         avg_debug.get("reward_mean", 0),   # ← was g_term_mean
                        "train/grad_norm":          grad_norm.item(),
                        "train/lr":                 lr_sched.get_last_lr()[0],
                        "train/epoch":              epoch,
                        "train/label_pos_count":    label_pos_count,
                        "train/label_neg_count":    label_neg_count,
                        "train/mse_gap_avg":        avg_mse_gap,

                        # ── Reward diagnostics ────────────────────────────────
                        "debug/reward_mean":        avg_debug.get("reward_mean",  0),  # ← was g_term_mean
                        "debug/reward_std":         avg_debug.get("reward_std",   0),  # ← was g_term_std
                        "debug/reward_safe":        avg_debug.get("reward_safe",  0),  # ← NEW
                        "debug/reward_unsafe":      avg_debug.get("reward_unsafe",0),  # ← NEW

                        # ── Reward shift δ ────────────────────────────────────
                        "debug/delta":              avg_debug.get("delta",        0),  # ← replaces kl_current
                        "debug/delta_raw":          avg_debug.get("delta_raw",    0),  # ← NEW
                        "debug/delta_ema":          reward_shift_ema.value,            # ← replaces kl_ema

                        # ── MSE diagnostics ───────────────────────────────────
                        "debug/mse_train_z0":       avg_debug.get("mse_train_z0", 0),
                        "debug/mse_ref_z0":         avg_debug.get("mse_ref_z0",   0),

                        # ── Satisfaction h per class ──────────────────────────
                        "debug/h_safe":             avg_debug.get("h_safe",      0),
                        "debug/h_nudity":           avg_debug.get("h_nudity",    0),   # ← was h_unsafe
                        "debug/h_violence":         avg_debug.get("h_violence",  0),   # ← NEW

                        # ── Component losses ──────────────────────────────────
                        "debug/bco_loss":           avg_debug.get("bco_loss",    0),   # ← was kto_loss
                        "debug/recon_loss":         avg_debug.get("recon_loss",  0),
                        "debug/identity_loss":      avg_debug.get("identity_loss",0),
                        "debug/identity_gap":       avg_debug.get("identity_gap", 0),

                        # ── Dynamic weight monitors (new in BCO) ─────────────
                        # These let you verify safe samples are anchored and
                        # unsafe samples have near-zero recon/identity pull.
                        "debug/recon_per_safe":     avg_debug.get("recon_per_safe",      0),  # ← NEW
                        "debug/recon_per_unsafe":   avg_debug.get("recon_per_unsafe",    0),  # ← NEW
                        "debug/identity_per_safe":  avg_debug.get("identity_per_safe",   0),  # ← NEW
                        "debug/identity_per_unsafe":avg_debug.get("identity_per_unsafe", 0),  # ← NEW
                        "debug/rw_mean_safe":       avg_debug.get("rw_mean_safe",        0),  # ← NEW
                        "debug/rw_mean_unsafe":     avg_debug.get("rw_mean_unsafe",      0),  # ← NEW

                        # ── REMOVED keys (no longer in debug dict) ───────────
                        # "debug/kl_current"  → debug/delta
                        # "debug/kl_ema"      → debug/delta_ema
                        # "debug/ref_mse_cap" → gone (BCO uses hinge internally)
                        # "debug/h_unsafe"    → split into h_nudity / h_violence
                    }, step=global_step)

                    print(
                        f"step={global_step} loss={avg_loss:.4f} "
                        f"h_S={h_s:.3f} ΔN={delta_n:.3f} ΔV={delta_v:.3f} "
                        f"δ_ema={reward_shift_ema.value:.4f} "       # ← was kl_ema_scalar
                        f"id_gap={avg_debug.get('identity_gap',0):.4f} "
                        f"lr={lr_sched.get_last_lr()[0]:.2e} beta={beta:.1f}"
                    )

                    accum_loss    = 0.0
                    mse_gap_sum   = 0.0
                    mse_gap_count = 0
                    debug_accum   = {}

                if global_step % cfg["training"]["save_every"] == 0:
                    save_fn(global_step, unet, optimizer, lr_sched, scaler, epoch)
                    visual_eval_fn(unet, pipe, val_vis_samples, global_step)

                if global_step >= cfg["training"]["max_steps"]:
                    return