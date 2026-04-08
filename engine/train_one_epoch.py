import torch
import torch.nn.functional as F

from models.unet_wrapper import unet_forward
from models.diffusion_utils import q_sample
from losses.kto_loss import kto_loss

def train_loop(
    unet,
    ref_unet,
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
    ref_unet.eval()
    vae.eval()
    text_enc.eval()

    global_step = 0
    accum_loss = 0.0
    grad_norm = torch.tensor(0.0, device=device)

    for epoch in range(999999):
        for batch in train_loader:
            z0 = batch["z0"].to(device)
            masked_latent = batch["masked_latent"].to(device)
            mask_l = batch["mask_latent"].to(device)
            input_ids = batch["input_ids"].to(device)
            label = batch["label"].to(device)

            with torch.no_grad():
                enc_hidden = text_enc(input_ids).last_hidden_state

            t = torch.randint(0, scheduler.config.num_train_timesteps, (z0.shape[0],), device=device)
            noise = torch.randn_like(z0)
            zt = q_sample(z0, t, noise, scheduler)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                pred_train = unet_forward(unet, zt, t, enc_hidden, mask_l, masked_latent)
                with torch.no_grad():
                    pred_ref = unet_forward(ref_unet, zt, t, enc_hidden, mask_l, masked_latent)
                loss = kto_loss(pred_train, pred_ref, noise, label, mask_l, beta=cfg["training"]["beta"])
                loss = loss / cfg["training"]["grad_accum_steps"]

            with torch.no_grad():
                mse_train = F.mse_loss(pred_train, noise, reduction="none").mean(dim=[1, 2, 3])
                mse_ref = F.mse_loss(pred_ref, noise, reduction="none").mean(dim=[1, 2, 3])
                log_ratio = -(mse_train - mse_ref)

                pos_mask = label.bool()
                neg_mask = ~pos_mask

                log_ratio_pos = log_ratio[pos_mask].mean() if pos_mask.any() else torch.tensor(0.0, device=device)
                log_ratio_neg = log_ratio[neg_mask].mean() if neg_mask.any() else torch.tensor(0.0, device=device)
                reward_gap = log_ratio_pos - log_ratio_neg

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            if (global_step + 1) % cfg["training"]["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), cfg["training"]["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                lr_sched.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % cfg["training"]["log_every"] == 0:
                wandb_log_fn({
                    "train/loss": accum_loss,
                    "train/log_ratio_pos": log_ratio_pos.item(),
                    "train/log_ratio_neg": log_ratio_neg.item(),
                    "train/reward_gap": reward_gap.item(),
                    "train/grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
                    "train/lr": lr_sched.get_last_lr()[0],
                    "train/epoch": epoch,
                }, step=global_step)
                print(f"step={global_step} loss={accum_loss:.4f} reward_gap={reward_gap.item():.4f}")
                accum_loss = 0.0

            if global_step % cfg["training"]["save_every"] == 0 and global_step > 0:
                save_fn(global_step, unet, optimizer, lr_sched, scaler, epoch)
                visual_eval_fn(unet, pipe, val_vis_samples, global_step)

            global_step += 1
            if global_step >= cfg["training"]["max_steps"]:
                return