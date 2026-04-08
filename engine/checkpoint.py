import os
import torch

def save_checkpoint(path, unet, optimizer, scheduler, scaler, global_step, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "unet": unet.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
    }, path)

def load_checkpoint(path, unet, optimizer=None, scheduler=None, scaler=None, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    unet.load_state_dict(ckpt["unet"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("global_step", 0), ckpt.get("epoch", 0)