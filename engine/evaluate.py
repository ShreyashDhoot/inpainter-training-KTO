import os
import torch
from PIL import Image

def decode_latent_to_pil(vae, latent):
    latent = latent.unsqueeze(0) if latent.dim() == 3 else latent
    latent = latent / vae.config.scaling_factor
    img = vae.decode(latent).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img[0].permute(1, 2, 0).detach().cpu().numpy()
    return Image.fromarray((img * 255).astype("uint8"))

def latent_mask_to_pil(mask_latent):
    from PIL import Image
    import numpy as np
    m = mask_latent.detach().cpu()
    if m.dim() == 3:
        m = m[0]
    m = m.numpy()
    if m.max() <= 1.0:
        m = (m * 255).astype("uint8")
    else:
        m = m.astype("uint8")
    return Image.fromarray(m)

def visual_eval(unet, pipe, val_vis_samples, step, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    unet.eval()
    with torch.no_grad():
        for i, sample in enumerate(val_vis_samples[:4]):
            z0 = sample["z0"]
            mask_latent = sample["mask_latent"]
            prompt = sample["prompt"]

            img = decode_latent_to_pil(pipe.vae, z0)
            mask = latent_mask_to_pil(mask_latent)

            out = pipe(
                prompt=prompt,
                image=img,
                mask_image=mask,
                num_inference_steps=25,
            ).images[0]
            out.save(os.path.join(out_dir, f"eval_step{step}_sample{i}.png"))
    unet.train()