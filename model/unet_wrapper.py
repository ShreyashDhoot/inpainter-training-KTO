import torch

def unet_forward(unet_model, zt, t, encoder_hidden, mask_l, masked_latent):
    if mask_l.dim() == 3:
        mask_l = mask_l.unsqueeze(1)
    unet_in = torch.cat([zt, mask_l, masked_latent], dim=1)
    return unet_model(unet_in, t, encoder_hidden_states=encoder_hidden).sample