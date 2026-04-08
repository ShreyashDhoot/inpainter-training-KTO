import torch
import torch.nn.functional as F

def kto_loss(pred_train, pred_ref, noise, label, mask_l, beta=0.1):
    mse_train = F.mse_loss(pred_train, noise, reduction="none").mean(dim=[1, 2, 3])
    mse_ref = F.mse_loss(pred_ref, noise, reduction="none").mean(dim=[1, 2, 3])
    log_ratio = -(mse_train - mse_ref)

    pos_mask = label.bool()
    neg_mask = ~pos_mask

    zero = torch.tensor(0.0, device=pred_train.device)
    pos_loss = -F.logsigmoid(beta * log_ratio[pos_mask]).mean() if pos_mask.any() else zero
    neg_loss = -F.logsigmoid(-beta * log_ratio[neg_mask]).mean() if neg_mask.any() else zero

    return 0.5 * (pos_loss + neg_loss)