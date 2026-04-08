def latent_collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = [b[k] for b in batch]
    return {k: __import__("torch").stack(v) for k, v in out.items()}