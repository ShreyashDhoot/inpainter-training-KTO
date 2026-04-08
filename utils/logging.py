import wandb

def init_wandb(project, config):
    return wandb.init(project=project, config=config)

def log_metrics(step, metrics):
    wandb.log(metrics, step=step)

def finish_wandb():
    wandb.finish()