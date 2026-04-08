import wandb

def init_wandb(project, config):
    """Initialize Weights & Biases logging.
    
    Sets up wandb project and logs the configuration for the training run.
    
    Args:
        project (str): The wandb project name.
        config (dict): Configuration dictionary from YAML config file.
    
    Returns:
        wandb.run: The wandb run object for the current session.
    """
    return wandb.init(project=project, config=config)

def log_metrics(step, metrics):
    """Log training metrics to Weights & Biases.
    
    Args:
        step (int): The current training step number.
        metrics (dict): Dictionary of metric names and values to log.
    """
    wandb.log(metrics, step=step)

def finish_wandb():
    """Finish the Weights & Biases logging session.
    
    Closes the wandb run and uploads final logs.
    """
    wandb.finish()