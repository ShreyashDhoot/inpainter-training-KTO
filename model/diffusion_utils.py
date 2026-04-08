def q_sample(z0, t, noise, scheduler):
    return scheduler.add_noise(z0, noise, t)