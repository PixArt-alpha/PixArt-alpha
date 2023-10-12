import torch
from .model import gaussian_diffusion as gd
from .model.dpm_solver import model_wrapper, DPM_Solver, NoiseScheduleVP


def DPMS(
        model,
        condition,
        uncondition,
        cfg_scale,
        model_type='noise',  # or "x_start" or "v" or "score"
        noise_schedule="linear",
        guidance_type='classifier-free',
        model_kwargs={},
        diffusion_steps=1000
):
    betas = torch.tensor(gd.get_named_beta_schedule(noise_schedule, diffusion_steps))

    ## 1. Define the noise schedule.
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

    ## 2. Convert your discrete-time `model` to the continuous-time
    ## noise prediction model. Here is an example for a diffusion model
    ## `model` with the noise prediction type ("noise") .
    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type=model_type,
        model_kwargs=model_kwargs,
        guidance_type=guidance_type,
        condition=condition,
        unconditional_condition=uncondition,
        guidance_scale=cfg_scale,
    )
    ## 3. Define dpm-solver and sample by multistep DPM-Solver.
    return DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")