# Copied from https://github.com/huggingface/diffusers/blob/fc6acb6b97e93d58cb22b5fee52d884d77ce84d8/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
# with the following modifications:
# - It uses the patched version of `ddim_step_with_logprob` from `ddim_with_logprob.py`. As such, it only supports the
#   `ddim` scheduler.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.
import pdb

from tqdm.auto import tqdm
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers.utils.torch_utils import randn_tensor

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
)
from PIL import Image
from .turbo_inference_with_logprob import turbo_step_with_logprob


def _get_x0_from_noise(sample, model_output, alphas_cumprod, timestep):
    alpha_prod_t = alphas_cumprod[timestep.long()].reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample


def prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
    ## for turbo, the sample_size for unet is 64
    shape = (
        batch_size,
        num_channels_latents,
        64,
        64,
    )
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device)

    return latents


@torch.no_grad()
def sdxl_turbo_pipeline_with_logprob(
    accelerator,
    vae,
    unet,
    noise_scheduler,
    height,
    width,
    num_inference_steps: int = 4,
    guidance_scale: float = 0.,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    add_time_ids: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
):  
    with torch.autocast('cuda'):
        # pdb.set_trace()
        # 0. Default height and width to unet
        height = height 
        width = width

        # 2. Define call parameters
        batch_size = prompt_embeds.shape[0]

        # 5. Prepare new latents or use the given one
        num_channels_latents = accelerator.unwrap_model(unet).config.in_channels
        latents = prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            prompt_embeds.device,
            generator,
            latents,
        )
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * noise_scheduler.init_noise_sigma

        # set timesteps
        noise_scheduler.set_timesteps(num_inference_steps, device=prompt_embeds.device)
        timesteps = noise_scheduler.timesteps

        # add timeid
        unet_added_conditions = {
                "time_ids": add_time_ids,
                "text_embeds": pooled_prompt_embeds
            }

        # 7. Denoising loop
        all_latents = [latents]
        all_model_input_latents = []
        all_log_probs = []
        # for i, t in tqdm(enumerate(timesteps)):
        for i, t in enumerate(timesteps):            

            # current_timesteps = torch.ones(bsz, device=prompt_embeds.device, dtype=torch.long) * t
            # latent_model_input = noise_scheduler.scale_model_input(latents, t)
            sigma = noise_scheduler.sigmas[i]
            latent_model_input = latents / ((sigma**2 + 1) ** 0.5)
            noise_scheduler.is_scale_input_called = True
            

            ## noise prediction
            noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=unet_added_conditions,
                        return_dict=False,
                    )[0]

            # pdb.set_trace()
            # compute the previous noisy sample x_t -> x_t-1
            latents, log_prob = turbo_step_with_logprob(
                                noise_scheduler,
                                noise_pred, 
                                t.unsqueeze(0),
                                latents,
                                generator=generator,
                                device=latents.device)

            # latents = noise_scheduler.step(noise_pred, t, latents, generator=generator, return_dict=False)[0]
            # # pdb.set_trace()
            if i != num_inference_steps - 1:
                all_model_input_latents.append(latent_model_input)
                all_latents.append(latents)
                all_log_probs.append(log_prob)


        ## vae decode
        # pdb.set_trace()
        if not output_type == "latent":
            image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        

        return image, all_latents, all_log_probs, all_model_input_latents
