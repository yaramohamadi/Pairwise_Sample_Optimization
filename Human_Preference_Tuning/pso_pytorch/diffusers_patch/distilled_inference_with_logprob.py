# Copied from https://github.com/huggingface/diffusers/blob/fc6acb6b97e93d58cb22b5fee52d884d77ce84d8/src/diffusers/schedulers/scheduling_ddim.py
# with the following modifications:
# - It computes and returns the log prob of `prev_sample` given the UNet prediction.
# - Instead of `variance_noise`, it takes `prev_sample` as an optional argument. If `prev_sample` is provided,
#   it uses it to compute the log prob.
# - Timesteps can be a batched torch.Tensor.
import pdb
from typing import Optional, Tuple, Union

import math
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput, DDIMScheduler
from diffusers import DDPMScheduler


def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)


def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep.cpu()).to(timestep.device)
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0, self.alphas_cumprod.gather(0, prev_timestep.cpu()), self.final_alpha_cumprod
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance


def _get_x0_from_noise(sample, model_output, alphas_cumprod, timestep):
    # pdb.set_trace()
    alpha_prod_t = alphas_cumprod[timestep.long()].reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample


def distilled_step_with_logprob(
    self: DDPMScheduler,
    model_output: torch.FloatTensor,
    timestep: torch.tensor,
    prev_timestep: torch.tensor,
    sample: torch.FloatTensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
    device=torch.device('cuda')
) -> Union[DDIMSchedulerOutput, Tuple]:
    """
    Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        eta (`float`): weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
            predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
            `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
            coincide with the one provided as input and `use_clipped_model_output` will have not effect.
        generator: random number generator.
        variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
            can directly provide the noise for the variance itself. This is useful for methods such as
            CycleDiffusion. (https://arxiv.org/abs/2210.05559)
        return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

    Returns:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
        returning a tuple, the first element is the sample tensor.

    """
    ## x0 prediction
    pred_original_sample = _get_x0_from_noise(
        sample, model_output, self.alphas_cumprod.to(device), timestep
    ).to(sample.dtype)


    # ## Clip or threshold "predicted x_0"
    # if self.config.thresholding:
    #     pred_original_sample = self._threshold_sample(pred_original_sample)
    # elif self.config.clip_sample:
    #     pred_original_sample = pred_original_sample.clamp(
    #         -self.config.clip_sample_range, self.config.clip_sample_range
    #     )

    ## add noise to previous timesteps (prev_timestep), with DDPM forward process
    self.alphas_cumprod = self.alphas_cumprod.to(device=pred_original_sample.device)
    alphas_cumprod = self.alphas_cumprod.to(dtype=pred_original_sample.dtype)
    prev_timestep = prev_timestep.to(pred_original_sample.device)

    sqrt_alpha_prod_t_prev = alphas_cumprod[prev_timestep.long()] ** 0.5
    sqrt_alpha_prod_t_prev = sqrt_alpha_prod_t_prev.flatten()
    while len(sqrt_alpha_prod_t_prev.shape) < len(pred_original_sample.shape):
        sqrt_alpha_prod_t_prev = sqrt_alpha_prod_t_prev.unsqueeze(-1)

    sqrt_one_minus_alpha_prod_t_prev = (1 - alphas_cumprod[prev_timestep.long()]) ** 0.5
    sqrt_one_minus_alpha_prod_t_prev = sqrt_one_minus_alpha_prod_t_prev.flatten()
    while len(sqrt_one_minus_alpha_prod_t_prev.shape) < len(pred_original_sample.shape):
        sqrt_one_minus_alpha_prod_t_prev = sqrt_one_minus_alpha_prod_t_prev.unsqueeze(-1)

    prev_sample_mean = sqrt_alpha_prod_t_prev * pred_original_sample # + sqrt_one_minus_alpha_prod_t_prev * added_noise


    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        ## added noise
        variance_noise = randn_tensor((1, pred_original_sample.shape[1], pred_original_sample.shape[2], pred_original_sample.shape[3]), 
                                generator=generator, device=device, dtype=sample.dtype)
        
        prev_sample = prev_sample_mean + sqrt_one_minus_alpha_prod_t_prev * variance_noise

    # Gaussian log prob of prev_sample given mean and variance
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (sqrt_one_minus_alpha_prod_t_prev**2))
        - torch.log(sqrt_one_minus_alpha_prod_t_prev)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample.type(sample.dtype), log_prob
