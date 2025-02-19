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
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from diffusers import DDPMScheduler


def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)



def turbo_step_with_logprob(
    self: EulerAncestralDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: torch.tensor,
    sample: torch.FloatTensor,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
    device=torch.device('cuda')
):
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
    # pdb.set_trace()
    ## init step index to transform timestep
    step_indices = []
    for _t in timestep:
        step_indices.append((_t == self.timesteps).nonzero()[0].item())
    next_step_indices = [s+1 for s in step_indices]

    sigma = self.sigmas[step_indices].reshape(-1, 1, 1, 1).to(model_output.device)

    ## upcast sample
    sample = sample.to(torch.float32)

    ## x0 prediction from epsilon, for VE SDE
    # pdb.set_trace()
    pred_original_sample = sample - sigma * model_output

    ## 
    # pdb.set_trace()
    sigma_from = self.sigmas[step_indices]
    sigma_to = self.sigmas[next_step_indices]
    sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

    sigma_from = sigma_from.reshape(-1, 1, 1, 1).to(model_output.device)
    sigma_to = sigma_to.reshape(-1, 1, 1, 1).to(model_output.device)
    sigma_up = sigma_up.reshape(-1, 1, 1, 1).to(model_output.device)
    sigma_down = sigma_down.reshape(-1, 1, 1, 1).to(model_output.device)

    # 2. Convert to an ODE derivative
    derivative = (sample - pred_original_sample) / sigma

    dt = sigma_down - sigma

    prev_sample_mean = sample + derivative * dt

    if prev_sample is None:
        ## if previous sample is not given,
        device = model_output.device
        noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)

        prev_sample = prev_sample_mean + noise * sigma_up
    else:
        ## upcast the given previous sample
        prev_sample = prev_sample.to(torch.float32)
        
    # # upon completion increase step index by one
    # self._step_index += 1

    # Gaussian log prob of prev_sample given mean and variance
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (sigma_up**2))
        - torch.log(sigma_up)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample.to(model_output.dtype), log_prob
