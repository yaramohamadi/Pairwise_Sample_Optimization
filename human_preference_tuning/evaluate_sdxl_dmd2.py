from collections import defaultdict
import contextlib
import os
import gc
import copy
import argparse
import datetime
from concurrent import futures
import time
import sys
import pdb

from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
import numpy as np
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
import copy
import numpy as np
from huggingface_hub import hf_hub_download

import transformers
from transformers import AutoTokenizer, PretrainedConfig
import diffusers
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
    LCMScheduler
)


import ImageReward as RM


def flatten_list(scores_list):
    return [_list for _list in scores_list]


@torch.no_grad()
def log_validation_val_dataset(accelerator, pipeline, device, weight_dtype, val_prompts, total_val_prompts,
                            pickscore_scorer, clip_scorer, imagereward_scorer, aes_scorer,
                            output_path=None, num_pad=0, args=None):
    print(f"Running validation... \n Generating images with val prompts.")
    
    ## image save dir
    os.makedirs(output_path, exist_ok=True)
    print(f'Save images to: {output_path}')

    """
        run inference with SDXL-DMD, 4 steps
    """
    generator = torch.Generator(device=device).manual_seed(0)
    images = []
    pick_scores = []
    clip_scores = []
    imagereward_scores = []
    aesthetic_scores = []

    for p_idx, prompt in enumerate(val_prompts):
        # print(prompt)
        with torch.autocast('cuda'):
            image=pipeline(
                prompt=prompt, 
                num_inference_steps=4, 
                guidance_scale=0, 
                timesteps=[999, 749, 499, 249],
                cross_attention_kwargs={"scale": 1.0},
                generator=generator,
            ).images[0]
        
        images.append(image)
        # plot_img(image, img_save_dir)

        ps_score = pickscore_scorer.score(image, prompt)
        pick_scores.append(ps_score)
        clip_score = clip_scorer.score([image], prompt)
        clip_scores.append(clip_score) 
        imagereward = imagereward_scorer.score(prompt, image)
        imagereward_scores.append(imagereward)
        aesthetic_score = aes_scorer(image)
        aesthetic_scores.append(aesthetic_score)

        if (p_idx +1) % 50 == 0:
            if accelerator.is_main_process:
                print(f"Sampling with {(p_idx+1)*accelerator.num_processes} Val Prompts")


    accelerator.wait_for_everyone()
    print('gathering results for ')
    ## need to pad for the last process
    pick_scores = torch.as_tensor(np.concatenate(pick_scores), device=device)
    if accelerator.process_index == accelerator.num_processes - 1:
        pick_scores = torch.concat([pick_scores, torch.zeros(num_pad, device=device, dtype=pick_scores.dtype)])
    pick_scores = accelerator.gather(pick_scores)[:total_val_prompts]

    # print(pick_scores)
    clip_scores = torch.as_tensor(np.stack(clip_scores), device=device)
    if accelerator.process_index == accelerator.num_processes - 1:
        clip_scores = torch.concat([clip_scores, torch.zeros(num_pad, device=device, dtype=pick_scores.dtype)])
    clip_scores = accelerator.gather(clip_scores)[:total_val_prompts]
    
    # print(clip_scores)
    imagereward_scores = torch.tensor(imagereward_scores, device=device)
    if accelerator.process_index == accelerator.num_processes - 1:
        imagereward_scores = torch.concat([imagereward_scores, torch.zeros(num_pad, device=device, dtype=pick_scores.dtype)])
    imagereward_scores = accelerator.gather(imagereward_scores)[:total_val_prompts]
    
    # print(imagereward_scores)
    aesthetic_scores = torch.concatenate(aesthetic_scores)
    if accelerator.process_index == accelerator.num_processes - 1:
        aesthetic_scores = torch.concat([aesthetic_scores, torch.zeros(num_pad, device=device, dtype=pick_scores.dtype)])
    aesthetic_scores = accelerator.gather(aesthetic_scores)[:total_val_prompts]

    ## show PickScore
    if accelerator.is_main_process:
        print(f'PickScore on PickaPic test set: {torch.mean(pick_scores):.8f}')
        print(f'CLIP Score on PickaPic test set: {torch.mean(clip_scores):.8f}')
        print(f'ImageReward Score on PickaPic test set: {torch.mean(imagereward_scores):.8f}')
        print(f'Aesthetic Score on PickaPic test set: {torch.mean(aesthetic_scores):.8f}')

    ## save some images to local dir
    # if accelerator.is_main_process:
    #     # for i in range(0, 50):
    #     for i, img in enumerate(images):
    #         images[i].save(os.path.join(output_path, f'{i}_test.png'))
    # del pipeline


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--lora_path', default=None, type=str, )
    parser.add_argument('--val_dataset', default='pickapic_test',)

    args = parser.parse_args()

    ## accelerator
    accelerator_config = ProjectConfiguration(
        project_dir='./',
    )
    accelerator = Accelerator(project_config=accelerator_config,)
    num_gpus = accelerator.num_processes
    proc_id = accelerator.process_index

    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

    ## device & precision
    # device = torch.device(f'cuda:{args.device}')
    device = accelerator.device
    weight_dtype = torch.float16

    ## base output path
    baseline_output_path = 'output/baseline_sdxl_dmd2_4step'
    os.makedirs(baseline_output_path, exist_ok=True)

    ## load dmd unet
    dmd_repo = "tianweiy/DMD2"
    dmd_file = "dmd2_sdxl_4step_unet_fp16.bin"
    unet = UNet2DConditionModel.from_config(
                    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet")
    unet.load_state_dict(torch.load(
                    hf_hub_download(dmd_repo, dmd_file, 
                        local_dir='../../Distilled_Diffusion_Finetune/pretrained_models')))
    unet.to(device, weight_dtype)
    
    ## load right vae
    vae = AutoencoderKL.from_pretrained(
        'madebyollin/sdxl-vae-fp16-fix',
    )
    vae.to(device, weight_dtype)
    
    ## build pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        unet=unet,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    
    # load lora weights
    pipeline.load_lora_weights("ZichenMiao/PSO", weight_name="SDXL_DMD2/pytorch_lora_weights.safetensors")
    baseline_output_path = 'output/sdxl_dmd2_4step_pso_online'
    

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)




    ## load val prompts
    if args.val_dataset == 'pickapic_test':
        val_dataset = load_dataset(
            "yuvalkirstain/pickapic_v1_no_images",
            split="test_unique"
        )
        val_prompts = val_dataset['caption']
    else:
        raise ValueError('......')


    num_prompts = len(val_prompts)
    print(f'Number of Val prompts: {num_prompts}')
    num_prompts_per_proc = num_prompts // num_gpus
    if num_prompts % num_gpus != 0:
        num_prompts_per_proc += 1
    num_pad = num_prompts_per_proc * num_gpus - num_prompts

    #  fetch the per-process val prompts
    val_prompts_proc = val_prompts[proc_id*num_prompts_per_proc: 
                                   (proc_id+1)*num_prompts_per_proc]
    baseline_output_path = os.path.join(baseline_output_path, args.val_dataset)


    ## load evaluators
    # Can do clip_utils, aes_utils, hps_utils
    from pso_pytorch.pickscore_utils import Selector
    ps_selector = Selector(device=device, cache_dir=None)
    from pso_pytorch.clip_utils import Selector
    clip_selector = Selector(device=device, cache_dir=None)
    imagereward_scorer = RM.load("ImageReward-v1.0", device=device)
    from pso_pytorch.aesthetic_scorer import AestheticScorer
    aes_scorer = AestheticScorer(torch.float16).to(device)

    ## validation
    log_validation_val_dataset(accelerator, pipeline, device, weight_dtype, val_prompts_proc, num_prompts,
                               ps_selector, clip_selector, imagereward_scorer, aes_scorer,
                                output_path=baseline_output_path, num_pad=num_pad)

