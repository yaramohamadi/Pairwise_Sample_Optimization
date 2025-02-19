from collections import defaultdict
import contextlib
import os
import gc
import copy
import datetime
from concurrent import futures
import time
import sys
import pdb
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
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

from transformers import AutoTokenizer, PretrainedConfig
import diffusers
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
    EulerAncestralDiscreteScheduler
)
from diffusers.loaders import StableDiffusionXLLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig, PeftModel
from peft.utils import get_peft_model_state_dict

from pso_pytorch.prompt_dataset import PromptDataset
import pso_pytorch.rewards
from pso_pytorch.diffusers_patch.sdxl_turbo_with_logprob import sdxl_turbo_pipeline_with_logprob
from pso_pytorch.diffusers_patch.turbo_inference_with_logprob import turbo_step_with_logprob

os.environ['WANDB_API_KEY'] = 'bca7ee0e7cb437e3cb4d484a435f9bc8c137fd1e'

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str=None, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def tokenize_captions(tokenizers, examples):
    captions = []
    for caption in examples["caption"]:
        captions.append(caption)

    tokens_one = tokenizers[0](
        captions, truncation=True, padding="max_length", max_length=tokenizers[0].model_max_length, return_tensors="pt"
    ).input_ids
    tokens_two = tokenizers[1](
        captions, truncation=True, padding="max_length", max_length=tokenizers[1].model_max_length, return_tensors="pt"
    ).input_ids

    return tokens_one, tokens_two


@torch.no_grad()
def encode_prompt(text_encoders, text_input_ids_list):
    prompt_embeds_list = []

    # pdb.set_trace()
    for i, text_encoder in enumerate(text_encoders):
        text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


@torch.no_grad()
def log_validation_val_dataset(config, vae, accelerator, weight_dtype, epoch, 
                               lora_dir, val_prompts, pickscore_scorer, clip_scorer):
    logger.info(f"Running validation... \n Generating images with val prompts.")
    
    if config.mixed_precision == "fp16":
        vae.to(weight_dtype)

    # create pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        config.pretrained.pretrained_model_name_or_path,
        vae=vae,
        torch_dtype=weight_dtype,
        cache_dir=config.general_cache_dir
    )

    ## load lora weights
    pipeline.load_lora_weights(lora_dir, weight_name="pytorch_lora_weights.safetensors")
    
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    """
        run inference with SDXL-Turbo, 4 steps
    """
    generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)
    images = []
    pick_scores = []
    clip_scores = []

    for p_idx, prompt in enumerate(val_prompts):
        with torch.autocast('cuda'):
            image=pipeline(
                prompt=prompt, 
                num_inference_steps=4, 
                guidance_scale=0., 
                generator=generator,
            ).images[0]
        
        images.append(image)

        ps_score = pickscore_scorer.score(image, prompt)
        pick_scores.append(ps_score)
        clip_score = clip_scorer.score([image], prompt)
        clip_scores.append(clip_score) 

        if (p_idx +1) % 50 == 0:
            logger.info(f"Sampling with [{p_idx+1}/{len(val_prompts)}] Val Prompts")

    for tracker in accelerator.trackers:
        assert tracker.name == "wandb"
        ## save the first 20 images in the val prompt list
        tracker.log(
            {
                "validation": [
                    wandb.Image(image.resize((256, 256)), caption=f"{i}: {val_prompts[i]}") 
                    for i, image in enumerate(images[:10])
                ]
            }, commit=False
        )

    ## save some images to local dir
    for i in range(5):
        images[i].save(os.path.join(lora_dir, f'{i}.png'))

    ## save PickScore
    tracker.log({
        "PickScore on PickaPic test set": np.mean(pick_scores),
        "CLIP Score on PickaPic test set": np.mean(clip_scores)})

    ## show PickScore
    logger.info(f'PickScore on PickaPic test set: {np.mean(pick_scores):.8f}')
    logger.info(f'CLIP Score on PickaPic test set: {np.mean(clip_scores):.8f}')

    del pipeline


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config
    # print(config)

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    # if not config.run_name:
    #     config.run_name = unique_id
    # else:
    #     config.run_name += "_" + unique_id

    ## change output dir, append dataset information
    effective_batch_size_pergpu = config.train.gradient_accumulation_steps * config.train.batch_size
    samples_per_epoch_pergpu = config.sample.num_batches_per_epoch * config.sample.batch_size
    config.run_name = f'SDXL_Turbo4_Reward_PS_Only_{config.sample.num_steps}steps_{samples_per_epoch_pergpu}sample_pergpu_{config.train.distilled_train_steps}steps_lorarank{config.train.lora_rank}_lr{config.train.learning_rate}_beta{config.train.beta}_train_bspergpu{effective_batch_size_pergpu}'
    config.output_dir = os.path.join(config.output_dir, config.run_name)
    logging_dir = os.path.join(config.output_dir, 'logs')


    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.train.distilled_train_steps)

 
    assert num_train_timesteps == (config.sample.num_steps - 1)

    accelerator_config = ProjectConfiguration(
        project_dir=config.output_dir,
        logging_dir=logging_dir,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps, 
    )
    
    logger.info(f"\n{config}")
    
    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        config.pretrained.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
        cache_dir=config.general_cache_dir
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        config.pretrained.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        use_fast=False,
        cache_dir=config.general_cache_dir
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        config.pretrained.pretrained_model_name_or_path
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        config.pretrained.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                    config.pretrained.pretrained_model_name_or_path, 
                    subfolder="scheduler",
                    cache_dir=config.general_cache_dir)
    
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        config.pretrained.pretrained_model_name_or_path, 
        subfolder="text_encoder", 
        cache_dir=config.general_cache_dir
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        config.pretrained.pretrained_model_name_or_path, 
        subfolder="text_encoder_2", 
        cache_dir=config.general_cache_dir
    )
    vae_path = (
        config.pretrained.pretrained_model_name_or_path
        if config.pretrained.pretrained_vae_model_name_or_path is None
        else config.pretrained.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if config.pretrained.pretrained_vae_model_name_or_path is None else None,
        cache_dir=config.general_cache_dir
    )
    ## sdxl turbo unet
    unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.pretrained_model_name_or_path, 
                subfolder="unet",
                cache_dir=config.general_cache_dir
            )

    

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet and text_encoders to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # The VAE is always in float32 to avoid NaN losses.
    if config.pretrained.pretrained_vae_model_name_or_path is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    else:
        vae.to(accelerator.device, dtype=torch.float32)

    # time ids
    def compute_time_ids(original_size=512, crops_coords_top_left=0,):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (512, 512)
        original_size = (original_size, original_size)
        crops_coords_top_left = (crops_coords_top_left, crops_coords_top_left)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    # Set up LoRA.
    """
      change the optimized lora
    """
    unet_lora_config = LoraConfig(
        r=config.train.lora_rank,
        lora_alpha=config.train.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config)
    if config.mixed_precision == "fp16":
        for param in unet.parameters():
            # only upcast trainable parameters (LoRA) into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

    ## if not use lora, copy for reference unet 
    # ref =  copy.deepcopy(pipeline.unet)
    # for param in ref.parameters():
    #     param.requires_grad = False
    
    ## enable unet gradient checkpointing
    unet.enable_gradient_checkpointing()

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        # logger.info(type(weights))
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None

            for model in models:
                # logger.info(type(model))
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if len(weights) > 0:
                    weights.pop()

            StableDiffusionXLLoraLoaderMixin.save_lora_weights(output_dir, unet_lora_layers=unet_lora_layers_to_save)

    def load_model_hook(models, input_dir):
        unet_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = StableDiffusionXLLoraLoaderMixin.lora_state_dict(input_dir)
        StableDiffusionXLLoraLoaderMixin.load_lora_into_unet(
            lora_state_dict, network_alphas=network_alphas, unet=unet_
        )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


    def sample_compare(a, b):
        ## a, b are multi-dim rewards with shape [bs, m]
        ## for each sample in bs, use one of m to compare rewards
        bs, num_rewards = a.shape
        random_reward_indices = torch.randint(0, num_rewards, (bs,), device=a.device)
        proc_a = a[torch.arange(bs, device=a.device), random_reward_indices]
        proc_b = b[torch.arange(bs, device=a.device), random_reward_indices]

        a_dominates = proc_a <= proc_b
        b_dominates = proc_b < proc_a
        
        c = torch.zeros([a.shape[0],2], dtype=torch.float, device=a.device)
        c[a_dominates] = torch.tensor([-1., 1.],device=a.device)
        c[b_dominates] = torch.tensor([1., -1.],device=a.device)

        return c


    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    """
        prepare prompt and reward fn
    """

    # Can do clip_utils, aes_utils, hps_utils
    from pso_pytorch.pickscore_utils import Selector
    ps_selector = Selector(device=accelerator.device, cache_dir=config.general_cache_dir)
    from pso_pytorch.clip_utils import Selector
    clip_selector = Selector(device=accelerator.device, cache_dir=config.general_cache_dir)

    # validation prompts
    val_dataset = load_dataset(
        config.val_dataset,
        cache_dir=config.cache_dir_val,
        split=config.val_split_name
    )
    val_prompts = val_dataset['caption']

    ## training prompt dataset
    prompt_dataset = PromptDataset()
    collate_fn = partial(
        prompt_dataset.sdxl_collate_fn,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
    )

    data_loader = torch.utils.data.DataLoader(
        prompt_dataset,
        collate_fn=collate_fn,
        batch_size=config.sample.batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
    )
    

    # # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # # more memory
    # autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    unet, optimizer, data_loader = accelerator.prepare(
        unet, optimizer, data_loader
    )
    # executor to perform callbacks asynchronously.
    # executor = futures.ThreadPoolExecutor(max_workers=2)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="Online_PSO", 
            config=config.to_dict(), 
            init_kwargs={"wandb": {"name": config.run_name}}
        )

    # Train!
    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    first_epoch = 0

    """
      decide the training & sampling timesteps
    """
    # # pdb.set_trace()
    # step_ratio = 1000 // config.sample.num_steps
    # distill_timesteps = (np.arange(config.sample.num_steps, 0, -1) * step_ratio).round() - 1
    # # distill_timesteps = distill_timesteps[::-1]
    # # [1, num_steps]
    # distill_timesteps = torch.tensor(distill_timesteps, 
    #                                  device=accelerator.device, 
    #                                  dtype=weight_dtype).long()
    

    """
      For online SDXL-Turbo, remove the last timestep in training, which is deterministic
    """
    global_step = 0
    for epoch in range(first_epoch, config.num_epochs):
        train_loss = 0.0
        train_ratio_win = 0.0
        train_ratio_lose = 0.0

        #################### SAMPLING #################### 
        samples = []
        iter_train_prompts = []
        prompt_metadata = None
        ## iterate over train prompts
        unet.eval()
        for batch_idx, batch in enumerate(data_loader):
            train_prompts = batch['prompts']
            bsz = len(batch['prompts'])
                
            ## tokenize prompts
            add_time_ids = torch.cat(
                    [compute_time_ids(512, 0) for _ in range(bsz)]
            )

            ## Get the text embedding for conditioning
            # pdb.set_trace()
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                [text_encoder_one, text_encoder_two], 
                [batch["input_ids_one"], batch["input_ids_two"]]
            )

            ## sample a pair of trajectories
            images1, latents1, log_probs1, input_latents1 = sdxl_turbo_pipeline_with_logprob(
                accelerator,
                vae,
                unet,
                noise_scheduler=noise_scheduler,
                height=512,
                width=512,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                add_time_ids=add_time_ids,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=0.,
                output_type="pt",
            )
            # pdb.set_trace()
            latents1 = torch.stack(latents1, dim=1)
            log_probs1 = torch.stack(log_probs1, dim=1)
            input_latents1 = torch.stack(input_latents1, dim=1)

            # for the second trajectory, use the same starting latents, prompt embeddings, and added conditions
            images2, latents2, log_probs2, input_latents2 = sdxl_turbo_pipeline_with_logprob(
                accelerator,
                vae,
                unet,
                noise_scheduler=noise_scheduler,
                height=512,
                width=512,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                add_time_ids=add_time_ids,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=0.,
                output_type="pt",
            )
            latents2 = torch.stack(latents2, dim=1)
            log_probs2 = torch.stack(log_probs2, dim=1)
            input_latents2 = torch.stack(input_latents2, dim=1)

            latents = torch.stack([latents1, latents2], dim=1)  # (batch_size, 2, num_steps, 4, 64, 64)
            log_probs = torch.stack([log_probs1, log_probs2], dim=1)  # (batch_size, 2, num_steps-1, 1)
            input_latents = torch.stack([input_latents1, input_latents2], dim=1)  # (batch_size, 2, num_steps-1, 1)
            """
              fetch sample timesteps from noise scheduler,
              REMOVE the last timestep (e.g., 249)
            """
            timesteps = noise_scheduler.timesteps[:-1].unsqueeze(0).repeat(config.sample.batch_size, 1)  # (batch_size, num_steps-1)

            prompt_embeds = torch.stack([prompt_embeds, prompt_embeds], dim=1) # (batch_size, 2, token_len, dim)
            pooled_prompt_embeds = torch.stack([pooled_prompt_embeds, pooled_prompt_embeds], dim=1) # (batch_size, 2, dim)
            add_time_ids = torch.stack([add_time_ids, add_time_ids], dim=1) # (batch_size, 2, dim)
            current_latents = latents[:, :, :-1]
            next_latents = latents[:, :, 1:]

            ## [2, bs, 3, H, W] -> [2*bs, 3, H, W]
            images = torch.stack([images1, images2], dim=0).reshape(-1, 3, 512, 512) # (batch_size*2, C, H, W)
            
            
            # pdb.set_trace()

            # compute rewards separately
            images1 = ((images1 + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            images2 = ((images2 + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

            # pdb.set_trace()

            ## each with [bs], suppose we have m rewards
            rewards1_ps = ps_selector.score([Image.fromarray(img) for img in images1], train_prompts)

            rewards2_ps = ps_selector.score([Image.fromarray(img) for img in images2], train_prompts)


            rewards1 = np.stack([rewards1_ps], axis=1)
            rewards2 = np.stack([rewards2_ps], axis=1)

            # multi-dimension rewards
            #  [bs, 2, m]
            rewards = np.stack([rewards1, rewards2], axis=1)
            
            eval_rewards = None
            train_prompts = list(train_prompts)
            iter_train_prompts += train_prompts
            samples.append(
                {
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "add_time_ids": add_time_ids,
                    "timesteps": timesteps,
                    "latents": current_latents,  # each entry is the latent before timestep t
                    "next_latents": next_latents,  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "input_latents": input_latents,
                    "images": images,
                    "rewards": torch.as_tensor(rewards, device=accelerator.device),
                }
            )
            # pdb.set_trace()

            logger.info(f'Epoch {epoch}: Sampling {bsz * accelerator.num_processes} pairs of trajectories, Batch [{batch_idx+1}/{config.sample.num_batches_per_epoch}]')

            ## break sample iterations
            if (batch_idx+1) == config.sample.num_batches_per_epoch:
                break

        ## concat samples on each process
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        
        ## collect overall rewards
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()
        
        ## logging
        accelerator.log(
                {
                    "reward1_mean": rewards[:, :, 0].mean(), 
                    "reward1_std": rewards[:, :, 0].std(),
                },
            )
        
        # ## save sampled images & rewards from the main proces
        # if epoch % 100 == 0:
        #     if accelerator.is_main_process:
        #         # pdb.set_trace()
        #         images = ((samples["images"] + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        #         ## show the first 4 pairs
        #         images = images[:4]
                        
        #         accelerator.log(
        #             {
        #                 "Train Sampled Pair Images":[
        #                         wandb.Image(Image.fromarray(images[i]).resize((256, 256)),
        #                                     caption=f"reward: {rewards[:, :, 0].reshape(-1)[i]:.6f}") 
        #                         for i in range(images.shape[0])
        #                     ]
        #             }
        #         )

        # save prompts
        del samples["images"]
        del images1
        del images2
        

        # pdb.set_trace()
        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert total_batch_size == config.sample.batch_size * config.sample.num_batches_per_epoch
        assert num_timesteps == num_train_timesteps
        orig_sample = copy.deepcopy(samples)
        orig_prompts = copy.deepcopy(iter_train_prompts)
        
        ## reset sample collectors
        del samples
        del iter_train_prompts
        samples = []
        iter_train_prompts = []
        gc.collect()
        torch.cuda.empty_cache()
         
            
        #################### TRAINING ####################
        unet.train()
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in orig_sample.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)]
            )
            for key in ["latents", "next_latents", 'input_latents']:
                tmp = samples[key].permute(0,2,3,4,5,1)[torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]
                samples[key] = tmp.permute(0,5,1,2,3,4)
            samples["timesteps"] = samples["timesteps"][torch.arange(total_batch_size, device=accelerator.device)[:, None], perms].unsqueeze(1).repeat(1,2,1)
            tmp = samples["log_probs"].permute(0,2,1)[torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]
            samples["log_probs"] = tmp.permute(0,2,1)
            # rebatch for training
            samples_batched = {k: v.reshape(-1, config.train.batch_size, *v.shape[1:]) for k, v in samples.items()}
            # dict of lists -> list of dicts for easier iteration
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]
            # train
            unet.train()
            info = defaultdict(list)
            
            ## iterate over all sampled trajectory pairs
            for i in range(0, total_batch_size, config.train.batch_size):
                sample_0 = {}
                sample_1 = {}
                for key, value in samples.items():
                    sample_0[key] = value[i:i+config.train.batch_size, 0]
                    sample_1[key] = value[i:i+config.train.batch_size, 1]
                
                embeds_0 = sample_0["prompt_embeds"]
                embeds_1 = sample_1["prompt_embeds"]
                pool_embeds_0 = sample_0["pooled_prompt_embeds"]
                pool_embeds_1 = sample_1["pooled_prompt_embeds"]
                add_ids_0 = sample_0["add_time_ids"]
                add_ids_1 = sample_1["add_time_ids"]
                
                ## decide sample-level random preference
                # human_prefer = sample_compare(sample_0['rewards'], sample_1['rewards'])
                for j in range(num_train_timesteps):
                    # pdb.set_trace()  
                    with accelerator.accumulate(unet):
                        ## online distilled model pred
                        noise_pred_0 = unet(
                                sample_0["input_latents"][:, j], 
                                sample_0["timesteps"][:, j], 
                                embeds_0,
                                added_cond_kwargs={"time_ids": add_ids_0, "text_embeds": pool_embeds_0},
                        ).sample

                        noise_pred_1 = unet(
                                sample_1["input_latents"][:, j], 
                                sample_1["timesteps"][:, j], 
                                embeds_1,
                                added_cond_kwargs={"time_ids": add_ids_1, "text_embeds": pool_embeds_1},
                        ).sample

                        ## ref distilled model prediction
                        accelerator.unwrap_model(unet).disable_adapters()
                        with torch.no_grad():
                            noise_ref_pred_0 = unet(
                                sample_0["input_latents"][:, j], 
                                sample_0["timesteps"][:, j], 
                                embeds_0,
                                added_cond_kwargs={"time_ids": add_ids_0, "text_embeds": pool_embeds_0},
                            ).sample
                            
                            noise_ref_pred_1 = unet(
                                sample_1["input_latents"][:, j], 
                                sample_1["timesteps"][:, j], 
                                embeds_1,
                                added_cond_kwargs={"time_ids": add_ids_1, "text_embeds": pool_embeds_1},
                            ).sample
                        accelerator.unwrap_model(unet).enable_adapters()


                        # compute the log prob of next_latents given latents under the current model
                        # pdb.set_trace()
                        _, total_prob_0 = turbo_step_with_logprob(
                            noise_scheduler,
                            model_output=noise_pred_0,
                            timestep=sample_0["timesteps"][:, j],
                            sample=sample_0["latents"][:, j],
                            prev_sample=sample_0["next_latents"][:, j],
                        )
                        _, total_ref_prob_0 = turbo_step_with_logprob(
                            noise_scheduler,
                            model_output=noise_ref_pred_0,
                            timestep=sample_0["timesteps"][:, j],
                            sample=sample_0["latents"][:, j],
                            prev_sample=sample_0["next_latents"][:, j],
                        )
                        _, total_prob_1 = turbo_step_with_logprob(
                            noise_scheduler,
                            model_output=noise_pred_1,
                            timestep=sample_1["timesteps"][:, j],
                            sample=sample_1["latents"][:, j],
                            prev_sample=sample_1["next_latents"][:, j],
                        )
                        _, total_ref_prob_1 = turbo_step_with_logprob(
                            noise_scheduler,
                            model_output=noise_ref_pred_1,
                            timestep=sample_1["timesteps"][:, j],
                            sample=sample_1["latents"][:, j],
                            prev_sample=sample_1["next_latents"][:, j],
                        )

                        ## binarize reward for win-lose pairs
                        ## step-random sample
                        # pdb.set_trace()
                        human_prefer = sample_compare(sample_0['rewards'], sample_1['rewards'])
                        # \pi_\theta(a_t | s_t) - \pi_\ref(a_t | s_t)
                        ratio_0 = torch.clamp(torch.exp(total_prob_0-total_ref_prob_0), 1 - config.train.eps, 1 + config.train.eps)
                        ratio_1 = torch.clamp(torch.exp(total_prob_1-total_ref_prob_1), 1 - config.train.eps, 1 + config.train.eps)
                        
                        loss = -torch.log(torch.sigmoid(
                            config.train.beta*(torch.log(ratio_0))*human_prefer[:, 0] + \
                            config.train.beta*(torch.log(ratio_1))*human_prefer[:, 1]
                        )).mean()


                        # log loss
                        train_loss += accelerator.gather(loss).mean().item() / accelerator.gradient_accumulation_steps

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            grad_norm = accelerator.clip_grad_norm_(params_to_optimize, config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

            
                    if accelerator.sync_gradients:
                        # # Checks if the accelerator has performed an optimization step behind the scenes
                        # assert (j == num_train_timesteps - 1) and \
                        #        (i + 1) % config.train.gradient_accumulation_steps == 0
                        
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        # info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        # info.update({f'train_loss: {train_loss:.5f}'})
                        
                        accelerator.log({'train_loss': train_loss}, step=global_step)
                        accelerator.log(info, step=global_step)
                        logger.info(f'Total Epoch {epoch}--Inner Train Epoch {inner_epoch}--Iter {i}: Global step:{global_step}, Loss: {train_loss:.4f}, Grad Norm: {accelerator.gather(grad_norm).mean().item():.4f}')

                        global_step += 1
                        info = defaultdict(list)
                        train_loss = 0.0

                        # make sure we did an optimization step at the end of the inner epoch
                        assert accelerator.sync_gradients

                        if accelerator.is_main_process:
                            if global_step % config.checkpointing_steps == 0 or global_step == 1:
                                save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                                accelerator.save_state(save_path)
                                logger.info(f"Saved state to {save_path}")
                        
                                ## only use val prompts
                                # log_validation_val_dataset(config, vae,
                                #                             accelerator, weight_dtype, global_step,
                                #                             lora_dir=save_path, val_prompts=val_prompts,
                                #                             pickscore_scorer=ps_selector, 
                                #                             clip_scorer=clip_selector)
                                log_validation_val_dataset(config, vae,
                                                            accelerator, weight_dtype, global_step,
                                                            lora_dir=save_path, val_prompts=val_prompts,
                                                            pickscore_scorer=ps_selector, 
                                                            clip_scorer=clip_selector)


        # if epoch!=0 and (epoch+1) % config.save_freq == 0 and accelerator.is_main_process:
        #     accelerator.save_state()


if __name__ == "__main__":
    app.run(main)
