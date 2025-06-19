import os
from collections import defaultdict
import contextlib
import datetime
import time
import wandb

# import swanlab
# swanlab.sync_wandb()
from functools import partial
import tempfile
from PIL import Image
import tqdm
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
import logging
import yaml

import copy
import pickle, gzip

import math
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src',"nablagflow"))

import diffusers
from diffusers import DDIMScheduler, EulerDiscreteScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DPMSolverSinglestepScheduler,DPMSolverMultistepScheduler
# from diffusers import DPMSolverSinglestepScheduler,DPMSolverMultistepScheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.import_utils import is_xformers_available

from packaging import version
from peft import LoraConfig, BOFTConfig
from peft.utils import get_peft_model_state_dict,set_peft_model_state_dict
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from src.utils.util import get_timestamp



# import scripts
# from nablagflow import scripts
# from nablagflow import alignment
import src.nablagflow.scripts as scripts
import src.nablagflow.alignment as alignment

from src.nablagflow.scripts.distributed import init_distributed_singlenode, set_seed, setup_for_distributed

import alignment.prompts
import alignment.rewards
import alignment.rewards_normal
from src.nablagflow.alignment.diffusers_patch.diffsplat_pas_pipeline_with_score import diffSplat_pas_pipeline_with_score
from src.nablagflow.alignment.diffusers_patch.step_with_score import  step_with_score, pred_orig_latent, _get_scale, get_one_minus_alpha_prod_t, get_alpha_prod_t

from transformers import T5EncoderModel, T5Tokenizer
from src.models import GSAutoencoderKL, GSRecon, ElevEst

from extensions.diffusers_diffsplat import PixArtTransformerMV2DModel, PixArtSigmaMVPipeline

import src.nablagflow.alignment.prompt_embedding_dataset as alignment_prompt_embedding_dataset

import einops
# import argparse
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

from src.utils.system_utils import make_source_code_snapshot

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=False)
flags.DEFINE_string("exp_name", "", "Experiment name.")
flags.DEFINE_string("caption", "", "Description of current experiments.")
flags.DEFINE_integer("seed", 0, "Seed.")


def save_checkpoint(transformer, epoch, output_dir, logger,is_local_main_process):
    save_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}")
    unwrapped_unet = unwrap_model(transformer)
    unet_lora_state_dict = get_peft_model_state_dict(unwrapped_unet, adapter_name="pf")
    StableDiffusionPipeline.save_lora_weights(
        save_path,
        unet_lora_state_dict,
        is_main_process=is_local_main_process,
        safe_serialization=True,
    )
    logger.info(f"Saved state to {save_path}")
def get_diffSlat_model(logger,load_text_encoder=False):


    import src.utils.util as util
    from src.options import opt_dict
    import src.utils.util as util


       
    config_file = "extensions/diffusers_diffsplat/configs/gsdiff_pas.yaml"
    configs = util.get_configs(config_file)  # change yaml configs by `extras`

    # Parse the option dict
    opt = opt_dict[configs["opt_type"]]
    if "opt" in configs:
        for k, v in configs["opt"].items():
            setattr(opt, k, v)
    



    output_dir = "output"
    checkpoint_dir = os.path.join("output", "diffSplatHFCKPT")
    transformer_ckpt_dir = os.path.join(checkpoint_dir , "gsdiff_gobj83k_pas_fp16__render", "checkpoints") ## transformer checkpoint
    infer_from_iter=13020





    # Initialize the model, optimizer and lr scheduler
    in_channels = 4  # hard-coded for SD 1.5/2.1
    if opt.input_concat_plucker:
        in_channels += 6
    if opt.input_concat_binary_mask:
        in_channels += 1
    transformer_from_pretrained_kwargs = {
        "sample_size": opt.input_res // 8,  # `8` hard-coded for PixArt-Sigma
        "in_channels": in_channels,
        "out_channels": 8,  # hard-coded for PixArt-Sigma
        "zero_init_conv_in": opt.zero_init_conv_in,
        "view_concat_condition": opt.view_concat_condition,
        "input_concat_plucker": opt.input_concat_plucker,
        "input_concat_binary_mask": opt.input_concat_binary_mask,
    }
    tokenizer = T5Tokenizer.from_pretrained("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="tokenizer")
    if load_text_encoder:
        text_encoder = T5EncoderModel.from_pretrained("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="text_encoder")
    else:
        text_encoder = None
    half_precision = True
    if opt.load_fp16vae_for_sdxl and half_precision:  # fixed fp16 VAE for SDXL
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")


    opt.vis_normals =True
    gsvae = GSAutoencoderKL(opt)
    gsrecon = GSRecon(opt)
    # elevest = ElevEst(opt)
    scheduler_type = "DPMsolver"

    # scheduler_type = "ddim"
    # assert scheduler_type == "ddim", f"Unsupported scheduler type: {scheduler_type}"
    if scheduler_type == "ddim":
        noise_scheduler = DDIMScheduler.from_pretrained("extensions/assets/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="scheduler")
    else:
        noise_scheduler = DPMSolverMultistepScheduler.from_pretrained("extensions/assets/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="scheduler")
        noise_scheduler.config.final_sigmas_type = 'sigma_min'
        noise_scheduler.config.algorithm_type = "sde-dpmsolver++"
        noise_scheduler.config.solver_order = 1
    



    # Load checkpoint
    # logger.info(f"Load checkpoint from iteration [{args.infer_from_iter}]\n")
    if not os.path.exists(os.path.join(transformer_ckpt_dir, f"{infer_from_iter:06d}")):
        infer_from_iter = util.load_ckpt(
            transformer_ckpt_dir,
            infer_from_iter,
            None,
            None,  # `None`: not load model ckpt here
        )
    path = os.path.join(transformer_ckpt_dir, f"{infer_from_iter:06d}")
    # os.system(f"python3 extensions/merge_safetensors.py {path}/transformer_ema")  # merge safetensors for loading
    transformer, loading_info  = PixArtTransformerMV2DModel.from_pretrained_new(path, subfolder="transformer_ema",
        low_cpu_mem_usage=False, ignore_mismatched_sizes=True, output_loading_info=True, **transformer_from_pretrained_kwargs)
    for key in loading_info.keys():
        assert len(loading_info[key]) == 0  # no missing_keys, unexpected_keys, mismatched_keys, error_msgs



    controlnet = None

    # Freeze all models
    if text_encoder is not None:
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        
    vae.requires_grad_(False)
    gsvae.requires_grad_(False)
    gsrecon.requires_grad_(False)
    # elevest.requires_grad_(False)
    transformer.requires_grad_(True)
    # transformer.requires_grad_(False)
    vae.eval()
    gsvae.eval()
    gsrecon.eval()
    # elevest.eval()
    transformer.eval()


    
    load_pretrained_gsvae ='gsvae_gobj265k_sdxl_fp16'
    load_pretrained_gsvae_ckpt = 39030
    load_pretrained_gsrecon ='gsrecon_gobj265k_cnp_even4'
    load_pretrained_gsrecon_ckpt = 124320
    load_pretrained_elevest ='elevest_gobj265k_b_C25'
    load_pretrained_elevest_ckpt= 62160
    
    # Load pretrained reconstruction and gsvae models
    logger.info(f"Load GSVAE checkpoint from [{load_pretrained_gsvae}] iteration [{load_pretrained_gsvae_ckpt:06d}]\n")
    gsvae = util.load_ckpt(
        os.path.join(checkpoint_dir, load_pretrained_gsvae, "checkpoints"),
        load_pretrained_gsvae_ckpt,
        None ,
        gsvae,
    )
    logger.info(f"Load GSRecon checkpoint from [{load_pretrained_gsrecon}] iteration [{load_pretrained_gsrecon_ckpt:06d}]\n")
    gsrecon = util.load_ckpt(
        os.path.join(checkpoint_dir, load_pretrained_gsrecon, "checkpoints"),
        load_pretrained_gsrecon_ckpt,
        None ,
        gsrecon,
    )
    logger.info(f"Load ElevEst checkpoint from [{load_pretrained_elevest}] iteration [{load_pretrained_elevest_ckpt:06d}]\n")
    # elevest = util.load_ckpt(
    #     os.path.join(checkpoint_dir, load_pretrained_elevest, "checkpoints"),
    #     load_pretrained_elevest_ckpt,
    #     None ,
    #     elevest,
    # )

    text_encoder = text_encoder
    vae = vae
    gsvae = gsvae
    gsrecon = gsrecon
    # elevest = elevest
    transformer = transformer


    # Set diffusion pipeline
    V_in = opt.num_input_views
    pipeline = PixArtSigmaMVPipeline(
        text_encoder=text_encoder, tokenizer=tokenizer,
        vae=vae, transformer=transformer,
        scheduler=noise_scheduler,
    )
    
    model_dict = {
                "pipeline": pipeline,
                "text_encoder": text_encoder,"tokenizer":tokenizer,
                "vae": vae,
                "transformer": transformer,
                "noise_scheduler": noise_scheduler,
                "gsvae": gsvae,
                "gsrecon": gsrecon,
                "transformer_config": transformer_from_pretrained_kwargs,
                
                  }
    return model_dict
                  
                
    
    

def compute_grad_norm(model, norm_type=2):
    """
    Computes the norm of the gradients for all parameters in the model.
    
    Args:
        model (torch.nn.Module): The model for which gradients are computed.
        norm_type (float or int): The type of the norm. Default is 2 for L2 norm.
        
    Returns:
        float: The gradient norm.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.flatten().norm()
            total_norm += param_norm.pow(2)
    total_norm = total_norm.sqrt()
    return total_norm

def unwrap_model(model):
    model = model.module if isinstance(model, DDP) else model
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def main(args):
    train()

def train():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # config = yaml.safe_load(open(f"config/{args.config}.yaml"))['parameters']
    config = FLAGS.config

    local_rank, global_rank, world_size = init_distributed_singlenode(timeout=36000)
    num_processes = world_size
    is_local_main_process = local_rank == 0
    setup_for_distributed(is_local_main_process)

    config.gpu_type = torch.cuda.get_device_name() \
                            if torch.cuda.is_available() else "CPU"
    if is_local_main_process:
        logger.info(f"GPU type: {config.gpu_type}")

    # config.config_name = f"{FLAGS.config}"
    if FLAGS.seed is not None:
        config.seed = FLAGS.seed
    else:
        config.seed = 0

    n_reward_avg = config.model.n_reward_avg
    mb_dropout = config.model.timestep_fraction
    assert mb_dropout > 0


    
    name_head = ".".join(config.experiment.reward_fn.split('_'))
    exp_group_name = f"{name_head}_{FLAGS.exp_name}"
    # exp_group_name = f"{config.experiment.reward_fn.split('_')[0]}_{FLAGS.exp_name}"
    wandb_name = f"{exp_group_name}_seed{config.seed}"


    if config.logging.use_wandb:
        wandb_proj_name = 'Nabla-R2D3'
        local_wandb_path = './wandb'
        wandb.login()
        wandb.init(project=wandb_proj_name, name=wandb_name, config=config.to_dict(),
           dir=local_wandb_path,settings=wandb.Settings(init_timeout=320),
           save_code=True, 
            # mode = "offline" if is_local_main_process else "disabled")
           mode="online" if is_local_main_process else "disabled")
    

    timestamp = get_timestamp()
    output_dir = os.path.join('./output_align3D',FLAGS.exp_name,timestamp)
    if is_local_main_process:
        make_source_code_snapshot(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"\n{config}")
    set_seed(config.seed)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.

    weight_dtype = torch.float32
    if config.training.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif config.training.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    device = torch.device(local_rank)
    
    trick_mix_precesion =  bool(os.environ.get("Transformer_FP32_Softmax", False))
    if trick_mix_precesion:
        logger.info(f"================enable trick_mix_precesion: {trick_mix_precesion}====================")
        logger.info(f"================compulsory fp32 softmax ====================")
    
    ####################################################################################
    ############## NOTE: Load DiffSlat Model and Prepare config. start #########################################
    #####################################################################################
    ###################################################################################
    
    """_summary_

    Raises:
        ValueError: _description_
        ImportError: _description_
        e: _description_

    Returns:
        _type_: _description_
    """

    config.model.reverse_loss_scale =0.0 
        
    model_dict = get_diffSlat_model(logger)
    pipeline:PixArtSigmaMVPipeline = model_dict["pipeline"]
    
    gsvae:AutoencoderKL= model_dict["gsvae"]
    gsrecon:GSRecon = model_dict["gsrecon"]
    # scheduler_config.update(pipeline.scheduler.config)
    num_inference_steps = config.sampling.num_steps  # Adjust as needed

    gsvae=gsvae.requires_grad_(False)
    gsrecon= gsrecon.requires_grad_(False)
    gsvae=gsvae.to(device, dtype=weight_dtype) 
    gsrecon=gsrecon.to("cpu")
    # gsrecon=gsrecon.to(device, dtype=weight_dtype)
    pipeline.vae.requires_grad_(False)
    # pipeline.text_encoder.requires_grad_(False)
    pipeline.vae.to("cpu")
    ## keeep this cond cpu to reduce G-RAM  usage.
    # pipeline.text_encoder.to("cpu")

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    transformer:PixArtTransformerMV2DModel = pipeline.transformer
    transformer.requires_grad_(False)
    for name, param in transformer.named_parameters():
        if 'conv_out' in name:
            param.requires_grad_(True)
            logger.info(f"--------------------Conv_out in Trainable parameter: {name}")
        else:
            param.requires_grad_(False)
    assert config.model.use_peft
    transformer.to(device, dtype=weight_dtype)

    transformer_lora_config = LoraConfig(
        r=config.model.lora_rank,
        init_lora_weights="gaussian",
        target_modules=[
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "proj",
            "linear",
            "linear_1",
            "linear_2",
        ],
        # use_dora=config.model.use_dora,
        # use_rslora=config.model.use_rslora, ## TODO: check this
    )

    if is_local_main_process:
        transformer_lora_config.save_pretrained(os.path.join(output_dir, "lora_config"))
    
    transformer.add_adapter(transformer_lora_config, adapter_name="pf")
    
    
    ####################################################################################
    ##############################Load Checkpoints##########################################
    ####################################################################################
    load_ckpt= config.training.load_ckpt
    if load_ckpt==True:
        lora_dir = config.training.lora_dir
        assert os.path.exists(lora_dir)
        ## FIXME: hard code the lora_dir
        import glob 
        from safetensors.torch import load_file, save_file
        if not os.path.exists(os.path.join(lora_dir,"adapter_model.safetensors")):      
            state_dict_file = glob.glob(os.path.join(lora_dir, "pytorch_lora_weights.safetensors"))
            # state_dict_file = glob.glob(os.path.join(lora_dir, "pytorch_lora_weights.safetensors"))
            state_dict_file = [os.path.join(lora_dir, "pytorch_lora_weights.safetensors"),]
            assert len(state_dict_file) == 1
            state_dict = load_file(state_dict_file[0])
            new_state_dict = {}
            for key, tensor in state_dict.items():
                # new_state_dict["base_model.model"+key.lstrip("unet")]=tensor
                new_state_dict[key.lstrip("unet").lstrip(".")]=tensor
        load_results = set_peft_model_state_dict(transformer, new_state_dict,adapter_name="pf")
        # print(load_results)
        logger.info(f"Loaded Lora weights from {state_dict_file[0]}")
        logger.info(f"mismatch lora keys {load_results[1]}")
    else:
        logger.info(f"Not loading any checkpoints, training from scratch")
    ####################################################################################
    ##############################Load Checkpoints##########################################
    ####################################################################################

    
    import src.utils.geo_util as geo_util
    default_cam_info = geo_util.default_4_cams(config.diffsplat,device=device)
    H = W = config.diffsplat.render_res
    plucker, _ = geo_util.plucker_ray(H, W, default_cam_info["input_C2W"].unsqueeze(0), default_cam_info["input_fxfycxcy"].unsqueeze(0))
    plucker = plucker.squeeze(0)  # (V_in, 6, H, W)
    
    def decode_gsvae(latents, cam_info:dict,clamp:bool=True,num_views:int=4):
        assert cam_info is not None and latents is not None, "cam_info and latents must be provided"

        
        
        render_C2W, render_fxfycxcy = cam_info["input_C2W"], cam_info["input_fxfycxcy"]
        height, width = cam_info["height"], cam_info["width"]
        V,d1,d2=render_C2W.shape
        render_C2W = render_C2W.unsqueeze(0)
        assert V==num_views,"error"
        B = latents.shape[0]//num_views
        render_C2W = render_C2W.expand(B, V, d1, d2)
        render_fxfycxcy = render_fxfycxcy.unsqueeze(0)
        render_fxfycxcy = render_fxfycxcy.expand(B, V, 4)
        
        input_C2W, input_fxfycxcy = default_cam_info["input_C2W"], default_cam_info["input_fxfycxcy"]
        input_C2W = input_C2W.unsqueeze(0)
        input_C2W = input_C2W.expand(B, V, d1, d2)
        input_fxfycxcy = input_fxfycxcy.unsqueeze(0)
        input_fxfycxcy = input_fxfycxcy.expand(B, V, 4)
        
        latents = latents / gsvae.scaling_factor + gsvae.shift_factor
        render_outputs = gsvae.decode_and_render_gslatents(
            gsrecon,
            latents, 
            input_C2W, input_fxfycxcy,
            C2W=render_C2W, fxfycxcy=render_fxfycxcy,
            height=height, width=width,
            # opacity_threshold=args.opacity_threshold,
        )
        images = render_outputs["image"] # torch.Size([B, V, C, H, W])
        if clamp:
            images = images.clamp(0, 1)
        return images, render_outputs
    
    num_views=config.diffsplat.num_views
    
    if config.training.reward_context=="inference":
        reward_context = torch.inference_mode   
    elif config.training.reward_context=="train" or config.training.reward_context=="none":
        reward_context = contextlib.nullcontext
    else:
        raise ValueError("reward_context must be either 'inference' or 'train'")
    
    ####################################################################################
    ############## NOTE: Load DiffSlat Model and Prepare config. End#########################################
    #####################################################################################
    ###################################################################################
    

    transformer.set_adapter("pf")
    if config.training.mixed_precision in ["fp16", "bf16"]:
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(transformer, dtype=torch.float32)

    

    ############################################################################################################
    pf_params = [param for name, param in transformer.named_parameters() if '.pf.' in name]
    scaler = None
    if config.training.mixed_precision in ["fp16", "bf16"]:
        scaler = torch.cuda.amp.GradScaler(

            growth_interval=config.training.gradscaler_growth_interval
        )


    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.training.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.allow_tf32 is True by default
        torch.backends.cudnn.benchmark = True

    if config.training.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # prepare prompt and reward fn
    
    if "pixart_cache_prompt_embedding_dataset" in config.experiment.prompt_fn: 
        

        if "normal" in config.experiment.reward_fn:
            dataset = alignment_prompt_embedding_dataset.pixart_cache_prompt_embedding_dataset(config.experiment.caching_dir,
                                                                                    read_first_n=config.embedding_cache.read_first_n,
                                                                                    is_apply_reward_threshold=config.experiment.is_apply_reward_threshold,
                                                                                    reward_dir =config.experiment.reward_dir, #####   normal             low reward prompts
                                                                                    reward_threshold=config.experiment.reward_threshold ## inf
                                                                                    )
        else:
            dataset = alignment_prompt_embedding_dataset.pixart_cache_prompt_embedding_dataset(config.experiment.caching_dir,
                                                                                    read_first_n=config.embedding_cache.read_first_n)
            
        
        prompt_fn=lambda batch_size: dataset.get_batch(batch_size)
    
    else:    
        raise ValueError(
            f"Unknown prompt_fn: {config.experiment.prompt_fn}. "
            "Please use a valid prompt function from alignment.prompts."
        )   


    if "normal" in config.experiment.reward_fn:
        fov= 40.0
        reward_fn = getattr(alignment.rewards_normal, config.experiment.reward_fn)(torch.float32, device,fov=fov,
                                                                                   H=config.diffsplat.render_res,
                                                                                   W=config.diffsplat.render_res,
                                                                                   apply_angle_mask=config.experiment.apply_angle_mask,
                                                                                   context_manager=reward_context
                                                                                   )
    else:
        logging.warning("transfer reward_fn to weight dtype")
        reward_fn = getattr(alignment.rewards, config.experiment.reward_fn)(torch.float32, device)


    neg_promptembeds_path = os.path.join(config.experiment.caching_dir,"negative_prompt_embeds.pt")
    neg_promptmask_path =os.path.join(config.experiment.caching_dir, "negative_prompt_attention_mask.pt")
    neg_prompt_embed = torch.load(neg_promptembeds_path).to(device, dtype=weight_dtype)
    neg_prompt_mask = torch.load(neg_promptmask_path).to(device,dtype=weight_dtype)
    
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sampling.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.training.batch_size, 1, 1)
    
    sample_neg_prompt_msk =neg_prompt_mask.repeat(config.sampling.batch_size, 1,)
    train_neg_prompt_msk = neg_prompt_mask.repeat(config.training.batch_size, 1,)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    def func_autocast():
        return torch.cuda.amp.autocast(dtype=weight_dtype)

    def flow_cast_float32():
        return torch.cuda.amp.autocast(dtype=torch.float32)
        # return torch.cuda.amp.autocast(dtype=weight_dtype)

    if config.model.use_peft:
        # LoRA weights are actually float32, but other part of SD are in bf16/fp16
        autocast = contextlib.nullcontext
    else:
        autocast = func_autocast

    ref_compute_mode = torch.inference_mode

        
        

    transformer.to(device)
    transformer = DDP(transformer, device_ids=[local_rank])


    params = [
        {"params": pf_params, "lr": config.training.lr},
    ]

    optimizer = optimizer_cls(
        params,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.adam_weight_decay,
        eps=config.training.adam_epsilon,
    )

    index_list = []

    ### HARDCODED: 20-step inference
    for i in range(5):
        index_list.extend([i, 5 + i, 10 + i, 15 + i])

    result = defaultdict(dict)
    result["config"] = config.to_dict()
    start_time = time.time()

    #######################################################
    # Start!
    samples_per_epoch = (
        config.sampling.batch_size * num_processes
        * config.sampling.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.training.batch_size * num_processes
        * config.training.gradient_accumulation_steps
    )

    if is_local_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {config.training.num_epochs}")
        logger.info(f"  Sample batch size per device = {config.sampling.batch_size}")
        logger.info(f"  Train batch size per device = {config.training.batch_size}")
        logger.info(
            f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}"
        )
        logger.info("")
        logger.info(f"  Total number of samples per epoch = test_bs * num_batch_per_epoch * num_process = {samples_per_epoch}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = train_bs * grad_accumul * num_process = {total_train_batch_size}"
        )
        logger.info(
            f"  Number of gradient updates per inner epoch = samples_per_epoch // total_train_batch_size = {samples_per_epoch // total_train_batch_size}"
        )
        logger.info(f"  Number of inner epochs = {config.training.num_inner_epochs}")

    assert config.sampling.batch_size >= config.training.batch_size
    assert config.sampling.batch_size % config.training.batch_size == 0 # not necessary
    assert samples_per_epoch % total_train_batch_size == 0

    first_epoch = -1
    global_step = 0
    curr_samples = None


    num_diffusion_steps = config.sampling.num_steps
    pipeline.scheduler.set_timesteps(num_diffusion_steps, device=device)  # set_timesteps(): 1000 steps -> 50 steps
    scheduler_dt = pipeline.scheduler.timesteps[0] - pipeline.scheduler.timesteps[1]
    num_train_timesteps = int(num_diffusion_steps * config.model.timestep_fraction)
    accumulation_steps = config.training.gradient_accumulation_steps * num_train_timesteps
    
    ###
    plucker = plucker.unsqueeze(0)  # (V_in, 6, H, W) -> (1, V_in, 6, H, W)
    plucker_cache = plucker
    _,v_,c,h,w=plucker.shape
    plucker= plucker.expand(config.sampling.batch_size, v_, c, h, w)  # (1, V_in, 6, H, W) -> (B, V_in, 6, H, W)

    from extensions.diffusers_diffsplat import  PixArtSigmaMVPipeline
    from extensions.diffusers_diffsplat import   PixArtTransformerMV2DModel
    def prepare_embedding(self:PixArtTransformerMV2DModel,latent_model_input,batch_size,do_classifier_free_guidance=True):
        num_views=4
        latent_model_input = einops.rearrange(latent_model_input, "(b v) c h w -> b v c h w", v=num_views)
        
        # if self.config.input_concat_plucker:
        plucker= plucker_cache.expand(batch_size, v_, c, h, w)  # (1, V_in, 6, H, W) -> (B, V_in, 6, H, W)
        plucker=einops.rearrange(plucker, "b v c h w -> (b v) c h w")
        assert plucker.shape[0] == batch_size * (num_views)
        plucker = pipeline.prepare_plucker(plucker, 1, do_classifier_free_guidance)

        plucker = F.interpolate(plucker, size=latent_model_input.shape[-2:], mode="bilinear", align_corners=False)
        plucker = einops.rearrange(plucker, "(b v) c h w -> b v c h w", v=num_views)
        latent_model_input = torch.cat([latent_model_input, plucker], dim=2)  # (B, V_in(+V_cond), 4+6, H', W')
        # plucker = einops.rearrange(plucker, "b v c h w -> (b v) c h w")
            
        latent_model_input = einops.rearrange(latent_model_input, "b v c h w -> (b v) c h w")
            
        # return latent
        # pass sssss
        return latent_model_input #torch.Size([, 10, 32, 32])
    def inter_repeat_list(lst,batch_size):
        result = []
        for element in lst:
            result.extend([element] * batch_size)
        return result
    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
    random_cams=None  
    if config.training.reward_aggregate_func=="max":
        aggregate_reward_func = lambda x: torch.max(x,dim=-1)[0]
    elif config.training.reward_aggregate_func=="mean":
        
        aggregate_reward_func = lambda x: torch.mean(x,dim=-1)
    else:
        raise ValueError("reward_aggregate_func must be either 'max' or 'mean'")
    
    
    # memo_efficient= "normal" in config.experiment.reward_fn and "yoso" in config.experiment.reward_fn
    memo_efficient= config.training.memo_efficient_mode
    
    if memo_efficient:
        logger.info(" +++++++++++++++++++ memory efficient Reward func++++++++++++++++++++")
    
    for epoch in range(first_epoch, config.training.num_epochs):

        #################### SAMPLING ####################
        torch.cuda.empty_cache()
        transformer.zero_grad()
        transformer.eval()


        if True:
            samples = []
            prompts = []
            for i in tqdm(
                range(config.sampling.num_batches_per_epoch),
                desc=f"Epoch {epoch}: sampling",
                disable=not is_local_main_process,
                position=0,
            ):
                with torch.inference_mode(): # similar to torch.no_grad() but also disables autograd.grad()
                    # generate prompts
                    
                    assert "prompt_embedding_dataset" in config.experiment.prompt_fn, "only cache prompt_fn is supported"
                    if "prompt_embedding_dataset" in config.experiment.prompt_fn:
                        prompt_dict =  prompt_fn(config.sampling.batch_size,)
                        
                        prompts = prompt_dict["prompts"]
                        prompt_metadata = prompt_dict["metadata"]
                        prompt_embeds = torch.cat(prompt_dict["embeds"]).to(device,weight_dtype)
                        prompts_at_masks =torch.cat( prompt_dict["masks"]).to(device,weight_dtype)
                    
                    else:
                        raise ValueError("Not supported prompt_fn")

                    # sample
                    with autocast():
                        ret_tuple = diffSplat_pas_pipeline_with_score(      
                            pipeline,
                            prompt_embeds=prompt_embeds, #torch.Size([Batch, 77, 768])
                            negative_prompt_embeds=sample_neg_prompt_embeds,#torch.Size([Batch, 77, 768])
                            prompt_attention_mask=prompts_at_masks,
                            negative_prompt_attention_mask=sample_neg_prompt_msk,
                            num_inference_steps=num_diffusion_steps,
                            guidance_scale=config.sampling.guidance_scale,
                            eta=config.sampling.eta,
                             plucker=plucker,# (Batch,V_in, 6, H, W)
                            init_std=config.diffsplat.init_std, 
                            init_noise_strength=config.diffsplat.init_noise_strength, 
                            init_bg=config.diffsplat.init_bg,
                            num_views=config.diffsplat.num_views,
                            output_type="latent",
                            return_unetoutput=config.model.unet_reg_scale > 0.,
                            device=device,
                                )

                    if config.model.unet_reg_scale > 0: ##  latents[ list of [(b v) c h w]]
                        _images, _, latents, scores, log_probs, unet_outputs = ret_tuple
                        unet_outputs = torch.stack(unet_outputs, dim=1)  # (batch_size, num_steps, 3, 32, 32)
                    else:
                        _images, _, latents, scores, log_probs = ret_tuple
                    
                    random_sample_cams=None    
                    if config.diffsplat.random_view:
                        random_sample_cams = geo_util.random_sample_cams(config.diffsplat,device=device)
                    
                   
                    images,render_outputs = decode_gsvae(_images, 
                                          cam_info= random_sample_cams  if random_sample_cams  is not None else default_cam_info
                                          )

                    latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
                    log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
                    scores = torch.stack(scores, dim=1)  # (batch_size, num_steps, 1)
                    timesteps = pipeline.scheduler.timesteps.repeat(
                        config.sampling.batch_size, 1
                    )  # (bs, num_steps)  (981, 961, ..., 21, 1) corresponds to "next_latents"
                    
                    step_index = torch.range(0, timesteps.size(1) - 1, device=timesteps.device, dtype=torch.int64).view(1, -1).expand(timesteps.size(0), -1)
                
                
                with torch.inference_mode():
                    B_,V_,C_,H_,W_ = images.shape
                    images = einops.rearrange(images, 'b v c h w -> (b v) c h w')
                    if "normal" in config.experiment.reward_fn:
                        normals =-1*einops.rearrange( render_outputs["raw_normal"],'b v c h w -> (b v) c h w')
                        masks = einops.rearrange(render_outputs["alpha"],'b v c h w -> (b v) c h w')
                        images_normals =(normals +1.0)*0.5
                        rewards,pred_normal = reward_fn(images.float(),normals,render_pkg=render_outputs,masks = masks,) # (reward, reward_metadata)
                        pred_normal = (pred_normal +1.0)*0.5
                        rewards= aggregate_reward_func(einops.rearrange(rewards,"(b v) -> b v", v=V_))
                        rewards=(rewards,{})
                    else:
                        rewards = reward_fn(images.float(),inter_repeat_list(prompts,num_views),inter_repeat_list(prompt_metadata,num_views) ) # (reward, reward_metadata)
                        # rewards[0] = einops.rearrange(rewards[0],"(b v) -> b v",b=B_, v=V_) 
                        rewards = (aggregate_reward_func(einops.rearrange(rewards[0],"(b v) -> b v",b=B_, v=V_)) ,rewards[1])
                        ###    V     mean
                    
                    if config.diffsplat.vis_normal:
                        normals =-1*einops.rearrange( render_outputs["raw_normal"],'b v c h w -> (b v) c h w')
                        masks = einops.rearrange(render_outputs["alpha"],'b v c h w -> (b v) c h w')
                        images_normals =(normals +1.0)*0.5
                        images = torch.cat([images,images_normals],dim=2)
                        if "normal" in config.experiment.reward_fn:
                            images = torch.cat([images,pred_normal],dim=2)
                    
                    samples.append(
                        {
                            "prompts": prompts, 
                            "prompt_metadata": prompt_metadata,
                            "prompt_embeds": prompt_embeds,
                            "prompt_at_masks": prompts_at_masks,
                            "timesteps": timesteps,
                            "latents": latents[
                                :, :-1
                            ],  
                            "next_latents": latents[
                                :, 1:
                            ],  
                            "scores": scores,
                            "log_probs": log_probs,
                            "rewards": rewards,
                            "step_index": step_index
                        }
                    )
                    if config.model.unet_reg_scale > 0:
                        samples[-1]["unet_outputs"] = unet_outputs


            # wait for all rewards to be computed
            for sample in tqdm(
                samples,
                desc="Waiting for rewards",
                disable=not is_local_main_process,
                position=0,
            ):
                rewards, reward_metadata = sample["rewards"]
                sample["rewards"] = torch.as_tensor(rewards, device=device)

            # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
            new_samples = {}
            for k in samples[0].keys():
                if k in ["prompts", "prompt_metadata"]:
                    # list of tuples [('cat', 'dog'), ('cat', 'tiger'), ...] -> list ['cat', 'dog', 'cat', 'tiger', ...]
                    new_samples[k] = [item for s in samples for item in s[k]]
                else:

                    new_samples[k] = torch.cat([s[k] for s in samples])
            samples = new_samples

            if epoch >= 0: ## 
                # this is a hack to force wandb to log the images as JPEGs instead of PNGs
                images = einops.rearrange(images, '(b v) c h w -> b  c h (v w)', b=B_, v=V_) 
               
                # tempdir_head = os.path.join("output_align3D/tempfolder") 
                with tempfile.TemporaryDirectory() as tmpdir:
                    if epoch == 0:
                        logger.info(f"Saving images to {tmpdir}")
                    for i, image in enumerate(images):
                    # for i, image in enumerate(images[:1]):
                        # bf16 cannot be converted to numpy directly
                        pil = Image.fromarray(
                            (image.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        )
                        # pil = pil.resize((256, 256))
                        pil.save(os.path.join(tmpdir, f"{i}.jpg")) #os.path.join(tmpdir, f"{i}.jpg")
                    if config.logging.use_wandb and is_local_main_process:
                        wandb.log(
                            {
                                "images": [
                                    wandb.Image(
                                        os.path.join(tmpdir, f"{i}.jpg"),
                                        caption=f"{prompt} | {reward:.2f}",
                                    )
                                    for i, (prompt, reward) in enumerate(
                                        zip(prompts, rewards)
                                        # zip(prompts[:1], rewards[:1])
                                    )
                                ],
                            },
                            step=global_step,
                        )

                rewards = torch.zeros(world_size * len(samples["rewards"]),
                            dtype=samples["rewards"].dtype, device=device)
                dist.all_gather_into_tensor(rewards, samples["rewards"])
                rewards = rewards.detach().cpu().float().numpy()
                result["reward_mean"][global_step] = rewards.mean()
                result["reward_std"][global_step] = rewards.std()

                if is_local_main_process:
                    logger.info(f"global_step: {global_step}  rewards: {rewards.mean().item():.3f}")
                    if config.logging.use_wandb:
                        wandb.log(
                            {
                                "reward_mean": rewards.mean(), # samples["rewards"].mean()
                                "reward_std": rewards.std(),
                            },
                            step=global_step,
                        )

                # del samples["prompt_ids"]

                total_batch_size, num_timesteps = samples["timesteps"].shape
                assert (
                    total_batch_size
                    == config.sampling.batch_size * config.sampling.num_batches_per_epoch
                )
                assert num_timesteps == num_diffusion_steps


        if curr_samples is None:
            curr_samples = samples
            continue
        #################### TRAINING ####################
        for inner_epoch in range(config.training.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=device)
            for k, v in curr_samples.items():
                if k in ["prompts", "prompt_metadata"]:
                    curr_samples[k] = [v[i] for i in perm]
                elif k in ["unet_outputs"]:
                    curr_samples[k] = v[perm]
                else:
                    curr_samples[k] = v[perm]

            if config.model.timestep_fraction < 1:
                if config.sampling.low_var_subsampling:
                    num_subsampled_steps = int(num_timesteps * config.model.timestep_fraction)

                    ### HARDCODED: 20-step inference
                    perms = torch.stack(
                        [
                            torch.cat([
                                torch.randperm(5, device=device),
                                torch.randperm(5, device=device) + 5,
                                torch.randperm(5, device=device) + 10,
                                torch.randperm(5, device=device) + 15,
                            ]).flatten()
                            for _ in range(total_batch_size)
                        ]
                    ) # (total_batch_size, num_steps)
                    perms = torch.cat([num_timesteps - 1 + torch.zeros_like(perms[:, :1]), perms[:, index_list]], dim=1)
                else:
                    perms = torch.stack(
                        [
                            torch.randperm(num_timesteps - 1, device=device)
                            for _ in range(total_batch_size)
                        ]
                    ) # (total_batch_size, num_steps)
                    # perms = torch.cat([perms.size(1) + torch.zeros_like(perms[:, :1]), perms], dim=1)
                    perms = torch.cat([num_timesteps - 1 + torch.zeros_like(perms[:, :1]), perms], dim=1)
            else:
                perms = torch.stack(
                    [
                        torch.randperm(num_timesteps, device=device)
                        for _ in range(total_batch_size)
                    ]
                ) # (total_batch_size, num_steps)


            
            # "prompts" & "prompt_metadata" are constant along time dimension
            key_ls = ["timesteps", "latents", "next_latents", "log_probs", "scores", "step_index"]
            for key in key_ls:
                curr_samples[key] = curr_samples[key][torch.arange(total_batch_size, device=device)[:, None], perms]
            if config.model.unet_reg_scale > 0:
                curr_samples["unet_outputs"] = \
                    curr_samples["unet_outputs"][torch.arange(total_batch_size, device=device)[:, None], perms]

            ### rebatch for training
            samples_batched = {}
            for k, v in curr_samples.items():
                if k in ["prompts", "prompt_metadata"]:
                    samples_batched[k] = [v[i:i + config.training.batch_size]
                                for i in range(0, len(v), config.training.batch_size)]
                elif k in ["unet_outputs"]:
                    samples_batched[k] = v.reshape(-1, config.training.batch_size, *v.shape[1:])
                else:
                    samples_batched[k] = v.reshape(-1, config.training.batch_size, *v.shape[1:])

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ] # len = sample_bs * num_batches_per_epoch // train_bs = num_train_batches_per_epoch

            transformer.train()

            info = defaultdict(list)
            for i, sample in tqdm( ###  sample 
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_local_main_process,
            ):
                """
                sample: [
                ('prompts', list of strings, len=train_bs), ('prompt_metadata', list of dicts),
                (bf16) ('prompt_embeds', torch.Size([1, 77, 768])),
                (int64) ('timesteps', torch.Size([1, 50])),
                (bf16) ('latents', torch.Size([1, 50, 4, 64, 64])), ('next_latents', torch.Size([1, 50, 4, 64, 64])),
                ('log_probs', torch.Size([1, 50])),
                ]
                """
                batch_size = config.training.batch_size
                if config.training.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    neg_prompt_embeds= einops.repeat(train_neg_prompt_embeds, "b n d -> (b v) n d", v=num_views)
                    pos_prompt_embeds = einops.repeat(sample["prompt_embeds"], "b n d -> (b v) n d", v=num_views)
                    
                    negative_prompt_attention_mask = einops.repeat(train_neg_prompt_msk, "b n -> (b v) n", v=num_views)
                    pos_prompt_embeds_mask = einops.repeat(sample["prompt_at_masks"], "b n -> (b v) n", v=num_views)
                    # if do_classifier_free_guidance:
                    prompt_embeds = torch.cat([ neg_prompt_embeds, pos_prompt_embeds], dim=0)
                    prompt_embeds_mask = torch.cat([negative_prompt_attention_mask, pos_prompt_embeds_mask], dim=0)
                else:
                    raise NotImplementedError

                buffer = [] ###
                for j in tqdm(range(num_train_timesteps), desc="Timestep", position=1, leave=False, disable=not is_local_main_process):
                    with autocast():
                        latent_tmp = sample["latents"][:, j].clone().detach()
                        latent_tmp.requires_grad_(True)
                        latent_tmp=einops.rearrange(latent_tmp,"b (v c) h w -> (b v) c h w",v=num_views) ## FIXME _check 
                        # Before inference, disable the LoRA adapters
                        transformer.module.disable_adapters()  # This should deactivate any applied LoRA adapter
                        
                        if config.diffsplat.random_view:
                            random_cams = geo_util.random_sample_cams(config.diffsplat,device=device,azi_delta=config.diffsplat.azi_delta,elevation_delta=config.diffsplat.elevation_delta)
                        
                        with ref_compute_mode():
                            latent_input= prepare_embedding(transformer,torch.cat([latent_tmp] * 2),batch_size,do_classifier_free_guidance=config.sampling.guidance_scale > 0.0)
                            noise_pred_ref = transformer(                             
                                latent_input,
                                encoder_hidden_states=prompt_embeds,
                                encoder_attention_mask=prompt_embeds_mask,
                                timestep=torch.cat([sample["timesteps"][:, j]]).repeat(latent_input.shape[0]), ## FIXME: batch 1
                                added_cond_kwargs=added_cond_kwargs,
                                cross_attention_kwargs=dict( num_views=num_views),
                            ).sample
                            
                            noise_pred_uncond_ref, noise_pred_text_ref = noise_pred_ref.chunk(2)
                            noise_pred_ref = (
                                    noise_pred_uncond_ref
                                    + config.sampling.guidance_scale
                                    * (noise_pred_text_ref - noise_pred_uncond_ref)
                            )
                            noise_pred_uncond_ref = noise_pred_text_ref = None

                        with torch.inference_mode():
                            next_latent=einops.rearrange(sample["next_latents"][:, j],"b (v c) h w -> (b v) c h w",v=num_views)
                            cur_latent=einops.rearrange(sample["latents"][:, j],"b (v c) h w -> (b v) c h w",v=num_views)
                            if pipeline.transformer.config.out_channels // 2 == 4 :
                                noise_pred_ref = noise_pred_ref.chunk(2, dim=1)[0]
                            else:
                               noise_pred_ref = noise_pred_ref
                            
                            _, score_pf_ref, _ = step_with_score(
                                pipeline.scheduler, noise_pred_ref,
                                sample["timesteps"][:, j], # (train_bs, 50) -> (train_bs,)
                                cur_latent ,## FIXME: reshape latents
                                # sample["latents"][:, j],## FIXME: reshape latents
                                eta=config.sampling.eta, 
                                prev_sample=next_latent, calculate_pb=False,
                                strength=config.model.pretrained_strength,
                                step_index=sample["step_index"][:, j],
                                # step_index=num_inference_steps - sample["step_index"][:, j] - 1,
                            ) # log_pf :(bs,) ## score_pf_ref  in nabla_ torch.Size([B, 4, 64, 64]), 


                        latent_tmp = None
                        _ = noise_pred_ref = None

                        

                        
                        transformer.module.enable_adapters() 
                        transformer.module.set_adapter("pf")

                        latent_tmp = sample["latents"][:, j].clone().detach()
                        latent_tmp.requires_grad_(True)
                        latent_tmp=einops.rearrange(latent_tmp,"b (v c) h w -> (b v) c h w",v=num_views)
                        
                        
                        
                        latent_input= prepare_embedding(transformer,torch.cat([latent_tmp] * 2),batch_size,do_classifier_free_guidance=config.sampling.guidance_scale > 0.0)
                        noise_pred = transformer(
                            # torch.cat([latent_tmp] * 2),
                                latent_input,
                                encoder_hidden_states=prompt_embeds,
                                encoder_attention_mask=prompt_embeds_mask,
                                timestep=torch.cat([sample["timesteps"][:, j]]).repeat(latent_input.shape[0]),
                                added_cond_kwargs=added_cond_kwargs,
                                cross_attention_kwargs=dict( num_views=num_views),
                              ).sample
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = (
                                noise_pred_uncond
                                + config.sampling.guidance_scale
                                * (noise_pred_text - noise_pred_uncond)
                        )
                        noise_pred_uncond = noise_pred_text = None
                        

                            
                        if config.model.unet_reg_scale > 0: ##  reg term
                            # target = einops.rearrange(sample["unet_outputs"][:, j],"b (v c) h w -> (b v) c h w",v=num_views)
                            # unetdiff = (noise_pred - target).pow(2)
                            noise_pred_ = einops.rearrange(noise_pred,"(b v) c h w -> b (v c) h w",v=num_views)
                            unetdiff = (noise_pred_ - sample["unet_outputs"][:, j]).pow(2)
                            if config.training.unet_norm_clip >0:
                                unetdiff  = unetdiff.to(torch.float32)
                                # score_r_next_norm = score_r_next.pow(2).sum(dim=[1,2,3]).sqrt()
                                unetdiff_norm =unetdiff.sum(dim=[1,2,3],keepdims=True).sqrt()

                                unetdiff =  unetdiff*(torch.minimum(unetdiff_norm, torch.ones_like(unetdiff[:,:1,:1,:1])*config.training.unet_norm_clip)  / (1e-7+ unetdiff_norm)).pow(2)  
                            unetreg = torch.mean(unetdiff, dim=(1, 2, 3))
                            unetdiffnorm = unetdiff.sum(dim=[1,2,3]).sqrt() ##TODO: error here

                        if pipeline.transformer.config.out_channels // 2 == 4:
                            noise_pred = noise_pred.chunk(2, dim=1)[0]
                        else:
                            noise_pred = noise_pred
                        next_latent=einops.rearrange(sample["next_latents"][:, j],"b (v c) h w -> (b v) c h w",v=num_views)
                        _, score_pf, log_pf  = step_with_score(
                            pipeline.scheduler, noise_pred,
                            sample["timesteps"][:, j], # (train_bs, 50) -> (train_bs,)
                            latent_tmp,
                            eta=config.sampling.eta,
                            prev_sample=next_latent, calculate_pb=False,
                            step_index=sample["step_index"][:, j],
                            # step_index=num_inference_steps - sample["step_index"][:, j] - 1,
                        ) # log_pf :(bs,)
                        log_pb = None
                        _ = None


                        _ = None
                        
                        #######################################################
                        #################### GFN ALGORITHM ####################
                        #######################################################
                        timestep_next = torch.clamp(sample["timesteps"][:, j] - scheduler_dt, min=0)
                        end_mask = sample["timesteps"][:, j] == pipeline.scheduler.timesteps[-1] # RHS is 1

                        latent_next_tmp = sample["next_latents"][:, j].detach().clone()
                        latent_next_tmp.requires_grad_()
                        transformer.module.set_adapter("pf")

                        latent_next_tmp=einops.rearrange(latent_next_tmp,"b (v c) h w -> (b v) c h w",v=num_views)
                        latent_input_next= prepare_embedding(transformer,torch.cat([latent_next_tmp] * 2),batch_size,do_classifier_free_guidance=config.sampling.guidance_scale > 0.0)
                        noise_pred_next_tmp = transformer(
                                latent_input_next,
                                encoder_hidden_states=prompt_embeds,
                                encoder_attention_mask=prompt_embeds_mask,
                                timestep=torch.cat([timestep_next]).repeat(latent_input.shape[0]),
                                added_cond_kwargs=added_cond_kwargs,
                                cross_attention_kwargs=dict( num_views=num_views),    
                        ).sample
                        
                        noise_pred_uncond_next_tmp, noise_pred_next_text_tmp = noise_pred_next_tmp.chunk(2)
                        noise_pred_next_tmp = (
                                noise_pred_uncond_next_tmp
                                + config.sampling.guidance_scale
                                * (noise_pred_next_text_tmp - noise_pred_uncond_next_tmp)
                        )
                        noise_pred_uncond_next_tmp = noise_pred_next_text_tmp = None
                        if pipeline.transformer.config.out_channels // 2 == 4:
                            noise_pred_next_tmp = noise_pred_next_tmp.chunk(2, dim=1)[0]
                        else:
                            noise_pred_next_tmp = noise_pred_next_tmp
                        
                        pred_z0_next = pred_orig_latent(
                            pipeline.scheduler,
                            noise_pred_next_tmp,
                            latent_next_tmp,
                            timestep_next.repeat_interleave(num_views, dim=0) ## # timestep_next ## TODO:check this when training batch >=2
                        )
                        noise_pred_next_tmp = None
                        # pred_xdata_next = decode(pred_z0_next, clamp=True).unsqueeze(1).repeat_interleave(n_reward_avg, dim=1).float()
                        ## decode this pred_z0_next to images

                        pred_xdata_next,render_outputs = decode_gsvae(pred_z0_next,clamp=True,
                                                       cam_info= random_cams  if random_cams  is not None else default_cam_info
                                                       )
                        pred_xdata_next = pred_xdata_next.unsqueeze(1).repeat_interleave(n_reward_avg, dim=1).float()
                        ##  pred_xdata_next torch.Size([B, N_r, V, 3, 256, 256])
                        with torch.cuda.amp.autocast(enabled=False):
                            noise = torch.randn_like(pred_xdata_next) * 2e-3
                            pred_xdata_next = (pred_xdata_next + noise).flatten(0,2) ## (0,1)-->(0,1,2)
                            noise = None
                            # sample["prompts"]
                            if memo_efficient: ## How many views are used for reward calculation
                                rw_views = 2 ## use two view instead of 4, hard-coded
                                index = torch.randperm(4)[:rw_views]
                                index= index.repeat(n_reward_avg*batch_size)
                            else:
                                rw_views = num_views
                                index = torch.arange(pred_xdata_next.shape[0])                         
                            if not 'normal' in config.experiment.reward_fn: ## other reward
                                prompts_ = inter_repeat_list(sample["prompts"],rw_views)
                                prompts_metadata_aug_= inter_repeat_list(sample["prompt_metadata"],rw_views)
                                prompts_aug = [x for x in prompts_ for _ in range(n_reward_avg)]
                                prompts_metadata_aug = [x for x in prompts_metadata_aug_ for _ in range(n_reward_avg)]
                                logr_next_tmp = reward_fn(pred_xdata_next[index], prompts_aug, prompts_metadata_aug)[0]
                                
                            else: ## normal reward
                                normals =-1*einops.rearrange( render_outputs["raw_normal"],'b v c h w -> (b v) c h w')
                                masks = einops.rearrange(render_outputs["alpha"],'b v c h w -> (b v) c h w')
                                # images_normals =(-1*normals +1.0)*0.5
                                logr_next_tmp = reward_fn(pred_xdata_next[index], normals[index],render_pkg=render_outputs,masks = masks[index] )[0] # (reward, reward_metadata)
                            logr_next_tmp =aggregate_reward_func(einops.rearrange(logr_next_tmp,"(b r v) -> (b r) v ",r=n_reward_avg, v=rw_views))

                            score_r_next_tmp = torch.autograd.grad(
                                outputs=logr_next_tmp.sum(),    # The value whose gradient we want
                                inputs=latent_next_tmp,       # The intermediate node we want the gradient with respect to
                                retain_graph=False,        # Retain graph for further gradient computations
                                create_graph=False         # If higher-order gradients are needed
                            )[0].detach() / n_reward_avg
                            latent_next_tmp = None
                            score_r_next = config.model.reward_scale * score_r_next_tmp
                            
                            score_r_next= einops.rearrange(score_r_next,"(b v) c h w -> b (v c) h w",v=num_views)
                            # raise notImplementedError
                            if config.training.reward_norm_clip and config.training.reward_norm_value > 0:
                                score_r_next  = score_r_next.to(torch.float32)
                                # score_r_next_norm = score_r_next.pow(2).sum(dim=[1,2,3]).sqrt()
                                score_r_next_norm = score_r_next.pow(2).sum(dim=[1,2,3],keepdims=True).sqrt()
                                score_r_next= score_r_next / score_r_next_norm * torch.minimum(score_r_next_norm, torch.ones_like(score_r_next[:,:1,:1,:1]) * config.training.reward_norm_value) 

                                
                                
                            if config.model.reward_adaptive_scaling:
                                alpha_prod_next = get_alpha_prod_t(pipeline.scheduler, timestep_next, sample["next_latents"][:, j])
                                # alpha_prod_next = einops.rearrange(alpha_prod_next,"b (v c) h w -> (b v) c h w",v=num_views)
                                if config.training.alpha_prod_clip:
                                    alpha_prod_next= alpha_prod_next.pow(1.5)
                                    alpha_prod_next = alpha_prod_next*(alpha_prod_next>0.05) 
                                if config.model.reward_adaptive_mode == 'squared':
                                    score_r_next = score_r_next * alpha_prod_next
                                else:
                                    score_r_next = score_r_next * alpha_prod_next.sqrt()
                            score_r_next= einops.rearrange(score_r_next,"b (v c) h w -> (b v) c h w",v=num_views)

                        score_r_next_tmp = None


                        index =None


                    score_pf_target = (score_pf_ref.float() + score_r_next).float()

                    ##
                    end_mask=end_mask.repeat_interleave(num_views,dim=0)
                    score_pf_target[end_mask] = (score_pf_ref[end_mask].float() + score_r_next[end_mask].float()).detach()


                    with torch.inference_mode():
                        if config.model.residual_loss:
                            grad_norm_score_ref = score_pf_ref.pow(2).sum(dim=[1,2,3]).sqrt()
                            grad_norm_res_score = (score_pf - score_pf_ref).pow(2).sum(dim=[1,2,3]).sqrt()
                        grad_norm_score_r = score_r_next.pow(2).sum(dim=[1,2,3]).sqrt()

                        grad_norm_score_pf_target = score_pf_target.pow(2).sum(dim=[1,2,3]).sqrt()


                    score_pf_ref = None
                    score_r_next = None
                    score_r = None
                    score_pf_ref_reverse = None
                    score_pb_reverse = None



                    info["log_pf"].append(torch.mean(log_pf).detach())
                    if log_pb is not None:
                        info["log_pb"].append(torch.mean(log_pb).detach())



                    loss_terminal = torch.zeros(1, device=score_pf.device)

                    if config.training.loss_type == 'l2':
                        losses_forward = (score_pf - score_pf_target).pow(2)
                    elif config.training.loss_type == 'l1':
                        losses_forward = (score_pf - score_pf_target).abs()
                        
                    if config.training.final_step_rw_only:
                        losses_forward = losses_forward*end_mask.view(-1,1,1,1)
                        


                        
                    score_pf = score_pf_target = None
                    loss_forward_mean = losses_forward.mean()

                    info["loss_terminal"].append(loss_terminal)
                    info["loss_forward"].append(loss_forward_mean.detach())
                    loss_backward_mean = 0.0
                    losses_backward = torch.zeros_like(losses_forward)
                    info["loss"].append(loss_forward_mean + loss_backward_mean)

                    with torch.inference_mode():
                        if config.model.residual_loss:
                            info["norm_score_ref_mean"].append(grad_norm_score_ref.mean())
                            info["norm_score_ref_min"].append(grad_norm_score_ref.min())
                            info["norm_score_ref_max"].append(grad_norm_score_ref.max())
                            info["norm_score_residual_mean"].append(grad_norm_res_score.mean())
                            info["norm_score_residual_min"].append(grad_norm_res_score.min())
                            info["norm_score_residual_max"].append(grad_norm_res_score.max())
                        info["grad_norm_score_pf_target_mean"].append(grad_norm_score_pf_target.mean())
                        info["grad_norm_score_pf_target_min"].append(grad_norm_score_pf_target.min())
                        info["grad_norm_score_pf_target_max"].append(grad_norm_score_pf_target.max())
                        info["norm_score_r_mean"].append(grad_norm_score_r.mean())
                        info["norm_score_r_min"].append(grad_norm_score_r.min())
                        info["norm_score_r_max"].append(grad_norm_score_r.max())

                        if config.model.unet_reg_scale > 0:
                            info["norm_unet_diff_mean"].append(unetdiffnorm.mean())
                            info["norm_unet_diff_min"].append(unetdiffnorm.min())
                            info["norm_unet_diff_max"].append(unetdiffnorm.max())


                    info["losses_forward_max"].append(losses_forward.max())
                    info["losses_backward_max"].append(losses_backward.max())
                    info["losses_bidir_max"].append(losses_forward.max())

                    losses = (losses_forward  + loss_terminal).mean()

                    if config.model.unet_reg_scale > 0:
                        losses = losses + config.model.unet_reg_scale * unetreg.mean()
                        info["unetreg"].append(unetreg.mean().detach())
                    loss = torch.mean(losses)

                    # torch.cuda.empty_cache()
                    loss = loss / accumulation_steps
                    assert torch.isnan(loss).sum() == 0, f"Loss is NaN at timestep {j} in epoch {epoch}"
                    if scaler:
                        # Backward passes under autocast are not recommended
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # prevent OOM
                    image = None
                    noise_pred_uncond = noise_pred_text = noise_pred = None
                    logr_next_tmp = logr_tmp = None
                    _ = log_pf = log_pb = None
                    score_pb_reverse = None
                    unetreg = losses =  None
                    score_pf_ref = score_pf = None
                    score_r_next = score_r_next_tmp = None
                    noise_pred_uncond_ref = noise_pred_text_ref = noise_pred_ref = None
                    score_pf_target = None
                    res_logflow = res_logflow_next = None
                    grad_norm_score_pf_target = grad_norm_score_pf_reverse_target = None

    

                if ((j == num_train_timesteps - 1) and
                        (i + 1) % config.training.gradient_accumulation_steps == 0): ### FIXME CHECK THIS
                    if scaler:
                        scaler.unscale_(optimizer)
                        pf_update_grad = torch.nn.utils.clip_grad_norm_([p for name, p in transformer.named_parameters() if '.pf.' in name], config.training.max_grad_norm)

                        # scaler.step(optimizer)
                        optimizer.step() ### No NaN/Inf check
                        scaler.update()
                    else:
                        pf_update_grad = torch.nn.utils.clip_grad_norm_([p for name, p in transformer.named_parameters() if '.pf.' in name], config.training.max_grad_norm)

                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1


                    for param in transformer.parameters():
                        param.grad = None

                    # info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    old_info = info
                    info = {}
                    for k, v in old_info.items():
                        if '_min' in k:
                            info[k] = torch.min(torch.stack(v))
                        elif '_max' in k:
                            info[k] = torch.max(torch.stack(v))
                        else:
                            try:
                                info[k] = torch.mean(torch.stack(v))
                            except Exception as e:
                                print(k)
                                print(v)
                                raise e
                    info['pf_update_grad'] = pf_update_grad.detach()




                    dist.barrier()
                    for k, v in info.items():
                        if '_min' in k:
                            dist.all_reduce(v, op=dist.ReduceOp.MIN)
                        elif '_max' in k:
                            dist.all_reduce(v, op=dist.ReduceOp.MAX)
                        else:
                            dist.all_reduce(v, op=dist.ReduceOp.SUM)
                    info = {k: v / num_processes if ('_min' not in k and '_max' not in k) else v for k, v in info.items()}

                    for k, v in info.items():
                        result[k][global_step] = v.item()

                    info.update({"epoch": epoch})
                    info.update({"global_step": global_step})
                    result["epoch"][global_step] = epoch
                    result["time"][global_step] = time.time() - start_time


                    if is_local_main_process:
                        if scaler:
                            info.update({"grad_scale": scaler.get_scale()})
                            result["grad_scale"] = scaler.get_scale()


                    if is_local_main_process:
                        if config.logging.use_wandb:
                            wandb.log(info, step=global_step)
                        logger.info(f"global_step={global_step}  " +
                              " ".join([f"{k}={v:.6f}" for k, v in info.items()]))
                    info = defaultdict(list) # reset info dict


        curr_samples = samples
        if is_local_main_process:
            pickle.dump(result, gzip.open(os.path.join(output_dir, f"result.json"), 'wb'))
        dist.barrier()

        if epoch % config.logging.save_freq == 0 or epoch == config.training.num_epochs - 1:
            if is_local_main_process:
                save_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}")
                unwrapped_unet = unwrap_model(transformer)
                unet_lora_state_dict = get_peft_model_state_dict(unwrapped_unet, adapter_name="pf")

                StableDiffusionPipeline.save_lora_weights(
                    save_path,
                    unet_lora_state_dict,
                    is_main_process=is_local_main_process,
                    safe_serialization=True,
                )

                logger.info(f"Saved state to {save_path}")

            dist.barrier()
        
        if epoch % config.logging.every_eval_epoch==0 and is_local_main_process:
            pass
            





    if is_local_main_process:
        save_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}_end")
        unwrapped_unet = unwrap_model(transformer)
        unet_lora_state_dict = get_peft_model_state_dict(unwrapped_unet, adapter_name="pf")
        StableDiffusionPipeline.save_lora_weights(
            save_directory=save_path,
            unet_lora_layers=unet_lora_state_dict,
            is_main_process=is_local_main_process,
            safe_serialization=True,
        )

        logger.info(f"Saved state to {save_path}")
    dist.barrier()

    if config.logging.use_wandb and is_local_main_process:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == '__main__':
  app.run(main)
