import warnings
warnings.filterwarnings("ignore")  # ignore all warnings

from typing import *
import einops
import os
import argparse
import logging
import time
import sys 
# sys.path.append('..')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src',"nablagflow"))
from tqdm import tqdm
import numpy as np
import imageio
import torch
import torch.nn.functional as tF
from einops import rearrange
import accelerate
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, AutoencoderKL
from kiui.cam import orbit_camera

from src.options import opt_dict
from src.models import GSAutoencoderKL, GSRecon, ElevEst
import src.utils.util as util
import src.utils.op_util as op_util
import src.utils.geo_util as geo_util
import src.utils.vis_util as vis_util
from src.utils.vis_util import  tob8,show_img,minmax_norm
from extensions.diffusers_diffsplat import PixArtTransformerMV2DModel, PixArtSigmaMVPipeline
from peft.utils import get_peft_model_state_dict,load_peft_weights
from peft import LoraConfig, get_peft_model_state_dict, get_peft_model, PeftModel
# from src.nablagflow.alignment.diffusers_patch.pipeline_with_score import pipeline_with_score
from src.nablagflow.alignment.diffusers_patch.diffsplat_pas_pipeline_with_score import diffSplat_pas_pipeline_with_score
from safetensors.torch import load_file, save_file
from src.utils.geo_util import generate_ray_directions,depth_double_to_normal
from src.nablagflow.alignment.rewards_normal import get_angle_mask
import math
import src.nablagflow.alignment as alignment
import alignment.rewards
import alignment.rewards_normal

def load_loradict2Transformer(Transformer, lora_dir):
    transformer_lora_config = LoraConfig(
        r=8,
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
    import glob 
    if not os.path.exists(os.path.join(lora_dir,"adapter_model.safetensors")):      
        state_dict_file = glob.glob(os.path.join(lora_dir, "pytorch_lora_weights.safetensors"))
        # state_dict_file = glob.glob(os.path.join(lora_dir, "pytorch_lora_weights.safetensors"))
        state_dict_file = [os.path.join(lora_dir, "pytorch_lora_weights.safetensors"),]
        assert len(state_dict_file) == 1
        state_dict = load_file(state_dict_file[0])
        dir_path= os.path.dirname(state_dict_file[0])
        new_state_dict = {}
        for key, tensor in state_dict.items():
            new_state_dict["base_model.model"+key.lstrip("unet")]=tensor
        save_file(new_state_dict, os.path.join(dir_path,"adapter_model.safetensors"))
    
    model = PeftModel.from_pretrained(Transformer, lora_dir,adapter_name="pf",config=transformer_lora_config)

    return  model



def main():
    parser = argparse.ArgumentParser(
        description="Infer a diffusion model for 3D object generation"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the config file"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--hdfs_dir",
        type=str,
        default=None,
        help="Path to the HDFS directory to save checkpoints"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG"
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use"
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Use half precision for inference"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster training on Ampere GPUs"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to the image for reconstruction"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to the directory of images for reconstruction"
    )
    parser.add_argument(
        "--infer_from_iter",
        type=int,
        default=-1,
        help="The iteration to load the checkpoint from"
    )
    parser.add_argument(
        "--rembg_and_center",
        action="store_true",
        help="Whether or not to remove background and center the image"
    )
    parser.add_argument(
        "--rembg_model_name",
        default="u2net",
        type=str,
        help="Rembg model, see https://github.com/danielgatis/rembg#models"
    )
    parser.add_argument(
        "--border_ratio",
        default=0.2,
        type=float,
        help="Rembg output border ratio"
    )

    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="sde-dpmsolver++",
        help="Type of diffusion scheduler"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Diffusion steps for inference"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.5,
        help="Classifier-free guidance scale for inference"
    )
    parser.add_argument(
        "--triangle_cfg_scaling",
        action="store_true",
        help="Whether or not to use triangle classifier-free guidance scaling"
    )
    parser.add_argument(
        "--min_guidance_scale",
        type=float,
        default=1.,
        help="Minimum of triangle cfg scaling"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.,
        help="The weight of noise for added noise in diffusion step"
    )

    parser.add_argument(
        "--init_std",
        type=float,
        default=0.,
        help="Standard deviation of Gaussian grids (cf. Instant3D) for initialization"
    )
    parser.add_argument(
        "--init_noise_strength",
        type=float,
        default=0.98,
        help="Noise strength of Gaussian grids (cf. Instant3D) for initialization"
    )
    parser.add_argument(
        "--init_bg",
        type=float,
        default=0.,
        help="Gray background of Gaussian grids for initialization"
    )

    parser.add_argument(
        "--elevation",
        type=float,
        default=None,
        help="The elevation of rendering"
    )
    parser.add_argument(
        "--use_elevest",
        action="store_true",
        help="Whether or not to use an elevation estimation model"
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=1.4,
        help="The distance of rendering"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Caption prompt for generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        # default="worst quality, normal quality, low quality, low res, blurry, ugly, disgusting",
        default="",
        help="Negative prompt for better classifier-free guidance"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Path to the file of text prompts for generation"
    )

    parser.add_argument(
        "--render_res",
        type=int,
        default=None,
        help="Resolution of GS rendering"
    )
    parser.add_argument(
        "--opacity_threshold",
        type=float,
        default=0.,
        help="The min opacity value for filtering floater Gaussians"
    )
    parser.add_argument(
        "--opacity_threshold_ply",
        type=float,
        default=0.,
        help="The min opacity value for filtering floater Gaussians in ply file"
    )
    parser.add_argument(
        "--save_ply",
        action="store_true",
        help="Whether or not to save the generated Gaussian ply file"
    )
    parser.add_argument(
        "--save_mesh",
        action="store_true",
        help="Whether or not to construct mesh using  ply file"
    )
    parser.add_argument(
        "--possion_recon",
        action="store_true",
        help="Whether or not to construct mesh using  ply file"
    )
    parser.add_argument(
        "--tsdf_recon",
        action="store_true",
        help="Whether or not to construct mesh using  ply file"
    )
    parser.add_argument(
        "--output_video_type",
        type=str,
        default=None,
        help="Type of the output video"
    )

    parser.add_argument(
        "--name_by_id",
        action="store_true",
        help="Whether or not to name the output by the prompt/image ID"
    )
    parser.add_argument(
        "--eval_text_cond",
        action="store_true",
        help="Whether or not to evaluate text-conditioned generation"
    )

    parser.add_argument(
        "--load_pretrained_gsrecon",
        type=str,
        default="gsrecon_gobj265k_cnp_even4",
        help="Tag of a pretrained GSRecon in this project"
    )
    parser.add_argument(
        "--load_pretrained_gsrecon_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained GSRecon checkpoint"
    )
    parser.add_argument(
        "--load_pretrained_gsvae",
        type=str,
        default="gsvae_gobj265k_sdxl_fp16",
        help="Tag of a pretrained GSVAE in this project"
    )
    parser.add_argument(
        "--load_pretrained_gsvae_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained GSVAE checkpoint"
    )
    parser.add_argument(
        "--load_pretrained_elevest",
        type=str,
        default="elevest_gobj265k_b_C25",
        help="Tag of a pretrained GSRecon in this project"
    )
    parser.add_argument(
        "--load_pretrained_elevest_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained GSRecon checkpoint"
    )
    parser.add_argument(
        "--save_geometry",
        action="store_true",
        help="Use half precision for inference"
    )
    parser.add_argument(
        "--enable_lora",
        action="store_true",
        help="Use half precision for inference"
    )

    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--reward_fn",
        type=str,
        default="aesthetic_score",
    ) 
    # Parse the arguments
    args, extras = parser.parse_known_args()

    # Parse the config file
    configs = util.get_configs(args.config_file, extras)  # change yaml configs by `extras`

    # Parse the option dict
    opt = opt_dict[configs["opt_type"]]
    if "opt" in configs:
        for k, v in configs["opt"].items():
            setattr(opt, k, v)

    # Create an experiment directory using the `tag`
    if args.tag is None:
        args.tag = time.strftime("%Y-%m-%d_%H:%M") + "_" + \
            os.path.split(args.config_file)[-1].split()[0]  # config file name

    # Create the experiment directory
    exp_dir = os.path.join(args.output_dir,)
    args.checkpoint_dir = os.path.join("output", "diffSplatHFCKPT")
    ckpt_dir = os.path.join(args.checkpoint_dir,"gsdiff_gobj83k_pas_fp16__render", "checkpoints")
    infer_dir = os.path.join(exp_dir, args.tag, "inferenceSeed"+str(args.seed))
    # os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(infer_dir, exist_ok=True)
    if args.hdfs_dir is not None:
        args.project_hdfs_dir = args.hdfs_dir
        args.hdfs_dir = os.path.join(args.hdfs_dir, args.tag)

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(os.path.join(args.output_dir, args.tag, "log_infer.txt"))  # output to file
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)
    logger.propagate = True  # propagate to the root logger (console)

    # Set the random seed
    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)
        logger.info(f"You have chosen to seed([{args.seed}]) the experiment [{args.tag}]\n")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Set options for image-conditioned models
    if args.image_path is not None or args.image_dir is not None:
        opt.prediction_type = "v_prediction"
        opt.view_concat_condition = True
        opt.input_concat_binary_mask = True
        if args.guidance_scale > 3.:
            logger.info(
                f"WARNING: guidance scale ({args.guidance_scale}) is too large for image-conditioned models. " +
                "Please set it to a smaller value (e.g., 2.0) for better results.\n"
            )

    # Load the image for reconstruction
    if args.image_dir is not None:
        logger.info(f"Load images from [{args.image_dir}]\n")
        image_paths = [
            os.path.join(args.image_dir, filename)
            for filename in os.listdir(args.image_dir)
            if filename.endswith(".png") or filename.endswith(".jpg") or \
                filename.endswith(".jpeg") or filename.endswith(".webp")
        ]
        image_paths = sorted(image_paths)
    elif args.image_path is not None:
        logger.info(f"Load image from [{args.image_path}]\n")
        image_paths = [args.image_path]
    else:
        logger.info(f"No image condition\n")
        image_paths = [None]

    # Load text prompts for generation
    if args.prompt_file is not None:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f.readlines() if line.strip() != ""]
        negative_prompt = args.negative_prompt.replace("_", " ")
        negative_promts = [negative_prompt] * len(prompts)
    else:
        prompt = args.prompt.replace("_", " ")
        negative_prompt = args.negative_prompt.replace("_", " ")
        prompts, negative_promts = [prompt], [negative_prompt]

    # Initialize the model, optimizer and lr scheduler
    in_channels = 4  # hard-coded for PixArt-Sigma
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
    text_encoder = T5EncoderModel.from_pretrained("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="text_encoder").to(torch.bfloat16 if args.half_precision else torch.float32)
    if opt.load_fp16vae_for_sdxl and args.half_precision:  # fixed fp16 VAE for SDXL
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
    else:
        vae = AutoencoderKL.from_pretrained("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="vae")
    opt.vis_normals =args.save_geometry ### 
    gsvae = GSAutoencoderKL(opt)
    gsrecon = GSRecon(opt)

    if args.scheduler_type == "ddim":
        noise_scheduler = DDIMScheduler.from_pretrained("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="scheduler")
    elif "dpmsolver" in args.scheduler_type:
        noise_scheduler = DPMSolverMultistepScheduler.from_pretrained("extensions/assets/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/scheduler", subfolder="scheduler")
        noise_scheduler.config.final_sigmas_type = 'sigma_min'
        noise_scheduler.config.algorithm_type = "sde-dpmsolver++"
        noise_scheduler.config.solver_order = 1
        
        
    elif args.scheduler_type == "edm":
        noise_scheduler = EulerDiscreteScheduler.from_pretrained("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="scheduler")
    else:
        raise NotImplementedError(f"Scheduler [{args.scheduler_type}] is not supported by now")
    ##  FIXME:  
    opt.common_tricks=False
    ##  FIXME:  
    
    if opt.common_tricks:
        noise_scheduler.config.timestep_spacing = "trailing"
        noise_scheduler.config.rescale_betas_zero_snr = True
    if opt.prediction_type is not None:
        noise_scheduler.config.prediction_type = opt.prediction_type
    if opt.beta_schedule is not None:
        noise_scheduler.config.beta_schedule = opt.beta_schedule

    # Load checkpoint
    logger.info(f"Load checkpoint from iteration [{args.infer_from_iter}]\n")
    if not os.path.exists(os.path.join(ckpt_dir, f"{args.infer_from_iter:06d}")):
        args.infer_from_iter = util.load_ckpt(
            ckpt_dir,
            args.infer_from_iter,
            args.hdfs_dir,
            None,  # `None`: not load model ckpt here
        )
    path = os.path.join(ckpt_dir, f"{args.infer_from_iter:06d}")
    # os.system(f"python3 extensions/merge_safetensors.py {path}/transformer_ema")  # merge safetensors for loading
    transformer, loading_info = PixArtTransformerMV2DModel.from_pretrained_new(path, subfolder="transformer_ema",
        low_cpu_mem_usage=False, ignore_mismatched_sizes=True, output_loading_info=True, **transformer_from_pretrained_kwargs)
    for key in loading_info.keys():
        assert len(loading_info[key]) == 0  # no missing_keys, unexpected_keys, mismatched_keys, error_msgs

    # Freeze all models
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    gsvae.requires_grad_(False)
    gsrecon.requires_grad_(False)
    transformer.requires_grad_(False)
    text_encoder.eval()
    vae.eval()
    gsvae.eval()
    gsrecon.eval()
    transformer.eval()

    # Load pretrained reconstruction and gsvae models
    logger.info(f"Load GSVAE checkpoint from [{args.load_pretrained_gsvae}] iteration [{args.load_pretrained_gsvae_ckpt:06d}]\n")
    gsvae = util.load_ckpt(
        os.path.join(args.checkpoint_dir, args.load_pretrained_gsvae, "checkpoints"),
        args.load_pretrained_gsvae_ckpt,
        None if args.hdfs_dir is None else os.path.join(args.project_hdfs_dir, args.load_pretrained_gsvae),
        gsvae,
    )
    logger.info(f"Load GSRecon checkpoint from [{args.load_pretrained_gsrecon}] iteration [{args.load_pretrained_gsrecon_ckpt:06d}]\n")
    gsrecon = util.load_ckpt(
        os.path.join(args.checkpoint_dir, args.load_pretrained_gsrecon, "checkpoints"),
        args.load_pretrained_gsrecon_ckpt,
        None if args.hdfs_dir is None else os.path.join(args.project_hdfs_dir, args.load_pretrained_gsrecon),
        gsrecon,
    )

    text_encoder = text_encoder.to(f"cuda:{args.gpu_id}")
    vae = vae.to(f"cuda:{args.gpu_id}")
    gsvae = gsvae.to(f"cuda:{args.gpu_id}")
    gsrecon = gsrecon.to(f"cuda:{args.gpu_id}")
    transformer = transformer.to(f"cuda:{args.gpu_id}")
    
    
    if args.lora_dir is not None and args.enable_lora:
        transformer = load_loradict2Transformer(transformer, args.lora_dir)
        # transformer = load_loradict2Transformer(transformer, args.lora_dir)
        logger.info(f"LoRA adapter loaded from [{args.lora_dir}]")
        transformer.set_adapter("pf")
        # transformer.enable_adapters()


    # Set diffusion pipeline
    V_in = opt.num_input_views
    pipeline = PixArtSigmaMVPipeline(
        text_encoder=text_encoder, tokenizer=tokenizer,
        vae=vae, transformer=transformer,
        scheduler=noise_scheduler,
    )
    pipeline.set_progress_bar_config(disable=False)
    # pipeline.enable_xformers_memory_efficient_attention()

    if args.seed >= 0:
        generator = torch.Generator(device=f"cuda:{args.gpu_id}").manual_seed(args.seed)
    else:
        generator = None

        # Set rendering resolution
    if args.render_res is None:
        args.render_res = opt.input_res
    ##################################################
    ##################################################
    ##################################################

    device = f"cuda:{args.gpu_id}"
    weight_dtype = torch.float32
    fov= 40.0
    if "normal" in args.reward_fn:
        
        reward_fn = getattr(alignment.rewards_normal, args.reward_fn)(torch.float32, device,fov=fov,
                                                                                   H=args.render_res,
                                                                                   W=args.render_res,
                                                                                   apply_angle_mask=False,
                                                                                #    context_manager=None
                                                                                   )
    else:
        if "aesthetic_score" in args.reward_fn:
            reward_fn = getattr(alignment.rewards, args.reward_fn)(dtype=torch.float32, device="cuda", distributed=False)

        else:   
            reward_fn = getattr(alignment.rewards, args.reward_fn)(torch.float32, device) ## FIXME:    f32    f16         


    ##################################################
    ##################################################
    ##################################################
    

    # Save all experimental parameters of this run to a file (args and configs)
    _ = util.save_experiment_params(args, configs, opt, infer_dir)


    # Inference
    # for i in range(len(image_paths)):  # to save outputs with the same name as the input image

    image = None
    def inter_repeat_list(lst,batch_size):
        result = []
        for element in lst:
            result.extend([element] * batch_size)
        return result

    elevation = args.elevation if args.elevation is not None else 10.
    print("the elevation is ",elevation)


    # Get plucker embeddings
    fxfycxcy = torch.tensor([opt.fxfy, opt.fxfy, 0.5, 0.5], device=f"cuda:{args.gpu_id}").float()
    elevations = torch.tensor([-elevation] * 4, device=f"cuda:{args.gpu_id}").deg2rad().float()
    azimuths = torch.tensor([0., 90., 180., 270.], device=f"cuda:{args.gpu_id}").deg2rad().float()  # hard-coded
    radius = torch.tensor([args.distance] * 4, device=f"cuda:{args.gpu_id}").float()
    input_C2W = geo_util.orbit_camera(elevations, azimuths, radius, is_degree=False)  # (V_in, 4, 4)
    input_C2W[:, :3, 1:3] *= -1  # OpenGL -> OpenCV
    input_fxfycxcy = fxfycxcy.unsqueeze(0).repeat(input_C2W.shape[0], 1)  # (V_in, 4)
    if opt.input_concat_plucker:
        H = W = opt.input_res
        plucker, _ = geo_util.plucker_ray(H, W, input_C2W.unsqueeze(0), input_fxfycxcy.unsqueeze(0))
        plucker = plucker.squeeze(0)  # (V_in, 6, H, W)
        if opt.view_concat_condition:
            plucker = torch.cat([plucker[0:1, ...], plucker], dim=0)  # (V_in+1, 6, H, W)
    else:
        plucker = None
    IMAGES = []
    alpha_thresh = 0.5
    SAVE_TXT = False
    USE_ANGLE_MASK = True
    WEIGHTED_NORMAL = True
    ADD_ALPHA_CHANNEL = False
    ADD_DEPTH_CHANNEL= False
    rays_cam = generate_ray_directions(H,W,fov).to(device)
    rays_cam = rearrange(rays_cam, 'h w c -> c h w ').float()
    # with transformer.disable_adapter():



    for j in range(len(prompts)):
        

        prompt, negative_prompt = prompts[j], negative_promts[j]

        MAX_NAME_LEN=30
        
        
        with torch.no_grad():
                with torch.autocast("cuda", torch.bfloat16 if args.half_precision else torch.float32):
                    # custom_pipline = False

                        
                    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = pipeline.encode_prompt(
                        prompt=prompt, ## prompt='Posto BMN Petróleo Logo'
                        # prompt=prompt, ## prompt='A yellow and blue spotted toy lizard/crocodile figurine.'
                        # prompt=prompt, ## prompt='A rusty metal plaque with the name 'pierwsza' on it.'
                        do_classifier_free_guidance= True,
                        negative_prompt= "",
                        device=f"cuda:{args.gpu_id}",
                        clean_caption=True,
                    )
            
                    ret_tuple = diffSplat_pas_pipeline_with_score(        
                    pipeline,
                    prompt_embeds=prompt_embeds, #torch.Size([Batch, 77, 768])
                    negative_prompt_embeds=negative_prompt_embeds,#torch.Size([Batch, 77, 768])
                    prompt_attention_mask=prompt_attention_mask,
                    negative_prompt_attention_mask=negative_prompt_attention_mask,
                    num_inference_steps=20,
                    guidance_scale=args.guidance_scale,
                    eta=args.eta,
                    plucker=plucker,# (Batch,V_in, 6, H, W)  # plucker_= repeat(plucker, "v c h w -> (b v) c h w", b=3)
                    init_std=args.init_std, 
                    init_noise_strength=args.init_noise_strength, 
                    init_bg=args.init_bg,
                    num_views=4,
                    output_type="latent",
                    return_unetoutput=False,
                    device=f"cuda:{args.gpu_id}",
                        )
                    out=ret_tuple[0]

                    
                    # fxfycxcy = torch.tensor([1.0*(j+1), 1.0*(j+1), 0.5, 0.5], device=f"cuda:{args.gpu_id}").float()
                    render_fxfycxcy = fxfycxcy.unsqueeze(0).repeat(input_C2W.shape[0], 1)  # (V_in, 4)
                    render_C2w = input_C2W
                    out = out / gsvae.scaling_factor + gsvae.shift_factor
                    render_outputs = gsvae.decode_and_render_gslatents(
                        gsrecon,
                        # out, torch.cat([input_C2W.unsqueeze(0),]*3),torch.cat([ input_fxfycxcy.unsqueeze(0)]*3),
                        # out, torch.cat([input_C2W.unsqueeze(0),]),torch.cat([ input_fxfycxcy.unsqueeze(0)]),
                        out,input_C2W.unsqueeze(0),input_fxfycxcy.unsqueeze(0),
                        C2W=render_C2w.unsqueeze(0), fxfycxcy=input_fxfycxcy.unsqueeze(0),
                        height=args.render_res, width=args.render_res,
                        scaling_modifier=1.0,
                        opacity_threshold=args.opacity_threshold,
                    ) 

                    images = render_outputs["image"].squeeze(0)  # (V_in, 3, H, W)
                    if images.dim() == 5 and not  images.size(0) == 1:
                        images= rearrange(images, "b v c h w -> (b v) c h w",b=3)
                    
                    
                if "normal" in args.reward_fn:
                    V_ = V_in
                    normals =-1*einops.rearrange( render_outputs["raw_normal"],'b v c h w -> (b v) c h w')
                    masks = einops.rearrange(render_outputs["alpha"],'b v c h w -> (b v) c h w')
                    images_normals =(normals +1.0)*0.5
                    rewards,pred_normal = reward_fn(images.float(),normals,render_pkg=render_outputs,masks = masks,) # (reward, reward_metadata)
                    pred_normal = (pred_normal +1.0)*0.5
                    rewards= torch.mean(einops.rearrange(rewards,"(b v) -> b v", v=V_))
                else:
                    prompt_metadata =[{}]
                    V_ = V_in
                    rewards = reward_fn(images.float(),inter_repeat_list([prompt],V_in),inter_repeat_list(prompt_metadata,V_in) )[0] # (reward, reward_metadata)
                    rewards= torch.mean(einops.rearrange(rewards,"(b v) -> b v", v=V_))

                
                name = f"TP{j:02d}_{rewards.item():.4f}"+prompt.replace(".","").replace(" ","_").replace("'","").replace(":","").replace("/","").replace(R"\\","")[:MAX_NAME_LEN]
                IMAGES.append(images)
                images = vis_util.tensor_to_image(rearrange(images, "v c h w -> c h (v w)"))  # (H, V*W, 3)
                
                raw_normal= True
                # raw_normal= False
                normal_keys = ["normal"]
                if raw_normal:
                    normal_keys=["raw_normal","normal"]
                
                if args.save_geometry:
                    
                    for normal_key in normal_keys:
                        normals = render_outputs[normal_key].squeeze(0)  # (V_in, 3, H, W)
                        # normals = render_outputs["normal"].squeeze(0)  # (V_in, 3, H, W)
                        angle_mask = get_angle_mask(einops.repeat(rays_cam,"c h w -> b c h w",b= normals.shape[0]),
                                                -1*normals,
                                                thresh_angle=75.0)
                        normals = (-1*normals+1)*0.5
                        ########
                        if WEIGHTED_NORMAL:
                            BG_COLOR = 1.0
                            alpha_weight = render_outputs["alpha"].squeeze(0)
                            normals = (normals*alpha_weight) + (1-alpha_weight)*BG_COLOR
                        #########
                        normals = vis_util.tensor_to_image(rearrange(normals, "v c h w -> c h (v w)"))  # (H, V*W, 3)
                        images = np.concatenate([images,normals], axis=0)
                    
                    angle_mask =  (render_outputs["alpha"].squeeze(0)>alpha_thresh)*angle_mask
                    normals_angle_masked = (-1*render_outputs[normal_key].squeeze(0)*angle_mask+1)*0.5
                    normals_angle_masked = vis_util.tensor_to_image(rearrange(normals_angle_masked, "v c h w -> c h (v w)"))  # (H, V*W, 3)
                    images = np.concatenate([images,normals_angle_masked], axis=0)
                    
                    if ADD_DEPTH_CHANNEL:
                        # imageio.imwrite(os.path.join(infer_dir, f"{name}_gs_normal.png"), normals
                        depths = render_outputs["raw_depth"].squeeze(0)  # (V_in, 3, H, W)
                        # IMAGES.append(depths)
                        depths = minmax_norm(depths)
                        depths = vis_util.tensor_to_image(rearrange(depths, "v c h w -> c h (v w)"))  # (H, V*W, 3)
                        # imageio.imwrite(os.path.join(infer_dir, f"{name}_gs_depth.png"), depths)
                        images = np.concatenate([images,depths], axis=0)

                    alpha = render_outputs["alpha"].squeeze(0)  # (V_in, 3, H, W)
                    # IMAGES.append(alpha)
                    alpha = vis_util.tensor_to_image(rearrange(alpha, "v c h w -> c h (v w)"))  # (H, V*W, 3)
                    images = np.concatenate([images,alpha],  axis=0)

                    
                    
                    if "normal" in args.reward_fn:
                        pred_normal = vis_util.tensor_to_image(rearrange(pred_normal, "v c h w -> c h (v w)"))
                        images = np.concatenate([images,pred_normal],  axis=0)
                        
                    if ADD_ALPHA_CHANNEL:
                        alpha_channel= np.tile(alpha[...,:1],(images.shape[0]//alpha.shape[0],1,1))
                        # alpha_channel= np.repeat(alpha[...,:1],images.shape[0]//alpha.shape[0],axis=0)
                        images = np.concatenate([images,alpha_channel],  axis=-1)

                    
                    

                
                
                                            # Save Gaussian ply file
                imageio.imwrite(os.path.join(infer_dir, f"{name}_gs.png"), images)
                print(f"Saved image to [{os.path.join(infer_dir, f'{name}_gs.png')}]")
                
                # torch.save(out.detach().cpu(),os.path.join(infer_dir, f'{name}_tensor.pt'))
                if args.save_ply:
                    ply_path = os.path.join(infer_dir, f"{name}.ply")
                    render_outputs["pc"][0].save_ply(ply_path, args.opacity_threshold_ply)

                # Render video
                if args.output_video_type is not None:
                    fancy_video = "fancy" in args.output_video_type
                    save_gif = "gif" in args.output_video_type

                    if fancy_video:
                        render_azimuths = np.arange(0., 360., 10)
                        # render_azimuths = np.arange(0., 720., 4) 
                    else:
                        render_azimuths = np.arange(0., 360., 2)

                    C2W = []
                    for i in range(len(render_azimuths)):
                        c2w = torch.from_numpy(
                            orbit_camera(-elevation, render_azimuths[i], radius=args.distance, opengl=True)
                        ).to(f"cuda:{args.gpu_id}")
                        c2w[:3, 1:3] *= -1  # OpenGL -> OpenCV
                        C2W.append(c2w)
                    C2W = torch.stack(C2W, dim=0)  # (V, 4, 4)
                    W2C= C2W.inverse()
                    fxfycxcy_V = fxfycxcy.unsqueeze(0).repeat(C2W.shape[0], 1)
                    images = []
                    normals = []
                    depths = []
                    alphas = []
                    Mdepth_list = []
                    color_list =[]
                    
                    xyz_list = []
                    world_normal_list =[]
                    
                            
                    if not os.path.exists(os.path.join(infer_dir, f"{name}")):
                        os.makedirs(os.path.join(infer_dir, f"{name}"))
                    for v in tqdm(range(C2W.shape[0]), desc="Rendering", ncols=125):
                        render_outputs = gsvae.decode_and_render_gslatents(
                            gsrecon,
                            out,  # (V_in, 4, H', W')
                            input_C2W.unsqueeze(0),  # (1, V_in, 4, 4)
                            input_fxfycxcy.unsqueeze(0),  # (1, V_in, 4)
                            C2W[v].unsqueeze(0).unsqueeze(0),  # (B=1, V=1, 4, 4)
                            fxfycxcy_V[v].unsqueeze(0).unsqueeze(0),  # (B=1, V=1, 4)
                            height=args.render_res, width=args.render_res,
                            scaling_modifier= 1.,
                            opacity_threshold=args.opacity_threshold,
                        )
                        image = render_outputs["image"].squeeze(0).squeeze(0)  # (3, H, W)
                        image = vis_util.tensor_to_image(image.unsqueeze(0), return_pil=save_gif)  # (H, V*W, 3)
                        if args.save_geometry:
                            for normal_key in normal_keys:
                                normal_ = render_outputs[normal_key].squeeze(0).squeeze(0)
                                normal_ = (-1*normal_+1)*0.5
                                if WEIGHTED_NORMAL:
                                    alpha_weight = render_outputs["alpha"].squeeze(0).squeeze(0)
                                    normal_ = (normal_*alpha_weight) + (1-alpha_weight)*BG_COLOR
                                # IMAGES.append(normals)
                                # normals.append(vis_util.tensor_to_image(normal_ , return_pil=save_gif ) )# (H, V*W, 3)
                                normal = vis_util.tensor_to_image(normal_.unsqueeze(0) , return_pil=save_gif )# (H, V*W, 3)
                                # image=torch.cat([image,normal_], dim=1)
                                image = np.concatenate([image,normal], axis=1)
                            
                            
                            
                        
                            depth_ = render_outputs["raw_depth"].squeeze(0).squeeze(0)
                            depth_ = minmax_norm(depth_)
                            # depths.append(vis_util.tensor_to_image(depth_, return_pil=save_gif) ) # (H, V*W, 3) 
                            depth_=vis_util.tensor_to_image(depth_.unsqueeze(0), return_pil=save_gif)  # (H, V*W, 3) 
                            
                            image=np.concatenate([image,depth_], axis=1)


                            


                        
                        images.append(image)
                        image=None
                            
                    if save_gif:
                        images[0].save(
                            os.path.join(infer_dir, f"{name}.gif"),
                            save_all=True,
                            append_images=images[1:],
                            optimize=False,
                            duration=1000 // 30,
                            loop=0,
                        )
                    else:  # save mp4
                        # for idx, sub_image in enumerate(images):
                        #     imageio.imwrite(os.path.join(infer_dir, f"{name}",f"v{idx:03d}.png"), sub_image)
                        
                        images = np.stack(images, axis=0)  # (V, H, W, 3)
                        imageio.mimwrite(os.path.join(infer_dir, f"{name}.mp4"), images, fps=10)


                        
                        
                        



    logger.info(f"Done! Saved outputs to [{infer_dir}]\n")



if __name__ == "__main__":
    main()