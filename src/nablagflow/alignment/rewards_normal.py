import sys
import os
import torch
import numpy as np
print("current path",os.getcwd())
sys.path.append(os.path.join(os.getcwd(),"../../../"))
sys.path.insert(0, os.getcwd())
from src.utils.geo_util import generate_ray_directions,depth_double_to_normal
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange,repeat
from torchvision.transforms.functional  import to_pil_image
import contextlib

import logging
logger = logging.getLogger('root')


def get_angle_mask(rays_cam,normals,thresh_angle=60):
    
    cosine_angle = (rays_cam*normals).sum(dim=-3,keepdim=True) ## (B,3,H,W) * (B,3,H,W) = (B,1,H,W)

    mask = torch.abs(cosine_angle)> torch.cos(torch.deg2rad(torch.Tensor([thresh_angle]))).item()
    return mask
def normal_score(normals,normals_target,std=1):
    """_summary_

    Args:
        normals (Torch.Tensor): Batch of rendering normals, shape (B*V, 3, H, W)
        normals_target (Torch.Tensor): Batch of target normals, shape (B*V, 3, H, W)

    Returns:
        Score: A scalar score for the batch of images, shape (B* V)
    """
    diff_norm = torch.square(normals - normals_target)

    
    return -torch.sum(diff_norm, dim=[1,2, 3,])
def normal_score_cosine(normals,normals_target,std=1):
    """_summary_

    Args:
        normals (Torch.Tensor): Batch of rendering normals, shape (B*V, 3, H, W)
        normals_target (Torch.Tensor): Batch of target normals, shape (B*V, 3, H, W)

    Returns:
        Score: A scalar score for the batch of images, shape (B* V)
    """
    diff_norm = normals * normals_target

    
    return torch.sum(diff_norm, dim=[1,2, 3,])

def normal_d2n_cons(dtype=torch.float32, device="cuda", distributed=True,fov=60.0, H=256, W=256,
                        apply_erosion=False,
                        apply_angle_mask=False,
                        context_manager=torch.inference_mode):
    """_summary_

    Args:
        dtype (_type_, optional): _description_. Defaults to torch.float32.
        device (str, optional): _description_. Defaults to "cuda".
        distributed (bool, optional): _description_. Defaults to True.
        fov (float, optional): _description_. Defaults to 60.0.
        H (int, optional): _description_. Defaults to 256.
        W (int, optional): _description_. Defaults to 256.
        apply_erosion (bool, optional): _description_. Defaults to False.
        apply_angle_mask (bool, optional): _description_. Defaults to False.
        context_manager (_type_, optional): _description_. Defaults to None. Used to stop the gradient of the model. ||normal - sg(f(D(z)))||^2

    Returns:
        _type_: _description_
    """


    H, W = H, W

    if apply_angle_mask:
        ##  mask
        rays_cam = generate_ray_directions(H,W,fov).to(device)
        rays_cam = rearrange(rays_cam, 'h w c -> c h w ')
    else:
        rays_cam = None
        # repeat(cam_dir, "1 c h w ->")
        
        # pass
    FLIP=-1
    fov = torch.deg2rad(torch.Tensor([fov])).item()
    # @torch.inference_mode()
    def _fn(images,normals, render_pkg=None,masks=None):
        """_summary_

        Args:
            images (_type_): _description_ Batch of images, shape (B*V, C, H, W)
            normals (_type_): _description_ Batch of target normals, shape (B*V, 3, H, W)
            prompts (_type_): _description_
            metadata (_type_): _description_

        Returns:
            _type_: _description_
        """
        assert render_pkg is not None, "render_pkg is None"
        if masks is not None:
            masks = masks>0.5
        else:
            masks = torch.ones_like(normals[:,:1,...])

        if masks is not None:
            if apply_angle_mask:
                # angle_mask = get_angle_mask(repeat(rays_cam,"c h w -> b c h w",b= normals.shape[0]),
                #                             refer_normals,
                #                             thresh_angle=60.0)
                angle_mask = get_angle_mask(repeat(rays_cam,"c h w -> b c h w",b= normals.shape[0]),
                                            normals,
                                            thresh_angle=60.0)
                masks = torch.logical_and(masks,angle_mask)
            # refer_normals= refer_normals*masks
            # normals = normals * masks
        
        rendered_expected_depth: torch.Tensor = rearrange( render_pkg["raw_depth"],"b v c h w -> (b v) c h w")
        rendered_median_depth: torch.Tensor =rearrange( render_pkg["median_depth"],"b v c h w -> (b v) c h w")
        rendered_normal: torch.Tensor = rearrange( render_pkg["raw_normal"],"b v c h w -> (b v) c h w")
        
        if context_manager== torch.inference_mode:
            rendered_expected_depth = rendered_expected_depth.detach()
            rendered_median_depth = rendered_median_depth.detach()
        elif context_manager== contextlib.nullcontext:
            pass
            
        else:
            raise Exception("context_manager should be torch.inference_mode or contextlib.nullcontext")
        depth_middepth_normal = depth_double_to_normal(H,W,fov, rendered_expected_depth, rendered_median_depth)   
         
        ##  depth_middepth_normal (B,N,3,H,W)
        depth_ratio = 0.0
        # depth_ratio = 0.6
        normal_score =  (rendered_normal.unsqueeze(1) * depth_middepth_normal).sum(dim=2)
        normal_score = normal_score*masks
        depth_normal_score = (1-depth_ratio) * normal_score[:,0,...] + depth_ratio * normal_score[:,1,...]
        depth_normal_score = depth_normal_score.sum([1,2]) / masks.sum([1,2,3])
        # import pdb
        # pdb.set_trace()
        # scores = normal_score_cosine(normals,refer_normals)
        # scores=scores/torch.sum(masks,dim=[1,2,3])        
         
        return depth_normal_score, rearrange(FLIP*depth_middepth_normal.detach(),"b n c h w -> b c (n h) w")

    return _fn




def normal_yoso(dtype=torch.float32, device="cuda", distributed=True,fov=60.0, 
                        H=256, W=256,
                        apply_erosion=False,
                        apply_angle_mask=False,
                        context_manager=None):
    import stablenormal
    from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline
    from stablenormal.pipeline_stablenormal import StableNormalPipeline
    from stablenormal.scheduler.heuristics_ddimsampler import HEURI_DDIMScheduler
    
    logger.warning(f"The Dtype  {dtype} is not used in the normal_yoso function! StableNormal Data type are fixed to float16 by according to its official implementation")

    if context_manager is None:
        context_manager = contextlib.nullcontext
    x_start_pipeline = YOSONormalsPipeline.from_pretrained(
    'Stable-X/yoso-normal-v1-7', trust_remote_code=True, variant="fp16", torch_dtype=torch.float16).to(device)
    
    
    pipe = x_start_pipeline
    #####################
    dtype =torch.float32
    # dtype =torch.float32
    pipe = pipe.to(dtype)
    
    pipe.unet.to(torch.float16)
    pipe.controlnet.to(torch.float16)
    # pipe.vae = pipe.vae.to(dtype)
    # pipe.unet = pipe.unet.to(dtype)
    #################################
    if apply_angle_mask:
        rays_cam = generate_ray_directions(H,W,fov).to(device)
        rays_cam = rearrange(rays_cam, 'h w c -> c h w ')
    else:
        rays_cam = None
    # @torch.inference_mode()
    pipe.image_processor.config.do_range_check=False
    def _fn(images,normals,render_pkg=None,masks=None):
        """_summary_

        Args:
            images (_type_): _description_ Batch of images, shape (B*V, C, H, W)
            normals (_type_): _description_ Batch of target normals, shape (B*V, 3, H, W)
            prompts (_type_): _description_
            metadata (_type_): _description_

        Returns:
            _type_: _description_
        """
        if isinstance(images, torch.Tensor):
            if images.dim() == 5:
                B, V, C, _H, _W = images.shape
                images = rearrange(images, 'b v c h w -> (b v) c h w')
            pass # assume float tensor in [0, 1]
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
            

        # images = F.pad(images, lrtb, mode="constant", value=0.0)
        # images = normalize(images)
        if masks is not None:
            masks = masks>0.5
            normals = normals * masks
            

        
        
        
        
        with context_manager(): 
            images = (images*2.0-1.0).to(dtype) ## Hard code to float16 for YOSO model, Since  skip_preprocess=True, we should normalize the inputs.
            ########################
            out = pipe(images, ### FIXME:
                    match_input_resolution=True,
                    # match_input_resolution=False,
                    processing_resolution=max(H,W),
                    batch_size=images.shape[0],
                    output_type = "pt",
                    # skip_preprocess=True, ### 
                    #####################
                    vae_dtype=torch.float32,
                    unet_dtype=torch.float16,
                    skip_preprocess=True,
                    skip_postprocess=True,
                    #################
                    
            )
            refer_normals = out.prediction

    
        if masks is not None:
            # refer_normals= refer_normals*masks
            if apply_angle_mask:
                angle_mask = get_angle_mask(repeat(rays_cam,"c h w -> b c h w",b= normals.shape[0]),
                                            refer_normals,
                                            thresh_angle=60.0)
                # angle_mask = get_angle_mask(repeat(rays_cam,"c h w -> b c h w",b= normals.shape[0]),
                #                             normals,
                #                             thresh_angle=60.0)
                masks = torch.logical_and(masks,angle_mask)
            refer_normals= refer_normals*masks
            normals = normals * masks
        
        
        scores = normal_score_cosine(normals,refer_normals)
        scores=scores/torch.sum(masks,dim=[1,2,3])        
        
        return scores, refer_normals

    return _fn

