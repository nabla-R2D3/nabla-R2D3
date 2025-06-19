

try:
    from diffusers.utils import randn_tensor
except ImportError:
    from diffusers.utils.torch_utils import randn_tensor
from .step_with_score import step_with_score
from ..utils import image_postprocess






from extensions.diffusers_diffsplat import  StableMVDiffusionPipeline


from PIL.Image import Image as PILImage
from torch import Tensor

import PIL.Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import *

from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import *
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import *

# Copied from https://github.com/camenduru/GRM/blob/master/third_party/generative_models/instant3d.py
def build_gaussians(H: int, W: int, std: float, bg: float = 0.) -> Tensor:
    assert H == W  # TODO: support non-square latents

    x_vals = torch.arange(W)
    y_vals = torch.arange(H)
    x_vals, y_vals = torch.meshgrid(x_vals, y_vals, indexing="ij")
    x_vals = x_vals.unsqueeze(0).unsqueeze(0)
    y_vals = y_vals.unsqueeze(0).unsqueeze(0)
    center_x, center_y = W//2., H//2.

    gaussian = torch.exp(-((x_vals - center_x) ** 2 + (y_vals - center_y) ** 2) / (2 * (std * H) ** 2))  # cf. Instant3D A.5
    gaussian = gaussian / gaussian.max()
    gaussian = (gaussian + bg).clamp(0., 1.)  # gray background for `bg` > 0.
    gaussian = gaussian.repeat(1, 3, 1, 1)
    gaussian = 1. - gaussian    # (1, 3, H, W) in [0, 1]

    gaussian = torch.cat([gaussian, gaussian], dim=-1)
    gaussian = torch.cat([gaussian, gaussian], dim=-2)  # (1, 3, 2H, 2W)
    gaussians = F.interpolate(gaussian, (H, W), mode="bilinear", align_corners=False)
    gaussians = gaussians * 2. - 1.  # (1, 3, H, W) in [-1, 1]
    return gaussians


# Copied from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion
def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]



@torch.no_grad()
def diffSplat_pas_pipeline_with_score(
    self:PixArtSigmaPipeline,
    num_views: int = 4,###

    image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor] = None,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0, 
    strength: float = 1.0,

    plucker: Optional[torch.FloatTensor] = None,
    triangle_cfg_scaling: bool = False,
    min_guidance_scale: float = 1.0,
    max_guidance_scale: float = 3.0,
    init_std: Optional[float] = 0.,
    init_noise_strength: Optional[float] = 1.,
    init_bg: Optional[float] = 0.,

    batch_size = None, dtype=None,
    device = None,
    calculate_pb = False, logp_mean = True,
    return_unetoutput = False,




    timesteps: List[int] = None,
    sigmas: List[float] = None,
    
    prompt_attention_mask: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    
    callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    callback_steps: int = 1,
    clean_caption: bool = True,
    use_resolution_binning: bool = False,  # `True` for original PixArt
    max_sequence_length: int = 300,

    
    **kwargs,
):
    # 1. Check inputs. Raise error if not correct
    height = height or self.transformer.config.sample_size * self.vae_scale_factor
    width = width or self.transformer.config.sample_size * self.vae_scale_factor
    if use_resolution_binning:
        if self.transformer.config.sample_size == 256:
            aspect_ratio_bin = ASPECT_RATIO_2048_BIN
        elif self.transformer.config.sample_size == 128:
            aspect_ratio_bin = ASPECT_RATIO_1024_BIN
        elif self.transformer.config.sample_size == 64:
            aspect_ratio_bin = ASPECT_RATIO_512_BIN
        elif self.transformer.config.sample_size == 32:
            aspect_ratio_bin = ASPECT_RATIO_256_BIN
        else:
            raise ValueError("Invalid sample size")
        orig_height, orig_width = height, width
        height, width = self.image_processor.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)

    # to deal with lora scaling and other possible forward hooks

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_steps,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
    )


    self._interrupt = False

    V_cond = 0
    if image is not None:
        assert image.ndim == 5  # (B, V_cond, 3, H, W)
        V_cond = image.shape[1]
    cross_attention_kwargs = {"num_views": num_views + (V_cond if self.transformer.config.view_concat_condition else 0)}

    # 2. Define call parameters
    if batch_size is None:
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
    if device is None:
        device = self._execution_device


    do_classifier_free_guidance = (guidance_scale if not triangle_cfg_scaling else max_guidance_scale) > 1.0

    # 3. Encode input prompt
    
    
    if prompt_embeds is not None:
        # lora_scale = (
        # self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        # )
        pass 
        # prompt_embeds, prompt_attention_mask, negative_prompt_embeds,negative_prompt_attention_mask = self.encode_prompt(
        #     prompt,
        #     do_classifier_free_guidance,
        #     negative_prompt=negative_prompt,
        #     num_images_per_prompt=num_images_per_prompt,
        #     device=device,
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=negative_prompt_embeds,
        #     prompt_attention_mask=prompt_attention_mask,
        #     negative_prompt_attention_mask=negative_prompt_attention_mask,
        #     clean_caption=clean_caption,
        #     max_sequence_length=max_sequence_length,
        # )
    prompt_embeds = repeat(prompt_embeds, "b n d -> (b v) n d", v=num_views)
    prompt_attention_mask = repeat(prompt_attention_mask, "b n -> (b v) n", v=num_views)
    
    if do_classifier_free_guidance:        
        negative_prompt_embeds = repeat(negative_prompt_embeds, "b n d -> (b v) n d", v=num_views)
        negative_prompt_attention_mask = repeat(negative_prompt_attention_mask, "b n -> (b v) n", v=num_views )

    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        


    # 3.1 Prepare input image latents
    if self.transformer.config.view_concat_condition:
        pass 
        print("TODO: Implement this")
        raise NotImplementedError
    
    
    if self.transformer.config.view_concat_condition:
        if image is not None:
            image_latents = self.prepare_image_latents(image, device, num_images_per_prompt, do_classifier_free_guidance)
        else:
            image_latents = torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.transformer.config.out_channels // 2,  # `num_channels_latents`; self.transformer.config.in_channels
                    int(height) // self.vae_scale_factor,
                    int(width) // self.vae_scale_factor,
                ),
                dtype=prompt_embeds.dtype,
                device=device,
            )
            if V_cond > 0:
                image_latents = image_latents.unsqueeze(1).repeat(1, V_cond, 1, 1, 1)
            if do_classifier_free_guidance:
                image_latents = torch.cat([image_latents] * 2, dim=0)

    # 3.2 Prepare Plucker embeddings  
    # v,c,h,w = plucker.shape
    # plucker = plucker.expand(batch_size,v,c,h,w) 
    if plucker is not None:
        if plucker.dim() == 5:
            plucker = rearrange(plucker, "b v c h w -> (b v) c h w")
        assert plucker.shape[0] == batch_size * (num_views + (V_cond if self.transformer.config.view_concat_condition else 0))
        plucker = self.prepare_plucker(plucker, num_images_per_prompt, do_classifier_free_guidance)

    # 4. Prepare timesteps
    if  num_inference_steps is None:
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
    else:
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
    # 5. Prepare latent variables
    latent_channels = self.transformer.config.out_channels // 2  # self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt * num_views,
        latent_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    # 5.1 Gaussian blobs initialization; cf. Instant3D
    if init_std > 0. and init_noise_strength < 1.:
        row = int(num_views**0.5)
        col = num_views - row
        init_image = build_gaussians(row * height, col * width, init_std, init_bg).to(device=device, dtype=latents.dtype)
        init_image = rearrange(init_image, "b d (r h) (c w) -> (b r c) d h w", r=row, c=col)
        timesteps, num_inference_steps = self.get_timesteps_img2img(num_inference_steps, init_noise_strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        latents = self.prepare_latents_img2img(
            init_image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
        )
        
    

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 6.1 Prepare micro-conditions.
    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}


    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)


    # 5.2 Prepare guidance scale
    if triangle_cfg_scaling:
        # Triangle CFG scaling; the first view is input condition
        guidance_scale = torch.cat([
            torch.linspace(min_guidance_scale, max_guidance_scale, num_views//2 + 1).unsqueeze(0),
            torch.linspace(max_guidance_scale, min_guidance_scale, num_views - (num_views//2 + 1) + 2)[1:-1].unsqueeze(0)
        ], dim=-1)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_images_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.unsqueeze(1).ndim)  # (B, V, 1, 1, 1)
        guidance_scale = rearrange(guidance_scale, "b v c h w -> (b v) c h w")

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    
    self._num_timesteps = len(timesteps)
    
    B_,V_ = batch_size,num_views
    all_latents = [rearrange(latents,"(b v) c h w -> b (v c) h w",b=B_,v = V_)] 
    all_log_probs = []
    all_scores = []
    all_log_pbs = []
    unet_outputs = []
    
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # if self.interrupt:
            #     continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            #latent_model_inputtorch.Size([32, 4, 32, 32])
            # Concatenate input latents with others
            latent_model_input = rearrange(latent_model_input, "(b v) c h w -> b v c h w", v=num_views)
            if self.transformer.config.view_concat_condition:
                latent_model_input = torch.cat([image_latents, latent_model_input], dim=1)  # (B, V_in+V_cond, 4, H', W')
            if self.transformer.config.input_concat_plucker:
                    plucker = F.interpolate(plucker, size=latent_model_input.shape[-2:], mode="bilinear", align_corners=False)
                    plucker = rearrange(plucker, "(b v) c h w -> b v c h w", v=num_views + (V_cond if self.transformer.config.view_concat_condition else 0))
                    latent_model_input = torch.cat([latent_model_input, plucker], dim=2)  # (B, V_in(+V_cond), 4+6, H', W')
                    plucker = rearrange(plucker, "b v c h w -> (b v) c h w")

            if self.transformer.config.input_concat_binary_mask:
                if self.transformer.config.view_concat_condition:
                    latent_model_input = torch.cat([
                        torch.cat([latent_model_input[:, :V_cond, ...], torch.zeros_like(latent_model_input[:, :V_cond, 0:1, ...])], dim=2),
                        torch.cat([latent_model_input[:, V_cond:, ...], torch.ones_like(latent_model_input[:, V_cond:, 0:1, ...])], dim=2),
                    ], dim=1)  # (B, V_in+V_cond, 4+6+1, H', W')
                else:
                    latent_model_input = torch.cat([
                        torch.cat([latent_model_input, torch.ones_like(latent_model_input[:, :, 0:1, ...])], dim=2),
                    ], dim=1)  # (B, V_in, 4+6+1, H', W')
            latent_model_input = rearrange(latent_model_input, "b v c h w -> (b v) c h w")
            
            ## here: pluker torch.Size([32, 6, 32, 32])
            # predict the noise residual
            current_timestep = t
            if not torch.is_tensor(current_timestep):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = latent_model_input.device.type == "mps"
                if isinstance(current_timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
            elif len(current_timestep.shape) == 0:
                current_timestep = current_timestep[None].to(latent_model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            current_timestep = current_timestep.expand(latent_model_input.shape[0])

            # predict noise model_output
            noise_pred = self.transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=current_timestep,
                added_cond_kwargs=added_cond_kwargs,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # Only keep the noise prediction for the latents
            if self.transformer.config.view_concat_condition:
                noise_pred = rearrange(noise_pred, "(b v) c h w -> b v c h w", v=num_views+V_cond)
                noise_pred = rearrange(noise_pred[:, V_cond:, ...], "b v c h w -> (b v) c h w")
            

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            if return_unetoutput:
                unet_outputs.append(rearrange(noise_pred.detach(),"(b v) c h w -> b (v c) h w",b=B_,v = V_))
                # unet_outputs.append(noise_pred.detach())
                
            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            else:
                noise_pred = noise_pred

            # compute previous image: x_t -> x_t-1
            
            # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            
                        # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)
            

            # compute the previous noisy sample x_t -> x_t-1
            # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]


            # compute the previous noisy sample x_t -> x_t-1
            prev_timestep = timesteps[i + 1] if i < num_inference_steps-1 else None
            if calculate_pb:
                raise NotImplementedError

            else: #input =latents torch.Size([16, 4, 32, 32])
                latents, score, log_prob = step_with_score(
                    self.scheduler, noise_pred, t, latents,
                    prev_timestep=prev_timestep, #
                    strength=strength,
                    step_index=i,
                    **extra_step_kwargs
                )
            
            all_latents.append(rearrange(latents,"(b v) c h w -> b (v c) h w",b=B_,v = V_))
            # all_latents.append(latents)
            all_log_probs.append(log_prob)
            all_scores.append(score)
            






    if not output_type == "latent": ### TODO:fix  this if needed    
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
            0
        ]
        
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
    
    if prompt_embeds is not None:
        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )
        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()
    else:
        # image = (image / 2 + 0.5).clamp(0, 1)
        image = image_postprocess(image)
    
    self.maybe_free_model_hooks() ### TODO:fix  this if needed
    
    assert not (calculate_pb and return_unetoutput), "Cannot return both log_pb and unet_outputs"
    if calculate_pb:
        return image, has_nsfw_concept, all_latents, all_log_probs, all_scores, all_log_pbs
    if return_unetoutput:
        return image, has_nsfw_concept, all_latents, all_log_probs, all_scores, unet_outputs

    return image, has_nsfw_concept, all_latents, all_log_probs, all_scores

    # if not return_dict:
    #     return (image, has_nsfw_concept)

    # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
