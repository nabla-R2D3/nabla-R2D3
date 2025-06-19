from PIL.Image import Image as PILImage
from torch import Tensor

import PIL.Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import *
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import *

from extensions.diffusers_diffsplat import UNetMV2DConditionModel


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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline
class StableMVDiffusionPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNetMV2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        feature_extractor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModelWithProjection = None,
        safety_checker: StableDiffusionSafetyChecker = None,
        requires_safety_checker: bool = False,
    ):
        new_config = dict(unet.config)  # save the original unet config

        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            image_encoder,
            requires_safety_checker,
        )

        # Not automatically change unet config in `__init__` (for `unet.config.sample_size`)
        self.unet._internal_dict = FrozenDict(new_config)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps_img2img(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.prepare_latents
    def prepare_latents_img2img(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                if image.shape[0] < batch_size and batch_size % image.shape[0] == 0:
                    image = torch.cat([image] * (batch_size // image.shape[0]), dim=0)
                elif image.shape[0] < batch_size and batch_size % image.shape[0] != 0:
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {image.shape[0]} to effective batch_size {batch_size} "
                    )

                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents

    def prepare_image_latents(self, image, device, num_images_per_prompt, do_classifier_free_guidance):
        dtype = next(self.vae.parameters()).dtype

        assert isinstance(image, Tensor)
        assert image.ndim == 5 and image.shape[2] == 3

        V_cond = image.shape[1]
        image = rearrange(image, "b v c h w -> (b v) c h w")

        # VAE latent
        image = image.to(device).to(dtype)  # not resize like CLIP preprocessing
        image = image * 2. - 1.
        image_latents = self.vae.encode(image).latent_dist.mode() * self.vae.config.scaling_factor

        image_latents = rearrange(image_latents, "(b v) c h w -> b v c h w", v=V_cond)

        # duplicate image latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.unsqueeze(1)
        bs_latent, _, v, c, h, w = image_latents.shape
        image_latents = image_latents.repeat(1, num_images_per_prompt, 1, 1, 1, 1)
        image_latents = image_latents.view(bs_latent * num_images_per_prompt, v, c, h, w)

        if do_classifier_free_guidance:
            negative_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_latents, image_latents])

        return image_latents

    def prepare_plucker(self, plucker, num_images_per_prompt, do_classifier_free_guidance):
        plucker = plucker.to(dtype=self.unet.dtype, device=self.unet.device)

        # duplicate plucker embeddings for each generation per prompt, using mps friendly method
        plucker = plucker.unsqueeze(1)
        bs, _, c, h, w = plucker.shape
        plucker = plucker.repeat(1, num_images_per_prompt, 1, 1, 1)
        plucker = plucker.view(bs * num_images_per_prompt, c, h, w)

        if do_classifier_free_guidance:
            plucker = torch.cat([plucker]*2, dim=0)

        return plucker

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    # Refine for triangle cfg scaling
    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
        return self.guidance_scale.max() > 1 and self.unet.config.time_cond_proj_dim is None

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor] = None,
        prompt: Union[str, List[str]] = None,

        num_views: int = 4,
        plucker: Optional[torch.FloatTensor] = None,
        triangle_cfg_scaling: bool = False,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        init_std: Optional[float] = 0.,
        init_noise_strength: Optional[float] = 1.,
        init_bg: Optional[float] = 0.,

        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        if not height or not width:
            height = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[0]
            )
            width = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[1]
            )
            height, width = height * self.vae_scale_factor, width * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale if not triangle_cfg_scaling else max_guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        self._interrupt = False

        V_cond = 0
        if image is not None:
            assert image.ndim == 5  # (B, V_cond, 3, H, W)
            V_cond = image.shape[1]
        self.cross_attention_kwargs.update(num_views=num_views + (V_cond if self.unet.config.view_concat_condition else 0))

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        prompt_embeds = repeat(prompt_embeds, "b n d -> (b v) n d", v=num_views + (V_cond if self.unet.config.view_concat_condition else 0))
        if self.do_classifier_free_guidance:
            negative_prompt_embeds = repeat(negative_prompt_embeds, "b n d -> (b v) n d", v=num_views + (V_cond if self.unet.config.view_concat_condition else 0))

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
            image_embeds = repeat(image_embeds, "b n d -> (b v) n d", v=num_views)

        # 3.1 Prepare input image latents
        if self.unet.config.view_concat_condition:
            if image is not None:
                image_latents = self.prepare_image_latents(image, device, num_images_per_prompt, self.do_classifier_free_guidance)
            else:
                image_latents = torch.zeros(
                    (
                        batch_size * num_images_per_prompt,
                        self.unet.config.out_channels,  # `num_channels_latents`; self.unet.config.in_channels
                        int(height) // self.vae_scale_factor,
                        int(width) // self.vae_scale_factor,
                    ),
                    dtype=prompt_embeds.dtype,
                    device=device,
                )
                if V_cond > 0:
                    image_latents = image_latents.unsqueeze(1).repeat(1, V_cond, 1, 1, 1)
                if self.do_classifier_free_guidance:
                    image_latents = torch.cat([image_latents] * 2, dim=0)

        # 3.2 Prepare Plucker embeddings
        if plucker is not None:
            if plucker.dim() == 5:
                plucker = rearrange(plucker, "b v c h w -> (b v) c h w")
            assert plucker.shape[0] == batch_size * (num_views + (V_cond if self.unet.config.view_concat_condition else 0))
            plucker = self.prepare_plucker(plucker, num_images_per_prompt, self.do_classifier_free_guidance)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.out_channels  # self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt * num_views,
            num_channels_latents,
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

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 6.3 Prepare guidance scale
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

            self._guidance_scale = guidance_scale

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Concatenate input latents with others
                latent_model_input = rearrange(latent_model_input, "(b v) c h w -> b v c h w", v=num_views)
                if self.unet.config.view_concat_condition:
                    latent_model_input = torch.cat([image_latents, latent_model_input], dim=1)  # (B, V_in+V_cond, 4, H', W')
                if self.unet.config.input_concat_plucker:
                        plucker = F.interpolate(plucker, size=latent_model_input.shape[-2:], mode="bilinear", align_corners=False)
                        plucker = rearrange(plucker, "(b v) c h w -> b v c h w", v=num_views + (V_cond if self.unet.config.view_concat_condition else 0))
                        latent_model_input = torch.cat([latent_model_input, plucker], dim=2)  # (B, V_in(+V_cond), 4+6, H', W')
                        plucker = rearrange(plucker, "b v c h w -> (b v) c h w")
                if self.unet.config.input_concat_binary_mask:
                    if self.unet.config.view_concat_condition:
                        latent_model_input = torch.cat([
                            torch.cat([latent_model_input[:, :V_cond, ...], torch.zeros_like(latent_model_input[:, :V_cond, 0:1, ...])], dim=2),
                            torch.cat([latent_model_input[:, V_cond:, ...], torch.ones_like(latent_model_input[:, V_cond:, 0:1, ...])], dim=2),
                        ], dim=1)  # (B, V_in+V_cond, 4+6+1, H', W')
                    else:
                        latent_model_input = torch.cat([
                            torch.cat([latent_model_input, torch.ones_like(latent_model_input[:, :, 0:1, ...])], dim=2),
                        ], dim=1)  # (B, V_in, 4+6+1, H', W')
                latent_model_input = rearrange(latent_model_input, "b v c h w -> (b v) c h w")

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # Only keep the noise prediction for the latents
                if self.unet.config.view_concat_condition:
                    noise_pred = rearrange(noise_pred, "(b v) c h w -> b v c h w", v=num_views+V_cond)
                    noise_pred = rearrange(noise_pred[:, V_cond:, ...], "b v c h w -> (b v) c h w")

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
                ## - add callback_on_step_end_tensor_inputs
                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
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

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
