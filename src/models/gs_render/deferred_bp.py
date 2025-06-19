import torch
from torch.utils.checkpoint import _get_autocast_kwargs

from src.models.gs_render.gs_util import render


class DeferredBP(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
        xyz, rgb, scale, rotation, opacity,
        height, width, C2W, fxfycxcy,
        patch_size, gaussian_model,
        znear, zfar,
        bg_color,
        scaling_modifier,
        render_dn,
    ):
        """ Forward rendering. """
        assert (xyz.dim() == 3) and (rgb.dim() == 3) and (scale.dim() == 3) and (rotation.dim() == 3)
        assert height  % patch_size == 0 and width % patch_size == 0

        ctx.save_for_backward(xyz, rgb, scale, rotation, opacity)  # save tensors for backward
        ctx.height = height
        ctx.width = width
        ctx.C2W = C2W
        ctx.fxfycxcy = fxfycxcy
        ctx.patch_size = patch_size
        ctx.gaussian_model = gaussian_model
        ctx.znear = znear
        ctx.zfar = zfar
        ctx.bg_color = bg_color
        ctx.scaling_modifier = scaling_modifier
        ctx.render_dn = render_dn

        ctx.gpu_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs()
        ctx.manual_seeds = []

        with (
            torch.no_grad(),
            torch.autocast("cuda", **ctx.gpu_autocast_kwargs),
            torch.autocast("cpu", **ctx.cpu_autocast_kwargs),
        ):
            device, (B, V) = C2W.device, C2W.shape[:2]

            images = torch.zeros(B, V, 3, height, width, device=device)
            alphas = torch.zeros(B, V, 1, height, width, device=device)
            depths = torch.zeros(B, V, 1, height, width, device=device)
            normals = torch.zeros(B, V, 3, height, width, device=device)

            for i in range(B):
                ctx.manual_seeds.append([])
                pc = ctx.gaussian_model.set_data(xyz[i], rgb[i], scale[i], rotation[i], opacity[i])

                for j in range(V):
                    fxfycxcy_ij = fxfycxcy[i, j]
                    fx, fy, cx, cy = fxfycxcy_ij[0], fxfycxcy_ij[1], fxfycxcy_ij[2], fxfycxcy_ij[3]

                    for m in range(0, ctx.width//ctx.patch_size):
                        for n in range(0, ctx.height //ctx.patch_size):
                            seed = torch.randint(0, 2**32, (1,)).long().item()
                            ctx.manual_seeds[-1].append(seed)

                            # Transform intrinsics
                            center_x = (m*ctx.patch_size + ctx.patch_size//2) / ctx.width
                            center_y = (n*ctx.patch_size + ctx.patch_size//2) / ctx.height

                            scale_x = ctx.width // ctx.patch_size 
                            scale_y = ctx.height // ctx.patch_size
                            trans_x = 0.5 - scale_x * center_x 
                            trans_y = 0.5 - scale_y * center_y 

                            new_fx = scale_x * fx 
                            new_fy = scale_y * fy
                            new_cx = scale_x * cx + trans_x
                            new_cy = scale_y * cy + trans_y
                            
                            new_fxfycxcy = torch.stack([new_fx, new_fy, new_cx, new_cy], dim=0) 

                            render_results = render(pc, patch_size, patch_size, C2W[i, j], new_fxfycxcy, znear, zfar, bg_color, scaling_modifier, render_dn)
                            images[i, j, :, n*ctx.patch_size:(n+1)*ctx.patch_size, m*ctx.patch_size:(m+1)*ctx.patch_size] = render_results["image"]
                            alphas[i, j, :, n*ctx.patch_size:(n+1)*ctx.patch_size, m*ctx.patch_size:(m+1)*ctx.patch_size] = render_results["alpha"]
                            depths[i, j, :, n*ctx.patch_size:(n+1)*ctx.patch_size, m*ctx.patch_size:(m+1)*ctx.patch_size] = render_results["depth"]
                            normals[i, j, :, n*ctx.patch_size:(n+1)*ctx.patch_size, m*ctx.patch_size:(m+1)*ctx.patch_size] = render_results["normal"]

        return images, alphas, depths, normals

    @staticmethod
    def backward(ctx, grad_images, grad_alphas, grad_depths, grad_normals):
        """ Backward process. """
        xyz, rgb, scale, rotation, opacity = ctx.saved_tensors

        xyz_nosync = xyz.detach().clone()
        xyz_nosync.requires_grad = True
        xyz_nosync.grad = None

        rgb_nosync = rgb.detach().clone()
        rgb_nosync.requires_grad = True
        rgb_nosync.grad = None

        scale_nosync = scale.detach().clone()
        scale_nosync.requires_grad = True
        scale_nosync.grad = None

        rotation_nosync = rotation.detach().clone()
        rotation_nosync.requires_grad = True
        rotation_nosync.grad = None

        opacity_nosync = opacity.detach().clone()
        opacity_nosync.requires_grad = True
        opacity_nosync.grad = None

        with (
            torch.enable_grad(),
            torch.autocast("cuda", **ctx.gpu_autocast_kwargs),
            torch.autocast("cpu", **ctx.cpu_autocast_kwargs)
        ):
            B, V = ctx.C2W.shape[:2]

            for i in range(B):
                ctx.manual_seeds.append([])
                pc = ctx.gaussian_model.set_data(xyz_nosync[i], rgb_nosync[i], scale_nosync[i], rotation_nosync[i], opacity_nosync[i])

                for j in range(V):
                    fxfycxcy_ij = ctx.fxfycxcy[i, j]
                    fx, fy, cx, cy = fxfycxcy_ij[0], fxfycxcy_ij[1], fxfycxcy_ij[2], fxfycxcy_ij[3]

                    for m in range(0, ctx.width//ctx.patch_size):
                        for n in range(0, ctx.height //ctx.patch_size):
                            grad_images_split = grad_images[i, j, :, n*ctx.patch_size:(n+1)*ctx.patch_size, m*ctx.patch_size:(m+1)*ctx.patch_size]
                            grad_alphas_split = grad_alphas[i, j, :, n*ctx.patch_size:(n+1)*ctx.patch_size, m*ctx.patch_size:(m+1)*ctx.patch_size]
                            grad_depths_split = grad_depths[i, j, :, n*ctx.patch_size:(n+1)*ctx.patch_size, m*ctx.patch_size:(m+1)*ctx.patch_size]
                            grad_normals_split = grad_normals[i, j, :, n*ctx.patch_size:(n+1)*ctx.patch_size, m*ctx.patch_size:(m+1)*ctx.patch_size]

                            seed = torch.randint(0, 2**32, (1,)).long().item()
                            ctx.manual_seeds[-1].append(seed)

                            # Transform intrinsics
                            center_x = (m*ctx.patch_size + ctx.patch_size//2) / ctx.width
                            center_y = (n*ctx.patch_size + ctx.patch_size//2) / ctx.height

                            scale_x = ctx.width // ctx.patch_size
                            scale_y = ctx.height // ctx.patch_size
                            trans_x = 0.5 - scale_x * center_x
                            trans_y = 0.5 - scale_y * center_y

                            new_fx = scale_x * fx
                            new_fy = scale_y * fy
                            new_cx = scale_x * cx + trans_x
                            new_cy = scale_y * cy + trans_y

                            new_fxfycxcy = torch.stack([new_fx, new_fy, new_cx, new_cy], dim=0)

                            render_results = render(pc, ctx.patch_size, ctx.patch_size, ctx.C2W[i, j], new_fxfycxcy, ctx.znear, ctx.zfar, ctx.bg_color, ctx.scaling_modifier)
                            color_split = render_results["image"]
                            alpha_split = render_results["alpha"]
                            depth_split = render_results["depth"]
                            normal_split = render_results["normal"]

                            render_split = torch.cat([color_split, alpha_split, depth_split, normal_split], dim=0)
                            grad_split = torch.cat([grad_images_split, grad_alphas_split, grad_depths_split, grad_normals_split], dim=0) 
                            render_split.backward(grad_split)

        return xyz_nosync.grad, rgb_nosync.grad, scale_nosync.grad, rotation_nosync.grad, opacity_nosync.grad, None, None, None, None, None, None, None, None, None, None, None


def deferred_bp(
    xyz, rgb, scale, rotation, opacity,
    height, width, C2W, fxfycxcy,
    patch_size, gaussian_model,
    znear, zfar,
    bg_color,
    scaling_modifier,
    render_dn,
):
    return DeferredBP.apply(
        xyz, rgb, scale, rotation, opacity,
        height, width, C2W, fxfycxcy,
        patch_size, gaussian_model,
        znear, zfar,
        bg_color,
        scaling_modifier,
        render_dn,
    )
