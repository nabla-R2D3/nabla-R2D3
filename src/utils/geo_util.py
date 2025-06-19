from typing import *
from torch import Tensor
from einops import rearrange, repeat
import torch
import torch.nn.functional as tF
from ml_collections import config_dict
import math
# the following functions depths_double_to_points and depth_double_to_normal are adopted from https://github.com/hugoycj/2dgs-gaustudio/blob/main/utils/graphics_utils.py
def depths_double_to_points(H,W,fov, depthmap1, depthmap2):

    fy = fx = H / (2 * math.tan(fov / 2.)) ## TODO: assume fx=fy
    intrins_inv = torch.tensor(
        [[1/fx, 0.,-W/(2 * fx)],
        [0., 1/fy, -H/(2 * fy),],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).reshape(3, -1).float().cuda()
    rays_d = intrins_inv @ points
    B,C,H,W= depthmap1.shape
    rays_d = repeat(rays_d.unsqueeze(0), '1 c N -> B c N', B=B)
    
    points1 = rearrange(depthmap1,"b c h w -> b c (h w)") * rays_d
    points2 = rearrange(depthmap2,"b c h w -> b c (h w)") * rays_d
    return rearrange(points1,"b c (h w) -> b c h w",h=H,w=W),  rearrange(points2,"b c (h w) -> b c h w",h=H,w=W)
    # rays_d = repeat(rays_d, '3 N -> B 3 N', B=B)
    # points1 = depthmap1.reshape(1,-1) * rays_d
    # points2 = depthmap2.reshape(1,-1) * rays_d
    # return points1.reshape(3,H,W), points2.reshape(3,H,W)

def look_at(campos, target, opengl=True):
    """construct pose rotation matrix by look-at.

    Args:
        campos (np.ndarray): camera position, float [3]
        target (np.ndarray): look at target, float [3]
        opengl (bool, optional): whether use opengl camera convention (forward direction is target --> camera). Defaults to True.

    Returns:
        np.ndarray: the camera pose rotation matrix, float [3, 3], normalized.
    """
   
    if not opengl:
        # camera forward aligns with -z, camera --> target
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z, target --> camera
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


# elevation & azimuth to pose (cam2world) matrix
def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True, customize_pos=False):
    """construct a camera pose matrix orbiting a target with elevation & azimuth angle.

    Args:
        elevation (float): elevation in (-90, 90), from +y to -y is (-90, 90)
        azimuth (float): azimuth in (-180, 180), from +z to +x is (0, 90)
        radius (int, optional): camera radius. Defaults to 1.
        is_degree (bool, optional): if the angles are in degree. Defaults to True.
        target (np.ndarray, optional): look at target position. Defaults to None.
        opengl (bool, optional): whether to use OpenGL camera convention. Defaults to True.

    Returns:
        np.ndarray: the camera pose matrix, float [4, 4]
    """
    
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    if not customize_pos:
        campos = np.array([x, y, z]) + target # [3] original implementation, shift the camera pose together with the target
    else:
        campos = np.array([x, y, z]) # customized for tactile camera, keep the predefined camera pose
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T


def undo_orbit_camera(T, is_degree=True):
    """ undo an orbital camera pose matrix to elevation & azimuth

    Args:
        T (np.ndarray): camera pose matrix, float [4, 4], must be an orbital camera targeting at (0, 0, 0)!
        is_degree (bool, optional): whether to return angles in degree. Defaults to True.

    Returns:
        Tuple[float]: elevation, azimuth, and radius.
    """
    
    campos = T[:3, 3]
    radius = np.linalg.norm(campos)
    elevation = np.arcsin(-campos[1] / radius)
    azimuth = np.arctan2(campos[0], campos[2])
    if is_degree:
        elevation = np.rad2deg(elevation)
        azimuth = np.rad2deg(azimuth)
    return elevation, azimuth, radius

# perspective matrix
def get_perspective(fovy, aspect=1, near=0.01, far=1000):
    """construct a perspective matrix from fovy.

    Args:
        fovy (float): field of view in degree along y-axis.
        aspect (int, optional): aspect ratio. Defaults to 1.
        near (float, optional): near clip plane. Defaults to 0.01.
        far (int, optional): far clip plane. Defaults to 1000.

    Returns:
        np.ndarray: perspective matrix, float [4, 4]
    """
    # fovy: field of view in degree.
    
    y = np.tan(np.deg2rad(fovy) / 2)
    return np.array(
        [
            [1 / (y * aspect), 0, 0, 0],
            [0, -1 / y, 0, 0],
            [
                0,
                0,
                -(far + near) / (far - near),
                -(2 * far * near) / (far - near),
            ],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )




def point_double_to_normal( points1, points2):
    points = torch.stack([points1, points2],dim=1)
    output = torch.zeros_like(points)
    dx = points[...,2:, 1:-1] - points[...,:-2, 1:-1]
    dy = points[...,1:-1, 2:] - points[...,1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=2), dim=2)
    output[...,1:-1, 1:-1] = normal_map
    return output

def depth_double_to_normal(H,W,fov, depth1, depth2):
    points1, points2 = depths_double_to_points(H,W,fov, depth1, depth2)
    return point_double_to_normal( points1, points2)



def unproject_from_depthmap_torch(c2w,intrinsic,depth:torch.tensor,depth_mask=None):
    """depth: (h,w)"""
    (h,w)=depth.shape
    px, py = torch.meshgrid(
        
        torch.arange(0, w, dtype=torch.float32),torch.arange(0, h, dtype=torch.float32),indexing='xy')
    # print(px.shape,px.max())
    # print(py.shape,py.max())
    img_xy = torch.stack([px+0.5, py+0.5], axis=-1).to(depth.device)
    # print(px)
    # print(px+0.5)
    reverse_intrin = torch.linalg.inv(intrinsic).T
    cam_xy =  img_xy * depth[...,None]
    cam_xyz = torch.cat([cam_xy, depth[...,None]], -1)
    cam_xyz = torch.matmul(cam_xyz, reverse_intrin)
    mask_depth= cam_xyz[...,2]>1e-6
    # cam_xyz = cam_xyz[mask_depth > 1e-7,:]
    cam_xyz = torch.cat([cam_xyz, torch.ones_like(cam_xyz[...,:1])], axis=-1)
    world_xyz = torch.matmul(cam_xyz.reshape(-1,4), c2w.T)[...,:3]
    return world_xyz,cam_xyz,img_xy,mask_depth

def generate_ray_directions(H, W, fov: float = 60.0):

    fov = torch.deg2rad(torch.Tensor([fov]))
    focal_length = 0.5 * H / torch.tan(0.5 * fov)

    i, j = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'  # 'ij' 
    )

    # pixel 2 camera direction
    x = (j - W * 0.5) / focal_length
    y = -(i - H * 0.5) / focal_length  # NOTE： flip y 
    z = torch.ones_like(x)  # z axis = 1

    directions_ = torch.stack([x, y, z], dim=-1)

    # normalize
    directions = directions_ / torch.norm(directions_, dim=-1, keepdim=True)

    return directions ## HWC
    # return directions,directions_,i,j,focal_length
def normalize_normals(normals: Tensor, C2W: Tensor, i: int = 0) -> Tensor:
    """Normalize a batch of multi-view `normals` by the `i`-th view.

    Inputs:
        - `normals`: (B, V, 3, H, W)
        - `C2W`: (B, V, 4, 4)
        - `i`: the index of the view to normalize by

    Outputs:
        - `normalized_normals`: (B, V, 3, H, W)
    """
    _, _, R, C = C2W.shape  # (B, V, 4, 4)
    assert R == C == 4
    _, _, CC, _, _ = normals.shape  # (B, V, 3, H, W)
    assert CC == 3

    dtype = normals.dtype
    normals = normals.clone().float()
    transform = torch.inverse(C2W[:, i, :3, :3])  # (B, 3, 3)

    return torch.einsum("brc,bvchw->bvrhw", transform, normals).to(dtype)  # (B, V, 3, H, W)


def default_4_cams(opt:config_dict.ConfigDict=None,device=None): ## hard-coded
    fxfycxcy = torch.tensor([opt.fxfy, opt.fxfy, 0.5, 0.5]).float()
    elevations = torch.tensor([-opt.elevation] * 4).deg2rad().float()
    azimuths = torch.tensor([0., 90., 180., 270.]).deg2rad().float()  # hard-coded
    radius = torch.tensor([opt.distance] * 4).float()
    input_C2W = orbit_camera(elevations, azimuths, radius, is_degree=False)  # (V_in, 4, 4)
    input_C2W[:, :3, 1:3] *= -1  # OpenGL -> OpenCV
    input_fxfycxcy = fxfycxcy.unsqueeze(0).repeat(input_C2W.shape[0], 1)  # (V_in, 4)
    
    cam_info = {}
    cam_info['input_C2W'] = input_C2W.to(device)
    cam_info['input_fxfycxcy'] = input_fxfycxcy.to(device)
    cam_info["height"] = opt.render_res
    cam_info["width"] = opt.render_res
    return cam_info

def random_sample_cams(opt:config_dict.ConfigDict=None,device=None,num_views=None,azi_delta=0,fxfy=None,elevation_delta=None): ## hard-coded
    assert opt.num_views == 4
    num_views = opt.num_views
    if fxfy is not None:
        fxfycxcy = torch.tensor([fxfy, fxfy, 0.5, 0.5]).float()
    else:   
        fxfycxcy = torch.tensor([opt.fxfy, opt.fxfy, 0.5, 0.5]).float()
    if elevation_delta is not None:
        elevations = torch.tensor([elevation_delta] * num_views).deg2rad().float()
    else:
        elevations = torch.tensor([-opt.elevation] * num_views).deg2rad().float()
    azimuths = torch.tensor([0., 90., 180., 270.]).deg2rad().float()  # hard-coded
    
    azi_delta= (azi_delta*torch.rand(1)-0.5*azi_delta).deg2rad().float()
    azimuths = azimuths + azi_delta
    ele_delta = (torch.rand(1)*opt.elevation-0.5*opt.elevation).deg2rad().float()
    elevations = elevations + ele_delta
    
    radius = torch.tensor([opt.distance] * num_views).float()
    input_C2W = orbit_camera(elevations, azimuths, radius, is_degree=False)  # (V_in, 4, 4)
    input_C2W[:, :3, 1:3] *= -1  # OpenGL -> OpenCV
    input_fxfycxcy = fxfycxcy.unsqueeze(0).repeat(input_C2W.shape[0], 1)  # (V_in, 4)
    
    cam_info = {}
    cam_info['input_C2W'] = input_C2W.to(device)
    cam_info['input_fxfycxcy'] = input_fxfycxcy.to(device)
    cam_info["height"] = opt.render_res
    cam_info["width"] = opt.render_res
    return cam_info


# def random_sample_cams(num=4,opt:config_dict.ConfigDict=None):
    
#     raise NotImplementedError
    
#     fxfycxcy = torch.tensor([opt.fxfy, opt.fxfy, 0.5, 0.5]).float()
#     elevations = torch.tensor([-opt.elevation] * 4).deg2rad().float()
#     azimuths = torch.tensor([0., 90., 180., 270.]).deg2rad().float()  # hard-coded
#     radius = torch.tensor([opt.distance] * 4).float()
#     input_C2W = orbit_camera(elevations, azimuths, radius, is_degree=False)  # (V_in, 4, 4)
#     input_C2W[:, :3, 1:3] *= -1  # OpenGL -> OpenCV
#     input_fxfycxcy = fxfycxcy.unsqueeze(0).repeat(input_C2W.shape[0], 1)  # (V_in, 4)

def normalize_C2W(C2W: Tensor, i: int = 0, norm_radius: float = 0.) -> Tensor:
    """Normalize a batch of multi-view `C2W` by the `i`-th view.

    Inputs:
        - `C2W`: (B, V, 4, 4)
        - `i`: the index of the view to normalize by
        - `norm_radius`: the normalization radius

    Outputs:
        - `normalized_C2W`: (B, V, 4, 4)
    """
    _, _, R, C = C2W.shape  # (B, V, 4, 4)
    assert R == C == 4

    device, dtype = C2W.device, C2W.dtype
    C2W = C2W.clone().float()

    if abs(norm_radius) > 0.:
        radius = torch.norm(C2W[:, i, :3, 3], dim=1)  # (B,)
        C2W[:, :, :3, 3] *= (norm_radius / radius.unsqueeze(1).unsqueeze(2))

    # The `i`-th view is normalized to a canonical matrix as the reference view
    transform = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, norm_radius],
        [0, 0, 0, 1]  # canonical c2w in OpenGL world convention
    ], dtype=torch.float32, device=device) @ torch.inverse(C2W[:, i, ...])  # (B, 4, 4)

    return (transform.unsqueeze(1) @ C2W).to(dtype)  # (B, V, 4, 4)


def unproject_depth(depth_map: Tensor, C2W: Tensor, fxfycxcy: Tensor) -> Tensor:
    """Unproject depth map to 3D world coordinate.

    Inputs:
        - `depth_map`: (B, V, H, W)
        - `C2W`: (B, V, 4, 4)
        - `fxfycxcy`: (B, V, 4)

    Outputs:
        - `xyz_world`: (B, V, 3, H, W)
    """
    device, dtype = depth_map.device, depth_map.dtype
    B, V, H, W = depth_map.shape

    depth_map = depth_map.reshape(B*V, H, W).float()
    C2W = C2W.reshape(B*V, 4, 4).float()
    fxfycxcy = fxfycxcy.reshape(B*V, 4).float()
    K = torch.zeros(B*V, 3, 3, dtype=torch.float32, device=device)
    K[:, 0, 0] = fxfycxcy[:, 0]
    K[:, 1, 1] = fxfycxcy[:, 1]
    K[:, 0, 2] = fxfycxcy[:, 2]
    K[:, 1, 2] = fxfycxcy[:, 3]
    K[:, 2, 2] = 1

    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")  # OpenCV/COLMAP camera convention
    y = y.to(device).unsqueeze(0).repeat(B*V, 1, 1) / (H-1)
    x = x.to(device).unsqueeze(0).repeat(B*V, 1, 1) / (W-1)
    # NOTE: To align with `plucker_ray(bug=False)`, should be:
    # y = (y.to(device).unsqueeze(0).repeat(B*V, 1, 1) + 0.5) / H
    # x = (x.to(device).unsqueeze(0).repeat(B*V, 1, 1) + 0.5) / W
    xyz_map = torch.stack([x, y, torch.ones_like(x)], axis=-1) * depth_map[..., None]
    xyz = xyz_map.view(B*V, -1, 3)

    # Get point positions in camera coordinate
    xyz = torch.matmul(xyz, torch.transpose(torch.inverse(K), 1, 2))
    xyz_map = xyz.view(B*V, H, W, 3)

    # Transform pts from camera to world coordinate
    xyz_homo = torch.ones((B*V, H, W, 4), device=device)
    xyz_homo[..., :3] = xyz_map
    xyz_world = torch.bmm(C2W, xyz_homo.reshape(B*V, -1, 4).permute(0, 2, 1))[:, :3, ...].to(dtype)  # (B*V, 3, H*W)
    xyz_world = xyz_world.reshape(B, V, 3, H, W)
    return xyz_world


def plucker_ray(h: int, w: int, C2W: Tensor, fxfycxcy: Tensor, bug: bool = True) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    """Get Plucker ray embeddings.

    Inputs:
        - `h`: image height
        - `w`: image width
        - `C2W`: (B, V, 4, 4)
        - `fxfycxcy`: (B, V, 4)

    Outputs:
        - `plucker`: (B, V, 6, `h`, `w`)
        - `ray_o`: (B, V, 3, `h`, `w`)
        - `ray_d`: (B, V, 3, `h`, `w`)
    """
    device, dtype = C2W.device, C2W.dtype
    B, V = C2W.shape[:2]

    C2W = C2W.reshape(B*V, 4, 4).float()
    fxfycxcy = fxfycxcy.reshape(B*V, 4).float()

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")  # OpenCV/COLMAP camera convention
    y, x = y.to(device), x.to(device)
    if bug:  # BUG !!! same here: https://github.com/camenduru/GRM/blob/master/model/visual_encoder/vit_gs.py#L85
        y = y[None, :, :].expand(B*V, -1, -1).reshape(B*V, -1) / (h - 1)
        x = x[None, :, :].expand(B*V, -1, -1).reshape(B*V, -1) / (w - 1)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    else:
        y = (y[None, :, :].expand(B*V, -1, -1).reshape(B*V, -1) + 0.5) / h
        x = (x[None, :, :].expand(B*V, -1, -1).reshape(B*V, -1) + 0.5) / w
        x = (x - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # (B*V, h*w, 3)
    ray_d = torch.bmm(ray_d, C2W[:, :3, :3].transpose(1, 2))  # (B*V, h*w, 3)
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # (B*V, h*w, 3)
    ray_o = C2W[:, :3, 3][:, None, :].expand_as(ray_d)  # (B*V, h*w, 3)

    ray_o = ray_o.reshape(B, V, h, w, 3).permute(0, 1, 4, 2, 3)
    ray_d = ray_d.reshape(B, V, h, w, 3).permute(0, 1, 4, 2, 3)
    plucker = torch.cat([torch.cross(ray_o, ray_d, dim=2).to(dtype), ray_d.to(dtype)], dim=2)

    return plucker, (ray_o, ray_d)


def orbit_camera(
    elevs: Tensor, azims: Tensor, radius: Optional[Tensor] = None,
    is_degree: bool = True,
    target: Optional[Tensor] = None,
    opengl: bool=True,
) -> Tensor:
    """Construct a camera pose matrix orbiting a target with elevation & azimuth angle.

    Inputs:
        - `elevs`: (B,); elevation in (-90, 90), from +y to -y is (-90, 90)
        - `azims`: (B,); azimuth in (-180, 180), from +z to +x is (0, 90)
        - `radius`: (B,); camera radius; if None, default to 1.
        - `is_degree`: bool; whether the input angles are in degree
        - `target`: (B, 3); look-at target position
        - `opengl`: bool; whether to use OpenGL convention

    Outputs:
        - `C2W`: (B, 4, 4); camera pose matrix
    """
    device, dtype = elevs.device, elevs.dtype

    if radius is None:
        radius = torch.ones_like(elevs)
    assert elevs.shape == azims.shape == radius.shape
    if target is None:
        target = torch.zeros(elevs.shape[0], 3, device=device, dtype=dtype)

    if is_degree:
        elevs = torch.deg2rad(elevs)
        azims = torch.deg2rad(azims)

    x = radius * torch.cos(elevs) * torch.sin(azims)
    y = - radius * torch.sin(elevs)
    z = radius * torch.cos(elevs) * torch.cos(azims)

    camposes = torch.stack([x, y, z], dim=1) + target  # (B, 3)
    R = look_at(camposes, target, opengl=opengl)  # (B, 3, 3)
    C2W = torch.cat([R, camposes[:, :, None]], dim=2)  # (B, 3, 4)
    C2W = torch.cat([C2W, torch.zeros_like(C2W[:, :1, :])], dim=1)  # (B, 4, 4)
    C2W[:, 3, 3] = 1.
    return C2W


def look_at(camposes: Tensor, targets: Tensor, opengl: bool = True) -> Tensor:
    """Construct batched pose rotation matrices by look-at.

    Inputs:
        - `camposes`: (B, 3); camera positions
        - `targets`: (B, 3); look-at targets
        - `opengl`: whether to use OpenGL convention

    Outputs:
        - `R`: (B, 3, 3); normalized camera pose rotation matrices
    """
    device, dtype = camposes.device, camposes.dtype

    if not opengl:  # OpenCV convention
        # forward is camera -> target
        forward_vectors = tF.normalize(targets - camposes, dim=-1)
        up_vectors = torch.tensor([0., 1., 0.], device=device, dtype=dtype)[None, :].expand_as(forward_vectors)
        right_vectors = tF.normalize(torch.cross(forward_vectors, up_vectors), dim=-1)
        up_vectors = tF.normalize(torch.cross(right_vectors, forward_vectors), dim=-1)
    else:
        # forward is target -> camera
        forward_vectors = tF.normalize(camposes - targets, dim=-1)
        up_vectors = torch.tensor([0., 1., 0.], device=device, dtype=dtype)[None, :].expand_as(forward_vectors)
        right_vectors = tF.normalize(torch.cross(up_vectors, forward_vectors), dim=-1)
        up_vectors = tF.normalize(torch.cross(forward_vectors, right_vectors), dim=-1)

    R = torch.stack([right_vectors, up_vectors, forward_vectors], dim=-1)
    return R
