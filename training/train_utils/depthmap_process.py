import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def _gaussian_kernel_1d(sigma: float, device, dtype, max_ksize: int = 0):
    if sigma is None or sigma <= 0:
        return None
    ksize = int(round(sigma * 6)) | 1
    if max_ksize and ksize > max_ksize:
        ksize = max_ksize | 1
    x = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    g = torch.exp(-(x * x) / (2 * sigma * sigma))
    g = g / g.sum()
    return g

def _gaussian_blur(img: torch.Tensor, sigma: float, max_ksize: int = 0):
    """
    img: [N,C,H,W]  (N can be B*S)
    """
    k = _gaussian_kernel_1d(sigma, img.device, img.dtype, max_ksize)
    if k is None:
        return img
    N, C, H, W = img.shape
    kx = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    img = F.conv2d(img, kx, padding=(0, k.numel() // 2), groups=C)
    img = F.conv2d(img, ky, padding=(k.numel() // 2, 0), groups=C)
    return img

def make_normals_torch(
    depths: torch.Tensor,                      # (B,S,H,W) or (B,S,H,W,1)
    K: torch.Tensor,                           # (B,3,3) or (B,S,3,3)
    valid_mask: Optional[torch.Tensor] = None, # (B,S,H,W) bool (optional)
    depth_smooth_sigma: float = 1.0,
    normal_smooth_sigma: float = 0.0,
    max_ksize: int = 15,
    eps: float = 1e-6,
    face_camera: bool = True,
) -> torch.Tensor:
    """
    Range depth -> normal map for batched sequences.

    Returns:
        normals: (B,S,3,H,W) unit normals. invalid pixels -> 0.
    """
    # --- normalize depth shape ---
    if depths.dim() == 5 and depths.size(-1) == 1:
        depths_ = depths[..., 0]  # (B,S,H,W)
    else:
        depths_ = depths
    assert depths_.dim() == 4, f"depths must be (B,S,H,W) (got {depths_.shape})"
    B, S, H, W = depths_.shape

    # --- normalize K shape ---
    if K.dim() == 3:
        # (B,3,3) -> (B,S,3,3)
        K_ = K[:, None, :, :].expand(B, S, 3, 3).contiguous()
    elif K.dim() == 4:
        assert K.shape[:2] == (B, S) and K.shape[-2:] == (3, 3), f"K must be (B,S,3,3); got {K.shape}"
        K_ = K
    else:
        raise ValueError("K must be (B,3,3) or (B,S,3,3)")

    # --- mask ---
    if valid_mask is None:
        valid_mask = depths_ > 0
    else:
        assert valid_mask.shape == (B, S, H, W)
        valid_mask = valid_mask.to(dtype=torch.bool, device=depths_.device)

    # flatten B,S -> N
    N = B * S
    d = depths_.reshape(N, 1, H, W)
    m = valid_mask.reshape(N, 1, H, W)
    Kf = K_.reshape(N, 3, 3)

    # optional depth smoothing
    if depth_smooth_sigma and depth_smooth_sigma > 0:
        d = _gaussian_blur(d, depth_smooth_sigma, max_ksize=max_ksize)

    # zero invalid to keep things stable
    d = d * m.to(d.dtype)

    # pixel grid
    u = torch.arange(W, device=d.device, dtype=d.dtype).view(1, 1, 1, W).expand(N, 1, H, W)
    v = torch.arange(H, device=d.device, dtype=d.dtype).view(1, 1, H, 1).expand(N, 1, H, W)

    fx = Kf[:, 0, 0].view(N, 1, 1, 1)
    fy = Kf[:, 1, 1].view(N, 1, 1, 1)
    cx = Kf[:, 0, 2].view(N, 1, 1, 1)
    cy = Kf[:, 1, 2].view(N, 1, 1, 1)

    # unit ray direction
    rx = (u - cx) / (fx + eps)
    ry = (v - cy) / (fy + eps)
    rz = torch.ones_like(rx)
    ray_norm = torch.sqrt(rx * rx + ry * ry + rz * rz + eps)
    rxu, ryu, rzu = rx / ray_norm, ry / ray_norm, rz / ray_norm

    # backproject: P = depth * ray_unit
    X = rxu * d
    Y = ryu * d
    Z = rzu * d
    P = torch.cat([X, Y, Z], dim=1)  # (N,3,H,W)

    # central differences in 3D
    P_pad = F.pad(P, (1, 1, 1, 1), mode="replicate")
    dx = P_pad[:, :, 1:-1, 2:] - P_pad[:, :, 1:-1, :-2]
    dy = P_pad[:, :, 2:, 1:-1] - P_pad[:, :, :-2, 1:-1]

    n = torch.cross(dx, dy, dim=1)  # (N,3,H,W)
    n = n / torch.sqrt((n * n).sum(dim=1, keepdim=True) + eps)

    if face_camera:
        # camera coords: z forward. Flip so normals generally face camera => n_z < 0
        flip = torch.where(n[:, 2:3] > 0, -torch.ones_like(n[:, 2:3]), torch.ones_like(n[:, 2:3]))
        n = n * flip

    # optional normal smoothing
    if normal_smooth_sigma and normal_smooth_sigma > 0:
        n = _gaussian_blur(n, normal_smooth_sigma, max_ksize=max_ksize)
        n = n / torch.sqrt((n * n).sum(dim=1, keepdim=True) + eps)

    # mask invalid outputs
    n = n * m.to(n.dtype)

    # reshape back to (B,S,3,H,W)
    normals = n.reshape(B, S, 3, H, W)
    return normals

def make_sparse_depths_torch(
    depths: torch.Tensor,                 # (B,S,H,W) or (B,S,H,W,1)
    ratio: float = 0.99,
    valid_mask: Optional[torch.Tensor] = None,  # (B,S,H,W) bool
    keep_value: float = 0.0,
    return_mask: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Create sparse depth maps by random pixel sampling (Bernoulli).
    Returns:
    sparse_depths: same shape as depths (without last singleton if provided)
    sparse_mask: (B,S,H,W) bool (optional)
    """
    assert 0.0 < ratio <= 1.0
    if depths.dim() == 5 and depths.size(-1) == 1:
        depths_ = depths[..., 0]
    else:
        depths_ = depths
    assert depths_.dim() == 4, f"depths must be (B,S,H,W) (got {depths_.shape})"

    if valid_mask is None:
        valid_mask = depths_ > 0

    # random keep mask
    r = torch.rand(valid_mask.shape, device=depths_.device, generator=generator)
    keep_mask = (r < ratio) & valid_mask

    sparse = torch.full_like(depths_, fill_value=keep_value)
    sparse[keep_mask] = depths_[keep_mask]

    if return_mask:
        return sparse, keep_mask
    return sparse, None
