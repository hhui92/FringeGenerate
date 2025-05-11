import os
import cv2
import h5py
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial import cKDTree
from typing import Dict, Any, Tuple

# -----------------------------------------------------------------------------
# Calibration data (camera & projector intrinsic / extrinsic parameters)
# -----------------------------------------------------------------------------
cam_calib = loadmat('/john/PatternFringe1/data/calibration/CamCalibResult.mat')
prj_calib = loadmat('/john/PatternFringe1/data/calibration/PrjCalibResult.mat')

Kc = cam_calib['KK']  # 3×3 camera intrinsic matrix
Rc1 = cam_calib['Rc_1']  # 3×3 camera rotation
Tc1 = cam_calib['Tc_1']  # 3×1 camera translation
Kp = prj_calib['KK']  # 3×3 projector intrinsic
Rp1 = prj_calib['Rc_1']  # 3×3 projector rotation
Tp1 = prj_calib['Tc_1']  # 3×1 projector translation

fx_c, fy_c, cx_c, cy_c = Kc[0, 0], Kc[1, 1], Kc[0, 2], Kc[1, 2]
fx_p, fy_p, cx_p, cy_p = Kp[0, 0], Kp[1, 1], Kp[0, 2], Kp[1, 2]

Ac = Kc @ np.hstack((Rc1, Tc1))  # 3×4 camera projection
Ap = Kp @ np.hstack((Rp1, Tp1))  # 3×4 projector projection
cam_center_w = Rc1.T @ Tc1  # 3×1 camera center in world
proj_center_w = Rp1.T @ Tp1  # 3×1 projector center in world

# Resolution settings
prj_width, prj_height = 1920, 1200  # projector resolution
width, height = 1408, 1024  # captured image size


def mode1() -> Dict[str, Any]:
    """Default photometric configuration."""
    return {
        'dir_light_dir': np.array([0.0, -1.0, 0.0], dtype=np.float32),
        'dir_light_color': 1.5,
        'ambient': 0.7,
        'proj_light_color': 1.0,
        'kd': 0.6,
        'ks': 0.5,
        'shininess': 64.0,
        'scene_brightness': 1.0,
        'distance_factor': 1.0,
        'proj_center_world': proj_center_w.squeeze(),
        'F0': 0.04,
        'roughness': 0.3,
    }


def estimate_normals(xyz: np.ndarray) -> np.ndarray:
    """Compute per-pixel normals via central differences (mirror padding)."""
    pad = np.pad(xyz, ((1, 1), (1, 1), (0, 0)), mode='reflect')
    gx = (pad[1:-1, 2:, :] - pad[1:-1, :-2, :]) * 0.5
    gy = (pad[2:, 1:-1, :] - pad[:-2, 1:-1, :]) * 0.5
    n = np.cross(gx, gy)
    n /= (np.linalg.norm(n, axis=2, keepdims=True) + 1e-8)
    return n


def attenuate(dist: np.ndarray) -> np.ndarray:
    """Inverse‑square fall‑off."""
    return 1.0 / (dist ** 2 + 1e-6)


def cook_torrance_brdf(n_dot_l, n_dot_v, n_dot_h, roughness, F0):
    """Cook-Torrance BRDF (scalar version)."""
    alpha = roughness ** 2
    d = alpha ** 2 / (np.pi * (n_dot_h ** 2 * (alpha ** 2 - 1) + 1) ** 2)
    f = F0 + (1 - F0) * (1 - n_dot_l).pow(5)
    g = 4 / (1 + np.sqrt(1 + alpha ** 2 * (1 - n_dot_v ** 2) / n_dot_v ** 2))
    return (d * f * g) / (4 * n_dot_l * n_dot_v + 1e-6)


def blinn_phong_brdf(xyz_points: np.ndarray,
                     normals_map: np.ndarray,
                     valid_mask: np.ndarray,
                     cam_pos_world: np.ndarray,
                     proj_stripe_map: np.ndarray,
                     config: Dict[str, Any]) -> np.ndarray:
    """Compute shading (diffuse + Cook-Torrance specular)."""
    H, W, _ = xyz_points.shape

    dir_light_dir = config['dir_light_dir']
    ambient = config['ambient']
    kd = config['kd']
    ks = config['ks']
    scene_brightness = config['scene_brightness']
    proj_center = config['proj_center_world']
    roughness = config['roughness']
    F0 = config['F0']

    V = cam_pos_world.reshape(1, 1, 3) - xyz_points
    V = V / (np.linalg.norm(V, axis=2, keepdims=True) + 1e-8)

    Lp = proj_center.reshape(1, 1, 3) - xyz_points
    dist_proj = np.linalg.norm(Lp, axis=2)
    Lp_n = Lp / (dist_proj[..., None] + 1e-8)
    attenuation = attenuation(dist_proj)

    ndotl = np.clip(np.sum(normals_map * Lp_n, axis=2), 0.01, 0.999)
    diffuse_proj = kd * ndotl * proj_stripe_map * attenuation

    Ld = -dir_light_dir / (np.linalg.norm(dir_light_dir) + 1e-8)
    ndold = np.clip(np.sum(normals_map * Ld.reshape(1, 1, 3), axis=2), 0.01, 0.999)
    diffuse_dir = kd * ndold

    Hbis = V + Lp_n
    Hbis = Hbis / (np.linalg.norm(Hbis, axis=2, keepdims=True) + 1e-8)
    n_dot_h = np.clip(np.sum(normals_map * Hbis, axis=2), 0.01, 0.999)
    spec_proj = cook_torrance_brdf(ndotl,
                                   np.clip(np.sum(normals_map * V, axis=2), 0.01, 0.999),
                                   n_dot_h, roughness, F0) * ks * attenuation

    Hd = V + Ld.reshape(1, 1, 3)
    Hd = Hd / (np.linalg.norm(Hd, axis=2, keepdims=True) + 1e-8)
    n_dot_hd = np.clip(np.sum(normals_map * Hd, axis=2), 0.01, 0.999)
    spec_dir = cook_torrance_brdf(ndold,
                                  np.clip(np.sum(normals_map * V, axis=2), 0.01, 0.999),
                                  n_dot_hd, roughness, F0) * ks

    color_map = ambient + diffuse_proj + diffuse_dir + spec_proj + spec_dir
    color_map *= scene_brightness
    color_map *= valid_mask
    return np.clip(color_map, 0, None)


def compute_point_cloud(pha, config: Dict[str, Any]):
    # ---------------------------------------------------------------------
    # 1) Pre‑processing & mask generation
    # ---------------------------------------------------------------------
    x_p = pha * prj_width
    valid_mask = (pha != 0).astype(np.float32)

    # ---------------------------------------------------------------------
    # 2) Build per‑pixel stereo equations: A · X = b (camera & projector)
    # ---------------------------------------------------------------------
    vc, uc = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    uc = uc.astype(np.float32) * valid_mask
    vc = vc.astype(np.float32) * valid_mask
    up = x_p * valid_mask

    # Projector extrinsics (possibly perturbed)
    proj_r = config.get('proj_R', Rp1)
    proj_t = config.get('proj_T', Tp1)
    Ap = Kp @ np.hstack((proj_r, proj_t))  # (3x4)

    # Build batched 3×3 matrices A and RHS b for every pixel
    A = np.zeros((height, width, 3, 3), dtype=np.float64)
    b = np.zeros((height, width, 3), dtype=np.float64)
    A[:, :, 0, 0] = Ac[0, 0] - Ac[2, 0] * uc
    A[:, :, 0, 1] = Ac[0, 1] - Ac[2, 1] * uc
    A[:, :, 0, 2] = Ac[0, 2] - Ac[2, 2] * uc
    A[:, :, 1, 0] = Ac[1, 0] - Ac[2, 0] * vc
    A[:, :, 1, 1] = Ac[1, 1] - Ac[2, 1] * vc
    A[:, :, 1, 2] = Ac[1, 2] - Ac[2, 2] * vc
    A[:, :, 2, 0] = Ap[0, 0] - Ap[2, 0] * up
    A[:, :, 2, 1] = Ap[0, 1] - Ap[2, 1] * up
    A[:, :, 2, 2] = Ap[0, 2] - Ap[2, 2] * up
    b[:, :, 0] = Ac[2, 3] * uc - Ac[0, 3]
    b[:, :, 1] = Ac[2, 3] * vc - Ac[1, 3]
    b[:, :, 2] = Ap[2, 3] * up - Ap[0, 3]

    # 添加一个小的正则项以确保矩阵可逆
    a = A + 1e-8 * np.eye(3).reshape(1, 1, 3, 3)

    a_flat = a.reshape(-1, 3, 3)          # (N,3,3)
    b_flat = b.reshape(-1, 3, 1)              # (N,3,1)

    # Invert each 3×3 block and multiply once:  X = A⁻¹ · b
    invA = np.linalg.inv(a_flat)            # (N,3,3)
    xyz_w_flat = (invA @ b_flat)              # (N,3,1)

    # Back to (H,W,3)
    xyz_w = xyz_w_flat.reshape(height, width, 3)
    xyz_points = xyz_w * valid_mask[..., None]
    return xyz_points


def bilinear_interpolate(im: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return bilinearly-interpolated values from grayscale image *im* at floating
    pixel locations (*x*, *y*).  All inputs are HxW and returned array is HxW.
    Coordinates outside the image are clamped to the border.
    """
    H, W = im.shape

    # Clamp sampling coordinates to valid range
    x = np.clip(x, 0.0, W - 1.0)
    y = np.clip(y, 0.0, H - 1.0)

    x0 = np.floor(x).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.floor(y).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, H - 1)

    Ia = im[y0, x0]
    Ib = im[y0, x1]
    Ic = im[y1, x0]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def re_projection(pha: np.ndarray, stripe_img: np.ndarray, cfg: dict):
    """
    The function keeps the exact computational steps/outputs intact, replacing
    all Torch tensor ops with their NumPy equivalents.  All external matrices
    (Ac, Kp, Rc_p, Tc_p, Rc_1, Tc_1, etc.) are assumed to be *NumPy* arrays that
    already exist in the caller's scope - just like in the original function.
    """

    h, w = pha.shape
    xyz = compute_point_cloud(pha, cfg)

    # Optional scaling towards camera
    to_cam = xyz - cam_center_w.squeeze();
    xyz = cam_center_w.squeeze() + to_cam * cfg['distance_factor']

    # World → projector
    flat = xyz.reshape(-1, 3)
    proj_xyzw = (cfg.get('proj_R', Rp1) @ flat.T + cfg.get('proj_T', Tp1)).T.reshape(h, w, 3)
    xpj, ypj, zpj = proj_xyzw[..., 0], proj_xyzw[..., 1], proj_xyzw[..., 2]
    up = fx_p * (xpj / zpj) + cx_p;
    vp = fy_p * (ypj / zpj) + cy_p

    stripes = bilinear_interpolate(stripe_img, up, vp)[..., None]
    mask = (pha != 0).astype(np.float32)[..., None]
    stripes *= mask

    normals = estimate_normals(xyz)
    shading = blinn_phong_brdf(xyz, normals, mask.squeeze(), cam_center_w.squeeze(), stripes.squeeze(), cfg)
    surface = stripes.squeeze() * shading

    # World → camera (Z‑buffer)
    cam_pts = (Rc1 @ flat.T + Tc1).T.reshape(h, w, 3)
    x_c, y_c, z_c = cam_pts.transpose(2, 0, 1)
    u = np.round(fx_c * x_c / (z_c + 1e-8) + cx_c).astype(int)
    v = np.round(fy_c * y_c / (z_c + 1e-8) + cy_c).astype(int)
    u = np.clip(u, 0, w - 1);
    v = np.clip(v, 0, h - 1)

    img_out = np.zeros((h, w), dtype=np.float32)
    depthmap = np.full((h, w), np.inf, dtype=np.float32)
    idx_flat = v.flatten(), u.flatten()
    depth = z_c.flatten();
    color = surface.flatten()
    nearer = depth < depthmap[idx_flat]
    depthmap[idx_flat[0][nearer], idx_flat[1][nearer]] = depth[nearer]
    img_out[idx_flat[0][nearer], idx_flat[1][nearer]] = color[nearer]

    return (img_out * 255).astype(np.uint8)


def visualize_point_cloud(points_np: np.ndarray):
    """Visualize and optionally store a NumPy point cloud via Open3D."""
    pts = points_np.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1.0
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
    vis.add_geometry(axis)
    vis.run()
    vis.destroy_window()


def generate_fringe(freq, step_num, ):
    """Generate a 1-D cosine fringe (horizontal stripes) with N phase steps."""
    patterns = []
    for num in range(step_num):
        x = np.linspace(0, 1, prj_width)
        phase = (num / step_num) * 2 * np.pi
        stripe_pattern = 0.5 + 0.5 * np.cos(2 * np.pi * freq * x + phase)
        fringe = np.tile(stripe_pattern, (prj_height, 1))
        patterns.append(fringe)
    return patterns


def multi_frequency_pattern(freq_list):
    ptns = {}
    for f in freq_list:
        ptn_list = generate_fringe(step=4, step_num=4)
        ptns[f] = ptn_list
    return ptns


def get_phase_from_fringe(frng_dct):
    """Dummy absolute phase estimator (replace with actual algorithm)."""
    raise NotImplementedError("Phase retrieval not implemented")


if __name__ == "__main__":
    freq_lst = [70, 64, 59]
    ptn_dct = multi_frequency_pattern(freq_lst)

    # Please use the real phase value to replace 'unwrapped_phase'.
    # unwrapped_phase = load(path)
    unwrapped_phase = None

    frng_dct = {}
    for k, v in frng_dct.items():
        frng_dct[k] = []
        for frng in v:
            final_imagee = re_projection(unwrapped_phase, frng, mode1)
            frng_dct[k].append(final_imagee)

    unwrapped_phi = get_phase_from_fringe(frng_dct)
    cloud_points = compute_point_cloud(unwrapped_phi, config={})
    visualize_point_cloud(cloud_points)
