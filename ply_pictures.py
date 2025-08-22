#Requires: open3d, plyfile, numpy  #Headless use needs EGL/OpenGL
import os, math, argparse, numpy as np
import open3d as o3d
try:  # Prefer Agg for headless matplotlib fallback
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False
from plyfile import PlyData

def load_ply_as_o3d(path):
    """Load a PLY as an Open3D point cloud with robust color normalization.

    Color handling:
      - If red/green/blue present as integers:
          * max<=1     -> assume already normalized
          * max<=255   -> divide by 255
          * max<=65535 -> divide by 65535
          * else       -> divide by max value
      - If floats in [0,1] assume already normalized.
      - Otherwise fallback to neutral gray.
    """
    ply = PlyData.read(path); v = ply["vertex"].data
    pts = np.c_[v["x"], v["y"], v["z"]].astype(np.float64)
    if {"red", "green", "blue"}.issubset(v.dtype.names):
        r = np.asarray(v["red"])
        g = np.asarray(v["green"])
        b = np.asarray(v["blue"])
        # Determine scaling
        if r.dtype.kind in "fi" and g.dtype.kind in "fi" and b.dtype.kind in "fi":
            max_val = float(np.max([r.max(), g.max(), b.max()]))
            if max_val <= 1.0:  # Already normalized
                pass
            else:
                # Treat as integer-like range
                if max_val <= 255:
                    scale = 255.0
                elif max_val <= 65535:
                    scale = 65535.0
                else:
                    scale = max_val
                r = r / scale; g = g / scale; b = b / scale
        else:  # Mixed dtypes; coerce to float and scale heuristically
            r = r.astype(np.float64)
            g = g.astype(np.float64)
            b = b.astype(np.float64)
            max_val = float(np.max([r.max(), g.max(), b.max()]))
            if max_val > 1.0:
                if max_val <= 255:
                    scale = 255.0
                elif max_val <= 65535:
                    scale = 65535.0
                else:
                    scale = max_val
                r /= scale; g /= scale; b /= scale
        cols = np.c_[r, g, b]
        cols = np.nan_to_num(cols, nan=0.5).clip(0.0, 1.0)
    else:
        cols = np.full((pts.shape[0], 3), 0.7, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    return pcd

def render_views_offscreen(pcd, out_dir, w, h, azimuths, elev_deg, radius_scale, point_size):
    """Primary renderer using OffscreenRenderer. Raises RuntimeError on failure."""
    renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
    scene = renderer.scene
    scene.set_background([1, 1, 1, 1])
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"; mat.point_size = float(point_size)
    scene.add_geometry("pcd", pcd, mat)
    aabb = pcd.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    extent = (aabb.get_max_bound() - aabb.get_min_bound())
    radius = np.linalg.norm(extent) * radius_scale if np.any(extent > 0) else 1.0
    scene.camera.set_projection(
        60.0, w / float(h), 0.01, radius * 10.0,
        o3d.visualization.rendering.Camera.FovType.Vertical)
    up = [0, 0, 1]
    for az in azimuths:
        azr = math.radians(az); elr = math.radians(elev_deg)
        cx = math.cos(elr) * math.cos(azr); cy = math.cos(elr) * math.sin(azr); cz = math.sin(elr)
        eye = center + radius * np.array([cx, cy, cz])
        scene.camera.look_at(center, eye, up)
        img = renderer.render_to_image()
        o3d.io.write_image(os.path.join(out_dir, f"view_az{az:03d}_el{elev_deg}.png"), img)


def render_views_legacy(pcd, out_dir, w, h, azimuths, elev_deg, radius_scale, point_size):
    """Fallback renderer using the legacy Visualizer (no OffscreenRenderer / EGL).

    This creates a hidden window and uses camera parameters to save images.
    """
    aabb = pcd.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    extent = (aabb.get_max_bound() - aabb.get_min_bound())
    radius = np.linalg.norm(extent) * radius_scale if np.any(extent > 0) else 1.0
    # Vertical FOV 60 deg -> fy
    fov_v = math.radians(60.0)
    fy = h / (2.0 * math.tan(fov_v / 2.0))
    fx = fy
    cx = w / 2.0
    cy = h / 2.0
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    up_world = np.array([0, 0, 1.0])

    for az in azimuths:
        azr = math.radians(az); elr = math.radians(elev_deg)
        cxv = math.cos(elr) * math.cos(azr)
        cyv = math.cos(elr) * math.sin(azr)
        czv = math.sin(elr)
        eye = center + radius * np.array([cxv, cyv, czv])
        front = (center - eye)
        front_norm = front / (np.linalg.norm(front) + 1e-9)
        right = np.cross(front_norm, up_world)
        right /= (np.linalg.norm(right) + 1e-9)
        up = np.cross(right, front_norm)
        # Build extrinsic (world to camera) matching OpenGL look-at
        R = np.stack([right, up, -front_norm], axis=0)  # 3x3
        t = -R @ eye.reshape(3,)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t

        params = o3d.camera.PinholeCameraParameters()
        params.extrinsic = extrinsic
        params.intrinsic = intrinsic

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=w, height=h)
        vis.add_geometry(pcd)
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])
        opt.point_size = float(point_size)
        vis.poll_events(); vis.update_renderer()
        out_path = os.path.join(out_dir, f"view_az{az:03d}_el{elev_deg}.png")
        vis.capture_screen_image(out_path, do_render=True)
        vis.destroy_window()


def render_views_mpl(pcd, out_dir, w, h, azimuths, elev_deg, radius_scale, point_size, max_points=200000, tight=False, margin=4, pad_frac=0.05):
    """Fallback using matplotlib 3D scatter (orthographic-ish).

    Parameters:
        tight: if True, post-process image to crop surrounding whitespace.
        margin: pixel margin to retain after cropping (white background assumed).
    """
    if not _HAVE_MPL:
        raise RuntimeError("matplotlib not available; install it or choose another backend")
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if len(pcd.colors) else np.full_like(pts, 0.7)
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx]
    aabb = pcd.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    extent = (aabb.get_max_bound() - aabb.get_min_bound())
    radius = np.linalg.norm(extent) * radius_scale if np.any(extent > 0) else 1.0
    for az in azimuths:
        azr = math.radians(az); elr = math.radians(elev_deg)
        # Camera direction
        cxv = math.cos(elr) * math.cos(azr)
        cyv = math.cos(elr) * math.sin(azr)
        czv = math.sin(elr)
        forward = np.array([cxv, cyv, czv])
        up = np.array([0, 0, 1.0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward); up /= np.linalg.norm(up)
        # Simple rotation matrix (camera axes as rows)
        R = np.stack([right, up, forward], 0)
        pts_centered = pts - center
        cam_pts = (R @ pts_centered.T).T
        # Orthographic projection ignoring z (depth used for size ordering)
        order = np.argsort(cam_pts[:, 2])  # back to front (so nearer points drawn last if we reversed)
        cam_pts = cam_pts[order]
        cols_p = cols[order]
        fig = plt.figure(figsize=(w/100.0, h/100.0), dpi=100)
        ax = fig.add_axes([0,0,1,1])  # full-bleed
        ax.axis('off')
        s = point_size  # Marker size
        ax.scatter(cam_pts[:,0], cam_pts[:,1], c=cols_p, s=s, marker='o', linewidths=0)
        # Tighten bounds around actual projected data rather than using spherical radius
        min_x, max_x = cam_pts[:,0].min(), cam_pts[:,0].max()
        min_y, max_y = cam_pts[:,1].min(), cam_pts[:,1].max()
        span_x = max_x - min_x; span_y = max_y - min_y
        if span_x == 0: span_x = 1e-6
        if span_y == 0: span_y = 1e-6
        pad_x = span_x * pad_frac
        pad_y = span_y * pad_frac
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(min_y - pad_y, max_y + pad_y)
        ax.set_aspect('equal', adjustable='box')
        out_path = os.path.join(out_dir, f"view_az{az:03d}_el{elev_deg}.png")
        fig.savefig(out_path, dpi=100, facecolor='white')
        plt.close(fig)
        if tight:
            try:
                from PIL import Image, ImageChops
                im = Image.open(out_path)
                bg = Image.new(im.mode, im.size, (255, 255, 255))
                diff = ImageChops.difference(im, bg).convert('L')
                # Binarize
                diff = diff.point(lambda p: 255 if p > 10 else 0)
                bbox = diff.getbbox()
                if bbox:
                    left = max(bbox[0]-margin, 0)
                    upper = max(bbox[1]-margin, 0)
                    right = min(bbox[2]+margin, im.width)
                    lower = min(bbox[3]+margin, im.height)
                    im.crop((left, upper, right, lower)).save(out_path)
            except Exception as e:
                print(f"[WARN] Tight crop failed: {e}")


def render_views(pcd, out_dir, w=1024, h=768, azimuths=(0,60,120,180,240,300), elev_deg=20, radius_scale=1.2, point_size=2.0, backend="auto", max_points=200000, tight=False, margin=4, pad_frac=0.05):
    """Render multiple orbit views.

    backend: 'auto' (try offscreen then legacy), 'offscreen', or 'legacy'.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Quick no-op for empty point cloud
    if len(pcd.points) == 0:
        print("[WARN] Point cloud is empty; nothing to render.")
        return

    if backend not in {"auto", "offscreen", "legacy", "mpl"}:
        raise ValueError("backend must be one of auto|offscreen|legacy|mpl")

    if backend in ("offscreen", "auto"):
        try:
            print("[INFO] Attempting OffscreenRenderer backend...")
            render_views_offscreen(pcd, out_dir, w, h, azimuths, elev_deg, radius_scale, point_size)
            return
        except Exception as e:
            if backend == "offscreen":
                raise
            print(f"[WARN] OffscreenRenderer failed ({e}); falling back to legacy Visualizer.")
    if backend in ("legacy", "auto"):
        try:
            print("[INFO] Using legacy Visualizer backend.")
            render_views_legacy(pcd, out_dir, w, h, azimuths, elev_deg, radius_scale, point_size)
            return
        except Exception as e:
            if backend == "legacy":
                raise
            print(f"[WARN] Legacy Visualizer failed ({e}); falling back to matplotlib.")
    print("[INFO] Using matplotlib fallback backend.")
    render_views_mpl(pcd, out_dir, w, h, azimuths, elev_deg, radius_scale, point_size, max_points=max_points, tight=tight, margin=margin, pad_frac=pad_frac)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Render orbit views of a point cloud PLY.")
    parser.add_argument("ply_path", help="Input PLY file path")
    parser.add_argument("--out_dir", default="renders", help="Output directory for images")
    parser.add_argument("--width", type=int, default=1280, help="Image width")
    parser.add_argument("--height", type=int, default=960, help="Image height")
    parser.add_argument("--point_size", type=float, default=2.5, help="Point size in render")
    parser.add_argument("--backend", choices=["auto","offscreen","legacy","mpl"], default="auto", help="Rendering backend selection")
    parser.add_argument("--max_points", type=int, default=200000, help="Max points for matplotlib fallback (sampling)")
    parser.add_argument("--radius_scale", type=float, default=1.2, help="Distance scale for camera radius (smaller -> tighter framing)")
    parser.add_argument("--tight", action="store_true", help="Crop whitespace tightly (mpl backend)")
    parser.add_argument("--margin", type=int, default=4, help="Whitespace margin (pixels) when --tight is used")
    parser.add_argument("--pad_frac", type=float, default=0.05, help="Fractional padding around projected points (mpl backend)")
    args=parser.parse_args()
    # Build descriptive output directory name including key parameters
    stem = os.path.splitext(os.path.basename(args.ply_path))[0]
    # Format floats compactly
    def fmt(x):
        return ("%g" % x)
    out_dir_full = f"{args.out_dir}_{stem}_{args.width}_{args.height}_{fmt(args.point_size)}_{args.max_points}_{fmt(args.radius_scale)}_{args.tight}_{args.margin}"
    pcd=load_ply_as_o3d(args.ply_path)
    render_views(pcd, out_dir_full, w=args.width, h=args.height, point_size=args.point_size, backend=args.backend, max_points=args.max_points, radius_scale=args.radius_scale, tight=args.tight, margin=args.margin, pad_frac=args.pad_frac)
    print(f"[INFO] Images written to: {out_dir_full}")