import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", layout_file="layouts/SfM.slides.json")


@app.cell(hide_code=True)
def _():
    import importlib
    from pathlib import Path
    import sys

    import cv2
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go
    import torch
    from matplotlib.lines import Line2D
    from PIL import Image

    return Image, Line2D, Path, cv2, go, importlib, mo, np, plt, sys, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Outline

    - What is the Structure from Motion (SfM) problem?
    - Affine SfM (Tomasi-Kanade)
      - Feature tracking
      - Measurement matrix construction
      - Centering (remove translation)
      - Rank-3 factorization (SVD)
      - Metric upgrade
    - Papers (options)
      - VGGT / MapAnything / VGG-SfM / MASt3R
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is the Structure from Motion (SfM) problem?

    - Recover 3D structure and camera motion from multiple 2D images.
    - Use correspondences of the same points across views as constraints.
    - Output is camera geometry + a 3D point cloud (up to gauge ambiguity).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Affine SfM (Tomasi-Kanade)

    - Assume an affine/orthographic camera so projection is linear.
    - Stack tracked 2D points across frames into a measurement matrix.
    - Factor the matrix into motion and structure, then enforce metric constraints.
    """)
    return


@app.cell(hide_code=True)
def _(Path, mo):
    dino_frame_paths = sorted(Path("data/dino").glob("viff.*.ppm"))
    n_frames = len(dino_frame_paths)
    next_frame = (
        mo.ui.button(
            value=0,
            on_click=lambda i: (i + 1) % n_frames,
            label="Next image",
            kind="neutral",
        )
        if n_frames
        else None
    )
    next_frame
    return dino_frame_paths, n_frames, next_frame


@app.cell(hide_code=True)
def _(Image, dino_frame_paths, mo, n_frames, next_frame):
    idx = next_frame.value
    current = dino_frame_paths[idx]
    mo.vstack(
        [
            mo.md(f"Frame **{idx + 1}/{n_frames}** — `{current.name}`"),
            mo.image(Image.open(current).convert("RGB"), width="50%", rounded=True),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TODO: what is optical flow?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature tracking

    - Detect salient points and follow the same points across frames.
        - SIFT for extracting keypoints
        - Optical Flow (Lucas-Kanade method) for tracking
    - Keep only trajectories visible in many frames for stability.
    - Reject obvious outliers/drift before building the matrix.
    """)
    return


@app.cell(hide_code=True)
def _(Image, cv2, dino_frame_paths, np):
    # extract test images
    selected_paths = [dino_frame_paths[i] for i in (7, 8, 9)]
    rgb_frames = [np.array(Image.open(p).convert("RGB")) for p in selected_paths]
    _gray_frames = [cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) for rgb in rgb_frames]

    # extract sift keypoints
    _sift = cv2.SIFT_create(nfeatures=400, contrastThreshold=0.08)
    sift_kps = [_sift.detectAndCompute(g, None)[0] for g in _gray_frames]
    sift_points = [np.array([kp.pt for kp in ks], dtype=np.float32) for ks in sift_kps]

    tracked_points = [sift_points[0].reshape(-1, 1, 2)]
    for _i in range(1, len(_gray_frames)):
        # get tracked SIFT keypoints from the last frame
        _p_prev = tracked_points[-1]
        # find optical flow to next frame and use it to track our prev keypoints
        _p_next, _s, _ = cv2.calcOpticalFlowPyrLK(
            _gray_frames[_i - 1], _gray_frames[_i], _p_prev, _p_prev.copy()
        )
        # find optical flow from next frame and use it to track our curr keypoints
        _p_back, _sb, _ = cv2.calcOpticalFlowPyrLK(
            _gray_frames[_i], _gray_frames[_i - 1], _p_next, _p_next.copy()
        )
        # only keep the keypoints which are within 1 pixel of the tracks in both directions
        _fb = np.linalg.norm((_p_back - _p_prev).reshape(-1, 2), axis=1)
        _ok = _s.reshape(-1).astype(bool) & _sb.reshape(-1).astype(bool) & (_fb < 1.0)
        # update our tracked points across all images
        tracked_points = [p[_ok] for p in tracked_points]
        tracked_points.append(_p_next[_ok])
    tracked_points = [p.reshape(-1, 2) for p in tracked_points]
    return rgb_frames, selected_paths, sift_kps, sift_points, tracked_points


@app.cell(hide_code=True)
def _(
    Line2D,
    plt,
    rgb_frames,
    selected_paths,
    sift_kps,
    sift_points,
    tracked_points,
):
    _fig_tracks, _ax_tracks = plt.subplots(
        1, len(selected_paths), figsize=(5 * len(selected_paths), 5)
    )
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=6,
            label="SIFT keypoints",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=7,
            label="Tracked points",
        ),
    ]
    for _i, _path in enumerate(selected_paths):
        _ax_tracks[_i].imshow(rgb_frames[_i])
        if len(sift_points[_i]):
            _ax_tracks[_i].scatter(
                sift_points[_i][:, 0], sift_points[_i][:, 1], s=5, c="red"
            )
        if len(tracked_points[_i]):
            _ax_tracks[_i].scatter(
                tracked_points[_i][:, 0], tracked_points[_i][:, 1], s=7, c="blue"
            )
        _ax_tracks[_i].set_title(
            f"{_path.name} ({len(sift_kps[_i])}, {len(tracked_points[0])} tracked)"
        )
        _ax_tracks[_i].axis("off")
    _ax_tracks[-1].legend(handles=legend, loc="lower right", framealpha=0.9)
    _fig_tracks.tight_layout()
    _fig_tracks
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Measurement matrix construction

    - Build \(W \in \mathbb{R}^{2F 	imes P}\) from tracked \((x, y)\) coordinates.
    - Each frame contributes two rows (x-row and y-row).
    - Columns represent a single 3D point trajectory across frames.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Centering (remove translation)

    - Subtract each frame's centroid from all tracked points.
    - This removes per-frame translation from the affine model.
    - Centered data should be approximately rank-3 in the noise-free case.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rank-3 factorization (SVD)

    - Core objective: \(\min_{\operatorname{rank}(\hat W)\le 3} \|W-\hat W\|_F^2\).
    - With SVD \(W = U\Sigma V^	op\), use \(\hat W = U_3\Sigma_3V_3^	op\).
    - Then set \(M = U_3\Sigma_3^{1/2}\), \(S = \Sigma_3^{1/2}V_3^	op\) so \(\hat W = MS\).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metric upgrade

    - Factorization is ambiguous up to any invertible 3x3 transform.
    - Solve for a transform that enforces orthonormality constraints on motion rows.
    - Apply it to obtain Euclidean-consistent motion and structure.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Engineering Details

    - We can merge multiple partial reconstructions to create a complete pointcloud
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## COLMAP

    **Industry Standard**

    - Works for perspective projection (not just affine)

    - Images come in any order

        - Not necessary one image

    - Images don't necessarily overlap
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    @everyone make one cell per concept, i think

    1. Feature Extraction (kinda the same)

    2. Feature Matching (SuperGlue, ???)

    3. Geometric Verification

    4. Triangulation

    5. Bundle Adjustment (and details)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # VGGT: Visual Geometry Grounded Transformer
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why **not** COLMAP (always, everywhere, all the time)?

    - Optimization of Visual Geometry is a computationally intensive task

      - Recent papers (DUSt3R, MASt3R, VGGSfM) has demonstrated learning-based approaches but do not automate the full pipeline

    - COLMAP takes 15s while VGGT takes <1s
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## VGGT Architecture

    **VGGT is a single feed-forward transformer**

    Given some images in the scene, it predicts the camera matrices, depth maps, 3D point maps and dense point cloud.

    ![VGGT Architecture Diagram](https://vgg-t.github.io/resources/architecture_v4.png)

    ### Pipeline Steps

    1. Patchification, tokenization for each image  (using DINOv2)

    2. Append 1 camera token + 4 register tokens

    3. Transformer blocks - alternating between frame-wise self attention and global self-attention

    4. Output tokens - camera head contains intrinsics + extrinsics, DPT head (image tokens) contains depth maps, point maps, tracking features
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Vision Transformers (in a nutshell)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Training and Losses

    VGGT is trained end-to-end with dense geometric supervision from posed multi-view data which is explicitly labeled.

    **Training data:** large-scale Internet and synthetic multi-view datasets with camera poses and depth/geometry supervision (sampled as image sets from the same scene).

    A compact view of the objective is:
    \[
    \mathcal{L}_{\text{total}} =
    \lambda_{\text{cam}}\,\mathcal{L}_{\text{cam}} +
    \lambda_{\text{depth}}\,\mathcal{L}_{\text{depth}} +
    \lambda_{\text{pts}}\,\mathcal{L}_{\text{3D}} +
    \lambda_{\text{track}}\,\mathcal{L}_{\text{track}}.
    \]

    where:
    \[
    \begin{aligned}
    \mathcal{L}_{\text{cam}}\; &=\; \alpha_R\,\left\lVert \log\!\left(R\hat{R}^{\top}\right) \right\rVert_1 + \alpha_t\,\left\lVert t-\hat{t} \right\rVert_1 + \alpha_K\,\left\lVert K-\hat{K} \right\rVert_1 \\
    \mathcal{L}_{\text{depth}} &=\; \frac{1}{|\Omega|}\sum_{p\in\Omega} \rho\!\left(\log d(p)-\log \hat d(p)\right) \\
    \mathcal{L}_{\text{3D}}\; &=\; \frac{1}{|\Omega|}\sum_{p\in\Omega} \rho\!\left(\|X(p)-\hat X(p)\|_2\right) \\
    \mathcal{L}_{\text{track}} &=\; \frac{1}{|\mathcal{M}|}\sum_{(p,q)\in\mathcal{M}} \rho\!\left(\|\pi_j(\hat X_i(p)) - q\|_2\right)
    \end{aligned}
    \]
    with \(\rho\) representing some robust penalty function (reducing the impact of outliers), \(\Omega\) valid pixels, and \(\mathcal{M}\) cross-view matches.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## VGGT demo: Dino reconstruction (CUDA)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    data_dir = mo.ui.text(value="data/dino", label="Image folder")
    point_density = mo.ui.slider(
        start=1000,
        stop=120000,
        step=1000,
        value=50000,
        label="Point density",
    )
    mo.vstack(
        [
            mo.hstack([mo.md("**VGGT data path**"), data_dir], justify="start"),
            mo.hstack([mo.md("**Displayed points**"), point_density], justify="start"),
        ]
    )
    return data_dir, point_density


@app.cell(hide_code=True)
def _(Image, Path, data_dir, importlib, np, sys, torch):
    _device = torch.device("cuda")

    _local_vggt_root = str((Path.cwd() / "vggt").resolve())
    if _local_vggt_root not in sys.path:
        sys.path.insert(0, _local_vggt_root)

    _VGGT = importlib.import_module("vggt.models.vggt").VGGT
    _pose_encoding_to_extri_intri = importlib.import_module(
        "vggt.utils.pose_enc"
    ).pose_encoding_to_extri_intri
    _unproject_depth_map_to_point_map = importlib.import_module(
        "vggt.utils.geometry"
    ).unproject_depth_map_to_point_map

    _image_paths = sorted(Path(data_dir.value).glob("*.ppm"))
    assert _image_paths, f"No .ppm images found in {data_dir.value}"
    _take_idx = np.linspace(
        0, len(_image_paths) - 1, min(12, len(_image_paths)), dtype=int
    )
    _sample_paths = [_image_paths[i] for i in _take_idx]

    _raw_frames = [
        np.array(Image.open(p).convert("RGB"), dtype=np.uint8) for p in _sample_paths
    ]
    _h = min(img.shape[0] for img in _raw_frames)
    _w = min(img.shape[1] for img in _raw_frames)
    _h = (_h // 14) * 14
    _w = (_w // 14) * 14
    _rgb_frames = [
        img[
            (img.shape[0] - _h) // 2 : (img.shape[0] + _h) // 2,
            (img.shape[1] - _w) // 2 : (img.shape[1] + _w) // 2,
        ]
        for img in _raw_frames
    ]
    _images = (
        torch.from_numpy(np.stack(_rgb_frames)).permute(0, 3, 1, 2).float().to(_device)
        / 255.0
    )

    _model = _VGGT.from_pretrained("facebook/VGGT-1B").eval().to(_device)
    with torch.inference_mode():
        _pred = _model(_images)

    _extr, _intr = _pose_encoding_to_extri_intri(_pred["pose_enc"], _images.shape[-2:])
    _points3d = _unproject_depth_map_to_point_map(_pred["depth"][0], _extr[0], _intr[0])

    points3d = _points3d.reshape(-1, 3)
    colors = _images.permute(0, 2, 3, 1).detach().cpu().numpy().reshape(-1, 3)
    _valid = np.isfinite(points3d).all(axis=1)
    points3d, colors = points3d[_valid], colors[_valid]
    _stride = max(1, len(points3d) // 120000)
    points3d = points3d[::_stride].astype(np.float32, copy=False)
    colors = colors[::_stride].astype(np.float32, copy=False)
    return colors, points3d


@app.cell(hide_code=True)
def _(colors, go, point_density, points3d):
    _target = min(max(int(point_density.value), 1000), 120000)
    _plot_stride = max(1, len(points3d) // _target)
    _plot_points = points3d[::_plot_stride]
    _plot_colors = colors[::_plot_stride]
    _fig_cloud = go.Figure(
        data=[
            go.Scatter3d(
                x=_plot_points[:, 0],
                y=_plot_points[:, 1],
                z=_plot_points[:, 2],
                mode="markers",
                marker={"size": 1.2, "color": _plot_colors, "opacity": 0.7},
            )
        ]
    )
    _fig_cloud.update_layout(
        title="VGGT Point Cloud Visualization",
        scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        height=760,
        uirevision="keep",
    )
    _fig_cloud
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
