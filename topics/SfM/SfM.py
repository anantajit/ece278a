import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


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
def _(mo):
    from pathlib import Path

    frame_paths = sorted(Path("data/dino").glob("viff.*.ppm"))
    n_frames = len(frame_paths)
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
    return frame_paths, n_frames, next_frame


@app.cell(hide_code=True)
def _(frame_paths, mo, n_frames, next_frame):
    from PIL import Image

    idx = next_frame.value
    current = frame_paths[idx]
    mo.vstack(
        [
            mo.md(f"Frame **{idx + 1}/{n_frames}** — `{current.name}`"),
            mo.image(Image.open(current).convert("RGB"), width="50%", rounded=True),
        ]
    )
    return (Image,)


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


@app.cell
def _(frame_paths):
    import numpy as np
    from PIL import Image as _Image
    import cv2

    # extract test images
    selected = [frame_paths[i] for i in (7, 8, 9)]
    rgbs = [np.array(_Image.open(p).convert("RGB")) for p in selected]
    grays = [cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) for rgb in rgbs]

    # extract sift keypoints
    sift = cv2.SIFT_create(nfeatures=400, contrastThreshold=0.08)
    kps = [sift.detectAndCompute(g, None)[0] for g in grays]
    pts = [np.array([kp.pt for kp in ks], dtype=np.float32) for ks in kps]

    tracked = [pts[0].reshape(-1, 1, 2)]
    for _i in range(1, len(grays)):
        # get tracked SIFT keypoints from the last frame
        p_prev = tracked[-1]
        # find optical flow to next frame and use it to track our prev keypoints
        p_next, s, _ = cv2.calcOpticalFlowPyrLK(grays[_i - 1], grays[_i], p_prev, None)
        # find optical flow from next frame and use it to track our curr keypoints
        p_back, sb, _ = cv2.calcOpticalFlowPyrLK(grays[_i], grays[_i - 1], p_next, None)
        # only keep the keypoints which are within 1 pixel of the tracks in both directions
        fb = np.linalg.norm((p_back - p_prev).reshape(-1, 2), axis=1)
        ok = s.reshape(-1).astype(bool) & sb.reshape(-1).astype(bool) & (fb < 1.0)
        # update our tracked points across all images
        tracked = [p[ok] for p in tracked]
        tracked.append(p_next[ok])
    tracked = [p.reshape(-1, 2) for p in tracked]
    return cv2, kps, np, pts, rgbs, selected, tracked


@app.cell(hide_code=True)
def _(kps, pts, rgbs, selected, tracked):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(1, len(selected), figsize=(5 * len(selected), 5))
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
    for _i, p in enumerate(selected):
        ax[_i].imshow(rgbs[_i])
        if len(pts[_i]):
            ax[_i].scatter(pts[_i][:, 0], pts[_i][:, 1], s=5, c="red")
        if len(tracked[_i]):
            ax[_i].scatter(tracked[_i][:, 0], tracked[_i][:, 1], s=7, c="blue")
        ax[_i].set_title(f"{p.name} ({len(kps[_i])}, {len(tracked[0])} tracked)")
        ax[_i].axis("off")
    ax[-1].legend(handles=legend, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig
    return (plt,)


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

    **Industry Standard Practical SfM System**

    COLMAP is the practical version of the SfM pipeline we have been discussing.

    Compared with affine Tomasi--Kanade, COLMAP:

    - works with **perspective projection**, not just affine projection
    - accepts **ordered or unordered** image collections
    - does not assume all images overlap
    - detects and matches features automatically
    - performs geometric verification, triangulation, and bundle adjustment

    In this section, we use **PyCOLMAP** for the actual COLMAP pipeline and small visual demos to explain each concept.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## COLMAP: Practical Structure from Motion

    COLMAP turns a set of unordered images into a **sparse 3D reconstruction** by solving for camera geometry and scene structure simultaneously.

    The full pipeline is:

    $$
    \underbrace{\mathcal{I}_1, \mathcal{I}_2, \ldots, \mathcal{I}_n}_{\text{images}}
    \xrightarrow{\text{extract}}
    \underbrace{\{(k_i, d_i)\}}_{\text{features}}
    \xrightarrow{\text{match}}
    \underbrace{\mathcal{C}_{ij}}_{\text{candidates}}
    \xrightarrow{\text{verify}}
    \underbrace{\mathcal{M}_{ij}}_{\text{inliers}}
    \xrightarrow{\text{triangulate}}
    \underbrace{\mathbf{X}_j \in \mathbb{R}^3}_{\text{3D points}}
    \xrightarrow{\text{refine}}
    \underbrace{\{\hat{P}_i, \hat{\mathbf{X}}_j\}}_{\text{reconstruction}}
    $$

    Each arrow is a distinct algorithmic step. We cover all five below, with both visual demos (OpenCV) and the real pipeline (PyCOLMAP).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calling COLMAP from Python

    PyCOLMAP exposes the full COLMAP pipeline as three function calls:

    ```python
    pycolmap.extract_features(database_path, image_dir)   # step 1
    pycolmap.match_exhaustive(database_path)               # step 2
    maps = pycolmap.incremental_mapping(                   # steps 3–5
        database_path, image_dir, output_dir
    )
    ```

    Internally, `incremental_mapping` registers images one at a time, triangulates new 3D points after each registration, and runs **bundle adjustment** to keep reprojection error low throughout.

    The surrounding cells use OpenCV to visualise each sub-step in isolation.
    """)
    return


@app.cell
def _():
    try:
        import pycolmap as pycolmap_module

        colmap_pycolmap_available = True
        colmap_pycolmap_error = None
        colmap_pycolmap_version = getattr(pycolmap_module, "__version__", "unknown")
    except Exception as _pycolmap_exc:
        pycolmap_module = None
        colmap_pycolmap_available = False
        colmap_pycolmap_error = str(_pycolmap_exc)
        colmap_pycolmap_version = None

    print("PyCOLMAP available:", colmap_pycolmap_available)
    print("PyCOLMAP version:", colmap_pycolmap_version)
    if not colmap_pycolmap_available:
        print("PyCOLMAP import error:", colmap_pycolmap_error)
        print("Install with: pip install pycolmap")
    return colmap_pycolmap_available, pycolmap_module


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## COLMAP demo dataset: Sacré-Cœur

    The initial SfM/Tomasi--Kanade section still uses the Dino sequence.

    For the COLMAP section, we switch to a more realistic unordered image collection:

    ```text
    data/sacre_coeur/*.jpg
    ```

    This is closer to how COLMAP is normally used: a folder of overlapping perspective images of the same scene.

    We use two Sacré-Cœur images for the visual feature/matching demos, and the full folder subset for the actual PyCOLMAP reconstruction.
    """)
    return


@app.cell(hide_code=True)
def _():
    from pathlib import Path as _ColmapPath

    _colmap_sacre_dir = _ColmapPath("data/building_front")

    colmap_image_paths = sorted(_colmap_sacre_dir.glob("*.jpg"))
    colmap_image_paths += sorted(_colmap_sacre_dir.glob("*.jpeg"))
    colmap_image_paths += sorted(_colmap_sacre_dir.glob("*.png"))
    colmap_image_paths += sorted(_colmap_sacre_dir.glob("*.JPG"))
    colmap_image_paths += sorted(_colmap_sacre_dir.glob("viff.*.ppm"))

    # Pick two images for the visual demos. Use nearby entries if possible,
    # otherwise fall back to the first two images.
    if len(colmap_image_paths) >= 3:
        colmap_img1_path = colmap_image_paths[0]
        colmap_img2_path = colmap_image_paths[1]
    elif len(colmap_image_paths) >= 2:
        colmap_img1_path = colmap_image_paths[0]
        colmap_img2_path = colmap_image_paths[1]
    else:
        colmap_img1_path = None
        colmap_img2_path = None

    print("Sacré-Cœur image folder:", _colmap_sacre_dir)
    print("Number of Sacré-Cœur images:", len(colmap_image_paths))
    print("COLMAP demo image 1:", colmap_img1_path)
    print("COLMAP demo image 2:", colmap_img2_path)

    if len(colmap_image_paths) < 2:
        print("Expected images in: data/sacre_coeur/*.jpg")
    return colmap_image_paths, colmap_img1_path, colmap_img2_path


@app.cell(hide_code=True)
def _(Image, colmap_image_paths, plt):
    _colmap_n_preview = min(8, len(colmap_image_paths))

    if _colmap_n_preview > 0:
        _colmap_cols = min(4, _colmap_n_preview)
        _colmap_rows = (_colmap_n_preview + _colmap_cols - 1) // _colmap_cols
        colmap_fig_dataset, _colmap_axes_dataset = plt.subplots(
            _colmap_rows,
            _colmap_cols,
            figsize=(4 * _colmap_cols, 3 * _colmap_rows),
        )

        if _colmap_n_preview == 1:
            _colmap_axes_flat = [_colmap_axes_dataset]
        else:
            _colmap_axes_flat = list(_colmap_axes_dataset.ravel())

        for _colmap_ax_i, _colmap_path_i in zip(
            _colmap_axes_flat, colmap_image_paths[:_colmap_n_preview]
        ):
            _colmap_img_i = Image.open(_colmap_path_i).convert("RGB")
            _colmap_ax_i.imshow(_colmap_img_i)
            _colmap_ax_i.set_title(_colmap_path_i.name, fontsize=9)
            _colmap_ax_i.axis("off")

        for _colmap_ax_i in _colmap_axes_flat[_colmap_n_preview:]:
            _colmap_ax_i.axis("off")

        colmap_fig_dataset.suptitle("Sacré-Cœur images used for the COLMAP section")
        colmap_fig_dataset.tight_layout()
    else:
        colmap_fig_dataset, _colmap_ax_dataset = plt.subplots(figsize=(7, 4))
        _colmap_ax_dataset.text(
            0.5,
            0.5,
            "No images found in data/sacre_coeur",
            ha="center",
            va="center",
        )
        _colmap_ax_dataset.axis("off")

    colmap_fig_dataset
    return


@app.cell(hide_code=True)
def _(Image, colmap_img1_path, colmap_img2_path, cv2, np):
    if colmap_img1_path is not None and colmap_img2_path is not None:
        colmap_img1_rgb = np.array(Image.open(colmap_img1_path).convert("RGB"))
        colmap_img2_rgb = np.array(Image.open(colmap_img2_path).convert("RGB"))

        colmap_img1_gray = cv2.cvtColor(colmap_img1_rgb, cv2.COLOR_RGB2GRAY)
        colmap_img2_gray = cv2.cvtColor(colmap_img2_rgb, cv2.COLOR_RGB2GRAY)
    else:
        colmap_img1_rgb = None
        colmap_img2_rgb = None
        colmap_img1_gray = None
        colmap_img2_gray = None
    return colmap_img1_gray, colmap_img1_rgb, colmap_img2_gray, colmap_img2_rgb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1 — Feature Extraction

    Each image $\mathcal{I}_i$ is mapped to a set of *keypoints* with associated descriptors:

    $$
    \mathcal{I}_i \;\longmapsto\; \bigl\{\,(k_{ij},\, \mathbf{d}_{ij})\bigr\}_{j=1}^{N_i}
    $$

    where $k_{ij} = (u, v, \sigma, \theta)$ encodes **location**, **scale** $\sigma$, and **orientation** $\theta$, and $\mathbf{d}_{ij} \in \mathbb{R}^{128}$ is the SIFT descriptor.

    **Why scale and orientation matter.** SIFT builds its descriptor in a *normalised patch* centred on $k_{ij}$: the patch is rotated by $-\theta$ and resized to a fixed scale. This makes $\mathbf{d}_{ij}$ invariant to in-plane rotation and scale change — the same physical point yields nearly identical descriptors across very different viewpoints.

    Circles in the plot below encode $\sigma$ (radius) and $\theta$ (tick direction).
    """)
    return


@app.cell(hide_code=True)
def _(colmap_img1_gray, colmap_img1_rgb, cv2, plt):
    if colmap_img1_gray is not None and colmap_img1_rgb is not None:
        try:
            _colmap_detector_feature = cv2.SIFT_create(nfeatures=500)
            colmap_detector_name = "SIFT"
        except Exception:
            _colmap_detector_feature = cv2.ORB_create(nfeatures=500)
            colmap_detector_name = "ORB"

        colmap_kp1, colmap_desc1 = _colmap_detector_feature.detectAndCompute(
            colmap_img1_gray,
            None,
        )

        _colmap_keypoint_vis = cv2.drawKeypoints(
            colmap_img1_rgb,
            colmap_kp1,
            None,
            color=(0, 255, 80),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        # Overdraw a filled dot at each center so small keypoints are visible
        for _kp in colmap_kp1:
            _cx, _cy = int(_kp.pt[0]), int(_kp.pt[1])
            cv2.circle(_colmap_keypoint_vis, (_cx, _cy), 8, (255, 80, 0), -1)

        colmap_fig_feature, _colmap_ax_feature = plt.subplots(figsize=(14, 10))
        _colmap_ax_feature.imshow(_colmap_keypoint_vis)
        _colmap_ax_feature.set_title(
            f"Step 1: Feature Extraction ({colmap_detector_name})"
        )
        _colmap_ax_feature.axis("off")
    else:
        colmap_detector_name = "None"
        colmap_kp1 = []
        colmap_desc1 = None

        colmap_fig_feature, _colmap_ax_feature = plt.subplots(figsize=(14, 10))
        _colmap_ax_feature.text(
            0.5,
            0.5,
            "No image loaded",
            ha="center",
            va="center",
        )
        _colmap_ax_feature.axis("off")

    print("Detector:", colmap_detector_name)
    print("Number of keypoints in image 1:", len(colmap_kp1))
    print("Descriptor shape:", None if colmap_desc1 is None else colmap_desc1.shape)

    colmap_fig_feature
    return colmap_desc1, colmap_detector_name, colmap_kp1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 — Feature Matching

    Given descriptors from two images, candidate correspondences are found by nearest-neighbour search in descriptor space:

    $$
    k_{1p} \;\leftrightarrow\; k_{2q}
    \quad \text{if} \quad
    \frac{\|\mathbf{d}_{1p} - \mathbf{d}_{2q}\|}{\|\mathbf{d}_{1p} - \mathbf{d}_{2q'}\|} < \tau
    $$

    This is **Lowe's ratio test**: a match is accepted only when the nearest neighbour is significantly closer than the second-nearest neighbour $q'$. A threshold of $\tau = 0.75$ is standard — it rejects ambiguous matches where two descriptors are similarly plausible.

    These are still *candidate* correspondences. Many will be outliers caused by repeated textures, reflections, or descriptor collisions. Geometric verification (next step) filters them.
    """)
    return


@app.cell(hide_code=True)
def _(
    colmap_desc1,
    colmap_detector_name,
    colmap_img1_rgb,
    colmap_img2_gray,
    colmap_img2_rgb,
    colmap_kp1,
    cv2,
    plt,
):
    if (
        colmap_img1_rgb is not None
        and colmap_img2_rgb is not None
        and colmap_img2_gray is not None
        and colmap_desc1 is not None
        and len(colmap_kp1) > 0
    ):
        if colmap_detector_name == "SIFT":
            _colmap_detector_match = cv2.SIFT_create(nfeatures=500)
            colmap_kp2, colmap_desc2 = _colmap_detector_match.detectAndCompute(
                colmap_img2_gray,
                None,
            )

            _colmap_matcher = cv2.BFMatcher(cv2.NORM_L2)
            _colmap_raw_knn_matches = _colmap_matcher.knnMatch(
                colmap_desc1,
                colmap_desc2,
                k=2,
            )

            colmap_matches = []
            for _colmap_pair in _colmap_raw_knn_matches:
                if len(_colmap_pair) == 2:
                    _colmap_m, _colmap_n = _colmap_pair
                    if _colmap_m.distance < 0.75 * _colmap_n.distance:
                        colmap_matches.append(_colmap_m)
        else:
            _colmap_detector_match = cv2.ORB_create(nfeatures=500)
            colmap_kp2, colmap_desc2 = _colmap_detector_match.detectAndCompute(
                colmap_img2_gray,
                None,
            )

            _colmap_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            colmap_matches = _colmap_matcher.match(colmap_desc1, colmap_desc2)
            colmap_matches = sorted(colmap_matches, key=lambda _m: _m.distance)

        colmap_matches = colmap_matches[:100]

        # Assign a unique color per match ranked by descriptor distance
        _n_show = min(40, len(colmap_matches))
        _sorted_matches = sorted(colmap_matches, key=lambda m: m.distance)[:_n_show]

        _colmap_match_vis = cv2.drawMatches(
            colmap_img1_rgb,
            colmap_kp1,
            colmap_img2_rgb,
            colmap_kp2,
            _sorted_matches,
            None,
            matchColor=(0, 255, 120),  # bright green lines
            singlePointColor=(80, 80, 80),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        # Overdraw thicker lines and endpoint dots manually for visibility
        _h, _w = colmap_img1_rgb.shape[:2]
        for _m in _sorted_matches:
            _p1 = tuple(map(int, colmap_kp1[_m.queryIdx].pt))
            _p2 = (
                int(colmap_kp2[_m.trainIdx].pt[0]) + _w,
                int(colmap_kp2[_m.trainIdx].pt[1]),
            )
            cv2.line(_colmap_match_vis, _p1, _p2, (0, 230, 255), 4)  # cyan, thickness=2
            cv2.circle(_colmap_match_vis, _p1, 5, (255, 80, 0), -1)  # orange dot left
            cv2.circle(_colmap_match_vis, _p2, 5, (255, 80, 0), -1)  # orange dot right

        colmap_fig_match, _colmap_ax_match = plt.subplots(figsize=(24, 10))
        _colmap_ax_match.imshow(_colmap_match_vis)
        _colmap_ax_match.set_title(
            f"Step 2: Candidate Feature Matches ({len(colmap_matches)} shown/kept)"
        )
        _colmap_ax_match.axis("off")
    else:
        colmap_kp2 = []
        colmap_desc2 = None
        colmap_matches = []

        colmap_fig_match, _colmap_ax_match = plt.subplots(figsize=(24, 10))
        _colmap_ax_match.text(
            0.5,
            0.5,
            "Feature matching unavailable\nRun feature extraction first.",
            ha="center",
            va="center",
        )
        _colmap_ax_match.axis("off")

    print("Keypoints in image 1:", len(colmap_kp1))
    print("Keypoints in image 2:", len(colmap_kp2))
    print("Candidate matches:", len(colmap_matches))

    colmap_fig_match
    return colmap_kp2, colmap_matches


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 — Geometric Verification

    Two cameras viewing the same 3D point must satisfy the **epipolar constraint**. For corresponding image points $\mathbf{x} \in \mathcal{I}_1$ and $\mathbf{x}' \in \mathcal{I}_2$:

    $$
    \mathbf{x}'^{\top} \mathbf{F} \mathbf{x} = 0
    $$

    $\mathbf{F} \in \mathbb{R}^{3 \times 3}$ is the **fundamental matrix** — rank 2, seven degrees of freedom. It encodes the complete relative geometry between the two cameras without knowing their intrinsics.

    When intrinsics $\mathbf{K}$ are known, $\mathbf{F}$ factors into the **essential matrix** $\mathbf{E} = \mathbf{K}'^{\top}\mathbf{F}\mathbf{K}$, which has only five degrees of freedom and decomposes into a relative rotation and translation:

    $$
    \mathbf{E} = [\mathbf{t}]_\times \mathbf{R}, \qquad \mathbf{E} = \mathbf{U}\,\text{diag}(1,1,0)\,\mathbf{V}^\top
    $$

    **RANSAC** estimates $\mathbf{F}$ (or $\mathbf{E}$) robustly: it repeatedly samples the minimum number of point pairs (7 for $\mathbf{F}$, 5 for $\mathbf{E}$), fits the matrix, and counts inliers whose epipolar distance falls below a pixel threshold $\epsilon$.
    """)
    return


@app.cell(hide_code=True)
def _(colmap_kp1, colmap_kp2, colmap_matches, cv2, np):
    if len(colmap_matches) >= 8:
        colmap_pts1 = np.float32([colmap_kp1[_m.queryIdx].pt for _m in colmap_matches])
        colmap_pts2 = np.float32([colmap_kp2[_m.trainIdx].pt for _m in colmap_matches])

        colmap_F, colmap_inlier_mask = cv2.findFundamentalMat(
            colmap_pts1,
            colmap_pts2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=1.0,
            confidence=0.99,
        )

        if colmap_inlier_mask is not None:
            colmap_inlier_mask = colmap_inlier_mask.ravel().astype(bool)
            colmap_inlier_matches = [
                _m for _m, _keep in zip(colmap_matches, colmap_inlier_mask) if _keep
            ]
        else:
            colmap_inlier_matches = []
    else:
        colmap_pts1 = np.empty((0, 2))
        colmap_pts2 = np.empty((0, 2))
        colmap_F = None
        colmap_inlier_mask = np.array([], dtype=bool)
        colmap_inlier_matches = []

    print("Candidate matches:", len(colmap_matches))
    print("Geometrically verified inliers:", len(colmap_inlier_matches))
    print("Estimated fundamental matrix:")
    print(colmap_F)
    return colmap_F, colmap_inlier_matches


@app.cell(hide_code=True)
def _(
    colmap_img1_rgb,
    colmap_img2_rgb,
    colmap_inlier_matches,
    colmap_kp1,
    colmap_kp2,
    cv2,
    plt,
):
    if colmap_img1_rgb is not None and colmap_img2_rgb is not None:
        import numpy as _np_match

        _n_show = min(40, len(colmap_inlier_matches))
        _sorted_inlier_matches = sorted(
            colmap_inlier_matches, key=lambda m: m.distance
        )[:_n_show]

        _colmap_verified_vis = cv2.drawMatches(
            colmap_img1_rgb,
            colmap_kp1,
            colmap_img2_rgb,
            colmap_kp2,
            _sorted_inlier_matches,
            None,
            matchColor=(0, 230, 255),  # cyan lines
            singlePointColor=(80, 80, 80),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        # Overdraw thicker lines manually
        _h, _w = colmap_img1_rgb.shape[:2]
        for _m in _sorted_inlier_matches:
            _p1 = tuple(map(int, colmap_kp1[_m.queryIdx].pt))
            _p2 = (
                int(colmap_kp2[_m.trainIdx].pt[0]) + _w,
                int(colmap_kp2[_m.trainIdx].pt[1]),
            )
            cv2.line(_colmap_verified_vis, _p1, _p2, (0, 230, 255), 2)

        colmap_fig_verify, _colmap_ax_verify = plt.subplots(figsize=(24, 10))
        _colmap_ax_verify.imshow(_colmap_verified_vis)
        _colmap_ax_verify.set_title(
            f"Step 3: Geometrically Verified Matches ({len(colmap_inlier_matches)} inliers)",
            fontsize=18,
        )
        _colmap_ax_verify.axis("off")
    else:
        colmap_fig_verify, _colmap_ax_verify = plt.subplots(figsize=(24, 10))
        _colmap_ax_verify.text(0.5, 0.5, "No images loaded", ha="center", va="center")
        _colmap_ax_verify.axis("off")

    colmap_fig_verify
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4 — Triangulation

    Given camera projection matrices $\mathbf{P}_1, \mathbf{P}_2$ and verified 2D correspondences $(\mathbf{x}_1, \mathbf{x}_2)$, we recover a 3D point $\mathbf{X}$ from the pair of ray equations:

    $$
    \mathbf{x}_1 \times (\mathbf{P}_1 \mathbf{X}) = \mathbf{0}, \qquad
    \mathbf{x}_2 \times (\mathbf{P}_2 \mathbf{X}) = \mathbf{0}
    $$

    Stacking rows from each cross-product gives an overdetermined linear system $\mathbf{A}\mathbf{X} = \mathbf{0}$ (the **DLT**). The solution minimising $\|\mathbf{A}\mathbf{X}\|$ subject to $\|\mathbf{X}\|=1$ is the last right singular vector of $\mathbf{A}$:

    $$
    \hat{\mathbf{X}} = \arg\min_{\mathbf{X}} \|\mathbf{A}\mathbf{X}\|_2
    $$

    Two rays in $\mathbb{R}^3$ generically *skew* — they do not intersect exactly due to noise. The DLT solution minimises the algebraic residual; **optimal triangulation** minimises the reprojection error directly, which is a non-linear problem solved via the Sampson approximation.
    """)
    return


@app.cell(hide_code=True)
def _(
    colmap_F,
    colmap_img1_gray,
    colmap_inlier_matches,
    colmap_kp1,
    colmap_kp2,
    cv2,
    np,
):
    if (
        colmap_F is not None
        and len(colmap_inlier_matches) >= 8
        and colmap_img1_gray is not None
    ):
        _colmap_h, _colmap_w = colmap_img1_gray.shape

        colmap_focal = 0.9 * max(_colmap_h, _colmap_w)
        colmap_K = np.array(
            [
                [colmap_focal, 0, _colmap_w / 2],
                [0, colmap_focal, _colmap_h / 2],
                [0, 0, 1],
            ],
            dtype=float,
        )

        colmap_pts1_in = np.float32(
            [colmap_kp1[_m.queryIdx].pt for _m in colmap_inlier_matches]
        )
        colmap_pts2_in = np.float32(
            [colmap_kp2[_m.trainIdx].pt for _m in colmap_inlier_matches]
        )

        colmap_E, colmap_E_mask = cv2.findEssentialMat(
            colmap_pts1_in,
            colmap_pts2_in,
            colmap_K,
            method=cv2.RANSAC,
            threshold=1.0,
            prob=0.999,
        )

        if colmap_E is not None:
            _colmap_pose_ok, colmap_R, colmap_t, colmap_pose_mask = cv2.recoverPose(
                colmap_E,
                colmap_pts1_in,
                colmap_pts2_in,
                colmap_K,
            )

            _colmap_P1 = colmap_K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            _colmap_P2 = colmap_K @ np.hstack([colmap_R, colmap_t])

            _colmap_points4D = cv2.triangulatePoints(
                _colmap_P1,
                _colmap_P2,
                colmap_pts1_in.T,
                colmap_pts2_in.T,
            )

            colmap_points3D = (_colmap_points4D[:3] / _colmap_points4D[3]).T
        else:
            colmap_R = np.eye(3)
            colmap_t = np.zeros((3, 1))
            colmap_pose_mask = None
            colmap_points3D = np.empty((0, 3))
    else:
        colmap_K = None
        colmap_E = None
        colmap_E_mask = None
        colmap_R = np.eye(3)
        colmap_t = np.zeros((3, 1))
        colmap_pose_mask = None
        colmap_pts1_in = np.empty((0, 2))
        colmap_pts2_in = np.empty((0, 2))
        colmap_points3D = np.empty((0, 3))

    print("Triangulated 3D points:", colmap_points3D.shape)
    return (
        colmap_K,
        colmap_R,
        colmap_points3D,
        colmap_pts1_in,
        colmap_pts2_in,
        colmap_t,
    )


@app.cell(hide_code=True)
def _(colmap_points3D, np, plt):
    colmap_fig_tri = plt.figure(figsize=(24, 10))
    _colmap_ax_tri = colmap_fig_tri.add_subplot(111, projection="3d")

    if len(colmap_points3D) > 0:
        _colmap_pts3d_vis = colmap_points3D.copy()
        _colmap_finite = np.isfinite(_colmap_pts3d_vis).all(axis=1)
        _colmap_pts3d_vis = _colmap_pts3d_vis[_colmap_finite]

        if len(_colmap_pts3d_vis) > 0:
            _colmap_norms = np.linalg.norm(_colmap_pts3d_vis, axis=1)
            _colmap_keep = _colmap_norms < np.percentile(_colmap_norms, 90)
            _colmap_pts3d_vis = _colmap_pts3d_vis[_colmap_keep]

            _colmap_ax_tri.scatter(
                _colmap_pts3d_vis[:, 0],
                _colmap_pts3d_vis[:, 1],
                _colmap_pts3d_vis[:, 2],
                s=10,
            )
            _colmap_ax_tri.set_title("Step 4: Triangulated Sparse 3D Points")
            _colmap_ax_tri.set_xlabel("X")
            _colmap_ax_tri.set_ylabel("Y")
            _colmap_ax_tri.set_zlabel("Z")
        else:
            _colmap_ax_tri.text(0.5, 0.5, 0.5, "No stable 3D points")
    else:
        _colmap_ax_tri.text(0.5, 0.5, 0.5, "Triangulation unavailable")

    colmap_fig_tri.tight_layout()
    colmap_fig_tri
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5 — Bundle Adjustment

    All previous steps introduce drift. Bundle adjustment (BA) is the gold-standard joint refinement: it simultaneously optimises **all** camera poses, **all** intrinsic parameters, and **all** 3D point positions by minimising total reprojection error:

    $$
    \min_{\{\mathbf{R}_i, \mathbf{t}_i, \mathbf{K}_i\},\, \{\mathbf{X}_j\}}
    \sum_{(i,j) \in \mathcal{V}}
    \rho\!\left(\left\|\mathbf{x}_{ij} - \pi(\mathbf{K}_i,\, \mathbf{R}_i,\, \mathbf{t}_i,\, \mathbf{X}_j)\right\|^2\right)
    $$

    where $\pi$ is the perspective projection function, $\mathcal{V}$ is the set of visible point-image pairs, and $\rho$ is a **robust kernel** (e.g. Cauchy or Huber) that downweights residuals from outlier observations.

    The problem is solved with the **Levenberg–Marquardt** algorithm. The key computational insight is that the Jacobian has a *sparse block structure*: each 3D point only appears in the cameras that observe it. Exploiting this with the **Schur complement** reduces the $O((6m + 3n)^3)$ naive solve to roughly $O(m^3)$, where $m$ is the number of cameras and $n \gg m$ is the number of 3D points.

    COLMAP runs BA after every image registration, keeping the growing reconstruction numerically healthy.
    """)
    return


@app.cell(hide_code=True)
def _(
    colmap_K,
    colmap_R,
    colmap_points3D,
    colmap_pts1_in,
    colmap_pts2_in,
    colmap_t,
    cv2,
    np,
    plt,
):
    if colmap_K is not None and len(colmap_points3D) > 0:
        _colmap_n = min(
            len(colmap_points3D),
            len(colmap_pts1_in),
            len(colmap_pts2_in),
        )

        _colmap_X = colmap_points3D[:_colmap_n]
        _colmap_obs1 = colmap_pts1_in[:_colmap_n]
        _colmap_obs2 = colmap_pts2_in[:_colmap_n]

        _colmap_rvec1 = np.zeros((3, 1))
        _colmap_tvec1 = np.zeros((3, 1))
        _colmap_proj1, _ = cv2.projectPoints(
            _colmap_X,
            _colmap_rvec1,
            _colmap_tvec1,
            colmap_K,
            None,
        )
        _colmap_proj1 = _colmap_proj1.reshape(-1, 2)

        _colmap_rvec2, _ = cv2.Rodrigues(colmap_R)
        _colmap_proj2, _ = cv2.projectPoints(
            _colmap_X,
            _colmap_rvec2,
            colmap_t,
            colmap_K,
            None,
        )
        _colmap_proj2 = _colmap_proj2.reshape(-1, 2)

        _colmap_err1 = np.linalg.norm(_colmap_proj1 - _colmap_obs1, axis=1)
        _colmap_err2 = np.linalg.norm(_colmap_proj2 - _colmap_obs2, axis=1)

        colmap_reprojection_errors = np.concatenate([_colmap_err1, _colmap_err2])
        colmap_mean_reprojection_error = float(np.mean(colmap_reprojection_errors))
    else:
        colmap_reprojection_errors = np.array([])
        colmap_mean_reprojection_error = None

    colmap_fig_ba, _colmap_ax_ba = plt.subplots(figsize=(12, 6))

    if len(colmap_reprojection_errors) > 0:
        _colmap_ax_ba.hist(colmap_reprojection_errors, bins=30)
        _colmap_ax_ba.axvline(
            colmap_mean_reprojection_error,
            linestyle="--",
            label="Mean error",
        )
        _colmap_ax_ba.set_title("Step 5: Reprojection Error")
        _colmap_ax_ba.set_xlabel("Reprojection error in pixels")
        _colmap_ax_ba.set_ylabel("Count")
        _colmap_ax_ba.legend()
    else:
        _colmap_ax_ba.text(
            0.5,
            0.5,
            "No reprojection errors available",
            ha="center",
            va="center",
        )
        _colmap_ax_ba.axis("off")

    print("Mean reprojection error:", colmap_mean_reprojection_error)

    colmap_fig_ba.tight_layout()
    colmap_fig_ba
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Actual PyCOLMAP Reconstruction Demo

    The previous cells visualize the individual ideas.

    This cell runs the actual PyCOLMAP pipeline if PyCOLMAP is installed:

    1. `extract_features`
    2. `match_exhaustive`
    3. `incremental_mapping`

    This is the closest part of the notebook to real COLMAP.

    Set the image folder and max image count, then press the button to run it.

    By default, the folder is:

    ```text
    data/sacre_coeur
    ```

    It may take a little time depending on the image count and machine.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    colmap_dataset_path_input = mo.ui.text(
        value="data/sacre_coeur",
        label="Image folder",
        placeholder="Path to images",
    )
    colmap_max_images_input = mo.ui.number(
        value=30,
        start=2,
        step=1,
        label="Max images",
    )
    colmap_run_pycolmap_button = mo.ui.button(
        value=0,
        on_click=lambda value: value + 1,
        label="Run actual PyCOLMAP sparse reconstruction",
        kind="success",
    )

    mo.vstack(
        [
            mo.hstack([colmap_dataset_path_input, colmap_max_images_input]),
            colmap_run_pycolmap_button,
        ]
    )
    return (
        colmap_dataset_path_input,
        colmap_max_images_input,
        colmap_run_pycolmap_button,
    )


@app.cell(hide_code=True)
def _(colmap_dataset_path_input, colmap_max_images_input):
    from pathlib import Path as _ColmapPath
    import shutil as _colmap_shutil

    colmap_source_dir = _ColmapPath(colmap_dataset_path_input.value).expanduser()
    colmap_source_image_paths = []

    if colmap_source_dir.exists():
        for _pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.ppm"):
            colmap_source_image_paths += sorted(colmap_source_dir.glob(_pattern))
    else:
        print("Input image folder does not exist:", colmap_source_dir)

    colmap_max_images = int(colmap_max_images_input.value)

    colmap_pycolmap_workspace = _ColmapPath("pycolmap_sacre_coeur_workspace")
    colmap_pycolmap_image_dir = colmap_pycolmap_workspace / "images"
    colmap_pycolmap_database_path = colmap_pycolmap_workspace / "database.db"
    colmap_pycolmap_sparse_dir = colmap_pycolmap_workspace / "sparse"

    # Recreate the image folder each run so stale files from another dataset do not remain.
    colmap_pycolmap_workspace.mkdir(exist_ok=True)
    if colmap_pycolmap_image_dir.exists():
        _colmap_shutil.rmtree(colmap_pycolmap_image_dir)
    colmap_pycolmap_image_dir.mkdir(exist_ok=True)
    colmap_pycolmap_sparse_dir.mkdir(exist_ok=True)

    # Use a manageable subset for a live demo.
    colmap_pycolmap_images_used = colmap_source_image_paths[:colmap_max_images]

    for _colmap_src in colmap_pycolmap_images_used:
        _colmap_dst = colmap_pycolmap_image_dir / _colmap_src.name
        _colmap_shutil.copy2(_colmap_src, _colmap_dst)

    print("PyCOLMAP workspace:", colmap_pycolmap_workspace)
    print("Source image directory:", colmap_source_dir)
    print("Source images found:", len(colmap_source_image_paths))
    print("PyCOLMAP image directory:", colmap_pycolmap_image_dir)
    print(
        "Images copied for reconstruction:",
        len(list(colmap_pycolmap_image_dir.glob("*"))),
    )
    return (
        colmap_pycolmap_database_path,
        colmap_pycolmap_image_dir,
        colmap_pycolmap_images_used,
        colmap_pycolmap_sparse_dir,
    )


@app.cell(hide_code=True)
def _(
    colmap_pycolmap_available,
    colmap_pycolmap_database_path,
    colmap_pycolmap_image_dir,
    colmap_pycolmap_images_used,
    colmap_pycolmap_sparse_dir,
    colmap_run_pycolmap_button,
    np,
    pycolmap_module,
):
    colmap_pycolmap_status = "not_run"
    colmap_pycolmap_reconstruction = None
    colmap_pycolmap_points_xyz = np.empty((0, 3))
    colmap_pycolmap_num_images = 0
    colmap_pycolmap_num_points3D = 0

    if colmap_run_pycolmap_button.value == 0:
        print("PyCOLMAP reconstruction not run yet. Press the button above to run it.")
    elif len(colmap_pycolmap_images_used) < 2:
        colmap_pycolmap_status = "not_enough_images"
        print("Need at least 2 images in the selected folder.")
    elif not colmap_pycolmap_available:
        colmap_pycolmap_status = "pycolmap_not_installed"
        print("PyCOLMAP is not installed. Install with: pip install pycolmap")
    else:
        try:
            import shutil as _colmap_shutil

            if colmap_pycolmap_database_path.exists():
                colmap_pycolmap_database_path.unlink()

            if colmap_pycolmap_sparse_dir.exists():
                _colmap_shutil.rmtree(colmap_pycolmap_sparse_dir)
            colmap_pycolmap_sparse_dir.mkdir(exist_ok=True)

            print("Running pycolmap.extract_features(...) with more SIFT features")

            _colmap_feature_options = pycolmap_module.FeatureExtractionOptions()
            _colmap_feature_options.max_image_size = 1200
            _colmap_feature_options.num_threads = 8
            _colmap_feature_options.use_gpu = True

            _colmap_feature_options.sift.max_num_features = 4096
            _colmap_feature_options.sift.peak_threshold = 0.006
            _colmap_feature_options.sift.edge_threshold = 10.0
            _colmap_feature_options.use_gpu = True

            pycolmap_module.extract_features(
                colmap_pycolmap_database_path,
                colmap_pycolmap_image_dir,
                extraction_options=_colmap_feature_options,
            )

            print("Running pycolmap.match_exhaustive(...)")
            pycolmap_module.match_exhaustive(colmap_pycolmap_database_path)

            print("Running pycolmap.incremental_mapping(...)")
            _colmap_maps = pycolmap_module.incremental_mapping(
                colmap_pycolmap_database_path,
                colmap_pycolmap_image_dir,
                colmap_pycolmap_sparse_dir,
            )

            if _colmap_maps:
                _colmap_keys = list(_colmap_maps.keys())
                _colmap_first_key = sorted(_colmap_keys)[0]
                colmap_pycolmap_reconstruction = _colmap_maps[_colmap_first_key]

                colmap_pycolmap_num_images = len(colmap_pycolmap_reconstruction.images)
                colmap_pycolmap_num_points3D = len(
                    colmap_pycolmap_reconstruction.points3D
                )

                _colmap_xyz_list = []
                for _colmap_point in colmap_pycolmap_reconstruction.points3D.values():
                    _colmap_xyz_list.append(_colmap_point.xyz)

                if _colmap_xyz_list:
                    colmap_pycolmap_points_xyz = np.array(_colmap_xyz_list)

                colmap_pycolmap_status = "success"
            else:
                colmap_pycolmap_status = "no_reconstruction"

        except Exception as _colmap_pycolmap_exc:
            colmap_pycolmap_status = (
                f"error: {type(_colmap_pycolmap_exc).__name__}: {_colmap_pycolmap_exc}"
            )

    print("PyCOLMAP status:", colmap_pycolmap_status)
    print("Registered images:", colmap_pycolmap_num_images)
    print("Sparse 3D points:", colmap_pycolmap_num_points3D)
    return colmap_pycolmap_reconstruction, colmap_pycolmap_status


@app.cell(hide_code=True)
def _(colmap_pycolmap_reconstruction, colmap_pycolmap_status, mo, np):
    # interactive rotation/zoom, RGB-colored sparse points, and camera centers.
    try:
        import plotly.graph_objects as _colmap_go

        _colmap_plotly_available = True
        _colmap_plotly_error = None
    except Exception as _colmap_plotly_exc:
        _colmap_plotly_available = False
        _colmap_plotly_error = str(_colmap_plotly_exc)

    if not _colmap_plotly_available:
        colmap_fig_pycolmap = mo.md(
            f"""
            !!! warning "Plotly is not installed"

                Install it with:

                ```bash
                uv add plotly
                ```

                Import error: `{_colmap_plotly_error}`
            """
        )
    elif colmap_pycolmap_status != "success" or colmap_pycolmap_reconstruction is None:
        colmap_fig_pycolmap = mo.md(
            f"""
            !!! warning "PyCOLMAP reconstruction unavailable"

                Status: `{colmap_pycolmap_status}`
            """
        )
    else:
        _colmap_xyz_list = []
        _colmap_color_list = []

        for _colmap_point in colmap_pycolmap_reconstruction.points3D.values():
            _colmap_xyz = np.asarray(_colmap_point.xyz, dtype=float)
            if not np.isfinite(_colmap_xyz).all():
                continue

            _colmap_color = getattr(_colmap_point, "color", None)
            if _colmap_color is None:
                _colmap_color = getattr(_colmap_point, "rgb", None)

            if _colmap_color is not None:
                _colmap_rgb = np.asarray(_colmap_color).astype(int).clip(0, 255)
                _colmap_color_str = (
                    f"rgb({_colmap_rgb[0]},{_colmap_rgb[1]},{_colmap_rgb[2]})"
                )
            else:
                _colmap_color_str = "rgb(40,120,220)"

            _colmap_xyz_list.append(_colmap_xyz)
            _colmap_color_list.append(_colmap_color_str)

        if _colmap_xyz_list:
            _colmap_points_xyz = np.vstack(_colmap_xyz_list)
            _colmap_colors = np.array(_colmap_color_list, dtype=object)

            # Remove extreme outliers for a cleaner presentation view.
            _colmap_center = np.median(_colmap_points_xyz, axis=0)
            _colmap_dist = np.linalg.norm(_colmap_points_xyz - _colmap_center, axis=1)
            _colmap_keep = _colmap_dist < np.percentile(_colmap_dist, 98)
            _colmap_points_xyz = _colmap_points_xyz[_colmap_keep]
            _colmap_colors = _colmap_colors[_colmap_keep]
        else:
            _colmap_points_xyz = np.empty((0, 3))
            _colmap_colors = []

        _colmap_camera_centers = []
        _colmap_camera_names = []

        for _colmap_image in colmap_pycolmap_reconstruction.images.values():
            try:
                _colmap_has_pose = getattr(_colmap_image, "has_pose", True)
                if callable(_colmap_has_pose):
                    _colmap_has_pose = _colmap_has_pose()
                if not _colmap_has_pose:
                    continue

                _colmap_cam_from_world = getattr(_colmap_image, "cam_from_world", None)
                if callable(_colmap_cam_from_world):
                    _colmap_cam_from_world = _colmap_cam_from_world()

                _colmap_world_from_cam = _colmap_cam_from_world.inverse()
                _colmap_center_xyz = np.asarray(
                    _colmap_world_from_cam.translation, dtype=float
                )
            except Exception:
                # Older pycolmap versions expose a direct projection_center method.
                try:
                    _colmap_center_xyz = np.asarray(
                        _colmap_image.projection_center(), dtype=float
                    )
                except Exception:
                    continue

            if np.isfinite(_colmap_center_xyz).all():
                _colmap_camera_centers.append(_colmap_center_xyz)
                _colmap_camera_names.append(getattr(_colmap_image, "name", "camera"))

        if _colmap_camera_centers:
            _colmap_camera_centers = np.vstack(_colmap_camera_centers)
        else:
            _colmap_camera_centers = np.empty((0, 3))

        colmap_fig_pycolmap = _colmap_go.Figure()

        if len(_colmap_points_xyz) > 0:
            colmap_fig_pycolmap.add_trace(
                _colmap_go.Scatter3d(
                    x=_colmap_points_xyz[:, 0],
                    y=_colmap_points_xyz[:, 1],
                    z=_colmap_points_xyz[:, 2],
                    mode="markers",
                    name="Sparse 3D points",
                    marker=dict(
                        size=4,
                        color=_colmap_colors,
                        opacity=0.85,
                    ),
                    hoverinfo="skip",
                )
            )

        if len(_colmap_camera_centers) > 0:
            colmap_fig_pycolmap.add_trace(
                _colmap_go.Scatter3d(
                    x=_colmap_camera_centers[:, 0],
                    y=_colmap_camera_centers[:, 1],
                    z=_colmap_camera_centers[:, 2],
                    mode="markers+lines",
                    name="Registered cameras",
                    marker=dict(size=5, color="red", symbol="diamond"),
                    line=dict(color="red", width=3),
                    text=_colmap_camera_names,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        colmap_fig_pycolmap.update_layout(
            title=(
                "Actual PyCOLMAP Sparse Reconstruction "
                f"({_colmap_points_xyz.shape[0]} points, "
                f"{_colmap_camera_centers.shape[0]} cameras)"
            ),
            width=1300,
            height=1000,
            scene=dict(
                aspectmode="data",
                dragmode="orbit",
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                xaxis=dict(showbackground=True, backgroundcolor="rgb(245,245,245)"),
                yaxis=dict(showbackground=True, backgroundcolor="rgb(245,245,245)"),
                zaxis=dict(showbackground=True, backgroundcolor="rgb(245,245,245)"),
            ),
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=0, r=0, t=50, b=0),
        )

    colmap_fig_pycolmap
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## COLMAP Summary

    | Step | What it does | Demo in this notebook |
    |---|---|---|
    | Feature extraction | Detect repeatable local features | Keypoint visualization |
    | Feature matching | Match descriptors across images | Candidate match visualization |
    | Geometric verification | Reject matches that violate camera geometry | RANSAC inliers using the fundamental matrix |
    | Triangulation | Estimate 3D points from verified matches | Two-view sparse point demo |
    | Bundle adjustment | Refine cameras and 3D points | Reprojection error objective |
    | PyCOLMAP | Runs real COLMAP pipeline from Python | `extract_features`, `match_exhaustive`, `incremental_mapping` |

    The OpenCV cells explain the concepts visually.

    The PyCOLMAP cell is the actual COLMAP-based reconstruction pipeline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## VGGT [Anantajit]

    **Outline**

    1. What are the limitations for COLMAP (slow)

    2. Overview of VGGT

    3. Live Demo

    4. Weaknesses
    """)
    return


if __name__ == "__main__":
    app.run()
