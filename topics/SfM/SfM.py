import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", layout_file="layouts/SfM.slides.json")


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
            mo.image(
                Image.open(current).convert("RGB"), width="50%", rounded=True
            ),
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
        p_next, s, _ = cv2.calcOpticalFlowPyrLK(
            grays[_i - 1], grays[_i], p_prev, None
        )
        # find optical flow from next frame and use it to track our curr keypoints
        p_back, sb, _ = cv2.calcOpticalFlowPyrLK(
            grays[_i], grays[_i - 1], p_next, None
        )
        # only keep the keypoints which are within 1 pixel of the tracks in both directions
        fb = np.linalg.norm((p_back - p_prev).reshape(-1, 2), axis=1)
        ok = s.reshape(-1).astype(bool) & sb.reshape(-1).astype(bool) & (fb < 1.0)
        # update our tracked points across all images
        tracked = [p[ok] for p in tracked]
        tracked.append(p_next[ok])
    tracked = [p.reshape(-1, 2) for p in tracked]
    return kps, pts, rgbs, selected, tracked


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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Affine SfM
    We've tracked P feature points across F images $(x_{fp}, y_{fp})$ for frame f and point p. We want spatial information $X_p$,$Y_p$,$Z_p$

    Under an affine camera model (weak perspective) then we get something like $x = AX + t$


    $$ x_{fp} = m_{f}^T \begin{bmatrix}
    X_{p} \\
    Y_{p} \\
    Z_{p}
    \end{bmatrix} + t_f $$

    Where $m_f = \in \mathbb{R^{1 \times 3}}$ is from camera.

    This approximation is good if depth variation is small compared to distance from camera (approximately linear in Z). Will see later that for more general SfM, affine approximation is not good.


    <!--
    - Build \(W \in \mathbb{R}^{2F 	imes P}\) from tracked \((x, y)\) coordinates.
    - Each frame contributes two rows (x-row and y-row).
    - Columns represent a single 3D point trajectory across frames.
    -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Measurement Matrix

    For one frame we have
    $$
    \begin{bmatrix}
    x_{f1} & x_{f2} & \cdots & x_{fP}  \\
    y_{f1} & y_{f2} & \cdots & y_{fP}
    \end{bmatrix}
    = \begin{bmatrix}
    a_{f}^T  \\
    b_{f}^T
    \end{bmatrix}
    \begin{bmatrix}
    X_{1} & X_{2} & \cdots & X_P \\
    \end{bmatrix}
    +
    \begin{bmatrix}
    t_{xf}  \\
    t_{yf} \\
    \end{bmatrix} \mathbb{I}^T
    $$

    We **center** by subtracting the frame's centroid from all tracked points. This removes per-frame translation from the affine model and is crucial later.

    This allows us to write a matrix for this frame $W_f$ as

    $$
    W_f =
    \begin{bmatrix}
    a_{f}^T  \\
    b_{f}^T \\
    \end{bmatrix} S
    $$

    Where S is all the P 3D points.

    We stack all frames vertically and let $M = \begin{bmatrix}
    a_{1}^T  \\
    b_{1}^T \\
    a_{2}^T  \\
    \vdots
    \end{bmatrix} S$

    Then the measurement matrix $W = MS$, $W \in \mathbb{R}^{2F\times P}, M \in \mathbb{R}^{2F \times 3}, S \in \mathbb{R}^{3 \times P}$.
    Each image coordinate is linear function of 3D point.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rank and Factorization

    Tomasi and Kanade (1992) observed that under affine projection, the centered mesaurement matrix has at *most* rank 3. Our derivation holds then up to affine ambigutity.

    From SVD, we have $W = U \Sigma V^T$. Tomasi and Kanade showed that the factorization will always connect motion and 3D shape, that is, for rank 3 approximation:

    $$M = U_3 \Sigma_{3}^{1/2},  S = \Sigma_{3}^{1/2} V_{3}^T $$

    $$W = MS \in \mathbb{R}^{2F \times P}$$

    In the end, each column of $W$ is a 3P point tracked over time, and each pair of rows is a camera frame.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
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


@app.cell
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
