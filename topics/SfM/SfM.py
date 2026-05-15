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
            mo.image(
                Image.open(current).convert("RGB"), width="50%", rounded=True
            ),
        ]
    )
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
def _(frame_paths):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image as _Image
    import cv2
    from matplotlib.lines import Line2D

    # extract test images
    selected = [frame_paths[i] for i in (7, 8, 9)]
    rgbs = [np.array(_Image.open(p).convert("RGB")) for p in selected]
    grays = [cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) for rgb in rgbs]

    # extract sift keypoints
    sift = cv2.SIFT_create(nfeatures=400, contrastThreshold=0.08)
    kps = [sift.detectAndCompute(g, None)[0] for g in grays]
    pts = [np.array([kp.pt for kp in ks], dtype=np.float32) for ks in kps]

    tracked = [pts[0].reshape(-1, 1, 2)]
    for i in range(1, len(grays)):
        p_prev = tracked[-1]
        p_next, s, _ = cv2.calcOpticalFlowPyrLK(
            grays[i - 1], grays[i], p_prev, None
        )
        p_back, sb, _ = cv2.calcOpticalFlowPyrLK(
            grays[i], grays[i - 1], p_next, None
        )
        fb = np.linalg.norm((p_back - p_prev).reshape(-1, 2), axis=1)
        ok = s.reshape(-1).astype(bool) & sb.reshape(-1).astype(bool) & (fb < 1.0)
        tracked = [p[ok] for p in tracked]
        tracked.append(p_next[ok])
    tracked = [p.reshape(-1, 2) for p in tracked]

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
    for i, p in enumerate(selected):
        ax[i].imshow(rgbs[i])
        if len(pts[i]):
            ax[i].scatter(pts[i][:, 0], pts[i][:, 1], s=5, c="red")
        if len(tracked[i]):
            ax[i].scatter(tracked[i][:, 0], tracked[i][:, 1], s=7, c="blue")
        ax[i].set_title(f"{p.name} ({len(kps[i])}, {len(tracked[0])} tracked)")
        ax[i].axis("off")
    ax[-1].legend(handles=legend, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig
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
    ## Papers (options)

    - Compare modern methods against classical affine SfM assumptions.
    - Focus on what replaces tracking/factorization in learned pipelines.
    - Track tradeoffs: robustness, scale, priors, and compute.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## VGGT / MapAnything / VGG-SfM / MASt3R

    - Pick one as the primary baseline and one as a contrastive alternative.
    - Extract each method's input assumptions and output representations.
    - Align evaluation criteria to the same datasets and failure cases.
    """)
    return


if __name__ == "__main__":
    app.run()
