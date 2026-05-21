import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", layout_file="layouts/SfM.slides.json")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    from PIL import Image as _Image
    import cv2

    return cv2, np


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
    ## Optical flow

    Optical flow estimates how image points move between consecutive frames as a 2D velocity field \((u, v)\). Assuming brightness constancy and small motion,
    \[
    I(x,y,t) = I(x+u\,\Delta t,\, y+v\,\Delta t,\, t+\Delta t)
    \quad\Rightarrow\quad
    I_x u + I_y v + I_t = 0.
    \]

    Lucas-Kanade solves this underdetermined constraint by assuming nearly constant flow in a small patch and fitting \((u,v)\) over all pixels in that patch by least squares:
    \[
    \min_{u,v} \sum_{p\in\Omega} \left(I_x(p)u + I_y(p)v + I_t(p)\right)^2.
    \]
    This works because many nearby pixels provide enough equations to estimate one local motion vector robustly.
    """)
    return


@app.cell(hide_code=True)
def _(Image, cv2, frame_paths, mo, np):
    first_idx, second_idx = 7, 8
    first = np.array(Image.open(frame_paths[first_idx]).convert("RGB"))
    second = np.array(Image.open(frame_paths[second_idx]).convert("RGB"))

    gray_first = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    gray_second = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
    flow_init = np.zeros(
        (gray_first.shape[0], gray_first.shape[1], 2), dtype=np.float32
    )
    flow = cv2.calcOpticalFlowFarneback(
        gray_first, gray_second, flow_init, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    flow_vis = first.copy()
    step = 18
    for y in range(step // 2, gray_first.shape[0], step):
        for x in range(step // 2, gray_first.shape[1], step):
            dx, dy = flow[y, x]
            p0 = (x, y)
            p1 = (int(x + dx), int(y + dy))
            cv2.arrowedLine(flow_vis, p0, p1, (0, 255, 0), 1, tipLength=0.3)

    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(f"**Frame 1** (`{frame_paths[first_idx].name}`)"),
                    mo.image(Image.fromarray(first), rounded=True),
                ]
            ),
            mo.vstack(
                [
                    mo.md(f"**Frame 2** (`{frame_paths[second_idx].name}`)"),
                    mo.image(Image.fromarray(second), rounded=True),
                ]
            ),
            mo.vstack(
                [
                    mo.md("**Optical flow**"),
                    mo.image(Image.fromarray(flow_vis), rounded=True),
                    mo.md("_Legend: green arrows show motion direction and size._"),
                ]
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
def _(Image, cv2, frame_paths, np):
    selected = [frame_paths[i] for i in (7, 8, 9)]
    rgbs = [np.array(Image.open(p).convert("RGB")) for p in selected]
    grays = [cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) for rgb in rgbs]

    # extract sift keypoints
    sift = cv2.SIFT_create(nfeatures=400, contrastThreshold=0.08)  # type: ignore[attr-defined]
    kps = [sift.detectAndCompute(g, None)[0] for g in grays]
    pts = [np.array([kp.pt for kp in ks], dtype=np.float32) for ks in kps]

    tracked = [pts[0].reshape(-1, 1, 2)]
    for _i in range(1, len(grays)):
        # get tracked SIFT keypoints from the last frame
        p_prev = tracked[-1]
        # find optical flow to next frame and use it to track our prev keypoints
        p_next, s, _ = cv2.calcOpticalFlowPyrLK(
            grays[_i - 1], grays[_i], p_prev, p_prev.copy()
        )
        # find optical flow from next frame and use it to track our curr keypoints
        p_back, sb, _ = cv2.calcOpticalFlowPyrLK(
            grays[_i], grays[_i - 1], p_next, p_next.copy()
        )
        p_next = p_next.astype(np.float32)
        p_back = p_back.astype(np.float32)
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

    _fig, _ax = plt.subplots(1, len(selected), figsize=(5 * len(selected), 5))
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
        _ax[_i].imshow(rgbs[_i])
        if len(pts[_i]):
            _ax[_i].scatter(pts[_i][:, 0], pts[_i][:, 1], s=5, c="red")
        if len(tracked[_i]):
            _ax[_i].scatter(tracked[_i][:, 0], tracked[_i][:, 1], s=7, c="blue")
        _ax[_i].set_title(f"{p.name} ({len(kps[_i])}, {len(tracked[0])} tracked)")
        _ax[_i].axis("off")
    _ax[-1].legend(handles=legend, loc="lower right", framealpha=0.9)
    _fig
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
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    from PIL import Image as _Image
    import plotly.graph_objects as plt_g
    from scipy.linalg import sqrtm

    def main():
        frame_paths = sorted(Path("./data/dino").glob("*.ppm"))
        selected_paths = frame_paths[:10]

        rgb_frames, frames = load_images(selected_paths)

        tracked, F = get_keypoints(frames)

        W = construct_W(tracked, F)

        plot_tracks(tracked, rgb_frames, F)

        M, S = affineSFM(W,F)

        # remove outliers
        dist = np.linalg.norm(S, axis=0)
        mask = dist < np.percentile(dist, 90)
        S = S[:, mask]

        plot_shape(S)

    def construct_W(tracks, F):
        P = len(tracks)
        W = np.zeros((2*F, P))
        for j, track in enumerate(tracks):
            for i, pt in enumerate(track):
                x, y = pt
                W[2*i, j] = x
                W[2*i+1, j] = y

        # centering
        for i in range(F):
            W[2*i]   -= np.mean(W[2*i])
            W[2*i+1] -= np.mean(W[2*i+1])

        return W

    def affineSFM(W,F):
      Ud, Sd, Vd = np.linalg.svd(W)

      # keep up to rank 3
      U3 = Ud[:,0:3]
      S3 = np.diag(Sd[:3])
      V3 = Vd.T[:, :3]

      M = U3 @ sqrtm(S3)
      S = sqrtm(S3) @ V3.T

      C = metric_upgrade(F,M)

      M = M @ C
      S = np.linalg.inv(C) @ S

      return M, S

    def plot_shape(S):
        x = S[0,:]
        y = S[1,:]
        z = S[2,:]

        fig = plt_g.Figure(data=[
            plt_g.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=4,
                    color=z,
                    colorscale='Turbo',
                    opacity=0.9
                )
            )
        ])

        fig.update_layout(
            title='Affine SfM Reconstruction',
            scene=dict(
                aspectmode='data'
            )
        )

        fig.show()

        fig = plt.figure(figsize=(18.5, 6))
        fig.suptitle('Reconstructed image')
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(x, y, z, color='red', lw=1)
        ax1.view_init(130, 0)
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(x, y, z, color='red', lw=1)
        ax2.view_init(45, 180)
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(x, y, z, color='red', lw=1)
        ax3.view_init(-90, 90)
        plt.show()

    def get_keypoints(frames):
        MAX_CORNERS = 800
        QUALITY = 0.01
        MIN_DISTANCE = 8

        LK_PARAMS = dict(
            winSize=(21,21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS |
                      cv2.TERM_CRITERIA_COUNT,
                      30, 0.01)
        )

        ## define a mask to eliminate background junk
        mask = np.zeros_like(frames[0])
        mask[10:490, 10:500] = 255

        p0 = cv2.goodFeaturesToTrack(
            frames[0],
            mask=mask,
            maxCorners=MAX_CORNERS,
            qualityLevel=QUALITY,
            minDistance=MIN_DISTANCE
        )

        tracks = [ [pt.ravel()] for pt in p0 ]

        prev_pts = p0
        for i in range(1, len(frames)):

            prev_img = frames[i-1]
            curr_img = frames[i]

            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_img,
                curr_img,
                prev_pts,
                None,
                **LK_PARAMS
            )

            good_new = next_pts[status==1]
            _ = prev_pts[status==1]

            new_tracks = []

            idx = 0
            for t, s in zip(tracks, status):

                if s == 1:
                    t.append(good_new[idx])
                    new_tracks.append(t)
                    idx += 1

            tracks = new_tracks

            prev_pts = good_new.reshape(-1,1,2)

        tracks = [t for t in tracks if len(t) == len(frames)]

        F = len(frames)

        return tracks, F

    def metric_upgrade(F,M):

      def _symmetric_vec(a, b):
       return np.array([
           a[0]*b[0],
           a[0]*b[1] + a[1]*b[0],
           a[0]*b[2] + a[2]*b[0],
           a[1]*b[1],
           a[1]*b[2] + a[2]*b[1],
           a[2]*b[2]
       ])

      def _positiveDef(M):
        """
        method to compute nearest positive definite matrix
        """
        M = (M + M.T) * 0.5
        k = 0
        I = np.eye(M.shape[0])
        while True:
            try:
                _ = np.linalg.cholesky(M)
                break
            except np.linalg.LinAlgError:
                k += 1
                _, v = np.linalg.eig(M)
                min_eig = v.min()
                M += (-min_eig * k * k + np.spacing(min_eig)) * I
        return M

      A = []
      b = []

      for i in range(F):
          ai = M[2*i]
          bi = M[2*i+1]

          # ai^T L ai = 1
          A.append(_symmetric_vec(ai, ai))
          b.append(1)

          # bi^T L bi = 1
          A.append(_symmetric_vec(bi, bi))
          b.append(1)

          # ai^T L bi = 0
          A.append(_symmetric_vec(ai, bi))
          b.append(0)

      A = np.array(A)
      b = np.array(b)

      x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

      L = np.array([
          [x[0], x[1], x[2]],
          [x[1], x[3], x[4]],
          [x[2], x[4], x[5]]
      ])

      L = _positiveDef(L)

      C = np.linalg.cholesky(L)
      return C

    def load_images(paths):
        rgb_frames = [np.array(_Image.open(p).convert("RGB")) for p in paths]
        frames = []
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                raise RuntimeError(f"Failed to load {path}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        return rgb_frames, frames

    def plot_tracks(tracked,rgbs,F):
        P = len(tracked)

        tracked = np.array(tracked)

        tracked = np.transpose(tracked, (1,0,2))

        _, ax = plt.subplots(
            1,
            min(3, F),
            figsize=(15,5)
        )

        if F == 1:
            ax = [ax]

        show_idx = [0, F//2, F-1]

        for k, idx in enumerate(show_idx):

            ax[k].imshow(rgbs[idx])

            x = tracked[idx,:,0]
            y = tracked[idx,:,1]

            ax[k].scatter(
                x,
                y,
                s=8,
                c='cyan'
            )

            ax[k].set_title(
                f'Frame {idx} ({P} tracked)'
            )

            ax[k].axis('off')

        plt.tight_layout()
        plt.show()

    if __name__ == "__main__":
        main()
    return cv2, np


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
