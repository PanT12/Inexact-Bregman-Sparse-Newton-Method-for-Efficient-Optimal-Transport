from skimage import io, color
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def load_figures(src_path, tgt_path, size_hw=(256, 256), show=True):

    m, n = size_hw
    src = io.imread(src_path)
    tar = io.imread(tgt_path)
    src = resize(src, (m, n))
    tar = resize(tar, (m, n))

    if show:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(src)
        axs[1].imshow(tar)
        axs[0].set_axis_off()
        axs[1].set_axis_off()
        plt.tight_layout()
        plt.show()

    src_lab = color.rgb2lab(src)
    tar_lab = color.rgb2lab(tar)

    src_lumi = src_lab[:, :, 0]
    src_ab_img = src_lab[:, :, 1:]
    tar_lumi = tar_lab[:, :, 0]
    tar_ab_img = tar_lab[:, :, 1:]

    src_ab = np.reshape(src_ab_img, (-1, 2), order='F')
    tar_ab = np.reshape(tar_ab_img, (-1, 2), order='F')

    return src, tar, src_lab, tar_lab, src_lumi, src_ab, tar_lumi, tar_ab


def to_lab_kmeans(src_ab, tar_ab, K=128, scale_to_01=True):
    union_ab = np.vstack([src_ab, tar_ab])
    if scale_to_01:
        scaler = MinMaxScaler((0, 1))
        union_ab_scaled = scaler.fit_transform(union_ab)
    else:
        scaler = None
        union_ab_scaled = union_ab

    size = src_ab.shape[0]
    kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(union_ab_scaled)
    centers = kmeans.cluster_centers_  # (K, 2)
    labels = kmeans.labels_  # (2*size,)

    labels_src = labels[:size]
    labels_tar = labels[size:]

    a = np.bincount(labels_src, minlength=K).astype(float) + 1e-3
    a /= a.sum()
    b = np.bincount(labels_tar, minlength=K).astype(float) + 1e-3
    b /= b.sum()

    diff = centers[:, None, :] - centers[None, :, :]  # (K, K, 2)
    C = np.sum(diff * diff, axis=2)
    C /= np.max(C)

    return centers, a, b, labels_src, labels_tar, scaler, C


def run_one_experiment(C, a, b, algo_name, registry, eta=1e-3, tol=1e-7, time_max=np.inf, opt=None):
    import time
    start = time.time()
    maybe_eta = None
    if algo_name == "our":
        X, history, maybe_eta = registry[algo_name](C, a, b, eta=eta, tol=tol, time_max=time_max, opt=opt)
    else:
        X, history = registry[algo_name](C, a, b, eta=eta, tol=tol, opt=opt)
    runtime = time.time() - start

    iterations = len(history.get('time', [])) if isinstance(history, dict) else None
    objective = None
    if isinstance(history, dict):
        if 'obj' in history and len(history['obj']) > 0:
            objective = float(history['obj'][-1])
        elif 'err' in history and len(history['err']) > 0:
            objective = float(history['err'][-1])

    metrics = {
        'runtime': float(runtime),
        'iterations': int(iterations) if iterations is not None else None,
        'objective': objective
    }

    return X, history, metrics, maybe_eta


def remap_colors_from_transport(X, a, centers, labels_src, labels_tar, src_rgb, scaler=None):
    """
    Based on OT coupling matrix X, remap the colors of the source image and return the synthesized RGB image.

    Parameters：
    - X: (K,K) transport plan from source to target clusters
    - a: (K,) source weight vector
    - centers: (K,2) cluster centers in ab color space
    - labels_src, labels_tar: (K,) cluster labels for source and target pixels
    - src_rgb: original source RGB image, shape (H,W,3), range [0,255] or [0,1]

    Return：
    - synthesis_rgb: synthesized RGB image after color transfer, shape (H,W,3), range [0,1]
    """
    import numpy as np
    from skimage import color

    # The color of each target cluster: use cluster centers
    Avg_tar = centers.copy()

    # The color of each source cluster: weighted average of target colors
    row_sum = a.copy()
    row_sum[row_sum < 1e-12] = 1e-12
    Avg_src = (X @ Avg_tar) / row_sum[:, None]

    new_ab_scaled = Avg_src[labels_src]

    if scaler is not None:
        from sklearn.preprocessing import MinMaxScaler  # noqa: F401 (hint for type)
        new_ab = scaler.inverse_transform(new_ab_scaled)
    else:
        new_ab = new_ab_scaled

    # Reconstruct the new image in Lab space and convert back to RGB
    src_rgb_float = src_rgb.astype(np.float64)
    if src_rgb_float.max() > 1.0:
        src_rgb_float = src_rgb_float / 255.0
    src_lab = color.rgb2lab(src_rgb_float)
    H, W = src_lab.shape[0], src_lab.shape[1]
    src_lumi = src_lab[:, :, 0]

    new_img_arr = np.hstack([src_lumi.reshape(-1, 1, order='F'), new_ab])
    new_img_arr = new_img_arr.reshape(H, W, 3, order='F')
    synthesis = color.lab2rgb(new_img_arr)
    return synthesis
