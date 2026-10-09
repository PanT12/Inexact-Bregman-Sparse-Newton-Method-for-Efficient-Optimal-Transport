import numpy as np
from mnist.loader import MNIST
from scipy.spatial.distance import cdist
import os, glob
from pathlib import Path
from PIL import Image


def get_ot_parameters(dataset, source_indices, target_indices, norm_type='euclidean', eps=0.01):

    source_images = []
    target_images = []
    
    for idx in source_indices:
        if hasattr(dataset[idx], '__getitem__'):
            img = dataset[idx][0] if isinstance(dataset[idx], tuple) else dataset[idx]
        else:
            img = dataset[idx]
        
        if hasattr(img, 'numpy'):
            img_array = img.numpy().flatten()
        elif hasattr(img, 'cpu'):
            img_array = img.cpu().numpy().flatten()
        else:
            img_array = np.array(img).flatten()
        
        source_images.append(img_array)
    
    for idx in target_indices:
        if hasattr(dataset[idx], '__getitem__'):
            img = dataset[idx][0] if isinstance(dataset[idx], tuple) else dataset[idx]
        else:
            img = dataset[idx]
        
        if hasattr(img, 'numpy'):
            img_array = img.numpy().flatten()
        elif hasattr(img, 'cpu'):
            img_array = img.cpu().numpy().flatten()
        else:
            img_array = np.array(img).flatten()
        
        target_images.append(img_array)
    
    source_images = np.array(source_images)  # shape: (m, d)
    target_images = np.array(target_images)  # shape: (n, d)
    
    a = get_marginal_with_regularization(source_images, eps)
    b = get_marginal_with_regularization(target_images, eps)
    
    C = get_pixel_cost_matrix(source_images, target_images, norm_type)
    
    return C, a, b


def get_marginal_with_regularization(images, eps=0.01):
    mu = images.flatten()
    mu = mu / mu.sum()
    
    n = len(mu)
    mu = (1 - eps / 8) * mu + eps / (8 * n)
    
    return mu.astype(np.float64)


def get_pixel_cost_matrix(source_images, target_images, norm_type='euclidean'):
    d = source_images.shape[1]  # pixel numbers per image
    m_pixels = int(np.sqrt(d))  # image length
    
    # generate pixel coordinate grid
    coords = cartesian_product(np.arange(m_pixels), np.arange(m_pixels))
    
    # compute cost matrix based on chosen norm
    if norm_type == 'euclidean':
        C = cdist(coords, coords, metric='euclidean')
        C = C * C
    elif norm_type == 'manhattan':
        C = cdist(coords, coords, metric='cityblock')
    elif norm_type == 'cosine':
        C = cdist(coords, coords, metric='cosine')
    else:
        C = cdist(coords, coords, metric='euclidean')
        C = C * C
    
    C /= np.max(C)
    
    return C.astype(np.float64)


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def load_mnist(mnist_dir, p_id=0, q_id=1, norm_type='euclidean', eps=0.01):

    if not os.path.isdir(mnist_dir):
        raise FileNotFoundError(f"MNIST dataset does not exist: {mnist_dir}")

    mndata = MNIST(mnist_dir)
    images, _ = mndata.load_training()

    p = np.array(images[p_id], dtype=np.float64).reshape(1, -1)
    q = np.array(images[q_id], dtype=np.float64).reshape(1, -1)

    a = get_marginal_with_regularization(p, eps)
    b = get_marginal_with_regularization(q, eps)
    C = get_pixel_cost_matrix(p, q, norm_type)
    return C, a, b


def load_dotmark(dotmark_dir, category, size, p_id, q_id, normalize=True, norm_type='euclidean', eps=0.01):
    dotmark_dir = Path(f"{dotmark_dir}/Data/{category}")
    pattern = f"data{size}_*.csv"
    files = sorted(dotmark_dir.glob(pattern))

    matrices = []

    for f in files:
        file_path = os.path.join(dotmark_dir, f)
        mat = np.loadtxt(file_path, delimiter=",")
        if normalize:
            s = mat.sum()
            if s != 0:
                mat = mat / s
        matrices.append(mat)

    stack = np.stack(matrices, axis=0)

    p = np.array(stack[p_id], dtype=np.float64).reshape(1, -1)
    q = np.array(stack[q_id], dtype=np.float64).reshape(1, -1)

    a = get_marginal_with_regularization(p, eps)
    b = get_marginal_with_regularization(q, eps)
    C = get_pixel_cost_matrix(p, q, norm_type)

    return C, a, b


def load_imagenet(imagenet_dir, p_id=0, q_id=1, image_size=100, norm_type='euclidean', eps=0.01):
    imagenet_dir = Path(imagenet_dir)
    if not imagenet_dir.is_dir():
        raise FileNotFoundError(
            f"ImageNet image folder does not exist: {imagenet_dir}. "
            "Put images under data/ImageNet or pass --data-dir /path/to/images."
        )

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(
        f for f in imagenet_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in image_exts
    )

    if not files:
        raise FileNotFoundError(
            f"No image files found under: {imagenet_dir}. "
            "Supported extensions: .jpg, .jpeg, .png, .bmp, .webp."
        )
    if p_id >= len(files) or q_id >= len(files):
        raise IndexError(
            f"ImageNet indices out of range: p_id={p_id}, q_id={q_id}, number of images={len(files)}"
        )

    def _load_grayscale(path):
        with Image.open(path) as img:
            img = img.convert("L").resize((image_size, image_size), Image.BILINEAR)
            return np.asarray(img, dtype=np.float64).reshape(1, -1)

    p = _load_grayscale(files[p_id])
    q = _load_grayscale(files[q_id])

    a = get_marginal_with_regularization(p, eps)
    b = get_marginal_with_regularization(q, eps)
    C = get_pixel_cost_matrix(p, q, norm_type)

    return C, a, b, str(files[p_id]), str(files[q_id])
