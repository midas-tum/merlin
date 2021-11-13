import pathlib
import h5py
import numpy as np

def center_crop(data, shape, channel_last=False):
    """
    [source] https://github.com/facebookresearch/fastMRI/blob/master/data/transforms.py
    Apply a center crop to the input real image or batch of real images.

    Args:
        data (numpy.array): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        numpy.array: The center cropped image
    """
    if channel_last:
        dim0 = data.shape[-3]
        dim1 = data.shape[-2]
    else:
        dim0 = data.shape[-2]
        dim1 = data.shape[-1]

    assert 0 < shape[0] <= dim0
    assert 0 < shape[1] <= dim1
    w_from = (dim0 - shape[0]) // 2
    h_from = (dim1 - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    if channel_last:
        return data[..., w_from:w_to, h_from:h_to, :]
    else:
        return data[..., w_from:w_to, h_from:h_to]

def random_crop(data, shape, channel_last=False):
    """
    Apply a random crop to the input real image or batch of real images.

    Args:
        data (numpy.array): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        numpy.array: The center cropped image
    """
    if channel_last:
        dim0 = data.shape[-3]
        dim1 = data.shape[-2]
    else:
        dim0 = data.shape[-2]
        dim1 = data.shape[-1]

    assert 0 < shape[0] <= dim0
    assert 0 < shape[1] <= dim1
    w_from = np.random.randint(0, dim0 - shape[0])
    h_from = np.random.randint(0, dim1 - shape[1])
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    if channel_last:
        return data[..., w_from:w_to, h_from:h_to, :]
    else:
        return data[..., w_from:w_to, h_from:h_to]

def save_reconstructions(reconstructions, out_dir):
    """
    [Code from https://github.com/facebookresearch/fastMRI]
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir = pathlib.Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, recons in reconstructions.items():
        fname = fname.split('/')[-1]
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)

def np_ensure_complex64(x):
    """
    [Code from https://github.com/khammernik/sigmanet]
    This function is used to recover complex dtype if current dtype is float16,
    otherwise dtype of x is unchanged.

    Args:
        x: Input data of any datatype.
    """
    if x.dtype == np.float16:
        return np.ascontiguousarray(x.astype(np.float32)).view(dtype=np.complex64)
    else:
        return x

def np_ensure_float32(x):
    """
    [Code from https://github.com/khammernik/sigmanet]
    This function is used to recover complex dtype if current dtype is float16,
    otherwise dtype of x is unchanged.

    Args:
        x: Input data of any datatype.
    """
    if x.dtype == np.float16:
        return np.ascontiguousarray(x.astype(np.float32))
    else:
        return x

def np_view_as_float16(x):
    """
    [Code from https://github.com/khammernik/sigmanet]
    This function is used to convert (complex) objects to float16.

    Args:
        x: Input data of any datatype.
    """
    if np.iscomplexobj(x):
        if x.dtype == np.complex128:
            x = x.astype(np.complex64)
        x = x.view(dtype=np.float32)
    x = np.float16(x)
    return x