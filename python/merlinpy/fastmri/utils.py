"""
Copyright (c) 2019 Imperial College London.
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import pathlib
import h5py

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
