import tensorflow as tf
import numpy as np
import os
import sys
from joblib import Parallel, delayed
from tqdm import tqdm

from merlinpy.datapipeline.utils_data import *



class DataGenerator(tf.keras.utils.Sequence):
    'Generates complex-valued data for keras model'

    def __init__(self, list_IDs, dim_in, n_augment=1, prefetch=True, scaleInput=2, batch_size=32, noise_level=0.1, accelerations=[2, 4], sampling_trajectory='PI', center=0.1, path='mnist.npz', shuffle=True, mode='train'):
        'Initialization'
        assert mode in ['train', 'val', 'test']

        self.batch_size = batch_size
        self.noise_level = noise_level
        self.shuffle = shuffle
        if not hasattr(self, 'dtype'):
            self.dtype = np.complex64

        # parsable list of data files
        self.list_IDs = list_IDs

        self.accelerations = accelerations  # acceleration ranges [list]
        self.sampling_trajectory = sampling_trajectory  # subsampling trajectory 'PI', 'radial'

        self.dim_in = dim_in  # input dimensions
        self.axes_in = (0,1)  # axes along which to operate fft/ifft
        self.scaleInput = scaleInput  # 1=slice-wise, 2=volume/patient-wise
        self.prefetch = prefetch
        self.n_augment = n_augment  # augment database (1 = none)

        if self.prefetch:
            self._prefetch()

        # prepare batch indexing
        self.on_epoch_end()

    def _prepare_data(self):
        'Data preparation'
        # convert uint8 to float32
        self.img = self.img.astype(np.float32)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.list_IDs) * self.n_augment) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        return self._data_generation(indexes)

    def _data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        # X : (n_samples, *dim, n_channels)
        target = np.empty((self.batch_size, *self.dim, 1), dtype=self.dtype)  # target/reference output
        kspace = np.empty((self.batch_size, *self.dim), dtype=self.dtype)  # rawdata/k-space for data consistency
        mask = np.empty((self.batch_size, *self.dim), dtype=np.float32)  # sampling mask
        noisy = np.empty((self.batch_size, *self.dim, 1), dtype=self.dtype)  # corrupted/aliased/noisy input

        # Generate data
        for i, ID in enumerate(indexes):
            # Get sample
            if self.prefetch:
                indices = [i for i, x in enumerate(self.list_IDs) if x == ID]
                target[i,] = self.targetimgs[indices,]
                noisy[i,] = self.imgsubs[indices,]
                kspace[i,] = self.kspacesubs[indices,]
                mask[i,] = self.masks[indices,]
            else:
                target[i,] = self.load_file('input_' + ID)  # load the whole patient -> for patient/volume-wise scaling
                target[i,] = self._normalize(target[i,])  # TODO: can put this into loading to enable for-loop with slice-wise scaling (if required)

                # get subsampled
                noisy[i,], kspace[i,], mask[i,] = self._data_augmentation(target[i,], np.random.choice(self.accelerations))

        return [noisy, kspace, mask], target

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _prefetch(self):
        print('Prefetching data')

        # Generate data
        lids = len(self.list_IDs)

        def preload(i):
            ID = self.list_IDs[i]
            pattmp = os.path.basename(ID).split('_')[1]   # splits patient ID from filename -> check if valid for all databases

            # get target and scale it
            targetimgtmp = self.load_file('input_' + ID)  # load the whole patient -> for patient/volume-wise scaling
            targetimgtmp = self._normalize(targetimgtmp)  # TODO: can put this into loading to enable for-loop with slice-wise scaling (if required)

            # get subsampled
            imgsubtmp, kspacesubtmp, masktmp = self._data_augmentation(targetimgtmp, np.random.choice(self.accelerations))

            return imgsubtmp, kspacesubtmp, masktmp, targetimgtmp, pattmp

        imgsubs, kspacesubs, masks, targetimgs, pats = zip(*Parallel(n_jobs=8)(delayed(preload)(i) for i in tqdm(range(lids))))
        self.imgsubs = np.asarray(imgsubs)
        self.kspacesubs = np.asarray(kspacesubs)
        self.masks = np.asarray(masks)
        self.targetimgs = np.asarray(targetimgs)
        pats = list(pats)

        print('done')

    def load_file(self, filename):
        # data extension specific loading
        # x = ....
        return None

    def _data_augmentation(self, X, acc):  # X: complex-valued: Time x X x Y x Z
        'Data augmentation subsampling - specific to trajectory'
        # generate subsampling mask
        if self.sampling_trajectory == 'VDCASPR':
            nSegments = np.random.choice(self.segments)  # determines timing
            mask = self._generate_mask(nSegments, acc)
            kspace = fftnc(X, axes=self.axes_in)
            kspace_sub = np.multiply(kspace, np.expand_dims(np.moveaxis(mask, -1, 0), axis=1))
            img_sub = ifftnc(kspace_sub, axes=self.axes_in)

        elif self.sampling_trajectory == 'PI':  # TODO
            mask[:, :, ::np.random.choice(self.accelerations), :] = 1
            kspace_sub = np.multiply(kspace, mask)
            img_sub = None # TODO
        elif self.sampling_trajectory == 'radial':  # TODO
            mask = []
            kspace_sub = []
            img_sub = []

        return img_sub, kspace_sub, mask

    def _normalize(self, x, min=0, max=1):
        'Normalization'
        # only scale the magnitude
        xabs = np.abs(x)
        assert np.max(xabs) > 1e-9
        return ((xabs - np.min(x)) / (np.max(x) - np.min(x)) * (max - min) + min) * np.exp(1j * np.angle(x))

    def _noisy(self, x):
        'Complex-valued noise simulation'
        # add complex Gaussian noise
        return x + self.noise_level / 2 * (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape))