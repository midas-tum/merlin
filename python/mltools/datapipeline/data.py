import tensorflow as tf
import numpy as np
import os
import sys
from joblib import Parallel, delayed
from tqdm import tqdm

from utils_data import *
import sampling.VD_CASPR_CINE as VD_CASPR_CINE


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
        self.sampling_trajectory = sampling_trajectory  # subsampling trajectory 'PI', 'VDCASPR', 'radial'

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

    def self.load_file(self, filename):
        # data extension specific loading
        x = ....

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
            img_sub =
        elif self.sampling_trajectory == 'radial':  # TODO
            mask = []
            kspace_sub = []
            img_sub = []

        return img_sub, kspace_sub, mask

    def _generate_mask(self, nSegments=12, acc=4):
        sType = 'CINE'
        sMode = 'interleaved'  # 'interleaved' (CINE), 'noninterleaved' (free-running, CMRA, T2Mapping, ...)
        numLin = self.dim_in[2]  # ky points (store it the other way round, for right dim)
        numPar = self.dim_in[3]  # kz points
        nRep = self.dim_in[0]  # time points
        # acc = 4  # acceleration
        isVariable = 1  # VDCASPR (=1) or CASPR (=0)
        isGolden = 2  # golden angle increment between spirals (=1), 0 = linear-linear, 1=golden-golden, 2=tinyGolden-golden, 3=linear-golden, 4=noIncr-golden
        isInOut = 1  # spiral in/out sampling => for isGolden=1 & isInOut=1: use tinyGolden-Golden-tinyGolden-Golden-... increments
        isCenter = 0  # sample center point
        isSamePattern = 0  # same sampling pattern per phase/contrast, i.e. no golden/tiny-golden angle increment between them (but still inside the pattern if isGolden==1)
        # nSegments = 10  # #segments = #rings in sampling pattern
        # nRep = 16  # number of repetitions (only free-running)
        nCenter = 15  # percentage of fully sampled center region
        iVerbose = 0  # 0=silent, 1=normal output, 2=all output
        lMask = np.zeros((numPar, numLin))
        kSpacePolar_LinInd = np.zeros((nRep * numLin * numPar, 1))
        kSpacePolar_ParInd = np.zeros((nRep * numLin * numPar, 1))
        out_parameter = np.zeros((3, 1), dtype='float32')
        parameter_list = np.asarray(
            [numLin, numPar, acc, nCenter, nSegments, nRep, isGolden, isVariable, isInOut, isCenter, isSamePattern,
             iVerbose], dtype='float32')
        res = VD_CASPR_CINE.run(parameter_list, lMask, kSpacePolar_LinInd, kSpacePolar_ParInd, out_parameter)
        n_SamplesInSpace = np.asscalar(out_parameter[0].astype(int))
        nSampled = np.asscalar(out_parameter[1].astype(int))
        nb_spiral = np.asscalar(out_parameter[2].astype(int))
        mask_rep = np.zeros((numPar, numLin, nRep))
        for iRep in range(1, nRep + 1):
            iVec = list()
            for iInner in range(nSegments):
                iVecTmp = [idx - 1 for idx in
                           range((iRep - 1) * nSegments + 1 + iInner, nSampled - nSegments + 1 + iInner + 1,
                                 nSegments * nRep)]
                iVec.extend(iVecTmp)
            # iVec = np.asarray(iVec)

            for iI in iVec:
                if (kSpacePolar_LinInd[iI] > 0) and (kSpacePolar_ParInd[iI] > 0):
                    mask_rep[np.asscalar(kSpacePolar_ParInd[iI].astype(int)), np.asscalar(
                        kSpacePolar_LinInd[iI].astype(int)), iRep - 1] += 1

        return mask_rep  # Z x Y x Time

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


class ComplexDataGeneratorMNIST(DataGeneratorMNIST):
    'Generates complex-valued data for keras model'

    def __init__(self, batch_size=32, noise_level=0.1, path='mnist.npz', shuffle=True, mode='train'):
        'Initialization'
        self.dtype = np.complex64
        super().__init__(batch_size=batch_size, noise_level=noise_level, path=path, shuffle=shuffle, mode=mode)

    def _prepare_data(self):
        'Data preparation'
        super()._prepare_data()
        # simulate some phase information
        phase = self._normalize(self.img[::-1], 0, 2 * np.pi)
        self.img = self.img.astype(np.complex64) * np.exp(1j * phase)
        self.img = self.img.astype(self.dtype)






class ComplexRawDataGeneratorMNIST(ComplexDataGeneratorMNIST):
    'Generates complex-valued data with raw data (for data consistency) for keras model'

    def __init__(self, batch_size=32, accelerations=[2, 4], accel_type='PI', center=0.1, path='mnist.npz', shuffle=True,
                 mode='train'):
        'Initialization'
        super().__init__(batch_size=batch_size, noise_level=0, path=path, shuffle=shuffle, mode=mode)
        self.accelerations = accelerations
        # data dimensions: kx x ky, only undersample along ky
        self.center = np.floor(center * self.dim[1])  # full sampled center

        # acceleration type: 'PI' = Parallel Imaging, 'CS' = Compressed Sensing
        self.accel_type = accel_type

    def _prepare_data(self):
        'Data preparation'
        super()._prepare_data()
        # normalize BEFORE creating the k-space
        self.img = self._normalize(self.img)
        # generate k-space
        self.kspace = fftnc(self.img, axes=(1, 2))

    def _subsample(self, kspace):
        'Retrospective undersampling/sub-Nyquist sampling'
        mask = np.zeros(self.dim, dtype=np.float32)

        # fully sampled center
        fscenter = (int(np.floor(self.dim[1] / 2 - self.center / 2)), int(np.floor(self.dim[1] / 2 + self.center / 2)))
        mask[:, fscenter[0]:fscenter[1]] = 1

        if self.accel_type == 'PI':
            # Parallel imaging undersampling
            # sample every n-th phase-encoding line
            mask[:, ::np.random.choice(self.accelerations)] = 1

        elif self.accel_type == 'CS':
            # Compressed Sensing like undersampling
            # calculate amount of points to sample, considering the fully sampled center
            to_sample = np.floor(self.dim[1] / np.random.choice(self.accelerations))
            nsampled = self.center
            # effective acceleration rate for high-frequency region. Considering fully sampled center and effective acceleration yields the overall desired acceleration
            # eff_accel = self.dim[1] / (to_sample - self.center)

            # Gaussian sampling
            x = np.arange(0, self.dim[1])
            stddev = np.floor(self.dim[1] / 4)
            xU, xL = x + 0.5, x - 0.5
            prob = scs.norm.cdf(xU, loc=np.floor(self.dim[1] / 2), scale=stddev) - scs.norm.cdf(xL, loc=np.floor(
                self.dim[1] / 2),
                                                                                                scale=stddev)  # calculate sampling prob. P(xL < x <= xU) = CDF(xU) - CDF(xL), with Gaussian CDF
            prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
            while (nsampled < to_sample):
                nums = np.random.choice(x, size=1, p=prob)
                if mask[0, nums] == 0:
                    mask[:, nums] = 1
                    nsampled += 1

        return kspace * mask, mask

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
            sample = self.img[ID,]

            # normalize to the range [0, 1]
            # sample = self._normalize(sample)

            # Store sample
            target[i, ..., 0] = sample

            kspace[i,], mask[i,] = self._subsample(self.kspace[ID,])

            # Generate noisy input
            noisy[i, ..., 0] = ifftnc(kspace[i,], axes=(0, 1))

        return [noisy, kspace, mask], target