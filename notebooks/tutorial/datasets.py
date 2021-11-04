import tensorflow as tf
import numpy as np
import scipy.stats as scs
import unittest

# N-dimensional FFT (considering ifftshift/fftshift operations)
def fftnc(x, axes=(0, 1)):
    for ax in axes:
        x = 1 / np.sqrt(x.shape[ax]) * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=ax), axis=ax), axes=ax)
    return x

# N-dimensional IFFT (considering ifftshift/fftshift operations)
def ifftnc(x, axes=(0, 1)):
    for ax in axes:
        x = np.sqrt(x.shape[ax]) * np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=ax), axis=ax), axes=ax)
    return x

class DataGeneratorMNIST(tf.keras.utils.Sequence):
    'Generates real-valued data for keras model'

    def __init__(self, batch_size=32, noise_level=0.1, path='mnist.npz', shuffle=True, mode='train'):
        'Initialization'
        assert mode in ['train', 'val', 'test']

        self.batch_size = batch_size
        self.noise_level = noise_level
        self.shuffle = shuffle
        if not hasattr(self, 'dtype'):
            self.dtype = np.float32

        if mode in ['train', 'val']:
            self.img = tf.keras.datasets.mnist.load_data(path='mnist.npz')[0][0]
        else:
            self.img = tf.keras.datasets.mnist.load_data(path='mnist.npz')[1][0]

        # ATTENTION!
        # This code part is performing a random database splitting for training/validation, as all MNIST images
        # are independent of each other. In case of MRI, a patient leave-out approach needs to be performed!!!
        # predefined 80/20 split. MNIST has 80000 samples
        cv_split = {'train': 48000, 'val': 12000}
        if mode == 'train':
            self.img = self.img[0:cv_split['train']]
        elif mode == 'val':
            self.img = self.img[-cv_split['val']:]

        # prepare data
        self._prepare_data()

        # save the size of the images
        self.dim = self.img.shape[1:]

        # get amount of data samples in train/val/test
        self.n_samples = self.img.shape[0]

        # prepare batch indexing
        self.on_epoch_end()

    def _prepare_data(self):
        'Data preparation'
        # convert uint8 to float32
        self.img = self.img.astype(np.float32)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        return self._data_generation(indexes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        # X : (n_samples, *dim, n_channels)
        target = np.empty((self.batch_size, *self.dim, 1), dtype=self.dtype)    # target/reference output
        noisy = np.empty((self.batch_size, *self.dim, 1), dtype=self.dtype)     # corrupted/noisy input

        # Generate data
        for i, ID in enumerate(indexes):
            # Get sample
            sample = self.img[ID,]

            # normalize to the range [0, 1]
            sample = self._normalize(sample)

            # Store sample
            target[i, ..., 0] = sample

            # Generate noisy input
            noisy[i, ..., 0] = self._noisy(sample)

        return noisy, target

    def _normalize(self, x, min=0, max=1):
        'Normalization'
        # normalize/scale the image
        return (x - np.min(x)) / (np.max(x) - np.min(x)) * (max - min) + min

    def _noisy(self, x):
        'Real-valued noise simulation'
        # add Gaussian noise
        return x + np.random.randn(*x.shape) * self.noise_level

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
        # !ATTENTION! Some weird numerical errors occur...
        # Set values in magn phase around 0 to eps=1e-12
        eps = 1e-12
        phase = self._normalize(self.img[::-1], -np.pi, np.pi)
        phase = np.sign(phase)*np.maximum(phase, eps)
        # magn = self._normalize(self.img)
        magn = np.maximum(self.img, eps)
        self.img = magn * np.exp(1j * phase)
        self.img = self.img.astype(self.dtype)
        
    def _normalize(self, x, min=0, max=1):
        'Normalization'
        # only scale the magnitude
        if np.iscomplexobj(x):
            xabs = np.abs(x)
            normed_magn = (xabs - np.min(xabs)) / (np.max(xabs) - np.min(xabs)) * (max - min) + min
            return normed_magn * np.exp(1j * np.angle(x))
        else:
            return (x - np.min(x)) / (np.max(x) - np.min(x)) * (max - min) + min

    def _noisy(self, x):
        'Complex-valued noise simulation'
        # add complex Gaussian noise
        return x + self.noise_level / 2 * ( np.random.randn(*x.shape) + 1j *  np.random.randn(*x.shape))
    
class ComplexRawDataGeneratorMNIST(ComplexDataGeneratorMNIST):
    'Generates complex-valued data with raw data (for data consistency) for keras model'

    def __init__(self, batch_size=32, accelerations=[2, 4], accel_type='PI', center=0.1, path='mnist.npz', shuffle=True, mode='train'):
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
        self.kspace = fftnc(self.img, axes=(1,2))

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
            #eff_accel = self.dim[1] / (to_sample - self.center)

            # Gaussian sampling
            x = np.arange(0, self.dim[1])
            stddev = np.floor(self.dim[1]/4)
            xU, xL = x + 0.5, x - 0.5
            prob = scs.norm.cdf(xU, loc=np.floor(self.dim[1]/2), scale=stddev) - scs.norm.cdf(xL, loc=np.floor(self.dim[1]/2), scale=stddev)  # calculate sampling prob. P(xL < x <= xU) = CDF(xU) - CDF(xL), with Gaussian CDF
            prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
            while(nsampled < to_sample):
                nums = np.random.choice(x, size=1, p=prob)
                if mask[0, nums] == 0:
                    mask[:, nums] = 1
                    nsampled += 1

        return kspace * mask, mask

    def _data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        # X : (n_samples, *dim, n_channels)
        target = np.empty((self.batch_size, *self.dim, 1), dtype=self.dtype)    # target/reference output
        kspace = np.empty((self.batch_size, *self.dim), dtype=self.dtype)       # rawdata/k-space for data consistency
        mask = np.empty((self.batch_size, *self.dim), dtype=np.float32)         # sampling mask
        noisy = np.empty((self.batch_size, *self.dim, 1), dtype=self.dtype)     # corrupted/aliased/noisy input

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
            noisy[i, ..., 0] = ifftnc(kspace[i,], axes=(0,1))

        return [noisy, kspace, mask], target

class TestDataGenerator(unittest.TestCase):
    def testComplex(self):
        import tensorflow as tf
        import numpy as np
        import matplotlib.pyplot as plt
        
        # initialize some parameters
        accelerations = [2, 4]  # simulate retrospectively accelerations in the range of e.g. 2x to 4x
        accel_type = 'PI'  # simulated undersampling strategy: 'PI' = Parallel Imaging, 'CS' = Compressed Sensing
        center = 0.1  # percent of fully sampled central region along ky phase-encoding, e.g. 0.1 := floor(10% * 28) ky center lines = 2 ky center lines 
        test_generator = ComplexRawDataGeneratorMNIST(batch_size=1, 
                                            accelerations=accelerations,
                                            accel_type=accel_type,
                                            center=center,
                                            shuffle=False,
                                            mode='test')

        inputs, outputs = test_generator.__getitem__(0)
        img = outputs[0,...,0]

        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.abs(img))
        plt.title('Magnitude')
        plt.subplot(1,2,2)
        plt.imshow(np.angle(img))
        plt.title('Phase')
        plt.savefig('test_complex_reconstruction.png')
        self.assertTrue(np.abs(np.max(np.abs(img)) - 1) <= 1e-6)

class TestDataGeneratorDenoising(unittest.TestCase):
    def testComplex(self):
        import tensorflow as tf
        import numpy as np
        import matplotlib.pyplot as plt
        
        # initialize some parameters
        test_generator = ComplexDataGeneratorMNIST(batch_size=1, 
                                                    shuffle=False,
                                                    mode='test')

        inputs, outputs = test_generator.__getitem__(0)
        img = outputs[0,...,0]

        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.abs(img))
        plt.title('Magnitude')
        plt.subplot(1,2,2)
        plt.imshow(np.angle(img))
        plt.title('Phase')
        plt.savefig('test_complex_denoising.png')
        self.assertTrue(np.abs(np.max(np.abs(img)) - 1) <= 1e-6)
        
                                            
    
if __name__ == "__main__":
    unittest.test()