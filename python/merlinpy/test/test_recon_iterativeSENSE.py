import unittest
import numpy as np
import tensorflow as tf
from merlinpy.recon.BART import setup_bart
from merlinpy.recon.iterativeSENSE import recon

class ItSenseTest(unittest.TestCase):
    def test_iterativeSENSE(self, acc=4):
        importsuccess = setup_bart('/home/gitaction/bart')
        if importsuccess:
            from bart import bart
            kspace = bart(1, 'phantom -x 64 -k -s 8')
            smap = bart(1, 'phantom -x 64 -S 8')

            kspace = np.expand_dims(kspace, 0).transpose((0, -1, 1, 2, 3))[..., 0]
            smap = np.expand_dims(smap, 0).transpose((0, -1, 1, 2, 3))[..., 0]

            reconimg = np.sum(recon(kspace, smap), 1)
            #import matplotlib.pyplot as plt
            #plt.imshow(np.abs(reconimg[0, :, :]))
            #plt.show()
            self.assertTrue((tf.shape(reconimg) == (1, 64, 64)).numpy().all())
        else:
            self.assertTrue(True)

if __name__ == "__main__":
    importsuccess = setup_bart('/home/gitaction/bart')
    if importsuccess:
        unittest.main()