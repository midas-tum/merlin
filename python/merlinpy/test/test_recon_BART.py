import unittest
import numpy as np
from merlinpy.recon.BART import setup_bart, recon

class BARTTest(unittest.TestCase):
    def test_BARTrecon(self, acc=4):
        importsuccess = setup_bart('/home/gitaction/bart')
        if importsuccess:
            from bart import bart
            kspace = bart(1, 'phantom -3 -x 64 -k -s 8')
            smap = bart(1, 'phantom -3 -x 64 -S 8')

            kspace = np.expand_dims(kspace, 0).transpose((0, -1, 1, 2, 3))
            smap = np.expand_dims(smap, 0).transpose((0, -1, 1, 2, 3))

            reconimg = recon(kspace, smap, None, None, '-d5 -m -S -R W:7:0:0.01 -R T:7:0:0.001', dim='3D')
            self.assertTrue(np.shape(reconimg) == (1, 64, 64, 64))
        else:
            self.assertTrue(True)

if __name__ == "__main__":
    importsuccess = setup_bart('/home/gitaction/bart')
    if importsuccess:
        unittest.main()

    #kspace = np.load('kspace.npy')
    #smap = np.load('smap.npy')
    #mask = np.load('mask.npy')
    #imgout = recon(kspace, smap, mask, '-d5 -m -S -R W:7:0:0.01 - R T:7:0:0.001')
