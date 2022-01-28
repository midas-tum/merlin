import unittest
import numpy as np
from merlinpy.datapipeline.sampling import (
    CASPR,
    PoissonDisc,
    Gaussian,
    VISTA,
    UIS,
    VRS
)

class SamplingTest(unittest.TestCase):
    def test_samplingCASPR2D(self, acc=4):
        self._test_sampling(CASPR([128, 64, 1], acc))  # non-interleaved and triggered/gated single-phase Cartesian with spiral ordering sampling

    def test_samplingCASPR2Dt(self, acc=4):
        self._test_sampling(CASPR([128, 64, 16], acc))  # non-interleaved and triggered/gated multi-phase Cartesian with spiral ordering sampling

    def test_samplingVDPD1D(self, acc=4):
        self._test_sampling(PoissonDisc([128, 1, 1], acc))  # 1D Poisson-Disc subsampling

    def test_samplingVDPD2D(self, acc=4):  # 2D Poisson-Disc subsampling
        self._test_sampling(PoissonDisc([128, 64, 1], acc))  # central ellipse, L2-norm VD distance
        self._test_sampling(PoissonDisc([128, 64, 1], acc, vd_type=1))  # no variable-density, i.e. uniform, L2-norm VD distance
        self._test_sampling(PoissonDisc([128, 64, 1], acc, vd_type=2))  # central point, L2-norm VD distance
        self._test_sampling(PoissonDisc([128, 64, 1], acc, vd_type=3))  # central block, L2-norm VD distance
        self._test_sampling(PoissonDisc([128, 64, 1], acc, vd_type=2, p=1, n=1))  # central point, L1-norm VD distance
        self._test_sampling(PoissonDisc([128, 64, 1], acc, vd_type=4, pF_value=0.75))  # ESPReSSo sampling 0.75, central ellipse, L2-norm VD distance

    def test_samplingVDPD2Dt(self, acc=4):  # 2D multi-phase Poisson-Disc subsampling
        self._test_sampling(PoissonDisc([128, 64, 16], acc))

    def test_samplingGaussian1D(self, acc=4):  # 1D Gaussian subsampling
        self._test_sampling(Gaussian([128, 1, 1], acc))  # central ellipse, L2-norm VD distance

    def test_samplingGaussian2D(self, acc=4):  # 2D Gaussian subsampling
        self._test_sampling(Gaussian([128, 64, 1], acc))  # central ellipse, L2-norm VD distance

    def test_samplingGaussian2Dt(self, acc=4):  # 2D multi-phase Gaussian subsampling
        self._test_sampling(Gaussian([128, 64, 16], acc))  # central ellipse, L2-norm VD distance

    def test_samplingVISTA(self, acc=16):
        self._test_sampling(VISTA([32, 1, 8], acc))  # VISTA subsampling

    def test_samplingUIS(self, acc=16):
        self._test_sampling(UIS([32, 1, 8], acc))  # UIS subsampling

    def test_samplingVRS(self, acc=16):
        self._test_sampling(VRS([32, 1, 8], acc))  # VRS subsampling

    def _test_sampling(self, sampler):
        mask = sampler.generate_mask()
        #sampler.plot_mask(asfigure=False)
        nsampled = sampler.get_accel()
        print('*** %s ***' % sampler.trajectory)
        print('Requested acceleration: %.2f' % sampler.acc)
        print('Obtained acceleration: %.2f' % nsampled)
        self.assertTrue(np.abs(nsampled - sampler.acc) / sampler.acc <= 0.05)  # < 5% deviation

if __name__ == "__main__":
    unittest.main()