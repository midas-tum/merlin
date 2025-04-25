import numpy as np
import math

from merlinpy.datapipeline.sampling.VISTA.main import sampling2dt
import matplotlib.pyplot as plt
try:
    from merlinpy.datapipeline.sampling import VD_CASPR_CINE as VD_CASPR_CINE
except:
    print('VD_CASPR_CINE sampling code is not compiled')
try:
    from merlinpy.datapipeline.sampling import VDPDGauss as VDPDGauss
except:
    print('VDPDGauss sampling code is not compiled')

class Sampling:  # abstract parent class
    # subsampling of phase-encoding directions (y/z) and along time (t)
    def __init__(self, dim, acc, trajectory, is_elliptical):
        self.dim = dim  # y - z - t
        self.acc = acc  # acceleration factor
        self.trajectory = trajectory  #  Poisson-Disc, Gaussian, GoldenRadial, TinyGoldenRadial,
        self.is_elliptical = is_elliptical
        self.mask = []

    def plot_mask(self, asfigure=True):
        numLin = self.dim[0]  # ky points
        numPar = self.dim[1]  # kz points
        nRep = self.dim[2]  # time points
        if asfigure:
            iMaxCol = 2
            iRows = np.ceil(nRep / iMaxCol)
            plt.figure()
            for iRep in range(1, nRep + 1):
                plt.subplot(iRows, iMaxCol, iRep)
                plt.imshow(self.mask[:, :, iRep - 1], cmap='gray')

                print('#samples repetition %d = %d' % (iRep, np.sum(self.mask[:, :, iRep - 1], axis=(0, 1))))
            plt.show()
        else:
            print('Sampling')
            print('    ', end='')
            for iPar in range(numPar):
                print('%03d ' % (iPar), end='')
            print('')

            for iLin in range(numLin):
                print('%03d ' % (iLin), end='')
                for iPar in range(numPar):
                    nsamples = np.sum(self.mask[iLin,iPar,:])
                    if nsamples > 0:
                        print('%03d ' % (nsamples), end='')
                    else:
                        print('--- ', end='')
                print('')

            for iRep in range(1, nRep + 1):
                print('*** Repetition = %d ***' % (iRep))
                print('#Samples repetition %d: %d' % (iRep, np.sum(self.mask[:, :, iRep - 1], axis=(0, 1))))

    def get_accel(self):
        # determine obtained acceleration factor of subsampling algorithm
        if self.is_elliptical:
            samples_elliptical = 0
            for iLin in range(self.dim[0]):
                for iPar in range(self.dim[1]):
                    if (((-1.0 + 2.0 * iPar / (self.dim[1] - 1.0)) * (-1.0 + 2.0 * iPar / (self.dim[1] - 1.0)) + (-1.0 + 2.0 * iLin / (self.dim[0] - 1.0)) * (-1.0 + 2.0 * iLin / (self.dim[0] - 1.0))) <= np.sqrt(1 + (-1.0 + (2.0 * (np.floor(self.dim[0] / 2.0)-3.0)) / (self.dim[0]-1)) * (-1.0 + (2.0 * (np.floor(self.dim[0] / 2.0)-3.0)) / (self.dim[0]-1)))):
                        samples_elliptical += 1
            return (samples_elliptical * self.dim[2])/np.count_nonzero(self.mask)
        else:
            return np.prod(self.dim)/np.count_nonzero(self.mask)

    def subsample(self, kspace):
        # subsample k-space, i.e. apply subsampling mask
        return np.multiply(kspace, self.mask)

    
class PoissonDisc(Sampling):
    # variable-density Poisson-Disc subsampling (1D / 2D / 2D+time)
    def __init__(self, dim, acc, nCenter=1, vd_type=4, pF_value=1, pF_x=0, smpl_type=1, ellip_mask=0, p=2, n=2, iVerbose=0):
        super().__init__(dim=dim, acc=acc, trajectory='Poisson-Disc', is_elliptical=(ellip_mask>0)&(dim[1]>1))

        self.mode = 1  # Poisson-Disc {1}, Gaussian {2}
        self.nCenter = nCenter  # percentage of fully sampled center region
        self.vd_type = vd_type  # variable-density options: none {1}, central point {2}, central block {3}, central ellipse {4}, local adaptive variable-density (needs external fraction.txt file) {5}
        self.pF_value = pF_value  # ESPReSSo / Partial Fourier compactification factor [0.5:1]
        self.pF_x = 0  # ESPReSSo/Partial Fourier direction along width {0}, height {1}
        self.smpl_type = smpl_type  # Poisson-Disc sampling options: 1D Poisson-Disc {0} (chosen automatically if nY=1), 2D Poisson-Disc {1} (strictly obeying neighbourhood criterion), 2D pseudo Poisson-Disc {2} (approximately obeying neighbourhood criterion)
        self.ellip_mask = ellip_mask  # sample in elliptical scan region (>0: create inscribing ellipse with these points, =0: do not use it)
        self.p = p  # power of variable-density scaling
        self.n = n  # root of variable-density scaling
        self.body_region = 0  # body-region adaptive sampling, requires local adaptive variable-density sampling and fraction.txt file; {0}: not used, keep at 0
        self.iso_fac = 1  # isometry, keep at 1
        self.iVerbose = iVerbose  # verbosity level

    def generate_mask(self):
        numLin = self.dim[0]  # ky points
        numPar = self.dim[1]  # kz points
        nRep = self.dim[2]  # time points

        out_parameter = np.zeros((3, 1), dtype='float32')
        parameter_list = np.asarray([numLin, numPar, self.acc, self.mode, self.nCenter/100, self.pF_value, self.pF_x, nRep, self.vd_type, self.smpl_type, self.ellip_mask, self.p, self.n, self.body_region, self.iso_fac], dtype='float32')

        lMask = np.zeros((numPar, numLin, nRep))
        res = VDPDGauss.run(parameter_list, lMask, out_parameter)

        self.mask = lMask
        return lMask  # Z x Y x Time

class Gaussian(Sampling):
    # variable-density Gaussian subsampling (1D / 2D / 2D+time)
    def __init__(self, dim, acc, nCenter=1, vd_type=4, pF_value=1, pF_x=0, smpl_type=1, ellip_mask=0, p=2, n=2, iVerbose=0):
        super().__init__(dim=dim, acc=acc, trajectory='Gaussian', is_elliptical=(ellip_mask>0)&(dim[1]>1))

        self.mode = 2  # Poisson-Disc {1}, Gaussian {2}
        self.nCenter = nCenter  # percentage of fully sampled center region
        self.vd_type = vd_type  # variable-density options: none {1}, central point {2}, central block {3}, central ellipse {4}, local adaptive variable-density (needs external fraction.txt file) {5}
        self.pF_value = pF_value  # ESPReSSo / Partial Fourier compactification factor [0.5:1]
        self.pF_x = 0  # ESPReSSo/Partial Fourier direction along width {0}, height {1}
        self.smpl_type = smpl_type  # Poisson-Disc sampling options: 1D Poisson-Disc {0} (chosen automatically if nY=1), 2D Poisson-Disc {1} (strictly obeying neighbourhood criterion), 2D pseudo Poisson-Disc {2} (approximately obeying neighbourhood criterion)
        self.ellip_mask = ellip_mask  # sample in elliptical scan region (>0: create inscribing ellipse with these points, =0: do not use it)
        self.p = p  # power of variable-density scaling
        self.n = n  # root of variable-density scaling
        self.body_region = 0  # body-region adaptive sampling, requires local adaptive variable-density sampling and fraction.txt file; {0}: not used, keep at 0
        self.iso_fac = 1  # isometry, keep at 1
        self.iVerbose = iVerbose  # verbosity level

    def generate_mask(self):
        numLin = self.dim[0]  # ky points
        numPar = self.dim[1]  # kz points
        nRep = self.dim[2]  # time points

        out_parameter = np.zeros((3, 1), dtype='float32')
        parameter_list = np.asarray(
            [numLin, numPar, self.acc, self.mode, self.nCenter / 100, self.pF_value, self.pF_x, nRep, self.vd_type,
             self.smpl_type, self.ellip_mask, self.p, self.n, self.body_region, self.iso_fac], dtype='float32')

        lMask = np.zeros((numPar, numLin, nRep))
        res = VDPDGauss.run(parameter_list, lMask, out_parameter)

        self.mask = lMask
        return lMask  # Z x Y x Time

class Sampling2Dt(Sampling):
    def __init__(self, dim, acc, typ='VISTA', alph=0.28, sd=10, nIter=120, g=None, uni=None,
                 ss=0.25, fl=None, fs=1, s=1.4, tf=0.0, dsp=5):
        super().__init__(dim=dim, acc=acc, trajectory='2Dt', is_elliptical=False)
        self.typ = typ  # 'UIS': uniform interleaved sampling, 'VRS': variable density random sampling, 'VISTA': Variable density incoherent spatiotemporal acquisition
        self.alph = alph  # variable-density spreading: 0<alpha<1
        self.sd = sd  # random number generator seed
        self.nIter = nIter  # number of VISTA iterations
        self.g = g  # resample onto integer Cartesian grid every g iterations Default value: floor(nIter/6)
        self.uni = uni  # At uni iteration, reset to equivalent uniform sampling. Default value: floor(nIter/2)
        self.ss = ss  # Step-size for gradient descent. Default value: 0.25
        self.fl = fl  # Start checking fully sampledness at fl^th iteration. Default value: floor(nIter*5/6)
        self.fs = fs  # Performed time average has to be fully sampled, 0 for no, 1 for yes. Only works with VISTA. Default value: 1
        self.s = s  # Exponent of the potential energy term. Default value 1.4
        self.tf = tf  # Step-size in time direction wrt to phase-encoding direction; use zero for constant temporal resolution. Default value: 0.0
        self.dsp = dsp  # Display the distribution at every dsp^th iteration

    def generate_mask(self):
        p = self.dim[0]  # Number of phase encoding steps
        t = self.dim[2]  # Number of frames
        R = self.acc  # acceleration
        if self.g == None:
            self.g = math.floor(self.nIter/6)
        if self.uni == None:
            self.uni = math.floor(self.nIter/2)
        if self.fl == None:
            self.fl = math.floor(self.nIter*5/6)

        self.mask = sampling2dt(p, t, R, self.typ, self.alph, self.sd, self.nIter, self.g, self.uni, self.ss, self.fl,
                           self.fs, self.s, self.tf, self.dsp)
        return self.mask

class VISTA(Sampling2Dt):
    def __init__(self, dim, acc, alph=0.28, sd=10, nIter=120, g=None, uni=None,
                 ss=0.25, fl=None, fs=1, s=1.4, tf=0.0, dsp=5):
        super().__init__(dim, acc, 'VISTA', alph, sd, nIter, g, uni, ss, fl, fs, s, tf, dsp)

class UIS(Sampling2Dt):
    def __init__(self, dim, acc, alph=0.28, sd=10, nIter=120, g=None, uni=None,
                 ss=0.25, fl=None, fs=1, s=1.4, tf=0.0, dsp=5):
        super().__init__(dim, acc, 'UIS', alph, sd, nIter, g, uni, ss, fl, fs, s, tf, dsp)

class VRS(Sampling2Dt):
    def __init__(self, dim, acc, alph=0.28, sd=10, nIter=120, g=None, uni=None,
                 ss=0.25, fl=None, fs=1, s=1.4, tf=0.0, dsp=5):
        super().__init__(dim, acc, 'VRS', alph, sd, nIter, g, uni, ss, fl, fs, s, tf, dsp)
