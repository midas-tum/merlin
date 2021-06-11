import numpy as np
import math
import VD_CASPR_CINE as VD_CASPR_CINE
from VISTA import vista
import VDPD as VDPD

class Sampling
    # subsampling of phase-encoding directions (y/z) and along time (t)
    def __init__(self, dim, acc, trajectory):
        self.dim = dim  # x - y - z - t - MRcha
        self.acc = acc  # acceleration factor
        self.trajectory = trajectory  # CASPR, Poisson-Disc, Gaussian, GoldenRadial, TinyGoldenRadial,
        self.mask = []

    def plot_mask(self):
        numLin = self.dim[1]  # ky points
        numPar = self.dim[2]  # kz points
        nRep = self.dim[3]  # time points
        iMaxCol = 2
        iRows = np.ceil(nRep / iMaxCol)
        plt.figure()
        for iRep in range(1, nRep + 1):
            plt.subplot(iRows, iMaxCol, iRep)
            plt.imshow(self.mask[:, :, iRep - 1], cmap='gray')

            print('#samples repetition %d = %d\n' % (iRep, np.sum(mask_rep[:, :, iRep - 1], axis=(0, 1))))
        plt.show()

    def subsample(self, kspace):
        return np.multiply(kspace, self.mask)

class CASPR(Sampling):
    # variable-density CASPR subsampling (2D / 2D+time)
    def __init__(self, dim, acc, mode='interleaved', nSegments=10, nCenter=15, isVariable=1, isGolden=2, isInOut=1, isCenter=0, isSamePattern=0, iVerbose=0):
        super().__init__(dim=dim, acc=acc, trajectory='CASPR')
        self.mode = mode  # 'interleaved' (CINE), 'noninterleaved' (free-running, CMRA, T2Mapping, ...)
        self.nSegments = nSegments  # #segments = #rings in sampling pattern
        self.nCenter = nCenter  # percentage of fully sampled center region
        self.isVariable = isVariable  # VDCASPR (=1) or CASPR (=0)
        self.isGolden = isGolden  # golden angle increment between spirals (=1), 0 = linear-linear, 1=golden-golden, 2=tinyGolden-golden, 3=linear-golden, 4=noIncr-golden
        self.isInOut = isInOut  # spiral in/out sampling => for isGolden=1 & isInOut=1: use tinyGolden-Golden-tinyGolden-Golden-... increments
        self.isCenter = isCenter  # sample center point
        self.isSamePattern = isSamePattern  # same sampling pattern per phase/contrast, i.e. no golden/tiny-golden angle increment between them (but still inside the pattern if isGolden==1)
        self.iVerbose = iVerbose  # 0=silent, 1=normal output, 2=all output

    def generate_mask(self):
        numLin = self.dim[1]  # ky points (store it the other way round, for right dim)
        numPar = self.dim[2]  # kz points
        nRep = self.dim[3]  # time points

        lMask = np.zeros((numPar, numLin))
        kSpacePolar_LinInd = np.zeros((nRep * numLin * numPar, 1))
        kSpacePolar_ParInd = np.zeros((nRep * numLin * numPar, 1))
        out_parameter = np.zeros((3, 1), dtype='float32')
        parameter_list = np.asarray(
            [numLin, numPar, self.acc, self.nCenter, self.nSegments, nRep, self.isGolden, self.isVariable, self.isInOut, self.isCenter, self.isSamePattern,
             self.iVerbose], dtype='float32')
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

        self.mask = mask_rep
        return mask_rep  # Z x Y x Time
    
class VDPD(Sampling):
    # variable-density Poisson-Disc subsampling (1D / 2D / 2D+time)
    def __init__(self, dim, acc, nCenter=1, vd_type=4, pF_value=1, pF_x=0, smpl_type=1, ellip_mask=0, p=2, n=2, iVerbose=0):
        super().__init__(dim=dim, acc=acc, trajectory='Poisson-Disc')

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
        numLin = self.dim[1]  # ky points
        numPar = self.dim[2]  # kz points
        nRep = self.dim[3]  # time points

        kSpacePolar_LinInd = np.zeros((nRep * numLin * numPar, 1))
        kSpacePolar_ParInd = np.zeros((nRep * numLin * numPar, 1))
        phaseInd = np.zeros((nRep * numLin * numPar, 1))
        out_parameter = np.zeros((3, 1), dtype='float32')
        parameter_list = np.asarray([numLin, numPar, self.acc, self.mode, self.nCenter/100, self.pF_value, self.pF_x, nRep, self.vd_type, self.smpl_type, self.ellip_mask, self.p, self.n, self.body_region, self.iso_fac, self.iVerbose], dtype='float32')

        res = VDPD.run(parameter_list, kSpacePolar_LinInd, kSpacePolar_ParInd, phaseInd, out_parameter)
        mask_rep = np.zeros((numPar, numLin, nRep))
        for iRep in range(1, nRep + 1):
            iVec = np.where(phaseInd == iRep - 1)
            # iVec = np.asarray(iVec)
            for iI in iVec[0]:
                if (kSpacePolar_LinInd[iI] > 0) and (kSpacePolar_ParInd[iI] > 0):
                    mask_rep[np.asscalar(kSpacePolar_ParInd[iI].astype(int)), np.asscalar(kSpacePolar_LinInd[iI].astype(int)), iRep - 1] += 1

        self.mask = mask_rep
        return mask_rep  # Z x Y x Time

class Gaussian(Sampling):
    # variable-density Gaussian subsampling (1D / 2D / 2D+time)
    def __init__(self, dim, acc, nCenter=1, vd_type=4, pF_value=1, pF_x=0, smpl_type=1, ellip_mask=0, p=2, n=2, iVerbose=0):
        super().__init__(dim=dim, acc=acc, trajectory='Gaussian')

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
        numLin = self.dim[1]  # ky points
        numPar = self.dim[2]  # kz points
        nRep = self.dim[3]  # time points

        kSpacePolar_LinInd = np.zeros((nRep * numLin * numPar, 1))
        kSpacePolar_ParInd = np.zeros((nRep * numLin * numPar, 1))
        phaseInd = np.zeros((nRep * numLin * numPar, 1))
        out_parameter = np.zeros((3, 1), dtype='float32')
        parameter_list = np.asarray([numLin, numPar, self.acc, self.mode, self.nCenter/100, self.pF_value, self.pF_x, nRep, self.vd_type, self.smpl_type, self.ellip_mask, self.p, self.n, self.body_region, self.iso_fac, self.iVerbose], dtype='float32')

        res = VDPD.run(parameter_list, kSpacePolar_LinInd, kSpacePolar_ParInd, phaseInd, out_parameter)
        mask_rep = np.zeros((numPar, numLin, nRep))
        for iRep in range(1, nRep + 1):
            iVec = np.where(phaseInd == iRep - 1)
            # iVec = np.asarray(iVec)
            for iI in iVec[0]:
                if (kSpacePolar_LinInd[iI] > 0) and (kSpacePolar_ParInd[iI] > 0):
                    mask_rep[np.asscalar(kSpacePolar_ParInd[iI].astype(int)), np.asscalar(kSpacePolar_LinInd[iI].astype(int)), iRep - 1] += 1

        self.mask = mask_rep
        return mask_rep  # Z x Y x Time
    
    
class VISTA(Sampling):
    def __init__(self, dim, acc, typ, alph=0.28, sd=10, nIter=120, g=None, uni=None,
                 ss=0.25, fl=None, fs=1, s=1.4, tf=0.0, dsp=5):
        self.typ = typ  # 'UIS', 'VRS', 'VISTA'
        self.alph = alph
        self.sd = sd
        self.nIter = nIter
        self.g = g
        self.uni = uni
        self.ss = ss
        self.fl = fl
        self.fs = fs
        self.s = s
        self.tf = tf
        self.dsp = dsp
        super().__init__(dim=dim, acc=acc, trajectory='VISTA')

    def generate_mask(self):
        p = self.dim[1]  # Number of phase encoding steps
        t = self.dim[3]  # Number of frames
        R = self.acc
        if self.g == None:
            self.g = math.floor(self.nIter/6)
        if self.uni == None:
            self.uni = math.floor(self.nIter/2)
        if self.fl == None:
            self.fl = math.floor(self.nIter*5/6)

        mask_VISTA = vista(p, t, R, self.typ, self.alph, self.sd, self.nIter, self.g, self.uni, self.ss, self.fl,
                           self.fs, self.s, self.tf, self.dsp)
        self.mask = mask_VISTA
        return mask_VISTA


class VDPD(Sampling):
    # variable-density Poisson-Disc subsampling (1D / 2D / 2D+time)
    def __init__(self, dim, acc, nCenter=1, vd_type=4, pF_value=1, pF_x=0, smpl_type=1, ellip_mask=0, p=2, n=2, iVerbose=0):
        super().__init__(dim=dim, acc=acc, trajectory='Poisson-Disc')

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
        numLin = self.dim[1]  # ky points
        numPar = self.dim[2]  # kz points
        nRep = self.dim[3]  # time points

        kSpacePolar_LinInd = np.zeros((nRep * numLin * numPar, 1))
        kSpacePolar_ParInd = np.zeros((nRep * numLin * numPar, 1))
        phaseInd = np.zeros((nRep * numLin * numPar, 1))
        out_parameter = np.zeros((3, 1), dtype='float32')
        parameter_list = np.asarray([numLin, numPar, self.acc, self.mode, self.nCenter/100, self.pF_value, self.pF_x, nRep, self.vd_type, self.smpl_type, self.ellip_mask, self.p, self.n, self.body_region, self.iso_fac, self.iVerbose], dtype='float32')

        res = VDPD.run(parameter_list, kSpacePolar_LinInd, kSpacePolar_ParInd, phaseInd, out_parameter)
        mask_rep = np.zeros((numPar, numLin, nRep))
        for iRep in range(1, nRep + 1):
            iVec = np.where(phaseInd == iRep - 1)
            # iVec = np.asarray(iVec)
            for iI in iVec[0]:
                if (kSpacePolar_LinInd[iI] > 0) and (kSpacePolar_ParInd[iI] > 0):
                    mask_rep[np.asscalar(kSpacePolar_ParInd[iI].astype(int)), np.asscalar(kSpacePolar_LinInd[iI].astype(int)), iRep - 1] += 1

        self.mask = mask_rep
        return mask_rep  # Z x Y x Time

class Gaussian(Sampling):
    # variable-density Gaussian subsampling (1D / 2D / 2D+time)
    def __init__(self, dim, acc, nCenter=1, vd_type=4, pF_value=1, pF_x=0, smpl_type=1, ellip_mask=0, p=2, n=2, iVerbose=0):
        super().__init__(dim=dim, acc=acc, trajectory='Gaussian')

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
        numLin = self.dim[1]  # ky points
        numPar = self.dim[2]  # kz points
        nRep = self.dim[3]  # time points

        kSpacePolar_LinInd = np.zeros((nRep * numLin * numPar, 1))
        kSpacePolar_ParInd = np.zeros((nRep * numLin * numPar, 1))
        phaseInd = np.zeros((nRep * numLin * numPar, 1))
        out_parameter = np.zeros((3, 1), dtype='float32')
        parameter_list = np.asarray([numLin, numPar, self.acc, self.mode, self.nCenter/100, self.pF_value, self.pF_x, nRep, self.vd_type, self.smpl_type, self.ellip_mask, self.p, self.n, self.body_region, self.iso_fac, self.iVerbose], dtype='float32')

        res = VDPD.run(parameter_list, kSpacePolar_LinInd, kSpacePolar_ParInd, phaseInd, out_parameter)
        mask_rep = np.zeros((numPar, numLin, nRep))
        for iRep in range(1, nRep + 1):
            iVec = np.where(phaseInd == iRep - 1)
            # iVec = np.asarray(iVec)
            for iI in iVec[0]:
                if (kSpacePolar_LinInd[iI] > 0) and (kSpacePolar_ParInd[iI] > 0):
                    mask_rep[np.asscalar(kSpacePolar_ParInd[iI].astype(int)), np.asscalar(kSpacePolar_LinInd[iI].astype(int)), iRep - 1] += 1

        self.mask = mask_rep
        return mask_rep  # Z x Y x Time