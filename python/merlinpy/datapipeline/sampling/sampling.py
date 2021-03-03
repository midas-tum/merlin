import numpy as np
import math
import VD_CASPR_CINE as VD_CASPR_CINE
from VISTA import vista

class Sampling
    def __init__(self, dim, acc, trajectory):
        self.dim = dim  # x - y - z - t - MRcha
        self.acc = acc  # acceleration factor
        self.trajectory = trajectory  # CASPR, Poisson-Disc, Gaussian, GoldenRadial, TinyGoldenRadial,

    def generate_mask(self):
        mask = []
        self.mask = mask
        return mask

    def subsample(self, kspace):
        return np.multiply(kspace, self.mask)

class CASPR(Sampling):
    def __init__(self, dim, acc, mode='interleaved', nSegments=10, nCenter=15, isVariable=1, isGolden=2, isInOut=1, isCenter=0, isSamePattern=0, iVerbose=0):
        self.mode = mode  # 'interleaved' (CINE), 'noninterleaved' (free-running, CMRA, T2Mapping, ...)
        self.nSegments = nSegments  # #segments = #rings in sampling pattern
        self.nCenter = nCenter  # percentage of fully sampled center region
        self.isVariable = isVariable  # VDCASPR (=1) or CASPR (=0)
        self.isGolden = isGolden  # golden angle increment between spirals (=1), 0 = linear-linear, 1=golden-golden, 2=tinyGolden-golden, 3=linear-golden, 4=noIncr-golden
        self.isInOut = isInOut  # spiral in/out sampling => for isGolden=1 & isInOut=1: use tinyGolden-Golden-tinyGolden-Golden-... increments
        self.isCenter = isCenter  # sample center point
        self.isSamePattern = isSamePattern  # same sampling pattern per phase/contrast, i.e. no golden/tiny-golden angle increment between them (but still inside the pattern if isGolden==1)
        self.iVerbose = iVerbose  # 0=silent, 1=normal output, 2=all output
        super().__init__(dim=dim, acc=acc, trajectory='CASPR')

    def generate_mask(self):
        sType = 'CINE'
        sMode = 'interleaved'  # 'interleaved' (CINE), 'noninterleaved' (free-running, CMRA, T2Mapping, ...)
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
