import numpy as np
import math
from VISTA import vista


class Sampling:
    def __init__(self, dim, acc, trajectory):
        self.dim = dim  # nFE, nPE, COIL_DIM, TIME_DIM, SLICE_DIM
        self.acc = acc  # acceleration factor
        self.trajectory = trajectory  # CASPR, Poisson-Disc, Gaussian, GoldenRadial, TinyGoldenRadial,

    def generate_mask(self):
        mask = []
        self.mask = mask
        return mask

    def subsample(self, kspace):
        return np.multiply(kspace, self.mask)


class VISTA_(Sampling):
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
        p = int(self.dim[1])  # Number of phase encoding steps
        t = int(self.dim[3])  # Number of frames
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

