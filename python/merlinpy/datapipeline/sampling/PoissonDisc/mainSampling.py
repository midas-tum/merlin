import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

lCompile = False
subsampleType = 1 # Poisson-Disc {1}, Gaussian {2}
numLin = 256  # ky points
numPar = 72  # kz points
acc = 20  # acceleration
nCenter = 1  # percentage of fully sampled center region
vd_type = 4  # variable-density options: none {1}, central point {2}, central block {3}, central ellipse {4}, local adaptive variable-density (needs external fraction.txt file) {5}
pF_value = 1  # ESPReSSo / Partial Fourier compactification factor [0.5:1]
pF_x = 0  # ESPReSSo/Partial Fourier direction along width {0}, height {1}
nRep = 4  # number of repetitions
smpl_type = 1  # Poisson-Disc sampling options: 2D Poisson-Disc {0} (chosen automatically if nY=1), 3D Poisson-Disc {1} (strictly obeying neighbourhood criterion), 3D pseudo Poisson-Disc {2} (approximately obeying neighbourhood criterion)
ellip_mask = 0  # sample in elliptical scan region
p = 2  # power of variable-density scaling
n = 2  # root of variable-density scaling
body_region = 0  # body-region adaptive sampling, keep at 0
iso_fac = 1  # isometry, keep at 1
lVerbose = 0 # verbosity level
kSpacePolar_LinInd = np.zeros((nRep*numLin*numPar, 1))
kSpacePolar_ParInd = np.zeros((nRep*numLin*numPar, 1))
phaseInd = np.zeros((nRep*numLin*numPar, 1))
out_parameter = np.zeros((3,1), dtype='float32')  # n_SamplesInSpace = [], nSampled = [], nb_spiral = []


if lCompile:
    os.system("./setup_VDPD.py clean")
    os.system("./setup_VDPD.py build")
    shutil.copyfile("build/lib.linux-x86_64-3.5/VDPD.cpython-35m-x86_64-linux-gnu.so", "./VDPD.so")

import VDPD as VDPD

parameter_list = np.asarray([numLin, numPar, acc, subsampleType, nCenter/100, pF_value, pF_x, nRep, vd_type, smpl_type, ellip_mask, p, n, body_region, iso_fac, lVerbose], dtype='float32')

res = VDPD.run(parameter_list, kSpacePolar_LinInd, kSpacePolar_ParInd, phaseInd, out_parameter)


# show and get sampling per repetition
mask_rep = np.zeros((numPar, numLin, nRep))
iMaxCol = 2
iRows = np.ceil(nRep/iMaxCol)
plt.figure()
for iRep in range(1, nRep+1):
    iVec = np.where(phaseInd == iRep-1)

    #iVec = np.asarray(iVec)

    for iI in iVec[0]:
        if (kSpacePolar_LinInd[iI] > 0) and (kSpacePolar_ParInd[iI] > 0):
            mask_rep[np.asscalar(kSpacePolar_ParInd[iI].astype(int)), np.asscalar(kSpacePolar_LinInd[iI].astype(int)), iRep-1] += 1

    plt.subplot(iRows, iMaxCol, iRep)
    #plt.xticks([])
    #plt.yticks([])
    #plt.grid(False)
    plt.imshow(mask_rep[:,:,iRep-1], cmap='gray')

    print('#samples repetition %d = %d\n' % (iRep, np.sum(mask_rep[:, :, iRep-1], axis=(0, 1))))

plt.show()