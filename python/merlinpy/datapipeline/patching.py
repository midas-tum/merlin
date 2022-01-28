'''
Copyright: 2016-2021 Thomas Kuestner (thomas.kuestner@med.uni-tuebingen.de) under Apache2 license
@author: Thomas Kuestner
'''

import numpy as np
import math

#########################################################################################################################################
#Module: compute_patchedshape (MAIN)                                                                                                    #
#Compute the shape of the patched array after patching depending on the patchSize and the patchOverlap.                                 #
#The patching-module contains two function:                                                                                             #
#Input: dicom_numpy_array ---> 3D dicom array (height, width, number of slices)                                                         #
#                              OR                                                                                                       #
#                              list/tuple of dicom_numpy_array shape                                                                    #
#########################################################################################################################################
def compute_patchedshape(dicom_numpy_array, patchSize, patchOverlap):
    if isinstance(dicom_numpy_array, (list, tuple)):
        dicom_numpy_array_shape = np.asarray(dicom_numpy_array)
    else:
        if type(dicom_numpy_array) is not np.ndarray:
            dicom_numpy_array = np.ndarray(dicom_numpy_array, dtype='f')
        dicom_numpy_array_shape = np.shape(dicom_numpy_array)

    if len(patchSize) > len(dicom_numpy_array_shape):
        print('Warning: dimension of patchSize (=%d) > image shape (=%d), cropping patchSize dimensions' % (len(patchSize), len(dicom_numpy_array_shape)))
        patchSize = patchSize[:np.ndim(dicom_numpy_array)]

    if patchOverlap < 1:
        dOverlap = np.multiply(patchSize, patchOverlap)
        dNotOverlap = np.round(np.multiply(patchSize, (1 - np.asarray(patchOverlap))))
    else:
        dOverlap = np.asarray(patchSize) - np.asarray(patchOverlap)
        dNotOverlap = np.multiply(np.ones_like(patchSize), patchOverlap)

    if len(patchSize) == 2:
        size_zero_pad = np.array(
            ([math.ceil((dicom_numpy_array_shape[0] - dOverlap[0]) / (dNotOverlap[0])) * dNotOverlap[0] + dOverlap[
                0],
              math.ceil((dicom_numpy_array_shape[1] - dOverlap[1]) / (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1]]))
        nbPatches = int(((size_zero_pad[0] - patchSize[0]) / (dNotOverlap[0]) + 1) * (
                    (size_zero_pad[1] - patchSize[1]) / (dNotOverlap[1]) + 1) * dicom_numpy_array_shape[2])

        return np.array((patchSize[0], patchSize[1], nbPatches))
    else:
        size_zero_pad = np.array(
            ([math.ceil((dicom_numpy_array_shape[0] - dOverlap[0]) / (dNotOverlap[0])) * dNotOverlap[0] + dOverlap[0],
              math.ceil((dicom_numpy_array_shape[1] - dOverlap[1]) / (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1],
              math.ceil((dicom_numpy_array_shape[2] - dOverlap[2]) / (dNotOverlap[2])) * dNotOverlap[2] + dOverlap[2]]))
        nbPatches_in_Y = int((size_zero_pad[0] - dOverlap[0]) / dNotOverlap[0])
        nbPatches_in_X = int((size_zero_pad[1] - dOverlap[1]) / dNotOverlap[1])
        nbPatches_in_Z = int((size_zero_pad[2] - dOverlap[2]) / dNotOverlap[2])
        nbPatches = nbPatches_in_X * nbPatches_in_Y * nbPatches_in_Z

        return np.array((patchSize[0], patchSize[1], patchSize[2], nbPatches))

#########################################################################################################################################
#Module: patching (MAIN)                                                                                                                #
#The module patching is responsible for splitting the dicom numpy array in patches depending on the patchSize and the                   #
#patchOverlap.                                                                                                                          #
#The patching-module contains two function:                                                                                             #
#patching2D: For 2D Patch-Splitting                                                                                                     #
#patching3D: For 3D Patch-Splitting                                                                                                     #
#########################################################################################################################################
def patching(dicom_numpy_array, patchSize, patchOverlap):
    if type(dicom_numpy_array) is not np.ndarray:
        dicom_numpy_array = np.ndarray(dicom_numpy_array, dtype='f')

    if len(patchSize) > np.ndim(dicom_numpy_array):
        print('Warning: dimension of patchSize (=%d) > image shape (=%d), cropping patchSize dimensions' % (len(patchSize), np.ndim(dicom_numpy_array)))
        patchSize = patchSize[:np.ndim(dicom_numpy_array)]

    if len(patchSize) == 2:
        return patching2D(dicom_numpy_array, patchSize, patchOverlap)
    else:
        return patching3D(dicom_numpy_array, patchSize, patchOverlap)

#########################################################################################################################################
#Function: patching2D                                                                                                                   #
#The function patching2D is responsible for splitting the dicom numpy array in patches depending on the patchSize and the patchOverlap. #                                                 #
#                                                                                                                                       #
#Input: dicom_numpy_array ---> 3D dicom array (height, width, number of slices)                                                         #
#       patchSize ---> size of patches, example: [40, 40], patchSize[0] = height, patchSize[1] = weight, height and weight can differ   #
#       patchOverlap ---> patchOverlap < 1: the ratio for overlapping, example: 0.25; patchOverlap >= 1: stride to move patch, e.g. 2   #
#                         can be scalar (same used for all dimensions) or list/tuple with same length as patchSize (i.e. #dimensions/   #
#                         rank of input)                                                                                                #
#Output: dPatches ---> 3D-Numpy-Array, which contain all Patches.                                                                       #
#                                                                                                                                       #
#########################################################################################################################################

def patching2D(dicom_numpy_array, patchSize, patchOverlap):
    if patchOverlap < 1:
        dOverlap = np.multiply(patchSize, patchOverlap)
        dNotOverlap = np.round(np.multiply(patchSize, (1 - np.asarray(patchOverlap))))
    else:
        dOverlap = np.asarray(patchSize) - np.asarray(patchOverlap)
        dNotOverlap = np.multiply(np.ones_like(patchSize), patchOverlap)

    size_zero_pad = np.array(([math.ceil((dicom_numpy_array.shape[0] - dOverlap[0]) / (dNotOverlap[0])) * dNotOverlap[0] + dOverlap[
        0], math.ceil((dicom_numpy_array.shape[1] - dOverlap[1]) / (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1]]))
    zero_pad = np.array(([int(size_zero_pad[0]) - dicom_numpy_array.shape[0], int(size_zero_pad[1]) - dicom_numpy_array.shape[1]]))
    zero_pad_part = np.array(([int(math.ceil(zero_pad[0] / 2)), int(math.ceil(zero_pad[1] / 2))]))

    # zero-padding of image so that correct amount of patches can be extracted
    Img_zero_pad = np.lib.pad(dicom_numpy_array, (
    (zero_pad_part[0], zero_pad[0] - zero_pad_part[0]), (zero_pad_part[1], zero_pad[1] - zero_pad_part[1]), (0, 0)),
                              mode='constant')

    nbPatches = int(((size_zero_pad[0]-patchSize[0])/(dNotOverlap[0])+1)*((size_zero_pad[1]-patchSize[1])/(dNotOverlap[1])+1)*dicom_numpy_array.shape[2])
    dPatches = np.zeros((patchSize[0], patchSize[1], nbPatches), dtype=float) #dtype=np.float32

    idxPatch = 0
    for iZ in range(0, dicom_numpy_array.shape[2], 1):
        for iY in range(0, int(size_zero_pad[0] - dOverlap[0]), int(dNotOverlap[0])):
            for iX in range(0, int(size_zero_pad[1] - dOverlap[1]), int(dNotOverlap[1])):
                dPatch = Img_zero_pad[iY:iY + patchSize[0], iX:iX + patchSize[1], iZ]
                dPatches[:,:,idxPatch] = dPatch
                idxPatch += 1

    #print("2D patching done!")
    return dPatches


#########################################################################################################################################
#Function: patching3D                                                                                                                   #
#The function patching3D is responsible for splitting the dicom numpy array in patches depending on the patchSize and the               #
#patchOverlap.                                                                                                                          #
#                                                                                                                                       #
#Input: dicom_numpy_array ---> 3D dicom array (height, width, number of slices)                                                         #
#       patchSize ---> size of patches, example: [40, 40, 40], patchSize[0] = height >=1, patchSize[1] = weight >=1,                    #
#                       patchSize[2] = depth >=1  #                                                                                     #
#       patchOverlap ---> patchOverlap < 1: the ratio for overlapping, example: 0.25; patchOverlap >= 1: stride to move patch, e.g. 2   #
#                         can be scalar (same used for all dimensions) or list/tuple with same length as patchSize (i.e. #dimensions/   #
#                         rank of input)                                                                                                #
#Output: dPatches ---> 3D-Numpy-Array, which contain all Patches.                                                                       #
#                                                                                                                                       #
#########################################################################################################################################

def patching3D(dicom_numpy_array, patchSize, patchOverlap):
    if patchOverlap < 1:
        dOverlap = np.multiply(patchSize, patchOverlap)
        dNotOverlap = np.round(np.multiply(patchSize, (1 - np.asarray(patchOverlap))))
    else:
        dOverlap = np.asarray(patchSize) - np.asarray(patchOverlap)
        dNotOverlap = np.multiply(np.ones_like(patchSize), patchOverlap)

    size_zero_pad = np.array(
        ([math.ceil((dicom_numpy_array.shape[0] - dOverlap[0]) / (dNotOverlap[0])) * dNotOverlap[0] + dOverlap[0],
          math.ceil((dicom_numpy_array.shape[1] - dOverlap[1]) / (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1],
          math.ceil((dicom_numpy_array.shape[2] - dOverlap[2]) / (dNotOverlap[2])) * dNotOverlap[2] + dOverlap[2]]))
    zero_pad = np.array(([int(size_zero_pad[0]) - dicom_numpy_array.shape[0],
                          int(size_zero_pad[1]) - dicom_numpy_array.shape[1],
                          int(size_zero_pad[2]) - dicom_numpy_array.shape[2]]))
    zero_pad_part = np.array(([int(math.ceil(zero_pad[0] / 2)),
                               int(math.ceil(zero_pad[1] / 2)),
                               int(math.ceil(zero_pad[2] / 2))]))

    Img_zero_pad = np.lib.pad(dicom_numpy_array,
                              ((zero_pad_part[0], zero_pad[0] - zero_pad_part[0]),
                               (zero_pad_part[1], zero_pad[1] - zero_pad_part[1]),
                               (zero_pad_part[2], zero_pad[2] - zero_pad_part[2])),
                              mode='constant')

    #nbPatches = ((size_zero_pad[0] - patchSize[0]) / (dNotOverlap[0]) + 1) * (
    #            (size_zero_pad[1] - patchSize[1]) / (dNotOverlap[1]) + 1) * (
    #                        (size_zero_pad[2] - patchSize[2]) / (np.round(dNotOverlap[2])) + 1)

    nbPatches_in_Y = int((size_zero_pad[0] - dOverlap[0]) / dNotOverlap[0])
    nbPatches_in_X = int((size_zero_pad[1] - dOverlap[1]) / dNotOverlap[1])
    nbPatches_in_Z = int((size_zero_pad[2] - dOverlap[2]) / dNotOverlap[2])
    nbPatches = nbPatches_in_X * nbPatches_in_Y * nbPatches_in_Z

    dPatches = np.zeros((patchSize[0], patchSize[1], patchSize[2], int(nbPatches)), dtype=float)
    idxPatch = 0
    for iZ in range(0, int(size_zero_pad[2] - dOverlap[2]), int(dNotOverlap[2])):
        for iY in range(0, int(size_zero_pad[0] - dOverlap[0]), int(dNotOverlap[0])):
            for iX in range(0, int(size_zero_pad[1] - dOverlap[1]), int(dNotOverlap[1])):
                dPatch = Img_zero_pad[iY:iY + patchSize[0], iX:iX + patchSize[1], iZ:iZ + patchSize[2]]
                dPatches[:, :, :, idxPatch] = dPatch
                idxPatch += 1

    #print("3D patching done!")
    return dPatches


#########################################################################################################################################
#Module: Unpatching (MAIN)                                                                                                              #
#The module Unpatching is responsible for reconstructing the probability/images. To reconstruct the images the means of all             #
#probabilities from overlapping patches are calculated and are assigned to every pixel-image. It's important to consider the order of   #
#dimensions within the algorithm of the module RigidPatching. In this case the order is: weight(x), height(y), depth(z)                 #
#The Unpatching-module contains two function:                                                                                           #
#unpatching2D: For 2D Patch-Splitting                                                                                                   #
#unpatching3D: For 3D Patch-Splitting                                                                                                   #
#########################################################################################################################################
def unpatching(patches, patchSize, patchOverlap, actualSize, overlapRegion='avg'):
    if type(patches) is not np.ndarray:
        patches = np.ndarray(patches, dtype='f')

    if len(patchSize) > len(actualSize):
        print('Warning: dimension of patchSize (=%d) > image shape (=%d), cropping patchSize dimensions' % (len(patchSize), len(actualSize)))
        patchSize = patchSize[:np.ndim(dicom_numpy_array)]

    if patches.ndim == 3:
        return unpatching2D(patches, patchSize, patchOverlap, actualSize, overlapRegion)
    else:
        return unpatching3D(patches, patchSize, patchOverlap, actualSize, overlapRegion)

#########################################################################################################################################
#Function: unpatching2D                                                                                                                 #
#The function unpatching2D has the task to reconstruct the images.  Every patch contains the probability of every class.                #                                                        #
#To visualize the probabilities it is important to reconstruct the probability-images. This function is used for 2D patching.           #                                                                                                                                    #
#Input: patches ---> patch array: patchSizeX x patchSizeY x nPatches (not one-hot encoded classes for seg masks!)                       #                                                                        #
#       patchSize ---> size of patches, example: [40, 40, 10], patchSize[0] = height, patchSize[1] = weight, patchSize[2] = depth       #
#       patchOverlap ---> patchOverlap < 1: the ratio for overlapping, example: 0.25; patchOverlap >= 1: stride to move patch, e.g. 2   #
#                         can be scalar (same used for all dimensions) or list/tuple with same length as patchSize (i.e. #dimensions/   #
#                         rank of input)                                                                                                #
#       actualSize ---> the actual size of the chosen mrt-layer: example: ab, t1_tse_tra_Kopf_0002; actual size = [256, 196, 40]        #                                                          #
#       overlapRegion --> handling of overlapping regions: 'avg' = average, 'add' = addition, 'owr' = overwrite                         #
#Output: unpatchImg ---> 3D-Numpy-Array, which contains the probability of every image pixel.                                           #
#########################################################################################################################################

def unpatching2D(patches, patchSize, patchOverlap, actualSize, overlapRegion='avg'):
    iCorner = [0, 0, 0]
    if patchOverlap < 1:
        dOverlap = np.multiply(patchSize, patchOverlap)
        dNotOverlap = np.round(np.multiply(patchSize, (1 - np.asarray(patchOverlap))))
    else:
        dOverlap = np.asarray(patchSize) - np.asarray(patchOverlap)
        dNotOverlap = np.multiply(np.ones_like(patchSize), patchOverlap)
    #dOverlap = np.round(np.multiply(patchSize, patchOverlap))
    #dNotOverlap = [patchSize[0] - dOverlap[0], patchSize[1] - dOverlap[1]]

    paddedSize = [int(math.ceil((actualSize[0] - dOverlap[0]) / (dNotOverlap[0])) * dNotOverlap[0] + dOverlap[0]),
                  int(math.ceil((actualSize[1] - dOverlap[1]) / (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1]),
                  actualSize[2]]

    unpatchImg = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))
    numVal = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))

    for iIndex in range(0, patches.shape[2], 1):
        lMask = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))

        lMask[iCorner[0]:iCorner[0] + int(patchSize[0]), iCorner[1]:iCorner[1] + int(patchSize[1]), iCorner[2]] = 1

        if overlapRegion == 'owr':
            unpatchImg[iCorner[0]:iCorner[0] + int(patchSize[0]), iCorner[1]:iCorner[1] + int(patchSize[1]), iCorner[2]] \
                = patches[:, :, iIndex]
        else:  # 'avg', 'add'
            unpatchImg[iCorner[0]:iCorner[0] + int(patchSize[0]), iCorner[1]:iCorner[1] + int(patchSize[1]), iCorner[2]] \
                = np.add(unpatchImg[iCorner[0]:iCorner[0] + int(patchSize[0]),
                         iCorner[1]:iCorner[1] + int(patchSize[1]),
                         iCorner[2]],
                         patches[:, :, iIndex])

        lMask = lMask == 1
        numVal[lMask] = numVal[lMask] + 1

        iCorner[1] = int(iCorner[1] + dNotOverlap[1])

        if iCorner[1] + patchSize[1] - 1 > paddedSize[1]:
            iCorner[1] = 0
            iCorner[0] = int(iCorner[0] + dNotOverlap[0])

        if iCorner[0] + patchSize[0] - 1 > paddedSize[0]:
            iCorner[0] = 0
            iCorner[1] = 0
            iCorner[2] = int(iCorner[2] + 1)
            #print(str(iCorner[2] / actualSize[2] * 100) + "%")

    if overlapRegion == 'avg':
        unpatchImg = np.divide(unpatchImg, numVal)

    if paddedSize == actualSize:
        pass
    else:
        pad_y = int((paddedSize[0] - actualSize[0]) / 2)
        pad_x = int((paddedSize[1] - actualSize[1]) / 2)

        pad_y_max = int(paddedSize[0] - (paddedSize[0] - actualSize[0] - pad_y))
        pad_x_max = int(paddedSize[1] - (paddedSize[1] - actualSize[1] - pad_x))

        unpatchImg = unpatchImg[pad_y:pad_y_max, pad_x:pad_x_max, :]

    return unpatchImg


#########################################################################################################################################
#Function: unpatching3D                                                                                                                 #
#The function unpatching3D has the task to reconstruct the images. Every patch contains the probability of every class.                 #
#To visualize the probabilities it is inportant to reconstruct the probability-images. This function is used for 3D patching.           #                                                                                                                                    #
#Input: patches ---> patch array: patchSizeX x patchSizeY x patchSizeZ x nPatches (not one-hot encoded classes for seg masks!)          #                                                                                            #
#       patchSize ---> size of patches, example: [40, 40, 10], patchSize[0] = height, patchSize[1] = weight, patchSize[2] = depth       #
#       patchOverlap ---> patchOverlap < 1: the ratio for overlapping, example: 0.25; patchOverlap >= 1: stride to move patch, e.g. 2   #
#                         can be scalar (same used for all dimensions) or list/tuple with same length as patchSize (i.e. #dimensions/   #
#                         rank of input)                                                                                                #
#       actualSize ---> the actual size of the chosen mrt-layer: example: ab, t1_tse_tra_Kopf_0002; actual size = [256, 196, 40]        #
#       overlapRegion --> handling of overlapping regions: 'avg' = average, 'add' = addition, 'owr' = overwrite                         #
#Output: unpatchImg ---> 3D-Numpy-Array, which contains the probability of every image pixel.                                           #
#########################################################################################################################################

def unpatching3D(patches, patchSize, patchOverlap, actualSize, overlapRegion='avg'):
    iCorner = [0, 0, 0]
    if patchOverlap < 1:
        dOverlap = np.multiply(patchSize, patchOverlap)
        dNotOverlap = np.round(np.multiply(patchSize, (1 - np.asarray(patchOverlap))))
    else:
        dOverlap = np.asarray(patchSize) - np.asarray(patchOverlap)
        dNotOverlap = np.multiply(np.ones_like(patchSize), patchOverlap)
    #dOverlap = np.round(np.multiply(patchSize, patchOverlap))
    #dNotOverlap = [patchSize[0] - dOverlap[0], patchSize[1] - dOverlap[1], patchSize[2] - dOverlap[2]]

    paddedSize = [int(math.ceil((actualSize[0] - dOverlap[0]) / (dNotOverlap[0])) * dNotOverlap[0] + dOverlap[
        0]), int(math.ceil((actualSize[1] - dOverlap[1]) / (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1]),
                  int(math.ceil((actualSize[2] - dOverlap[2]) / (dNotOverlap[2])) * dNotOverlap[2] + dOverlap[2])]

    unpatchImg = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))
    numVal = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))

    for iIndex in range(0, patches.shape[3], 1):
        #print(iIndex)
        lMask = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))
        lMask[iCorner[0]: iCorner[0] + patchSize[0], iCorner[1]: iCorner[1] + patchSize[1], iCorner[2]: iCorner[2] + patchSize[2]] = 1

        if overlapRegion == 'owr':
            unpatchImg[iCorner[0]:iCorner[0] + patchSize[0], iCorner[1]: iCorner[1] + patchSize[1],
            iCorner[2]: iCorner[2] + patchSize[2]] \
                =  patches[:, :, :, iIndex]
        else:  # 'avg', 'add'
            unpatchImg[iCorner[0]:iCorner[0] + patchSize[0], iCorner[1]: iCorner[1] + patchSize[1], iCorner[2]: iCorner[2] + patchSize[2]] \
                = np.add(unpatchImg[iCorner[0]: iCorner[0] + patchSize[0], iCorner[1]: iCorner[1] + patchSize[1],
                         iCorner[2]: iCorner[2] + patchSize[2]], patches[:,:,:,iIndex])

        lMask = lMask == 1
        numVal[lMask] = numVal[lMask] + 1

        iCorner[1] = int(iCorner[1] + dNotOverlap[1])

        if iCorner[1] + patchSize[1] - 1 > paddedSize[1]:
            iCorner[1] = 0
            iCorner[0] = int(iCorner[0] + dNotOverlap[0])

        if iCorner[0] + patchSize[0] - 1 > paddedSize[0]:
            iCorner[0] = 0
            iCorner[1] = 0
            iCorner[2] = int(iCorner[2] + dNotOverlap[2])
            #print(str(iCorner[2] / actualSize[2] * 100) + "%")

    if overlapRegion == 'avg':
        unpatchImg = np.divide(unpatchImg, numVal)

    if paddedSize == actualSize:
        pass
    else:
        pad_y = int((paddedSize[0] - actualSize[0]) / 2)
        pad_x = int((paddedSize[1] - actualSize[1]) / 2)
        pad_z = int((paddedSize[2] - actualSize[2]) / 2)

        pad_y_max = int(paddedSize[0] - (paddedSize[0] - actualSize[0] - pad_y))
        pad_x_max = int(paddedSize[1] - (paddedSize[1] - actualSize[1] - pad_x))
        pad_z_max = int(paddedSize[2] - (paddedSize[2] - actualSize[2] - pad_z))

        unpatchImg = unpatchImg[pad_y:pad_y_max, pad_x:pad_x_max, pad_z:pad_z_max]

    return unpatchImg
