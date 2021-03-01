
import numpy as np
import math

eps = np.spacing(1)

def isempty(number): #if array is empty, return 1
    if any(number) == True:
        num = 0
    else:
        num = 1
    return num


def computeU(P, T, p, R, alph, s=1.4):
    """
    Compute potential energy (U) of the distribution
    :param P:
    :param T:
    :param p: Number of phase encoding steps
    :param R: Net acceleration rate
    :param alph: 0<alph<1 controls sampling density; 0: uniform density, 1: maximally non-uniform density
    :param s: Exponent of the potenital energy term. Default value 1.4
    :return:
    """

    N = len(P)
    sig = p / 5                # Std of the Gaussian envelope for sampling density
    a = max(R / 10 + 0.25, 1)  # Scaling of time dimension; frames are "a" units apart

    U = 0
    for i in range(N):
        k = list(range(i))
        k.extend(list(range(i + 1, N)))
        k = np.array(k)
        U = U + 1 / 2 * ((1 - alph * math.exp(-P[i] ** 2 / (2 * sig ** 2))) * (
                    1 - alph * math.exp(-P[k] ** 2 / (2 * sig ** 2)))) / (
                        (P[i] - P[k]) ** 2 + (a * (T[i] - T[k])) ** 2) ** (s / 2)

    return U


def excludeOuter(tmp, p):
    """
    Remove the "voids" that are contiguous to the boundary
    """
    tmp = np.array(sorted(tmp, reverse=True))
    cnd = abs(np.diff(np.hstack((math.ceil(p / 2), tmp))))
    if max(cnd) == 1:
        tmp = np.array([])
    else:
        tmp = tmp[np.where(cnd > 1)[0][0] - 1:]

    tmp = np.array(sorted(tmp))
    cnd = abs(np.diff(np.hstack(-math.floor(p / 2) - 1, tmp)))
    if max(cnd) == 1:
        tmp = np.array([])
    else:
        tmp = tmp[np.where(cnd > 1)[0][0] - 1:]

    return tmp

def fillK(P, T, Pacc, Tacc, p, R, alph, s=1.4, fr=1):
    """
    Ensures time-average of VISTA is fully sampled(except for the outer mostregion)

    p: Number of phase encoding steps
    fr: What fraction of the time-averaged should be fully sampled. Default value: 1
    """

    # empty locations
    tmp = np.setdiff1d(np.array([x for x in range(-math.floor(fr * p / 2))]), P)
    tmp2 = np.array(sorted(list(map(abs, tmp))))
    ords = np.argsort(tmp)
    tmp2 = np.multiply(tmp2, np.sign(tmp[ords]))  # Sorted (from center-out) empty locations

    while len(tmp2) > 0:
        ind = tmp2[1]  # the hole (PE location) to be filled

        # ind:数，eps:数
        # P: 传入的ph，长度为N
        # T: 传入的ti，ndarray
        can = []
        for every_P in P:
            if np.sign(ind + eps) * every_P > np.sign(ind + eps) * ind:
                can.append(every_P)
        can = np.array(can)  # Find candidates which are on the boundary side of "ind"
        # can: ndarray
        # fi can is empty, then break
        if isempty(can) == 1:
            break
        else:
            Pcan = []
            for every_can in can:
                if np.sign(ind + eps) * (every_can - ind) == min(np.sign(ind + eps) * (every_can - ind)):
                    Pcan.append(every_can)
            Pcan = np.array(Pcan)
            Tcan = []
            for every_P_ind in range(len(P)):
                if P[every_P_ind] == Pcan[0]:
                    Tcan.append(T[every_P_ind])
            Tcan = np.array(Tcan)

            # Pacc: ph，ndarray
            U = np.Inf
            for i in range(len(Pcan)):
                Ptmp = Pacc
                Ttmp = Tacc
                lentmp = min(len(Pacc), len(Tacc))
                for ind_tmp in range(lentmp):
                    if Pacc[ind_tmp] == Pcan[i] and Tacc[ind_tmp] == Tcan[i]:
                        Ptmp[ind_tmp] = ind
                Utmp = computeU(Ptmp, Ttmp, p, R, alph, s)  # Compute engergy U for the ith candidate
                if Utmp <= U:
                    slc = i
                    U = Utmp

            lentmp2 = min(len(Pacc), len(Tacc))
            for ind_tmp2 in range(lentmp2):
                if Pacc[ind_tmp2] == Pcan[slc] and Tacc == Tcan[slc]:
                    Pacc[ind_tmp2] = ind  # Fill the hole with the appropriate candidate
            lentmp3 = min(len(P), len(T))
            for ind_tmp3 in range(lentmp3):
                if P[ind_tmp3] == Pcan[slc] and T[ind_tmp3] == Tcan[slc]:
                    P[ind_tmp3] = ind  # Fill the hole with the approprate candidate
            
            tmp = np.setdiff1d(np.array([x for x in range(-math.floor(fr * p / 2), math.ceil(fr * p / 2))]), P)
            tmp = excludeOuter(tmp, p)
            tmp2 = np.array(sorted(list(map(abs, tmp))))
            ords = np.argsort(tmp)
            tmp2 = np.multiply(tmp2, np.sign(tmp[ords]))  # Find new holes

    return Pacc, Tacc
