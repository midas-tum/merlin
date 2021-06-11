import math
import numpy as np


def square(num):
  return num*num


# Index to (x,y)
def ind2xy(ind, X):
    x = []
    y = []
    for every_ind in ind:
        x.append(every_ind - math.floor((every_ind-1)/X)*X)
        y.append(math.ceil(every_ind/X))
    x = np.array(x)
    y = np.array(y)
    return x, y


def ind2xy_single(ind, X):
    x = ind - math.floor((ind-1)/X)*X
    y = math.ceil(ind/X)
    return x, y


# For a given 'dupind', this function finds the nearest vacant location among 'empind'
def nearestvac(dupind, empind, p):
    x0, y0 = ind2xy_single(dupind, p)
    x, y = ind2xy(empind, p)
    dis1 = np.array(list(map(square, (x-x0)))).astype(float)
    dis2 = np.array(list(map(square, (y-y0)))).astype(float)
    eps = np.spacing(1)
    for dis2_ind in range(len(dis2)):
        if dis2[dis2_ind] > eps:
            dis2[dis2_ind] = np.Inf
    dis = []
    for dis12 in (dis1+dis2):
        dis_ = math.sqrt(dis12)
        dis.append(dis_)
    b = dis.index(min(dis))
    n = empind[b]
    return n


def dispdup(ph, ti, p, t):
    """
    If multiple samples occupy the same location, this routine displaces the
    duplicate samples to the nearest vacant location so that there is no more
    than one smaple per location on the k-t grid.

    p: Number of phase encoding steps
    t: Number of frames
    """
    ph = ph + math.ceil((p+1)/2)
    ti = ti + math.ceil((t+1)/2)
    pt = (ti-1)*p + ph

    uniquept, countOfpt = np.unique(pt, return_counts=True)
    repeatedValues = []
    for every_count in range(len(countOfpt)):
        if countOfpt[every_count] != 1:
            repeatedValues.append(uniquept[every_count])
    repeatedValues = np.array(repeatedValues)  # (169,)
    dupind = []
    for i in range(len(repeatedValues)):
        tmp = np.where(pt == repeatedValues[i])[0]
        dupind.extend((list(tmp[1:])))  # Indices of locations which have more than one sample

    empind = np.setdiff1d(np.array([x for x in range(0, p * t)]), pt)

    for i in range(len(dupind)):  # Let's go through all 'dupind' one by one
        newind = nearestvac(pt[dupind[i]], empind, p)
        pt[dupind[i]] = newind
        empind = np.setdiff1d(empind, np.array(newind))

    ph, ti = ind2xy(pt, p)
    ph = ph - math.ceil((p + 1) / 2)
    ti = ti - math.ceil((t + 1) / 2)

    return ph, ti
