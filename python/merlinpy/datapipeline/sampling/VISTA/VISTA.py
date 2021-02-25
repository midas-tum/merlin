
import numpy as np
from scipy.linalg import toeplitz
import math
import datetime
import matplotlib.pyplot as plt

"""
R: Net acceleration factor
p: Number of phase encoding steps
t: Number of frames
typ: Type of sampling
"""
p = 132
t = 25
R = 3


#Let's handle the special case of R = 1
if R == 1:
  samp = np.ones((p,t))

# Let's find uniform interleaved sampling (UIS)
def samp_UIS(p,t,R):
    ptmp = np.zeros((p, 1))  # (132,1)
    for i in list(range(0, p, R)):
        i = round(i)
        ptmp[i][0] = 1

    ttmp = np.zeros((t, 1))  # (25,1)
    for i in list(range(0, t, R)):
        i = round(i)
        ttmp[i][0] = 1

    Top = toeplitz(ptmp, ttmp)

    Top_list = np.reshape(Top, p * t)
    index = []
    for _ in np.where(Top_list == 1)[0]:
        index.append(_)

    ph = []
    for i in index:
        i_ph = (i - 1) % p - math.floor(p / 2)
        ph.append(i_ph)

    ti = []
    for i in index:
        i_ti = math.floor((i - 1) / p) - math.floor(t / 2)
        ti.append(i_ti)

    ind = []
    for i in range(len(index)):
        i_ph = ph[i]
        i_ti = ti[i]
        ind_ = round(p * (i_ti + math.floor(t / 2)) + (i_ph + math.floor(p / 2) + 1))
        ind.append(ind_)

    samp = np.zeros((p, t))
    samp = np.reshape(samp, p * t)
    for i in ind:
        samp[i] = 1
    samp = np.reshape(samp, (p, t))
    return samp




# Use VRS as initialization for VISTA(variable density random sampling)
alph = 0.28  # 0=uniform density, 0<alph<1 for variable density
sig = p / 5  # Std of the Gaussian envelope that defines variable density
tr = round(p/R) # Number of readout lines per frame (temporal resolution)
sd = 10 #sd: Seed for random number generation

p1 = []
for _ in range(-math.floor(p / 2), math.ceil(p / 2)):
    p1.append(_)
p1 = np.reshape(p1, (len(p1), 1))

t1 = []
ind=0
ti = np.zeros((tr*t,1))
ph = np.zeros((tr*t,1))
prob = []
for p in p1:
    a = (0.1 + alph/(1-alph+1e-10)*math.exp(-p**2/(1*sig**2)))
    prob.append(a)
prob = np.reshape(prob, (len(prob),1))

np.random.seed(sd)
tmpSd = [round(s) for s in 1e6*np.random.rand(t)]

def randp(P,sd,*args):
    #P:Prob
    #sd:seed
    #args: dimensions
    P = np.reshape(P, (len(P),1))
    P = P.flatten()
    cum_prob = []
    for i in np.concatenate([[0],np.cumsum(P)]):
        i_prob = i/sum(P)
        cum_prob.append(i_prob)
    length = np.prod(args)
    np.random.seed(sd)
    X = np.random.rand(length)
    ind = np.searchsorted(cum_prob, X, "right")
    return ind

c = math.floor(t/2)
d = math.ceil(t/2)
for i in range(-c,d):
    a = np.where(t1==i)[0]
    n_tmp = tr - len(a)
    prob_tmp = prob
    prob_tmp[a] = 0
    p_tmp = randp(prob_tmp, tmpSd[i+c+1], n_tmp, 1) - math.floor(p/2)-1
    p_tmp = np.reshape(p_tmp, (len(p_tmp),1))
    ti[ind : ind+n_tmp] = i
    ph[ind : ind+n_tmp] = p_tmp
    ind = ind + n_tmp

def samp_VRS(p,t):
    #[ph, ti] = dispdup(ph, ti, param)
    samp = np.zeros((p,t))
    samp = np.reshape(samp, (p*t,1))
    ind_list = p*(ti+c) + (ph+math.floor(p/2)+1)
    ind_list = ind_list.flatten()
    ind = []
    for ind_ in ind_list:
        ind_new = round(ind_)
        ind.append(ind_new)
    samp[ind] = 1
    return samp


# Displacement parameters
nIter = 120
W = max(R/10 + 0.25, 1)      #Scaling of time dimension; frames are "W" units apart
N = tr*t                     #Total number of samples after rounding off
g = math.floor(nIter/6)      #Every gth iteration is relocated on a Cartesian grid. Default value: floor(param.nIter/6)
uni = math.floor(nIter/2)    #At param.uni iteration, reset to equivalent uniform sampling. Default value: floor(param.nIter/2)
ss = 0.25                    #Step-size for gradient descent. Default value: 0.25;
fl = math.floor(nIter*5/6)   #Start checking fully sampledness at fl^th iteration. Default value: floor(param.nIter*5/6)
fs = 1                       #Does time average has to be fully sampled, 0 for no, 1 for yes. Only works with VISTA. Default value: 1
s = 1.4                      #Exponent of the potenital energy term. Default value 1.4

import datetime
print('Computing VISTA, plese wait as it may take a while ...', datetime.datetime.now())
stp = np.ones((1, nIter)) #Gradient descent displacement
a = W * np.ones((1,nIter)) #Temporal axis scaling

def square(num):
  return num*num

def s2(num):
  return num**(s+2)

def s_square(num):
  return num**s

def get_median(data):
  data.sort()
  half = len(data) // 2
  return (data[half] + data[-half]) / 2


np.random.seed(sd)
f = 1 + round(100 * np.random.rand())  # Figure index
dis_ext = np.zeros((N, 1));  # Extent of displacement
s = 1.4  # Exponent of the potenital energy term. Default value 1.4
tf = 0.0  # Step-size in time direction wrt to phase-encoding direction; use zero for constant temporal resolution. Default value: 0.0

for i in range(0, nIter):
    # [ph,ti] = tile(ph(1:N), ti(1:N), param);
    for j in range(0, N):
        # Distances -------------------------------------------------------
        m = np.array(list(map(square, abs(ph - ph[j]))))
        n = np.array(list(map(square, abs(a.flatten()[i] * (ti - ti[j])))))
        dis = []
        for dis_sqr in (m + n):
            dis_ = math.sqrt(dis_sqr)
            dis.append(dis_)
        dis = np.array(dis)
        nanloc = np.where(dis == 0)[0]
        dis[nanloc] = np.Inf

        # Scaling ---------------------------------------------------------
        scl_array = -np.array(list(map(square, ph))) / (2 * sig ** 2)
        scl = []
        for scl_ in scl_array:
            _scl_ = 1 - alph * math.exp(scl_)
            scl.append(_scl_)
        scl = np.array(scl)
        scl = scl + (1 - scl[0])
        dscl = 1 / sig ** 2 * alph * ph[j] * math.exp(
            -(ph[j] ** 2) / (2 * sig ** 2))  # Differentiation of scl wrt to "ph"

        # Force and resulting displacement --------------------------------
        fx = s * np.multiply((ph[j] - ph).flatten(), (scl[j] * scl) / np.array(list(map(s2, dis)))) - (
                    dscl * scl / np.array(list(map(s_square, dis))))
        fy = s * np.multiply(a.flatten()[i] ** 2 * (ti.flatten()[j] - ti.flatten()),
                             scl[j] * scl / np.array(list(map(s2, dis)))) * tf
        ph[j] = ph[j] + max(min(stp.flatten()[i] * sum(fx), R / 4), -R / 4)
        ti[j] = ti[j] + max(min(stp.flatten()[i] * sum(fy), R / 4), -R / 4)

        # Ensure that the samples stay in bounds --------------------------
        if ph[j] < -math.floor(p / 2) - 1 / 2:
            ph[j] = ph[j] + p
        elif ph[j] > math.ceil(p / 2) - 1 / 2:
            ph[j] = ph[j] - p

        if ti[j] < -math.floor(t / 2) - 1 / 2:
            ti[j] = ti[j] + t
        elif ti[j] > math.ceil(t / 2) - 1 / 2:
            ti[j] = ti[j] - t

        # Displacing samples to nearest Cartesian location
        if (i + 1) % g == 0 or (i + 1) == nIter:
            ph[j] = round(ph[j][0])
            ti[j] = round(ti[j][0])

        # Measuing the displacement
        if i == 1:
            dis_ext[j] = abs(stp.flatten()[i] * sum(fx))

        # Normalizing the step-size to a reasonable value
        if i == 2:
            stp = ss * (1 + R / 4) / get_median(list(dis_ext.flatten())) * stp

        # At uni-th iteration, reset to jittered uniform sampling
        ti = list(ti.flatten())[:N]
        ph = list(ph.flatten())[:N]
        if (i + 1) == uni:
            tmp = np.zeros((tr, t))
            for k in range(1, t + 1):
                tmp[:, k - 1] = np.array(sorted(list(ph.flatten())[(k - 1) * tr: k * tr]))
            tmp = np.array(list(
                map(round, np.mean(tmp, axis=1))))  # Find average distances between adjacent phase-encoding samples
            ph = np.tile(tmp, t)  # Variable density sampling with "average" distances
            np.random.seed(sd)
            rndTmp = np.random.rand(t, 1)
            for k in range(-math.floor(t / 2), math.ceil(t / 2)):
                tmp = (np.zeros(ti.shape)).flatten()
                ptmp = []
                for ti_ind in np.where(ti == k)[0]:
                    tmp[ti_ind - 1] = 1
                    ptmp_ = ph[ti_ind - 1] + round(
                        1 / 2 * R ** 1 * (rndTmp[k + math.floor(t / 2)][0] - 0.5))  # Add jitter
                    ptmp.append(ptmp_)
                ptmp = np.array(ptmp)
                for ptmp_ind in np.where(ptmp > math.ceil(p / 2) - 1)[0]:
                    ptmp[ptmp_ind] = ptmp[ptmp_ind] - p  # Ensure the samples don't run out of the k-t grid
                for ptmp_ind in np.where(ptmp < -math.floor(p / 2))[0]:
                    ptmp[ptmp_ind] = ptmp[ptmp_ind] + p
                ptmp_ind = 0
                ph = ph.flatten()
                for ph_ind in np.where(tmp == 1)[0]:
                    ph[ph_ind] = ptmp.flatten()[ptmp_ind]
                    ptmp_ind += 1

            # Temporarily stretch the time axis to avoid bad local minima
            a = a.flatten()
            nn = []
            for i_ in range(i, nIter):
                nn_i = 1 + math.exp(i_ - (i + 1)) / math.ceil(nIter / 60)
                nn.append(nn_i)
            nn = np.array(nn)
            a[i:] = np.multiply(a[i:], nn)
            # figure
            plt.imshow(np.reshape(a, (1, nIter)))
            plt.show()

        # Displace the overlapping points so that one location has only one sample
        if (i + 1) % g == 0 or (i + 1) == nIter:
            #[ph, ti] = dispdup(ph(1:N), ti(1:N), param)

        # Check/ensure time-average is fully sampled
        if ((i + 1) % g == 0 or (i + 1) == nIter) and (i + 1) >= fl:
            ph = ph[:N]
            ti = ti[:N]
            if fs == 1:  # Ensuring fully sampledness at average all
            #   [ph, ti] = fillK(ph, ti, ph, ti, param)
            elif fs > 1:  # Ensuring fully sampledness for "fs" frames
                for m in range(0, math.floor(t / fs)):
                    tmp = (m - 1) * tr * fs + 1