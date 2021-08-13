import numpy as np
from scipy.linalg import toeplitz
import math
import datetime
import matplotlib.pyplot as plt
from .dispdup import dispdup
from .fillK import fillK
from .randp import randp


def vista(p, t, R, typ, alph, sd, nIter, g, uni, ss, fl, fs, s, tf, dsp):
    """
    :param p: Number of phase encoding steps
    :param t: Number of frames
    :param R: Net acceleration factor
    :param typ: Type of sampling
    :param alph: 0=uniform density, 0<alph<1 for variable density
    :param sd: Seed for random number generation
    :param nIter: Number of iterations for VISTA (defualt: 120)
    :param g: Every gth iteration is relocated on a Cartesian grid. Default value: floor(nIter/6)
    :param uni: At uni iteration, reset to equivalent uniform sampling. Default value: floor(nIter/2)
    :param ss: Step-size for gradient descent. Default value: 0.25
    :param fl: Start checking fully sampledness at fl^th iteration. Default value: floor(nIter*5/6)
    :param fs: Does time average has to be fully sampled, 0 for no, 1 for yes. Only works with VISTA. Default value: 1
    :param s: Exponent of the potenital energy term. Default value 1.4
    :param tf: Step-size in time direction wrt to phase-encoding direction; use zero for constant temporal resolution. 
               Default value: 0.0
    :param dsp: Display the distribution at every dsp^th iteration
    :return: Variable Density Incoherent Spatiotemporal Acquisition (VISTA) 2D sampling pattern
    """

    sig = p / 5        # Std of the Gaussian envelope that defines variable density
    tr = round(p / R)  # Number of readout lines per frame (temporal resolution)

    # Displacement parameters
    W = max(R / 10 + 0.25, 1)  # Scaling of time dimension; frames are "W" units apart
    N = tr * t  # Total number of samples after rounding off

    # Let's handle the special case of R = 1
    def noacc(p, t):
        samp = np.ones((p, t))
        return samp

    # Let's find uniform interleaved sampling (UIS)
    def samp_UIS(p, t, R):
        ptmp = np.zeros((p, 1)).flatten()
        for i in list(range(0, p, R)):
            i = round(i)
            ptmp[i] = 1

        ttmp = np.zeros((t, 1)).flatten()
        for i in list(range(0, t, R)):
            i = round(i)
            ttmp[i] = 1

        Top = toeplitz(ptmp, ttmp)
        Top_flatten = np.reshape(Top, (p * t,), order='F')
        ind = []
        for _ in np.where(Top_flatten == 1)[0]:
            ind.append(_)
        ind = np.array(ind)

        ph = []
        for i in ind:
            i_ph = i % p - math.floor(p / 2)
            ph.append(i_ph)
        ph = np.array(ph)

        ti = []
        for i in ind:
            i_ti = math.floor((i / p)) - math.floor(t / 2)
            ti.append(i_ti)
        ti = np.array(ti)

        ph, ti = dispdup(ph, ti, p, t)

        samp = np.zeros((p, t))
        ind2 = []
        for i in range(len(ind)):
            ind_ = round(p * (ti[i] + math.floor(t / 2)) + (ph[i] + math.floor(p / 2) + 1))
            ind2.append(ind_)
        ind = np.array(ind2).astype('int64')

        samp = np.reshape(samp, (p * t,), order='F')
        samp[ind] = 1
        samp = np.reshape(samp, (p, t), order='F')

        return samp

    def tile(ph, ti, p, t):
        """
        Replicate the sampling pattern in each direction. Probablity, this is
        not an efficient way to impose periodic boundary condition because it
        makes the problem size grow by 9 fold.
        """
        po = np.concatenate((ph, ph - p, ph, ph + p, ph - p, ph + p, ph - p, ph, ph + p))
        to = np.concatenate((ti, ti - t, ti - t, ti - t, ti, ti, ti + t, ti + t, ti + t))
        return po, to

    if R == 1:
        return noacc(p, t)

    if typ == 'UIS':
        return samp_UIS(p, t, R)

    # Use VRS as initialization for VISTA(variable density random sampling)
    p1 = []
    for _ in range(-math.floor(p / 2), math.ceil(p / 2)):
        p1.append(_)
    p1 = np.array(p1)

    t1 = np.array([])
    ind = 0
    ti = np.zeros((tr * t,))
    ph = np.zeros((tr * t,))
    prob = []
    for every_p1 in p1:
        prob_ = (0.1 + alph / (1 - alph + 1e-10) * math.exp(-every_p1 ** 2 / (1 * sig ** 2)))
        prob.append(prob_)
    prob = np.array(prob) 

    np.random.seed(sd)
    tmpSd = [round(s) for s in 1e6 * np.random.rand(t)]  # Seeds for random numbers
    tmpSd = np.array(tmpSd)

    for i in range(-math.floor(t / 2), math.ceil(t / 2)):
        a = np.where(t1 == i)[0]
        n_tmp = tr - len(a)
        prob_tmp = prob
        prob_tmp[a] = 0
        p_tmp = randp(prob_tmp, tmpSd[i + math.floor(t / 2)], n_tmp, 1) - math.floor(p / 2) - 1
        ti[ind: ind + n_tmp] = i
        ph[ind: ind + n_tmp] = p_tmp
        ind = ind + n_tmp

    if typ == 'VRS':
        ph, ti = dispdup(ph, ti, p, t)
        samp = np.zeros((p, t))
        samp = np.reshape(samp, (p * t,), order='F')
        ind_list = p * (ti + math.floor(t / 2)) + (ph + math.floor(p / 2) + 1)
        for i in ind_list:
            ind = round(i) - 1
            samp[ind] = 1
        return np.reshape(samp, (p, t), order='F')

    print('Computing VISTA, please wait as it may take a while ...', datetime.datetime.now())
    stp = np.ones((1, nIter)).flatten()    # Gradient descent displacement, shape(120,)
    a = W * np.ones((1, nIter)).flatten()  # Temporal axis scaling

    def square(num):
        return num * num

    def s2(num):
        return num ** (s + 2)

    def s_square(num):
        return num ** s

    def get_median(data):
        data.sort()
        half = len(data) // 2
        return (data[half] + data[-half]) / 2

    np.random.seed(sd)
    f = round(100 * np.random.rand())     # Figure index
    dis_ext = np.zeros((N, 1)).flatten()  # Extent of displacement, shape(650,)

    for i in range(nIter):
        ph, ti = tile(ph[:N], ti[:N], p, t)
        for j in range(N):
            # Distances -------------------------------------------------------
            m = np.array(list(map(square, abs(ph - ph[j]))))
            n = np.array(list(map(square, abs(a[i] * (ti - ti[j])))))
            dis = []
            for dis_sqr in (m + n):
                dis_ = math.sqrt(dis_sqr)
                dis.append(dis_)
            dis = np.array(dis)
            nanloc = np.where(dis == 0)[0]
            dis[nanloc] = np.Inf

            # Scaling ---------------------------------------------------------
            scl_tmp = -np.array(list(map(square, ph))) / (2 * sig ** 2)
            scl = []
            for scl_ in scl_tmp:
                _scl_ = 1 - alph * math.exp(scl_)
                scl.append(_scl_)
            scl = np.array(scl)
            scl = scl + (1 - scl[0])
            dscl = 1 / sig ** 2 * alph * ph[j] * math.exp(
                -(ph[j] ** 2) / (2 * sig ** 2))  # Differentiation of scl wrt to "ph"

            # Force and resulting displacement --------------------------------
            fx = s * np.multiply((ph[j] - ph), (scl[j] * scl) / np.array(list(map(s2, dis)))) - (
                        dscl * scl / np.array(list(map(s_square, dis))))
            fy = s * np.multiply(a[i] ** 2 * (ti[j] - ti), scl[j] * scl / np.array(list(map(s2, dis)))) * tf
            ph[j] = ph[j] + max(min(stp[i] * sum(fx), R / 4), -R / 4)
            ti[j] = ti[j] + max(min(stp[i] * sum(fy), R / 4), -R / 4)

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
                ph[j] = round(ph[j])
                ti[j] = round(ti[j])

            # Measuing the displacement
            if i == 1:
                dis_ext[j] = abs(stp.flatten()[i] * sum(fx))

        # Normalizing the step-size to a reasonable value
        if i == 2:
            stp = ss * (1 + R / 4) / get_median(list(dis_ext)) * stp

        # At uni-th iteration, reset to jittered uniform sampling
        ti = ti[:N]
        ph = ph[:N]
        if (i + 1) == uni:
            tmp = np.zeros((tr, t))
            for k in range(t):
                tmp[:, k] = np.array(sorted(ph[k * tr: (k + 1) * tr]))
            tmp = np.array(list(map(round, np.mean(tmp,axis=1))))  
            # Find average distances between adjacent phase-encoding samples
            
            ph = np.tile(tmp, t)  # Variable density sampling with "average" distances
            np.random.seed(sd)
            rndTmp = np.random.rand(t, 1).flatten()
            for k in range(-math.floor(t / 2), math.ceil(t / 2)):
                tmp = np.where(ti == k)[0]
                ptmp = ph[tmp] + round(1 / 2 * R ** 1 * (rndTmp[k + math.floor(t / 2)] - 0.5))  # Add jitter
                for ptmp_ind in np.where(ptmp > math.ceil(p / 2) - 1)[0]:
                    ptmp[ptmp_ind] = ptmp[ptmp_ind] - p  # Ensure the samples don't run out of the k-t grid
                for ptmp_ind in np.where(ptmp < -math.floor(p / 2))[0]:
                    ptmp[ptmp_ind] = ptmp[ptmp_ind] + p
                ptmp_ind = 0
                for ph_ind in tmp:
                    ph[ph_ind] = ptmp[ptmp_ind]
                    ptmp_ind += 1

            # Temporarily stretch the time axis to avoid bad local minima
            nn = []
            for i_item in range(i + 1, nIter):
                nn_i = 1 + math.exp(-(i_item - (i + 1)) / math.ceil(nIter / 60))
                nn.append(nn_i)
            nn = np.array(nn)
            a[i + 1:] = np.multiply(a[i + 1:], nn)

            # figure
            # plt.scatter(a)
            # plt.show()

        # Displace the overlapping points so that one location has only one sample
        if (i + 1) % g == 0 or (i + 1) == nIter:
            ph, ti = dispdup(ph[:N], ti[:N], p, t)

        # Check/ensure time-average is fully sampled
        if ((i + 1) % g == 0 or (i + 1) == nIter) and (i + 1) >= fl:
            ph = ph[:N]
            ti = ti[:N]
            if fs == 1:  # Ensuring fully sampledness at average all
                ph, ti = fillK(ph, ti, ph, ti, p, R, alph)
            elif fs > 1:  # Ensuring fully sampledness for "fs" frames
                for m in range(math.floor(t / fs)):
                    tmp = (m - 1) * tr * fs + 1
                    ph, ti = fillK(ph[tmp], ti[tmp], ph, ti, p, R, alph)

        if (i + 1) == 1 or (i + 1) % dsp == 0 or (i + 1) == nIter:  # When to diplay the distribution
            plt.figure(f)
            plt.scatter(ti[:N], ph[:N], s=10)
            plt.title('Iter is %s,number of samples is %s' % (i + 1, N))
            plt.xlim((-math.floor(t / 2), math.ceil(t / 2) - 1))
            plt.ylim((-math.floor(p / 2), math.ceil(p / 2) - 1))
            plt.xlabel('time')
            plt.ylabel('phase')
            plt.show()

    ph, ti = dispdup(ph[:N], ti[:N], p, t)

    # From indices to 2D binary mask
    samp = np.zeros((p, t))
    ind = []
    for phti_ind in range(N):
        ind.append(round(p * (ti[phti_ind] + math.floor(t / 2)) + (ph[phti_ind] + math.floor(p / 2) + 1)))
    ind = np.array(ind).astype('int64')
    samp = np.reshape(samp, (p * t,), order='F')
    samp[ind] = 1
    samp = np.reshape(samp, (p, t), order='F')
    print('VISTA computed at', datetime.datetime.now())
    return samp