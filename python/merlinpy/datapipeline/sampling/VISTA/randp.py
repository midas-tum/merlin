
import numpy as np

def randp(P,sd,n,m):
    """
    RANDP - pick random values with relative probability
    :param P: input prob_tmp
    :param sd: seed
    :param n,m: dimensions
    :return:
    """
    np.random.seed(sd)
    X = np.random.rand(n*m)

    for i in P:
        if i<0:
            print('All probabilities should be 0 or larger.')

    cum_prob = []
    for i in np.concatenate([[0],np.cumsum(P)]):
        i_prob = i/sum(P)
        cum_prob.append(i_prob)

    ind = np.searchsorted(cum_prob, X, "right")
    return ind
