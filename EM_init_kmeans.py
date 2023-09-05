import numpy as np
from scipy.cluster.vq import kmeans2


def init(Data, nbStates):

    nbVar, nbData = Data.shape


    #centroids, labels = kmeans2(Data.T, nbStates, iter=500)
    centroids, labels = kmeans2(Data.T, nbStates, minit='++', iter=1000)
    #print('centro : ',centroids)
    Mu = centroids.T
    Priors = np.zeros(nbStates)
    Sigma = np.zeros((nbVar, nbVar, nbStates))

    for i in range(nbStates):
        idtmp = np.where(labels == i)[0]
        Priors[i] = len(idtmp)
        Sigma[:, :, i] = np.cov(Data[:, idtmp])
        Sigma[:, :, i] += 1E-5 * np.diag(np.ones(nbVar))

    Priors /= np.sum(Priors)


    return Priors, Mu, Sigma