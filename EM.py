import numpy as np
import gaussPDF

def learn(Data, Priors0, Mu0, Sigma0):
    loglik_threshold = 1e-10
    nbVar, nbData = Data.shape
    nbStates = Sigma0.shape[2]
    loglik_old = -np.inf
    nbStep = 0

    Mu = Mu0.copy()
    Sigma = Sigma0.copy()
    Priors = Priors0.copy()

    max_iter = 50000
    num_iter = 0
    while True and num_iter <= max_iter:
        num_iter += 1
        Pxi = np.zeros((nbData, nbStates))
        #print('Pxi :', Pxi.shape)
        Pix_tmp = np.zeros((nbData, nbStates))

        # E-step

        for i in range(nbStates):
            Pxi[:, i] = gaussPDF.PDF(Data, np.expand_dims(Mu0[:, i], axis=1), np.expand_dims(Sigma[:, :, i], axis=2))

        Pix_tmp = np.tile(Priors, (nbData, 1))  * Pxi
        Pix = Pix_tmp / np.tile(np.sum(Pix_tmp, axis=1, keepdims=True), (1, nbStates))
        E = np.sum(Pix, axis=0)

        # M-step
        for i in range(nbStates):
            Priors[i] = E[i] / nbData
            #Mu[:, i] = np.dot(Data, Pix[:, i]) / E[i]
            Mu[:, i] = np.dot(Data, Pix[:, i]) / E[i]
            #print('Pix[:, i] :', Pix[:, i])

            Mui=Mu[:, i][:, np.newaxis]

            Data_tmp1 = Data - np.tile(Mui, (1, nbData))
            Data_tmp2a = np.tile(np.reshape(Data_tmp1, (nbVar, 1, nbData)), (1, nbVar, 1))
            Data_tmp2b = np.tile(np.reshape(Data_tmp1, (1, nbVar, nbData)), (nbVar, 1, 1))
            Data_tmp2c = np.tile(np.reshape(Pix[:, i], (1, 1, nbData)), (nbVar, nbVar, 1))
            Sigma[:, :, i] = np.sum(Data_tmp2a * Data_tmp2b * Data_tmp2c, axis=2) / E[i]
            Sigma[:, :, i] += 1E-5 * np.diag(np.ones(nbVar))  # Add a tiny variance for numerical stability

        # Stopping criterion


        for i in range(nbStates):
            Mui=Mu[:, i][:, np.newaxis]
            Sigmai=Sigma[:, :, i][:,:, np.newaxis]

            Pxi[:, i] = gaussPDF.PDF(Data, Mui, Sigmai)
        F = np.dot(Pxi, Priors)
        F[F < np.finfo(float).eps] = np.finfo(float).eps
        loglik = np.sum(np.log(F))
        if abs((loglik / loglik_old) - 1) < loglik_threshold:
            break
        loglik_old = loglik
        nbStep += 1

    # Add a tiny variance for numerical stability
    #Sigma += 1E-5 * np.tile(np.diag(np.ones(nbVar)), (1, 1, nbStates))
    for i in range(nbStates):
        Sigma[:, :, i] = Sigma[:, :, i] + 1E-5 * np.diag(np.ones(nbVar))

    return Priors, Mu, Sigma, nbStep, loglik







def learn3d(Data, Priors0, Mu0, Sigma0):
    loglik_threshold = 1e-10
    nbVar, nbData = Data.shape
    nbStates = Sigma0.shape[2]
    loglik_old = -np.inf
    nbStep = 0

    Mu = Mu0.copy()
    Sigma = Sigma0.copy()
    Priors = Priors0.copy()

    max_iter = 50000
    num_iter = 0
    while True and num_iter <= max_iter:
        num_iter += 1
        Pxi = np.zeros((nbData, nbStates))
        #print('Pxi :', Pxi.shape)
        Pix_tmp = np.zeros((nbData, nbStates))

        # E-step

        for i in range(nbStates):
            Pxi[:, i] = gaussPDF.PDF(Data, Mu0[:, i:i+1], Sigma[:, :, i:i+1])

        Pix_tmp = np.tile(Priors, (nbData, 1))  * Pxi
        Pix = Pix_tmp / np.tile(np.sum(Pix_tmp, axis=1, keepdims=True), (1, nbStates))
        E = np.sum(Pix, axis=0)

        # M-step
        for i in range(nbStates):
            Priors[i] = E[i] / nbData
            #Mu[:, i] = np.dot(Data, Pix[:, i]) / E[i]
            Mu[:, i] = np.dot(Data, Pix[:, i]) / E[i]
            #print('Pix[:, i] :', Pix[:, i])

            Mui=Mu[:, i:i+1]

            Data_tmp1 = Data - np.tile(Mui, (1, nbData))
            Data_tmp2a = np.tile(np.reshape(Data_tmp1, (nbVar, 1, nbData)), (1, nbVar, 1))
            Data_tmp2b = np.tile(np.reshape(Data_tmp1, (1, nbVar, nbData)), (nbVar, 1, 1))
            Data_tmp2c = np.tile(np.reshape(Pix[:, i], (1, 1, nbData)), (nbVar, nbVar, 1))
            Sigma[:, :, i] = np.sum(Data_tmp2a * Data_tmp2b * Data_tmp2c, axis=2) / E[i]
            Sigma[:, :, i] += 1E-5 * np.diag(np.ones(nbVar))  # Add a tiny variance for numerical stability

        # Stopping criterion


        for i in range(nbStates):
            Mui=Mu[:, i:i+1]
            Sigmai=Sigma[:, :, i:i+1]

            Pxi[:, i] = gaussPDF.PDF(Data, Mui, Sigmai)
        F = np.dot(Pxi, Priors)
        F[F < np.finfo(float).eps] = np.finfo(float).eps
        loglik = np.sum(np.log(F))
        if abs((loglik / loglik_old) - 1) < loglik_threshold:
            break
        loglik_old = loglik
        nbStep += 1

    # Add a tiny variance for numerical stability
    #Sigma += 1E-5 * np.tile(np.diag(np.ones(nbVar)), (1, 1, nbStates))
    for i in range(nbStates):
        Sigma[:, :, i] = Sigma[:, :, i] + 1E-5 * np.diag(np.ones(nbVar))

    return Priors, Mu, Sigma, nbStep, loglik