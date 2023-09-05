import numpy as np
import gaussPDF


'''
    nbData = x.shape[1]
    nbVar = Mu.shape[0]
    nbStates = Sigma.shape[2]

    Pxi = np.zeros((nbData, nbStates))
    for i in range(nbStates):
        Pxi[:, i] = Priors[i] * gaussPDF(x, Mu[in_dims, i], Sigma[np.ix_(in_dims, in_dims, [i])])

    beta = Pxi / np.tile(np.sum(Pxi, axis=1)[:, np.newaxis] + np.finfo(float).tiny, (1, nbStates))

    y_tmp = np.zeros((len(out_dims), nbVar, nbStates))
    for j in range(nbStates):
        y_tmp[:, :, j] = np.tile(Mu[out_dims, j][:, np.newaxis], (1, nbData)) + Sigma[np.ix_(out_dims, in_dims, [j])] @ np.linalg.inv(Sigma[np.ix_(in_dims, in_dims, [j])]) @ (x - np.tile(Mu[in_dims, j][:, np.newaxis], (1, nbData)))

    beta_tmp = beta.reshape((1, 1) + beta.shape)
    y_tmp2 = np.tile(beta_tmp, (len(out_dims), 1, 1)) * y_tmp
    y = np.sum(y_tmp2, axis=2)

    if len(out_dims) > 1:
        Sigma_y_tmp = np.zeros((len(out_dims), len(out_dims), 1, nbStates))
        for j in range(nbStates):
            Sigma_y_tmp[:, :, 0, j] = Sigma[np.ix_(out_dims, out_dims, [j])] - Sigma[np.ix_(out_dims, in_dims, [j])] @ np.linalg.inv(Sigma[np.ix_(in_dims, in_dims, [j])]) @ Sigma[np.ix_(in_dims, out_dims, [j])]

        beta_tmp = beta.reshape((1, 1) + beta.shape)
        Sigma_y_tmp2 = np.tile(beta_tmp * beta_tmp, (len(out_dims), len(out_dims), 1, 1)) * np.tile(Sigma_y_tmp, (1, 1, nbData, 1))
        Sigma_y = np.sum(Sigma_y_tmp2, axis=3)
    else:
        Sigma_y = None
'''



def Pred(Priors, Mu, Sigma, x, in_dims, out_dims):



    Mui0=Mu[in_dims, 0][np.newaxis,np.newaxis]
    Sigmai0=Sigma[in_dims, in_dims, 0][np.newaxis,np.newaxis,np.newaxis]


    leno = 1
    leni = 1
    nbData = x.shape[1]
    print('x  ', x.shape)
    nbVar = Mu.shape[0]
    nbStates = Sigma.shape[2]
    #print('nbData nbVar nbStates', nbData, nbVar, nbStates)
    Pxi = np.zeros((nbData, nbStates))
    for i in range(nbStates):

        pdf = gaussPDF.PDF(x, Mu[in_dims, i][np.newaxis,np.newaxis], Sigma[in_dims, in_dims, i][np.newaxis,np.newaxis,np.newaxis])
        #print('Pxi', Pxi.shape)
        #print('Pxi[:, i]', (Priors[i] * pdf[:, np.newaxis]).shape)
        Pxi[:, i] = Priors[i] * pdf

    beta = Pxi / np.tile(np.sum(Pxi, axis=1)[:, np.newaxis] + np.finfo(float).tiny, (1, nbStates))

    y_tmp = np.zeros((leno, nbData, nbStates))
    for j in range(nbStates):
        Muoj = Mu[out_dims, j][np.newaxis, np.newaxis]
        Sigmaoj = Sigma[out_dims, in_dims, j][np.newaxis, np.newaxis]
        Sigmaij = Sigma[in_dims, in_dims, j][np.newaxis, np.newaxis]

        dxm = x - np.tile(Mu[in_dims, j][np.newaxis, np.newaxis], (1, nbData))
        #print('np.tile(Muoj, (1, nbData))', (np.tile(Muoj, (1, nbData)) + Sigmaoj / Sigmaij * dxm).shape)

        #print('y_tmp', y_tmp.shape)
        y_tmp[:, :, j] = np.tile(Muoj, (1, nbData)) + Sigmaoj / Sigmaij * dxm

    # beta_tmp = beta.reshape((1, 1) + beta.shape)
    beta_tmp = np.reshape(beta, (1, *beta.shape))
    y_tmp2 = np.tile(beta_tmp, (leno, 1, 1)) * y_tmp
    #print('beta : ', beta.shape)
    #print('beta_tmp : ', beta_tmp.shape)
    #print('y_tmp2 : ', y_tmp2.shape)
    y = np.sum(y_tmp2, axis=2)


    if leno > 1:
        Sigma_y_tmp = np.zeros((leno, leno, 1, nbStates))
        for j in range(nbStates):
            Sigma_y_tmp[:, :, 0, j] = Sigma[np.ix_(out_dims, out_dims, [j])] - Sigma[
                np.ix_(out_dims, in_dims, [j])] @ np.linalg.inv(Sigma[np.ix_(in_dims, in_dims, [j])]) @ Sigma[
                                          np.ix_(in_dims, out_dims, [j])]

        beta_tmp = beta.reshape((1, 1) + beta.shape)
        Sigma_y_tmp2 = np.tile(beta_tmp * beta_tmp, (leno, leno, 1, 1)) * np.tile(Sigma_y_tmp, (1, 1, nbData, 1))
        Sigma_y = np.sum(Sigma_y_tmp2, axis=3)
    else:
        Sigma_y = None


    return y, Sigma_y, beta


def Pred3d(Priors, Mu, Sigma, x, in_dims, out_dims):

    in_dims=len(in_dims)#0,1



    Mui0=Mu[:in_dims, 0:1]
    Sigmai0=Sigma[:in_dims, :in_dims, 0:1]


    leno = 1
    leni = 2
    nbData = x.shape[1]
    #print('x  ', x.shape)
    nbVar = Mu.shape[0]
    nbStates = Sigma.shape[2]
    #print('nbData nbVar nbStates', nbData, nbVar, nbStates)



    Pxi = np.zeros((nbData, nbStates))
    for i in range(nbStates):

      pdf = gaussPDF.PDF(x, Mu[:in_dims, i:i+1], Sigma[:in_dims, :in_dims, i:i+1])

      #print('Pxi', Pxi.shape)
      #print('Pxi[:, i]', (Priors[i] * pdf[:, np.newaxis]).shape)
      Pxi[:, i] = Priors[i] * pdf

    beta = Pxi / np.tile(np.sum(Pxi, axis=1)[:, np.newaxis] + np.finfo(float).tiny, (1, nbStates))

    y_tmp = np.zeros((leno, nbData, nbStates))
    for j in range(nbStates):
      Muoj = Mu[out_dims, j]
      Sigmaoj = Sigma[out_dims, :in_dims, j]#oij
      Sigmaij = Sigma[:in_dims, :in_dims, j]#iij
      dxm = x - np.tile(Mu[:in_dims, j:j+1], (1, nbData))
      #print('dxm',dxm.shape)

      y_tmp[:, :, j] = np.tile(Muoj, (1, nbData)) +np.dot(np.dot(Sigmaoj, np.linalg.inv(Sigmaij)),dxm)


    # beta_tmp = beta.reshape((1, 1) + beta.shape)
    beta_tmp = np.reshape(beta, (1, *beta.shape))
    y_tmp2 = np.tile(beta_tmp, (leno, 1, 1)) * y_tmp
    #print('beta : ', beta.shape)
    #print('beta_tmp : ', beta_tmp.shape)
    #print('y_tmp2 : ', y_tmp2.shape)
    y = np.sum(y_tmp2, axis=2)


    if leno > 1:
        Sigma_y_tmp = np.zeros((leno, leno, 1, nbStates))
        for j in range(nbStates):
            Sigma_y_tmp[:, :, 0, j] = Sigma[out_dims, out_dims, j] - Sigma[
                out_dims, :in_dims, j] @ np.linalg.inv(Sigma[:in_dims, :in_dims, j]) @ Sigma[
                                          :in_dims, out_dims, j]

        beta_tmp = beta.reshape((1, 1) + beta.shape)
        Sigma_y_tmp2 = np.tile(beta_tmp * beta_tmp, (leno, leno, 1, 1)) * np.tile(Sigma_y_tmp, (1, 1, nbData, 1))
        Sigma_y = np.sum(Sigma_y_tmp2, axis=3)
    else:
        Sigma_y = None


    return y, Sigma_y, beta