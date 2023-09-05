import EM_init_kmeans
import EM
import GMR


def Mot(Data, nbStates, x):

    Priors0, Mu0, Sigma0 = EM_init_kmeans.init(Data,nbStates)

    print('Priors :', Priors0.shape)
    print('data :', Data.shape)
    print('Mu0 :', Mu0.shape)
    print('sigma0 :',Sigma0.shape)
    print(Data.shape)


    Priors, Mu, Sigma, nbStep, loglik = EM.learn3d(Data, Priors0, Mu0, Sigma0)


    print('Priors :', Priors.shape)
    print('Mu :', Mu.shape)
    print('Sigma :',Sigma.shape)


    '''''
    next_x=x
    in_dims = [0, 1]
    out_dims = 2
    #x = Data[[0, 1], :]
    y, Sigma_y, beta = GMR.Pred3d(Priors, Mu, Sigma, x, in_dims, out_dims)
    next_x[0, 0] += x[1, 0]  # 速度累加
    next_x[1, 0] += y  # 加速度累加
    y = y[0, :]
    print('true', Data[2].shape)
    print('ytrue', y.shape)

    return next_x,y
    '''''
    return Priors, Mu, Sigma