import numpy as np
from sklearn.mixture import GaussianMixture
import GMR
import Exp
import Plot




def Train(Data = None, nbStates = None, Gauss_dim = None, Coupled = None):
    if Gauss_dim == None :
        Gauss_dim = Data.shape[1]*2+1
        print('Gauss_dim : ', Gauss_dim)
    ################################################################
    #数据获取
    nb_dof = Data.shape[1]  # 5个关节
    ################################################################
    #定义变量
    Phis = np.zeros((nb_dof, nbStates,))
    Mhis = np.zeros((nb_dof, Gauss_dim, nbStates,))
    Shis = np.zeros((nb_dof, Gauss_dim, Gauss_dim, nbStates,))

    Data5 = np.zeros((nb_dof, 1, 1))
    Data5_full = np.zeros((nb_dof, 1, 1))
    Data5_Coup = np.zeros((nb_dof, 1, 1))

    for i in np.arange(Data.shape[1]):

        an = Data[:, i].T
        d_an = np.gradient(an, axis=0)
        dd_an = np.gradient(d_an, axis=0)
        Data3 = np.stack((an, d_an, dd_an))

        an_Coup = Data.T
        d_an_Coup = np.gradient(an_Coup, axis=1)
        dd_an_Coup = np.gradient(d_an_Coup, axis=1)[i, :]
        Data3_Coup = np.vstack((an_Coup, d_an_Coup, dd_an_Coup))
        #print('VS', Data3_Coup.shape, Data3[2,:], Data3_Coup[4,:])


        Data5 = np.tile(Data5, (1, int(Data3.shape[0] / Data5.shape[1]), int(Data3.shape[1] / Data5.shape[2])))

        Data5_Coup = np.tile(Data5_Coup, (1, int(Data3_Coup.shape[0] / Data5_Coup.shape[1]), int(Data3_Coup.shape[1] / Data5_Coup.shape[2])))

        Data5[i, :] = Data3.copy()
        Data5_Coup[i, :] = Data3_Coup.copy()

    #Data5_CF = Exp.fill0(0.1, 0.1, 1, DataO=Data5, Data5=Data5_Coup, NboDataP=10000, Percentage=0.5, Gauss_dim=5)
    #print('Finished')
    #Plot.tdplot(Data5=Data5_CF, Mhis=Mhis, Mode='cXYf')
    print('Data5: ', Data5.shape, Data5[:, 1, :])
    for i in np.arange(Data.shape[1]):
        ################################################################
        #数据整理
        #an = Data[:, i].ravel()
        #d_an = np.gradient(an, axis=0)
        #dd_an = np.gradient(d_an, axis=0)
        #Data3 = np.stack((an, d_an, dd_an))

        ################################################################
        #fill
        # data_full = Exp.fill(Data.T, coef_a=1, coef_b=1, ExpX=10, ExpY=10, ExPerc=0.5)
        # data_full = Exp.fill(Data.T, coef_a=1, coef_b=1, ExpX=10, ExpY=10, ExPerc=0.5)


        #data_full = Data3.T.copy()
        #data_full = data_full.T

        ################################################################
        # 调整size，放入Array Data5
        #Data5 = np.tile(Data5, (1, int(Data3.shape[0] / Data5.shape[1]), int(Data3.shape[1] / Data5.shape[2])))
        #Data5_full = np.tile(Data5_full, (1, int(data_full.shape[0] / Data5_full.shape[1]), int(data_full.shape[1] / Data5_full.shape[2])))

        #Data5[i, :] = Data3.copy()
        #Data5_full[i, :] = data_full.copy()



        #######################################################################################
        # fit 混合高斯模型
        #gm = GaussianMixture(n_components=nbStates, random_state=0).fit(Data.T)
        if Coupled == 'false':
            gm = GaussianMixture(n_components=nbStates, random_state=0).fit(Data5[i, :].T)
        elif Coupled == 'true':
            gm = GaussianMixture(n_components=nbStates, random_state=0).fit(Data5_Coup[i, :].T)
        #gm = GaussianMixture(n_components=nbStates, random_state=0).fit(Data5[i, :].T)

        #gmf = GaussianMixture(n_components=nbStates, random_state=0).fit(Data5_CF[i, :].T)
        #gm = gmf

        Priors = gm.weights_.T.copy()
        Mu = gm.means_.T.copy()
        Sigma = gm.covariances_.T.copy()

        Phis[i, :] = Priors
        Mhis[i, :] = Mu
        Shis[i, :] = Sigma
        #######################################################################################


    return Phis, Mhis, Shis, Data5, Data5_Coup, Data5_full


def CountT(Data5=None):
    xData = Data5[:, 0, :]
    max_values_col = np.max(xData, axis=1) - np.min(xData, axis=1)
    dis = np.max(max_values_col)  # endpoint 到 initialpoint 在状态空间距离
    Threshold = dis * 1 / 100

    iniP = Data5[:, 0, 1]  # 演示数据初始点
    finP = Data5[:, 0, -1]  # 演示数据终点
    SpeedIni = Data5[:, 1, 1]  # 演示数据初速度
    return Threshold, iniP, finP, SpeedIni


def Predict(Coupled=None, x_cur=None, Phis=None, Mhis=None, Shis=None):
    ##########################################
    #整理维度
    Gauss_dim = Mhis.shape[1]
    nb_dof = Mhis.shape[0]
    ##########################################
    Gauss_dim_in = Gauss_dim - 1
    in_dims = [i for i in range(Gauss_dim_in)]
    out_dims = Gauss_dim_in
    if Coupled == 'true':
        oneline = x_cur.ravel()[:, np.newaxis]
        x_cur = np.tile(oneline, (1, nb_dof)).copy()

    '''''
    if Gauss_dim == 3:
        in_dims = [0, 1]
        out_dims = 2
    elif Gauss_dim == 5:
        in_dims = [0, 1, 2, 3]
        out_dims = 4
        x_coup = np.zeros((4, 1))
        x_coup[0, 0] = x_cur[0, 0]
        x_coup[1, 0] = x_cur[0, 1]
        x_coup[2, 0] = x_cur[1, 0]
        x_coup[3, 0] = x_cur[1, 1]

        y_coup = x_coup.copy()
        x_cur = np.hstack((x_coup, y_coup))
    '''''

    y_cur = np.zeros(nb_dof)
    # 从五个（机械臂状态空间维度=5）二维x状态和五组Priors，Mu，Sigma 得到五个y
    for i in np.arange(nb_dof):
        Priors = Phis[i, :]
        Mu = Mhis[i, :]
        Sigma = Shis[i, :]
        xin = x_cur[:, i:i + 1].copy()
        print('输入 xin : ', i, ' ', xin)
        y, Sigma_y, beta = GMR.Pred3d(Priors, Mu, Sigma, xin, in_dims, out_dims)
        y_cur[i] = y
    print('模型计算得加速度： ', y_cur)

    return y_cur
