"""
import numpy as np

def PDF(Data, Mu, Sigma):

    D, N = Data.shape
    K = Mu.shape[0]

    prob = np.zeros(N)

    for i in range(N):
        for j in range(K):
            diff = Data[:, i] - Mu[:, j]
            inv_sigma = np.linalg.inv(Sigma[:, :, j])
            exponent = -0.5 * np.dot(np.dot(diff.T, inv_sigma), diff)
            coef = 1 / np.sqrt((2 * np.pi) ** D * np.linalg.det(Sigma[:, :, j]))
            prob[i] += coef * np.exp(exponent)





    #nbVar, nbData = Data.shape

    # Center the data points
    #Data = Data.T - np.tile(Mu.T, (nbData, 1))

    # Calculate the Mahalanobis distance
    #prob = Data / Sigma
    #prob = np.sum((Data / Sigma) * Data, axis=1)

    # Calculate the exponent term of the Gaussian distribution
    #prob = np.exp(-0.5 * prob)

    # Calculate the normalization term of the Gaussian distribution
    #normalization = np.sqrt((2 * np.pi) ** nbVar * (np.abs(np.linalg.det(Sigma)) + np.finfo(float).eps))

    # Calculate the probabilities
    #prob /= normalization

    return prob
"""



import numpy as np

def PDF(Data, Mu, Sigma):
    """
    Calculate the probability density function (PDF) of the given data points under a Gaussian distribution.

% Inputs -----------------------------------------------------------------
%   o Data:  D x N array representing N datapoints of D dimensions.
%   o Mu:    D x K array representing the centers of the K GMM components.
%   o Sigma: D x D x K array representing the covariance matrices of the
%            K GMM components.
% Outputs ----------------------------------------------------------------
%   o prob:  1 x N array representing the probabilities for the
%            N datapoints.

    """

    D, N = Data.shape
    #print('Mu :', Mu.shape, Mu)
    K = Mu.shape[1]

    prob = np.zeros(N)

    for k in range(K):
        # 获取第 k 个高斯分量的中心和协方差矩阵
        mu_k = Mu[:, k]
        sigma_k = Sigma[:, :, k]

        # 计算协方差矩阵的逆和行列式，以便后续使用
        sigma_inv_k = np.linalg.inv(sigma_k)
        sigma_det_k = np.linalg.det(sigma_k)

        # 计算常数项 (2π)^(D/2) * |Σ|^(1/2)，其中 D 是数据维度
        constant_k = (2 * np.pi) ** (D / 2) * np.sqrt(sigma_det_k)

        for i in range(N):
            # 计算数据点到中心的差值
            diff = Data[:, i] - mu_k

            # 计算多元高斯概率密度函数
            exponent = -0.5 * np.dot(np.dot(diff.T, sigma_inv_k), diff)
            prob[i] += np.exp(exponent) / constant_k

    return prob