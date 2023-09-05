import numpy as np

import EM_init_kmeans
import EM
import GMR
import Motivation

import Exp

import Method
import Plot

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import pandas as pd


from sklearn.mixture import GaussianMixture






def main():
    # 导入数据
    # D1 = pd.read_excel('DataA.xlsx').to_numpy()
    # D1 = pd.read_excel('example3d.xlsx').to_numpy()
    # D1 = np.delete(D1, 4, axis=1)
    # D1 = pd.read_excel('Gdata.xlsx').to_numpy()
    D1 = pd.read_excel('Infdata.xlsx').to_numpy()
    D1 = pd.read_excel('3DTestData.xlsx').to_numpy()


    Phis, Mhis, Shis, Data5, Data5_Coup, Data5_full = Method.Train(Data=D1, nbStates=80, Coupled='true')




    print('Phis : ', Phis.shape)
    print('Mhis : ', Mhis.shape)
    print('Shis : ', Shis.shape)

    ##############################################################################################################
    Plot.tdplot(Data5=Data5, Mhis=Mhis, Mode='cXDX')

    ##############################################################################################################
    #'''''

    # Exp.fill(trajectory, 0.1, 0.2)

    ########################################################################################
    # 预测
    ##############################
    # 启动数据
    Threshold, iniP, finP, SpeedIni = Method.CountT(Data5=Data5)
    x_cur = np.stack((iniP, SpeedIni))  # 循环开始状态
###
###
###9.502512563	1.268689995	1.53802334
    #x_cur = np.array([[4, 15, 4], [float(1), float(1), float(0)]])  # 初始x为inip
    x_cur = np.array([[9.502512563, 1.268689995, 1.53802334], [float(0), float(0), float(0)]])  # 初始x为inip
    #x_cur = np.array([[-3, 0.5], [float(0), float(0)]])  # 初始x为inip
    #################################
    # 定义循环用变量
    #x_last = x_cur.copy()

    xHis = x_cur.copy()[np.newaxis, :]
    yHis = np.zeros(D1[0, :].shape)

    #in_dims = [0, 1]  # 3维高斯输入维度
    #out_dims = 2  # 高斯输出维度

    datacnt = 0
    ##############################

    #LastPs = [0, 0, 0, 0, 0, 0, 0]
    nearest_location, v_nearest, nearest_index, last_index, next_index = Exp.nearest(data5=Data5, current_location=x_cur[0, :],
                                                         current_velocity=x_cur[1, :])
    while datacnt < 300: #np.sqrt(np.sum(abs(finP - x_cur[0, :]) ** 2)) > Threshold and
        print('threshold ', np.sum(abs(finP - x_cur[0, :]) ** 2), Threshold)
        datacnt += 1
        y_cur = Method.Predict(Coupled='true', x_cur=x_cur, Phis=Phis, Mhis=Mhis, Shis=Shis)
        # TarTraj = np.vstack((Data5[0, 0, :], Data5[1, 0, :])).T
        #y_ret, y_dra, CurrPs, acceleration = Exp.draw(LastPs, data5=Data5, coefx=[10, 10], coefy=[1, 1], CurrSpeed=x_cur[1, :], CurrLoc=x_cur[0, :])
        acceleration, nearest_location = Exp.dynamic(last_nearest_loc=nearest_location, data5=Data5, coef=[500, 100, 0.6], CurrSpeed=x_cur[1, :], CurrLoc=x_cur[0, :])

        #print('check shape : ', acceleration.shape)
        #LastPs = CurrPs
        print('y_ret/y_cur : ', acceleration / y_cur, 'y_ret : ', acceleration, 'y_cur : ', y_cur)
###
###
###
        # y_cur = y_cur+y_dra.copy()
        #y_cur = y_cur + y_ret.copy() / 5
        y_cur = y_cur + acceleration / 5
        # print('Check y_dra 2 :                    ', y_ret, ' ', y_cur)
        # y_cur = y_dra.copy()
        # y_cur = y_ret.copy()

######################################################################################
        #输入加速度y_cur
        # 模拟下一步位置
        x_last = x_cur.copy()
        x_cur[1, :] += y_cur  # 加速度累加
        x_cur[0, :] += x_cur[1, :]  # 速度累加  更新位移
        # 更新速度
        x_cur[1, :] = x_cur[0, :] - x_last[0, :]  # 更新速度
        #输出状态x_cur
######################################################################################
        xHis = np.vstack((xHis, x_cur[np.newaxis, :]))  # 记录位移
        yHis = np.vstack((yHis, y_cur))  # 记录加速度
        print('xHis', xHis.shape, '    yHis : ', yHis.shape)
        # print('x   xd  xdd     Threshold    delta ：', x_cur[0, 0],'||', x_cur[1, 0],'||', y_cur[0],'   ',xHis.shape,'     ',EndP - x_cur[0,:])

    for i in np.arange(Data5.shape[0]):
        # 创建一个 3D 图形对象
        fig = plt.figure()

        # 绘制三维散点图
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(Data5[i, 0, :], Data5[i, 1, :], Data5[i, 2, :], c='r', marker='o', alpha=0.7)
        ax.scatter(xHis[:, 0, i], xHis[:, 1, i], yHis[:, i], c='b', marker='x', alpha=1)
        ax.set_xlabel('x')
        ax.set_ylabel('dx')
        ax.set_ylabel('ddx')
        plt.title('Sim and data ddx')

        ax2 = fig.add_subplot(122)
        ax2.plot(range(0, Data5[i, 0, :].shape[0]), Data5[i, 0, :], c='r', marker='o', label='Data x')
        ax2.plot(range(0, xHis[:, 0, i].shape[0]), xHis[:, 0, i], c='b', marker='x', label='Sim x')
        # 设置坐标轴标签
        ax2.set_xlabel('t')
        ax2.set_ylabel('x')

        plt.title('Sim and data x')

        # 显示图形
        plt.show()


    # 创建一个 3D 图形对象

    if D1.shape[1] == 2:
        Plot.tdplot(Data5=Data5, Mode='map', xHis=xHis)
    elif D1.shape[1] == 3:
        Plot.tdplot(Data5=Data5, Mode='map3d', xHis=xHis)




    return



if __name__ == '__main__':
    #import script

    main()






