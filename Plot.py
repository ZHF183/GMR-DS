import matplotlib.pyplot as plt


def tdplot (Data5=None, Mhis=None, Mode=None, xHis=None):

    if Mode == 'dcXDX':
        #三维高斯
        # 创建一个 3D 图形对象
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        # 绘制三维散点图
        ax.scatter(Data5[0, 0, :], Data5[0, 1, :], Data5[0, 2, :], c='red', marker='o', label='data', alpha=0.4)
        ax.scatter(Mhis[0, 0, :], Mhis[0, 1, :], Mhis[0, 2, :], c='blue', marker='x', label='mean', alpha=1)
        plt.legend()
        plt.title('Decoupled Means X')
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('dX')
        ax.set_zlabel('ddX')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(Data5[1, 0, :], Data5[1, 1, :], Data5[1, 2, :], c='red', marker='o', label='data', alpha=0.4)
        ax.scatter(Mhis[1, 0, :], Mhis[1, 1, :], Mhis[1, 2, :], c='blue', marker='x', label='mean', alpha=1)
        plt.legend()
        plt.title('Decoupled Means Y')
        # 设置坐标轴标签
        ax.set_xlabel('Y')
        ax.set_ylabel('dY')
        ax.set_zlabel('ddY')
        # 显示图形
        plt.show()
    elif Mode == 'dcXY':
        #三维高斯
        # 创建一个 3D 图形对象
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        # 绘制三维散点图
        ax.scatter(Data5[0, 0, :], Data5[1, 0, :], Data5[0, 2, :], c='red', marker='o', label='data', alpha=0.4)
        ax.scatter(Mhis[0, 0, :], Mhis[1, 0, :], Mhis[0, 2, :], c='blue', marker='x', label='mean', alpha=1)
        plt.legend()
        plt.title('Decoupled Means X')
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('ddX')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(Data5[0, 0, :], Data5[1, 0, :], Data5[1, 2, :], c='red', marker='o', label='data', alpha=0.4)
        ax.scatter(Mhis[0, 0, :], Mhis[1, 0, :], Mhis[1, 2, :], c='blue', marker='x', label='mean', alpha=1)
        plt.legend()
        plt.title('Decoupled Means Y')
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('ddY')
        # 显示图形
        plt.show()
    elif Mode == 'cXYf':
    # x y
        # 创建一个 3D 图形对象
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        # 绘制三维散点图
        ax.scatter(Data5[0, 0, :], Data5[0, 1, :], Data5[0, 4, :], c='red', marker='o', label='data', alpha=0.4)
        #ax.scatter(Mhis[0, 0, :], Mhis[1, 1, :], Mhis[0, 4, :], c='blue', marker='x', label='mean', alpha=1)
        plt.legend()
        plt.title('Coupled Means X')
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('ddX')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(Data5[1, 0, :], Data5[1, 1, :], Data5[1, 4, :], c='red', marker='o', label='data', alpha=0.4)
        #ax.scatter(Mhis[1, 0, :], Mhis[1, 1, :], Mhis[1, 4, :], c='blue', marker='x', label='mean', alpha=1)
        plt.legend()
        plt.title('Coupled Means Y')
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('ddY')
        # 显示图形
        plt.show()

    elif Mode == 'cXY':
    # x y
        # 创建一个 3D 图形对象
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        # 绘制三维散点图
        ax.scatter(Data5[0, 0, :], Data5[1, 0, :], Data5[0, 2, :], c='red', marker='o', label='data', alpha=0.4)
        ax.scatter(Mhis[0, 0, :], Mhis[1, 1, :], Mhis[0, 4, :], c='blue', marker='x', label='mean', alpha=1)
        plt.legend()
        plt.title('Coupled Means X')
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('ddX')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(Data5[0, 0, :], Data5[1, 0, :], Data5[1, 2, :], c='red', marker='o', label='data', alpha=0.4)
        ax.scatter(Mhis[1, 0, :], Mhis[1, 1, :], Mhis[1, 4, :], c='blue', marker='x', label='mean', alpha=1)
        plt.legend()
        plt.title('Coupled Means Y')
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('ddY')
        # 显示图形
        plt.show()

    elif Mode == 'cXDXf':
        #x x'
        # 创建一个 3D 图形对象
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        # 绘制三维散点图
        ax.scatter(Data5[0, 0, :], Data5[0, 2, :], Data5[0, 4, :], c='red', marker='o', label='data', alpha=0.4)
        #ax.scatter(Mhis[0, 0, :], Mhis[0, 2, :], Mhis[0, 4, :], c='blue', marker='x', label='mean', alpha=1)
        plt.legend()
        plt.title('Coupled Means X')
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('dX')
        ax.set_zlabel('ddX')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(Data5[1, 0, :], Data5[1, 2, :], Data5[1, 4, :], c='red', marker='o', label='data', alpha=0.4)
        #ax.scatter(Mhis[1, 1, :], Mhis[1, 3, :], Mhis[1, 4, :], c='blue', marker='x', label='mean', alpha=1)
        plt.legend()
        plt.title('Coupled Means y')
        # 设置坐标轴标签
        ax.set_xlabel('Y')
        ax.set_ylabel('dY')
        ax.set_zlabel('ddY')
        # 显示图形
        plt.show()

    elif Mode == 'cXDX':
        #x x'
        # 创建一个 3D 图形对象
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        # 绘制三维散点图
        ax.scatter(Data5[0, 0, :], Data5[0, 1, :], Data5[0, 2, :], c='red', marker='o', label='data', alpha=0.4)
        ax.scatter(Mhis[0, 0, :], Mhis[0, 2, :], Mhis[0, 4, :], c='blue', marker='x', label='mean', alpha=1)
        plt.legend()
        plt.title('Coupled Means X')
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('dX')
        ax.set_zlabel('ddX')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(Data5[1, 0, :], Data5[1, 1, :], Data5[1, 2, :], c='red', marker='o', label='data', alpha=0.4)
        ax.scatter(Mhis[1, 1, :], Mhis[1, 3, :], Mhis[1, 4, :], c='blue', marker='x', label='mean', alpha=1)
        plt.legend()
        plt.title('Coupled Means y')
        # 设置坐标轴标签
        ax.set_xlabel('Y')
        ax.set_ylabel('dY')
        ax.set_zlabel('ddY')
        # 显示图形
        plt.show()

    elif Mode == 'map3d':
        fig = plt.figure()

        # 绘制三维散点图
        ax3 = fig.add_subplot(projection='3d')
        ax3.scatter(Data5[0, 0, :], Data5[1, 0, :], Data5[2, 0, :], c='r', marker='o', alpha=0.7)
        ax3.scatter(xHis[:, 0, 0], xHis[:, 0, 1], xHis[:, 0, 2], c='b', marker='x', alpha=1)

        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('z')
        plt.title('Sim and data Map 3d')
        plt.show()



    elif Mode == 'map':
        # 创建一个 3D 图形对象
        fig = plt.figure()

        ax2 = fig.add_subplot()
        ax2.plot(Data5[0, 0, :], Data5[1, 0, :], c='r', marker='o', label='Data x')
        ax2.plot(xHis[:, 0, 0], xHis[:, 0, 1], c='b', marker='x', label='Sim x')
        # 设置坐标轴标签
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.grid(True)
        plt.title('Sim and data map')
        # 显示图形
        plt.show()

    return