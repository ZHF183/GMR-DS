import numpy as np
import Comp
import math
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def full_f(x_range, y_range, x_len, y_len, trajectory_data, coef_a, coef_b, endP):
  # 创建扩展区域网格
  x_range = tuple(float(element) for element in x_range)
  y_range = tuple(float(element) for element in y_range)

  #print('in full_f  x_range0   x_range1', type(x_range[0]), type(x_range[1]), x_len)
  x_extend = np.linspace(x_range[0], x_range[1], x_len)
  y_extend = np.linspace(y_range[0], y_range[1], y_len)
  X, Y = np.meshgrid(x_extend, y_extend)

  # 根据轨迹数据生成扩展区域的 Z 值
  a = coef_a
  b = coef_b

  endP = trajectory_data[-1, 0]

  #xmean = np.mean(endP-trajectory_data[:, 0])
  #ymean = np.mean(trajectory_data[:, 1])
  #zmean = np.mean(trajectory_data[:, 2])
  dx = np.max(trajectory_data[:, 0]) - np.min(trajectory_data[:, 0])
  dy = np.max(trajectory_data[:, 1]) - np.min(trajectory_data[:, 1])
  dz = np.max(trajectory_data[:, 2]) - np.min(trajectory_data[:, 2])

  #Z_extend = a*abs(zmean/xmean)*(endP-X)-b*abs(zmean/ymean)*Y
  #print('check endP : ', a*abs(zmean/xmean), ' , ', endP, ' , ', b*abs(zmean/ymean))

  Z_extend = a * abs(dz / dx) * (endP - X) - b * abs(dz / dy) * Y
  print('check endP : ', a * abs(dz / dx), ' , ', dx, ' , ', endP,  ' , ', b * abs(dz / dy))

  X, Y = np.meshgrid(x_extend, y_extend)
  #print('trajectory_data : ', trajectory_data.shape, trajectory_data)
  for point in trajectory_data:
      x, y, z = point
      mask = (X == x) & (Y == y)
      Z_extend[mask] = z

  data_array = np.column_stack((X.flatten(), Y.flatten(), Z_extend.flatten()))
  data_array = np.vstack((trajectory_data, data_array))
  unique_array = np.unique(data_array, axis=0)
  return unique_array
  #return Z_extend


def to_base_n(number, base):
    if number == 0:
        return np.array([0])
    ans = []
    while number > 0:
        remainder = number % base
        ans.append(remainder)

        number = number // base
    ans.reverse()
    return np.array(ans)




def fill0(a, b, c, DataO=None, Data5=None, NboDataP=None, Percentage=None, Gauss_dim=3):
    '''''
    Data5 5*3*199
    '''''
    ###############################################################
    #整理数据范围
    Data5_range = np.zeros((Data5.shape[0],Data5.shape[1]-1,2))
    for i in np.arange(Data5.shape[0]):
        for j in np.arange(Data5.shape[1]-1):
            DataMin = np.min(Data5[i, j, :])
            DataMax = np.max(Data5[i, j, :])
            Datalen = DataMax - DataMin
            DPedMin = DataMin - Datalen * Percentage
            DPedMax = DataMax + Datalen * Percentage
            Data5_range[i, j, :] = np.array([DPedMin, DPedMax])
    print('Data5_range : ', Data5_range.shape, Data5_range)
    ###############################################################
    n = Data5.shape[1]-1
    DaDensity = int(NboDataP ** (1/n))
    DataNb = DaDensity ** n  # Data Point数量
    DiXNb = np.zeros((n + 1, DataNb))
    GDNb = DiXNb.shape[0]  # 高斯维度数量
    DataOut = np.zeros((Data5.shape[0], GDNb, DataNb))  # 定义变量

    for i in np.arange(Data5.shape[0]):
        for m in range(DataNb):
            #print('m :', m)
            n_order = to_base_n(m, DaDensity)
            diff = abs(n - n_order.shape[0])
            n_order = np.concatenate((np.zeros(diff), n_order))#整理大小后
            xydxdxy = n_order.copy()
            #print('count : ', n_order.shape, type(n_order), n_order)
            for k in range(GDNb-1):

                min = Data5_range[i, k, 0]
                max = Data5_range[i, k, 1]
                maxData = DaDensity - 1
                #print('11',min,max,Data5_range)
                xydxdxy[k] = n_order[k] * (max - min) / maxData + min
            #print('111', xydxdxy)

            for k in range(GDNb):

                if k == GDNb-1:              ################################################加速度

                    if Gauss_dim == 3:
                        if i == 0:
                            x = n_order[k-2]
                            dx = n_order[k-1]
                        else:
                            y = n_order[k - 2]
                            dy = n_order[k - 1]
                        #n_order
                        DiXNb[k, m] = 0

                    elif Gauss_dim == 5:

                        x = xydxdxy[k - 4]
                        y = xydxdxy[k - 3]
                        dx = xydxdxy[k - 2]
                        dy = xydxdxy[k - 1]

                        CurrLoc = np.array([x, y])
                        CurrSpeed = np.array([dx, dy])
                        nearest_x, nearest_y, Velocity_x, Velocity_y, nearest_index, last_index, next_index = nearest(data5=DataO, CurrLoc=CurrLoc, CurrSpeed=CurrSpeed)

                        ax = a * (nearest_x - x) - b * dx + c * (Velocity_x - dx)
                        ay = a * (nearest_y - y) - b * dy + c * (Velocity_y - dy)
                        #print('Curr  x:', x, ' y:', y, ' dx:', dx, ' dy:', dy,'   Near x:', nearest_x, ' y', nearest_y, ' vx:', Velocity_x, 'vy', Velocity_y)
                        #print('Check Trajectory : ', nearest_x, ' ', nearest_y, ' ', Velocity_x, ' ', Velocity_y)
                        if i == 0:
                            DiXNb[k, m] = ax
                        else:
                            DiXNb[k, m] = ay

                else :        #前面的坐标

                    #print('trrr',k)
                    #DiXNb[k,m] = n_order[k]
                    DiXNb[k, m] = xydxdxy[k]


        DataOut[i,:] = DiXNb.copy()

    print('DataOut0 :', DataOut.shape, DataOut[0, :].T)
    print('DataOut1 :', DataOut.shape, DataOut[1, :].T)

    return DataOut




def fill(trajectory, coef_a=None, coef_b=None, ExpX=None, ExpY=None, ExPerc=None):
    a = coef_a
    b = coef_b

    xmin = np.min(trajectory[:,0])#zuixiao
    xmax = np.max(trajectory[:,0])  # zuida
    xlen = xmax - xmin
    ymin = np.min(trajectory[:,1])  # zuixiao
    ymax = np.max(trajectory[:,1])  # zuida
    ylen = ymax - ymin
    #print('xrange, yrange', xmin,'  ', xmax,'  ', ymin,'  ', ymax)
    xminfull = xmin - ExPerc * xlen
    xmaxfull = xmax + ExPerc * xlen
    yminfull = ymin - ExPerc * ylen
    ymaxfull = ymax + ExPerc * ylen

    x_range = (xminfull, xmaxfull)
    y_range = (yminfull, ymaxfull)

    #datanbfull = 5 #5个采样点

    datanbfullx = ExpX#x5个采样点， 全长2len
    datanbfully = ExpY#y5个采样点， 全长2len

    #print('x_range, y_range, datanbfullx, datanbfully', x_range, '  ', y_range, '  ', datanbfullx, '  ', datanbfully)

    endP = trajectory[-1,0]
    #print('endP : ', endP)

    print('Check trajectory :', trajectory.shape, trajectory)

    trajectory_full = full_f(x_range, y_range, datanbfullx, datanbfully, trajectory, coef_a, coef_b, endP)

    print('Check trajectory_full :', trajectory_full.shape, trajectory_full)

    x = trajectory_full[:, 0]
    y = trajectory_full[:, 1]
    z = trajectory_full[:, 2]

    ############################################################################################
    # 创建插值网格

    x_interp = np.linspace(xminfull, xmaxfull, datanbfullx*20)
    y_interp = np.linspace(yminfull, ymaxfull, datanbfully*20)
    print('Check x_interp :', x_interp.shape, x_interp)
    #print('插值 x_interp， x_interp', x_interp.shape,x_interp,'   ', y_interp.shape, y_interp)

    X_interp, Y_interp = np.meshgrid(x_interp, y_interp)

    # 使用 griddata 进行三维插值
    Z_interp = griddata((x, y), z, (X_interp, Y_interp), method='linear')
    #Z_interp = griddata((x, y), z, (X_interp, Y_interp), method='cubic')
    Z_interp = np.nan_to_num(Z_interp)#nan to number



    Z_interp = Comp.compress_2d_array(Z_interp, 63)
    X_interp = Comp.compress_2d_array(X_interp, 63)
    Y_interp = Comp.compress_2d_array(Y_interp, 63)
    data_full = Comp.flatten(Z_interp)

    xSize = np.array([xminfull, xmaxfull])
    ySize = np.array([yminfull, ymaxfull])

    print('xSize : ', xSize)

    data_full = Comp.resize(data_full, xSize, ySize)
    print('Check data_full :', data_full.shape)
    ############################################################
    #强调轨迹
    data_full = Comp.stack(data_full, trajectory, 1)

    print('Check data_full :', data_full.shape)



    #data_full = np.column_stack((X_interp.flatten(), Y_interp.flatten(), Z_interp.flatten()))
    #print('data_full  :  ', data_full.shape)
    '''''
    # 绘制原始数据点和插值结果
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(data_full[:, 0], data_full[:, 1], data_full[:, 2], c='red', marker='o', alpha=0.01)
    ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='blue', marker='^', alpha=0.2)
    ax.set_title('Interpolated Data Point')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    '''''
    # 绘制原始数据点和插值结果
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(x, y, z)
    ax.set_title('Original Data')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax_interp = fig.add_subplot(122, projection='3d')

    ax_interp.plot_surface(X_interp, Y_interp, Z_interp, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax_interp.set_title('Interpolated Data')


    plt.show()

    return data_full

def distance_to_line(px1, py1, px2, py2, x3, y3):
    numerator = ((py2 - py1) * x3 - (px2 - px1) * y3 + px2 * py1 - py2 * px1)
    denominator = math.sqrt((py2 - py1)**2 + (px2 - px1)**2)
    distance = numerator / denominator
    return distance

def nearest(data5=None, current_location=None, current_velocity=None):

    trajectory_location = data5[:, 0, :]
    trajectory_velocity = data5[:, 1, :]
    ##########################################################################
    #计算状态差
    # 将current_location从(3,)塑造成(3,200) # trajectory_location 减去 broadcasted_current_location 计算平方和
    broadcasted_current_location = np.tile(current_location[:, np.newaxis], (1, data5.shape[2]))
    distances_location = np.sum((trajectory_location - broadcasted_current_location) ** 2, axis=0)

    ##########################################################################
    # 计算速度差
    # 将current_velocity从(3,)塑造成(3,100) # trajectory_location 减去 broadcasted_current_location 计算平方和
    broadcasted_current_velocity = np.tile(current_velocity[:, np.newaxis], (1, data5.shape[2]))
    distances_velocity = np.sum((trajectory_velocity - broadcasted_current_velocity) ** 2, axis=0)
    ##########################################################################
    # 同时考虑状态和速度的最近点
    nearest_index = np.argmin(distances_location + distances_velocity)
    ##########################################################################

    if nearest_index == 0:
        last_index = nearest_index
        next_index = nearest_index + 1
    elif nearest_index >= data5.shape[2]-1:
        last_index = nearest_index - 1
        next_index = nearest_index
    else:
        last_index = nearest_index - 1
        next_index = nearest_index + 1

    nearest_location = trajectory_location[:, nearest_index]
    v_nearest = (trajectory_location[:, next_index] - trajectory_location[:, last_index]) / (next_index - last_index)

    return nearest_location, v_nearest, nearest_index, last_index, next_index

def vector_abs(vector):
    vectorabs = np.sqrt(np.sum(vector ** 2))
    return vectorabs

def dynamic(last_nearest_loc=None, data5=None, coef=None, CurrSpeed=None, CurrLoc=None):
    '''''
    LastPs[lx, ly, nearx, neary, nextx, nexty, LastAng]
    data5 2(x and y)*3(二阶)*NbOfDataP
    coefx 2
    coefy 2

    CurrSeed 2
    CurrLoc 2
    TarTraj 2(x and y)*NbOfDataP
    '''''
    #location_data = data5[:, 0, :]
    #####################################################################################################
    # CurrLoc = np.array([x_current, y_current])
    nearest_location, v_nearest, nearest_index, last_index, next_index = nearest(data5=data5, current_location=CurrLoc,
                                                         current_velocity=CurrSpeed)
    v_nearest_location = nearest_location - last_nearest_loc

    v_current = CurrSpeed
    current_location = CurrLoc
    displacement_vector = nearest_location - current_location

    # normal component of displacement_vector on v_nearest
    v_nearest_unit = v_nearest / vector_abs(v_nearest)
    displacement_tangential = np.dot(displacement_vector, v_nearest_unit) * v_nearest_unit
    displacement_normal = displacement_vector - displacement_tangential
    # normal component of v_current on v_nearest
    velocity_tangential = np.dot(v_current, v_nearest_unit) * v_nearest_unit
    velocity_normal = v_current - velocity_tangential
    # normal component of v_current on v_nearest
    velocity_tangential = np.dot(v_current, v_nearest_unit) * v_nearest_unit
    velocity_normal = v_current - velocity_tangential
    #切向相对速度
    d_velocity_tangential = velocity_tangential - v_nearest_location

    print('法向位移：', displacement_normal)
    # tanli = coefx[0] * displacement_normal
    # zuni =  -coefx[1] * velocity_normal
    #####################################################################################################
    # 数据范围
    datarange = np.dstack((np.min(data5, axis=2), np.max(data5, axis=2)))
    acc_disp_scale = (datarange[:, 2, 1] - datarange[:, 2, 0]) / (datarange[:, 0, 1] - datarange[:, 0, 0])
    acc_velo_scale = (datarange[:, 2, 1] - datarange[:, 2, 0]) / (datarange[:, 1, 1] - datarange[:, 1, 0])

    acc_disp_scale = vector_abs(datarange[:, 2, 1]) / (vector_abs(datarange[:, 0, 1] - datarange[:, 0, 0]))
    acc_velo_scale = vector_abs(datarange[:, 2, 1]) / (vector_abs(datarange[:, 1, 1]))

    spring_normal = coef[0] * displacement_normal * acc_disp_scale
    damping_normal = -coef[1] * velocity_normal * acc_velo_scale
    acc_normal = spring_normal + damping_normal

    spring_tangential = coef[0] * displacement_tangential * acc_disp_scale
    damping_tangential = -coef[1] * d_velocity_tangential * acc_velo_scale / 10
    acc_tangential = spring_tangential + damping_tangential

    acceleration = acc_normal + acc_tangential + coef[2] * (v_nearest - v_current)
    print('acc_disp_scale', acc_disp_scale, 'acc_velo_scale ：', acc_velo_scale)
    print('  沿着  位移 ：', displacement_tangential, '相对速度 ：', d_velocity_tangential)
    print('  垂直  位移 ：', displacement_normal,'速度 ： ', velocity_normal)
    print(' ')
    print('  沿着  弹力 : ', spring_tangential, '阻尼', damping_tangential)
    print('  垂直  弹力 : ', spring_normal, '阻尼', damping_normal)
    print('  总加速度  : ', acceleration)
    print('  总速度  : ', CurrSpeed)
    print(' ')
    print('  切向速度 : ', vector_abs(d_velocity_tangential), '  法向速度 ：', vector_abs(velocity_normal))
    print('  切向位移 : ', vector_abs(displacement_tangential), '  法向位移 ：', vector_abs(displacement_normal))
    print('  总位移 : ', displacement_tangential+displacement_normal, '  总速度 ：', d_velocity_tangential+velocity_normal)
    #####################################################################################################



    '''''
    ########################
    #plot
    fig = plt.figure()
    ax2 = fig.add_subplot(projection='3d')
    ax2.scatter(data5[0, 0, nearest_index], data5[1, 0, nearest_index], data5[2, 0, nearest_index], c='g', marker='x', label='nearest')
    #ax2.scatter([data5[0, 0, last_index], data5[0, 0, next_index]], [data5[1, 0, last_index], data5[1, 0, next_index]], [data5[2, 0, last_index], data5[2, 0, next_index]], c='y', marker='x', label='Data x')
    ax2.scatter(CurrLoc[0], CurrLoc[1], CurrLoc[2], c='r', marker='o', label='current',alpha=0.9)
    ax2.scatter(data5[0, 0, :], data5[1, 0, :], data5[2, 0, :], c='r', marker='o', label='trajectory',alpha=0.1)
    print('Check : current_x : ', CurrLoc)
    # 设置坐标轴标签
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    plt.title('Sim and data map 3d')
    ax2.grid(True)
    plt.show()
    '''''



    return acceleration, nearest_location


def draw(LastPs, data5=None, coefx=None, coefy=None, CurrSpeed=None, CurrLoc=None):
    '''''
    LastPs[lx, ly, nearx, neary, nextx, nexty, LastAng]
    data5 2(x and y)*3(二阶)*NbOfDataP
    coefx 2
    coefy 2
    
    CurrSeed 2
    CurrLoc 2
    TarTraj 2(x and y)*NbOfDataP
    '''''
    location_data  = data5[:, 0, :]
    print('location_data :', location_data.shape)
###############
###############
###############
    x_positions = data5[0, 0, :]
    y_positions = data5[1, 0, :]

    x_current = CurrLoc[0]
    y_current = CurrLoc[1]
#####################################################################################################
    #CurrLoc = np.array([x_current, y_current])
    nearest_location, v_nearest, nearest_index, last_index, next_index = nearest(data5=data5, current_location=CurrLoc, current_velocity=CurrSpeed)
    v_current = CurrSpeed
    current_location = CurrLoc
    displacement_vector = nearest_location - current_location

    # normal component of displacement_vector on v_nearest
    v_nearest_unit = v_nearest/vector_abs(v_nearest)
    displacement_normal = displacement_vector - np.dot(displacement_vector, v_nearest_unit) * v_nearest_unit
    # normal component of v_current on v_nearest
    velocity_normal = v_current - np.dot(v_current, v_nearest_unit) * v_nearest_unit

    print('法向位移：', displacement_normal)
    #tanli = coefx[0] * displacement_normal
    #zuni =  -coefx[1] * velocity_normal
    #####################################################################################################
    #数据范围
    datarange = np.dstack((np.min(data5, axis=2), np.max(data5, axis=2)))

    spring = coefx[0] * displacement_normal * (datarange[:, 2, 1] - datarange[:, 2, 0]) / (datarange[:, 0, 1] - datarange[:, 0, 0])
    damping = -coefx[1] * velocity_normal * (datarange[:, 2, 1] - datarange[:, 2, 0]) / (datarange[:, 1, 1] - datarange[:, 1, 0])

    acceleration = spring + damping + 0.6 * (v_nearest - v_current)
    print('  新  X弹力 : ', spring[0], '  Y弹力 ：', spring[1], 'X阻尼', damping[0], 'Y阻尼', damping[1])

    #####################################################################################################

    nearest_x = nearest_location[0]
    nearest_y = nearest_location[1]

    x_speed = CurrSpeed[0]
    y_speed = CurrSpeed[1]

    nearest_vx = v_nearest[0]
    nearest_vy = v_nearest[1]

    dv_x = nearest_vx - x_speed
    dv_y = nearest_vy - y_speed



    #print()
    if nearest_index == 0:
        last_x = x_positions[nearest_index]
        last_y = y_positions[nearest_index]
        next_x = x_positions[nearest_index + 1]
        next_y = y_positions[nearest_index + 1]
    elif nearest_index >= data5.shape[2]-1:
        last_x = x_positions[nearest_index - 1]
        last_y = y_positions[nearest_index - 1]
        next_x = x_positions[nearest_index]
        next_y = y_positions[nearest_index]

    else:
        last_x = x_positions[nearest_index - 1]
        last_y = y_positions[nearest_index - 1]
        next_x = x_positions[nearest_index + 1]
        next_y = y_positions[nearest_index + 1]

    tan = (next_y - last_y) / (next_x - last_x)
    angle_rad = math.atan(tan)  # 传入参数为1.0，计算反正切(1.0)，即求 π/4 的弧度值

    indx = (next_x - last_x) / abs(next_x - last_x)
    indy = (next_y - last_y) / abs(next_y - last_y)
    #print('indx 0 ', indx, last_x, next_x)
    if indx == -1:
        angle_rad += math.pi

    angle_deg = math.degrees(angle_rad)  # 将弧度转换为角度

    Last_deg = math.degrees(LastPs[6])

    #print('weisha qudao',Last_deg, angle_deg, abs(Last_deg - angle_deg))
    '''''
    if abs(Last_deg - angle_deg) > 35 and Last_deg != 0:
        angle_rad = LastPs[-1]
        last_x = LastPs[0]
        last_y = LastPs[1]
        nearest_x = LastPs[2]
        nearest_y = LastPs[3]
        next_x = LastPs[4]
        next_y = LastPs[5]
    '''''
    angle_deg = math.degrees(angle_rad)  # 将弧度转换为角度

    Move_A = math.atan(y_speed / x_speed)
    Move_A += (-0.5*(x_speed/abs(x_speed))+0.5)*math.pi
    #if x_speed <0:
        #Move_A +=math.pi
    d_A = angle_rad - Move_A

    dl = distance_to_line(last_x, last_y, next_x, next_y, x_current, y_current)
    nor_speed = x_speed * math.sin(angle_rad) - y_speed * math.cos(angle_rad)  # 法向速度 垂直于轨迹 右偏负
    tan_speed = x_speed * math.cos(angle_rad) + y_speed * math.sin(angle_rad)  # 切向速度 沿着轨迹

    #print('运动角度 ：', math.degrees(Move_A), '    轨迹角度 ', angle_deg, '   差值 ',  math.degrees(d_A))
    #print( '垂直速度 ：', nor_speed, '沿着速度 ：', tan_speed, '距离 ： ', dl)





    Xdx = np.max(data5[0, 0, :]) - np.min(data5[0, 0, :])
    Xdy = np.max(data5[0, 1, :]) - np.min(data5[0, 1, :])
    Xdz = np.max(data5[0, 2, :]) - np.min(data5[0, 2, :])

    Ydx = np.max(data5[1, 0, :]) - np.min(data5[1, 0, :])
    Ydy = np.max(data5[1, 1, :]) - np.min(data5[1, 1, :])
    Ydz = np.max(data5[1, 2, :]) - np.min(data5[1, 2, :])

    dz = math.sqrt(Xdz**2 + Ydz**2)
    #nor_ac1 = -coefx[0] * abs(Xdz / Xdx) * dl# + coefx[1] * abs(Xdz / Xdy) * nor_speed
    #nor_ac2 = -coefx[0] * abs(Xdz / Xdx) * dl# - coefx[1] * abs(Xdz / Xdy) * nor_speed
    nor_acT = -coefx[0] * abs(Xdz / Xdx) * dl
    nor_acZ = - coefx[1] * abs(Xdz / Xdy) * nor_speed
    nor_ac = nor_acT + nor_acZ
    x_ac = nor_ac * math.sin(angle_rad) + 0.5 * dv_x
    y_ac = -nor_ac * math.cos(angle_rad) + 0.5 * dv_y
    ac = np.array([x_ac, y_ac])
###
    print('  X弹力 : ', nor_acT*math.sin(angle_rad), '  Y弹力 ：', -nor_acT*math.cos(angle_rad), '  X阻尼：', nor_acZ*math.sin(angle_rad), '  Y阻尼：', -nor_acZ*math.cos(angle_rad))

    #存数据
    #LastPs[lx, ly, nearx, neary, nextx, nexty, LastAng] = last_x
    CurrPs = np.array([last_x, last_y, nearest_x, nearest_y, next_x, next_y, angle_rad])

    ax = coefx[0] * abs(Xdz / Xdx) * (nearest_x - x_current) - coefx[1] * abs(Xdz / Xdy) * x_speed
    ay = coefy[0] * abs(Ydz / Ydx) * (nearest_y - y_current) - coefy[1] * abs(Ydz / Ydy) * y_speed
    #print('Check : current_x : ', x_current, 'current_y : ', y_current)
    print('最近点x : ', nearest_x, '最近点x : ', nearest_y)
    print('现在x : ', x_current, '现在y : ', y_current)
    print( 'x坐标 : dx : ', (nearest_x - x_current), 'vx : ', x_speed, 'ax : ', ax)
    print( 'y坐标 : dy : ', (nearest_y - y_current), 'vy : ', y_speed, 'ay : ', ay)

    '''''
    ########################
    #plot
    fig = plt.figure()
    ax2 = fig.add_subplot()
    ax2.plot(nearest_x, nearest_y, c='y', marker='x', label='Data x')
    ax2.plot([last_x, next_x], [last_y, next_y], c='g', marker='x', label='Data x')
    ax2.plot(x_positions, y_positions, c='b', marker='x', label='Data x',alpha=0.3)
    ax2.plot(x_current, y_current, c='r', marker='o', label='Data x',alpha=0.6)
    print('Check : current_x : ', x_current, 'current_y : ', y_current)
    # 设置坐标轴标签
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.title('Sim and data map')
    ax2.grid(True)
    plt.show()
    '''''


    accD = np.array([ax, ay])


    return ac, accD, CurrPs, acceleration

def imitate(acc_5):
    x5  = 0
    return x5



