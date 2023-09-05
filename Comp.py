import numpy as np
def compress_2d_array(original_2d_array, compressed_size):
    if compressed_size >= len(original_2d_array):
        return original_2d_array

    row_step = len(original_2d_array) // compressed_size
    col_step = len(original_2d_array[0]) // compressed_size

    compressed_2d_array = [
        [original_2d_array[i][j] for j in range(0, len(original_2d_array[0]), col_step)]
        for i in range(0, len(original_2d_array), row_step)
    ]

    compressed_2d_array = np.array(compressed_2d_array)

    return compressed_2d_array

def flatten (Z_interp) :
    n = Z_interp.shape[0]
    x_values = np.tile(np.arange(n), n)
    y_values = np.repeat(np.arange(n), n)
    z_values = Z_interp.flatten()
    data_full = np.vstack((x_values, y_values, z_values)).T

    return data_full

def resize(array, xSize, ySize):

    arraynew = array.copy()

    xdata = array[:, 0]
    ydata = array[:, 1]
    xmin = np.min(xdata)
    xmax = np.max(xdata)
    ymin = np.min(ydata)
    ymax = np.max(ydata)

    xSmin = xSize[0]
    xSmax = xSize[1]
    ySmin = ySize[0]
    ySmax = ySize[1]

    xmul = (xSmax - xSmin) / (xmax - xmin) #array / size
    ymul = (ySmax - ySmin) / (ymax - ymin)

    xdnew = (xdata - xmin) * xmul + xSmin
    ydnew = (ydata - ymin) * ymul + ySmin

    #print('Check array size : ', xdata, xmin, xmul, xSmin)

    arraynew[:, 0] = xdnew
    arraynew[:, 1] = ydnew

    return arraynew

def stack (big_array, small_arrays, Percentage) :


    mul = int(Percentage * big_array.shape[0] / small_arrays.shape[0]) #   n * small = Percentage * big
    small_axn = [small_arrays for _ in range(mul)]
    # 使用 vstack 函数在垂直方向合并数组
    result_array = np.vstack((big_array, *small_axn))

    return result_array