import numpy as np
from scipy import ndimage
from subpixel_edges.edgepixel import EdgePixel
from subpixel_edges.edges_iterN import h_edges, v_edges


def main_iterN(image: np.ndarray, threshold: int | float, order: int):
    ep = EdgePixel()
    rows, cols = image.shape[:2]  # [30, 30]
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    max_valid_offset = 1  # 亚像素允许最大偏移值
    # 每次迭代都会从零生成新的强度图像和计数图像
    intensity_image = np.zeros(
        (rows, cols)
    )  # [30, 30] 强度图像, 每个像素值表示每个像素的强度的累积总和
    counter = np.zeros(
        (rows, cols)
    )  # [30, 30] 计数图像, 每个像素值表示包括该像素的子图像的数量

    # smooth image
    # image_smooth = image.copy()
    # image_smooth[1:rows - 1, 1:cols - 1] = (image[0:rows - 2, 0:cols - 2] + image[0:rows - 2, 1:cols - 1] + image[0:rows - 2, 2:cols] +
    #                                         image[1:rows - 1, 0:cols - 2] + image[1:rows - 1, 1:cols - 1] + image[1:rows - 1, 2:cols] +
    #                                         image[2:rows, 0:cols - 2] + image[2:rows, 1:cols - 1] + image[2:rows, 2:cols]) / 9
    weight = np.ones((3, 3), np.float64) / 9
    # w = 0.75
    # 调节平滑程度: w 的值用于调节图像处理中平滑操作的强度。较高的 w 值通常意味着在平滑过程中给予中心像素更大的权重，从而在边缘检测中帮助更明晰地识别边缘。
    # 影响梯度计算：在进行梯度计算时，w 可以用来平衡当前像素与其邻域像素的影响，以提高检测精度。
    w = (1 + 24 * weight[1, 2] + 48 * weight[1, 1]) / 12
    image_smooth = ndimage.convolve(image, weight, mode="constant")

    # 梯度计算
    Gx = np.zeros((rows, cols))  # Gx用来存储图像在x方向上的梯度.  [30, 30]
    # 计算Gx中除了边缘外所有像素在x方向上的梯度. 使用的是简单的一阶差分近似. 通过切片操作, 对 image 进行了边界处理, 避免了越界错误
    Gx[:, 1:-1] = 0.5 * (image_smooth[:, 2:] - image_smooth[:, :-2])
    Gy = np.zeros((rows, cols))  # Gy用来存储图像在y方向上的梯度.  [30, 30]
    # 计算Gy中除了边缘外所有像素在y方向上的梯度. 使用的是简单的一阶差分近似. 通过切片操作, 对image image进行了边界处理, 避免了越界错误
    Gy[1:-1, :] = 0.5 * (image_smooth[2:, :] - image_smooth[:-2, :])
    grad = np.sqrt(Gx**2 + Gy**2)  # 计算了每个像素点的梯度幅值.     [30, 30]

    # 取梯度的绝对值, 忽略梯度的方向（正负）, 只关注梯度的大小, 忽略边缘
    absGyInner = np.abs(Gy[5:-5, 2:-2])  # [20, 26] 上下少
    absGxInner = np.abs(Gx[2:-2, 5:-5])  # [26, 20] 左右少

    # 初始化了两个布尔类型的二维数组 Ey 和 Ex, 它们用于标记图像在 y 和 x 方向上的边缘位置
    Ey = np.zeros((rows, cols), dtype=bool)  # [30, 30]
    Ex = np.zeros((rows, cols), dtype=bool)  # [30, 30]

    # 使用 np.logical_and.reduce 来逻辑与多个条件, 设置Ey数组中对应位置的值为True, 如果这些条件同时满足:
    #   - 当前像素的梯度幅值 grad 大于某个阈值 threshold
    #   - 在 y 方向上的梯度分量的绝对值 absGyInner 大于或等于同一行上x方向梯度分量的绝对值 np.abs(Gx)
    #   - absGyInner 大于或等于该行上面一行和下面一行的 Gy 梯度分量的绝对值
    Ey[5:-5, 2:-2] = np.logical_and.reduce(
        [
            grad[5:-5, 2:-2] > threshold,
            absGyInner >= np.abs(Gx[5:-5, 2:-2]),
            absGyInner >= np.abs(Gy[4:-6, 2:-2]),
            absGyInner >= np.abs(Gy[6:-4, 2:-2]),
        ]
    )

    Ex[2:-2, 5:-5] = np.logical_and.reduce(
        [
            grad[2:-2, 5:-5] > threshold,
            absGxInner >= np.abs(Gy[2:-2, 5:-5]),
            absGxInner >= np.abs(Gx[2:-2, 4:-6]),
            absGxInner >= np.abs(Gx[2:-2, 6:-4]),
        ]
    )

    # [rows,cols] -> [rows*cols]
    # F 代表列优先
    image_flatten = image.ravel("F")  # [30, 30] -> [900]
    image_smooth_flatten = image_smooth.ravel("F")  # [30, 30] -> [900]

    Ey = Ey.ravel("F")  # [30, 30] -> [900]
    Ex = Ex.ravel("F")  # [30, 30] -> [900]
    y = y.ravel("F")  # [30, 30] -> [900]
    x = x.ravel("F")  # [30, 30] -> [900]

    # 在y和x方向上被检测为边缘的像素的位置. 由于图像是二维的, 每个边缘点的位置可以用行和列的索引来表示
    # 这里通过(x[Ey] * rows + y[Ey])计算出y方向上的边缘点的索引, 同理计算x方向上的边缘点的索引
    # [24] 24 个边缘点,递增的 index 索引, y方向的梯度找到的是水平方向的边缘
    h_edges_ = x[Ey] * rows + y[Ey]
    # [28] 28 个边缘点,递增的 index 索引, x方向的梯度找到的是竖直方向的边缘
    v_edges_ = x[Ex] * rows + y[Ex]

    Gx = Gx.ravel("F")  # x 方向梯度 [30, 30] -> [900]
    Gy = Gy.ravel("F")  # y 方向梯度 [30, 30] -> [900]

    # all shape: [24]
    (
        h_edges_,
        h_pixel_x,
        h_pixel_y,
        h_x,
        h_y,
        h_nx,
        h_ny,
        h_curv,
        h_i0,
        h_i1,
        intensity_image,
        counter,
    ) = h_edges(
        image,
        image_flatten,
        image_smooth_flatten,
        Gx,
        Gy,
        x,
        y,
        h_edges_,
        rows,
        order,
        w,
        threshold,
        max_valid_offset,
        intensity_image,
        counter,
    )

    # all shape: [28]
    (
        v_edges_,
        v_pixel_x,
        v_pixel_y,
        v_x,
        v_y,
        v_nx,
        v_ny,
        v_curv,
        v_i0,
        v_i1,
        intensity_image,
        counter,
    ) = v_edges(
        image,
        image_flatten,
        image_smooth_flatten,
        Gx,
        Gy,
        x,
        y,
        v_edges_,
        rows,
        order,
        w,
        threshold,
        max_valid_offset,
        intensity_image,
        counter,
    )

    # compute final subimage
    # intensity_image 强度图像, 每个像素值表示每个像素的强度的累积总和
    # counter         计数图像, 每个像素值表示包括该像素的子图像的数量
    # counter 中值大于0的像素表示至少包括在一个子图像中的像素. 在这种情况下, new_image(i, j) = intensity_image(i, j) / counter(i, j)
    counter_greater_than_zero = counter > 0
    intensity_image[counter_greater_than_zero] = intensity_image[counter_greater_than_zero] / counter[counter_greater_than_zero]
    # counter 中值为0的像素表示远离任何边界的像素. 在这种情况下, new_image(i, j) ＝ image_smooth(i, j)
    counter_equal_zero = counter == 0
    intensity_image[counter_equal_zero] = image_smooth[counter_equal_zero]

    # all shape: [24+28]
    # 边缘像素位置
    ep.pixel_x = np.concatenate((h_pixel_x, v_pixel_x), axis=0)
    ep.pixel_y = np.concatenate((h_pixel_y, v_pixel_y), axis=0)
    # 边缘亚像素位置
    ep.x = np.concatenate((h_x, v_x), axis=0)
    ep.y = np.concatenate((h_y, v_y), axis=0)
    # 边缘方向
    ep.nx = np.concatenate((h_nx, v_nx), axis=0)
    ep.ny = np.concatenate((h_ny, v_ny), axis=0)
    # 边缘像素位置 1D index inside image
    ep.position = np.concatenate((h_edges_, v_edges_), axis=0)
    # 边缘曲率
    ep.curv = np.concatenate((h_curv, v_curv), axis=0)
    # intensities
    ep.i0 = np.concatenate((h_i0, v_i0), axis=0)
    ep.i1 = np.concatenate((h_i1, v_i1), axis=0)

    # 这些行代码首先找出那些 x 或 y 坐标超出图像边界的边缘点索引, 然后使用 np.delete 函数删除这些边缘点的所有相关信息, 确保只保留图像边界内的边缘点
    # erase elements outside the image size
    # union1d: 并集
    index_to_erase1 = np.union1d(np.where(ep.x > cols), np.where(ep.y > rows))
    index_to_erase2 = np.union1d(np.where(ep.x < 0), np.where(ep.y < 0))
    index_to_erase = np.union1d(index_to_erase1, index_to_erase2)

    ep.x = np.delete(ep.x, index_to_erase)
    ep.y = np.delete(ep.y, index_to_erase)
    ep.pixel_x = np.delete(ep.pixel_x, index_to_erase)
    ep.pixel_y = np.delete(ep.pixel_y, index_to_erase)
    ep.nx = np.delete(ep.nx, index_to_erase)
    ep.ny = np.delete(ep.ny, index_to_erase)
    ep.position = np.delete(ep.position, index_to_erase)
    ep.curv = np.delete(ep.curv, index_to_erase)
    ep.i0 = np.delete(ep.i0, index_to_erase)
    ep.i1 = np.delete(ep.i1, index_to_erase)

    return ep, intensity_image, grad, absGxInner, absGyInner
