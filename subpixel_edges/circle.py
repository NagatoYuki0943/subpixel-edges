import numpy as np
from numba import njit


@njit(cache=True)
def circle_grid(
    x: np.ndarray,  # x 坐标, c 中大于0, 小于4的 x
    y: np.ndarray,  #  y坐标, c 中大于0, 小于4的 y
    radius2: float,  # radius的平方
    x_center: float,  # 圆心坐标 x
    y_center: float,  # 圆心坐标 y
    dx: np.ndarray,  # 每个像素点在 x 方向上的偏移量 [50, 50]
    dy: np.ndarray,  # 每个像素点在 y 方向上的偏移量 [50, 50]
) -> np.ndarray:
    """计算每个索引点是否在圆内，并返回一个数组p，其中包含每个索引点在圆内的平均值。"""
    num_pixels: int = x.shape[0]  # n

    if num_pixels > 0:
        p: np.ndarray = np.zeros((num_pixels, 1))  # [n, 1]
        for n in range(0, num_pixels):
            grid: np.ndarray = (
                (x[n] + dx - x_center) ** 2 + (y[n] + dy - y_center) ** 2 < radius2
            ) * 1
            p[n] = np.mean(grid)

    else:
        p = np.zeros((0, 0))

    return p


@njit(cache=True)
def circle_horizontal_window(
    x_window_center: int,  # 子像素边缘中心
    y_window_center: int,  # 子像素边缘中心
    x_center: float,  # 圆的中心 x 坐标
    y_center: float,  # 圆的中心 y 坐标
    radius: float,  # 圆的半径
    inner_intensity: float,  # 两侧强度
    outer_intensity: float,  # 两侧强度
    grid_resolution: int,
) -> np.ndarray:
    # compute pixels completely outside or inside
    r2 = radius**2

    # 生成 xy 索引, x 的每一列值都相同, y 的每一行值都相同 [3, 9]
    # x = np.zeros(
    #     ((np.arange(x_window_center - 4, x_window_center + 4 + 1)).size,
    #      (np.arange(y_window_center - 1, y_window_center + 1 + 1)).size),
    #     dtype=np.float64
    # ).T
    x = np.zeros(
        (
            (x_window_center + 4 + 1) - (x_window_center - 4),
            (y_window_center + 1 + 1) - (y_window_center - 1),
        ),
        dtype=np.float64,
    ).T
    y = x.copy()
    x[:, :] = np.arange(x_window_center - 4, x_window_center + 4 + 1)
    y_vect = np.arange(y_window_center - 1, y_window_center + 1 + 1).reshape((-1, 1))
    y[:, :] = y_vect

    # meshgrid 生成 xy 索引, 和上面代码相同, 但是 meshgrid 不支持 numba
    # x_ = np.arange(x_window_center - 4, x_window_center + 4 + 1)
    # y_ = np.arange(y_window_center - 1, y_window_center + 1 + 1)
    # x, y = np.meshgrid(x_, y_, indexing='xy')

    # 判断x,y在偏移0.5像素的范围是否在圆内部
    c = ((x - x_center - 0.5) ** 2 + (y - y_center - 0.5) ** 2 < r2).astype(
        np.float64
    )  # 左上
    c += (x - x_center - 0.5) ** 2 + (y - y_center + 0.5) ** 2 < r2  # 左下
    c += (x - x_center + 0.5) ** 2 + (y - y_center - 0.5) ** 2 < r2  # 右上
    c += (x - x_center + 0.5) ** 2 + (y - y_center + 0.5) ** 2 < r2  # 右下

    # bool_c0 指的是 BB 的区域, bool_c4 指的是 AA 的区域
    bool_c0 = np.where(c.ravel() == 0)
    bool_c4 = np.where(c.ravel() == 4)
    # bool_c0 = np.where(c == 0) # numba不支持多维索引
    # bool_c4 = np.where(c == 4)

    # c 中大于0小于4的index [3, 9]
    bool_c04 = np.logical_and(c > 0, c < 4)

    # 将 AA 和 BB 的值放入重建图像中
    i = np.copy(c)
    i.ravel()[bool_c0] = outer_intensity
    i.ravel()[bool_c4] = inner_intensity
    # i[bool_c0] = outer_intensity # numba不支持多维索引
    # i[bool_c4] = inner_intensity

    # compute contour pixels
    delta = 1 / (grid_resolution - 1)

    #  -0.5到0.5 的范围,作用是标识在这个范围内的坐标,用在 circle_grid 中确定 c 中大于0小于4的的像素 grid [50, 50]
    dx = np.zeros(
        ((np.arange(-0.5, 0.5, delta)).size, (np.arange(-0.5, 0.5, delta)).size),
        dtype=np.float64,
    ).T
    dy = dx.copy()
    dx[:, :] = np.arange(-0.5, 0.5, delta)
    dy_vect = np.arange(-0.5, 0.5, delta).reshape((-1, 1))
    dy[:, :] = dy_vect

    # 获取 c 中大于0小于4的 grid [n, 1]
    grid = circle_grid(
        x.ravel()[bool_c04.ravel()],
        # x[bool_c04], # numba不支持多维索引
        y.ravel()[bool_c04.ravel()],
        # y[bool_c04], # numba不支持多维索引
        r2,
        x_center,
        y_center,
        dx,
        dy,
    )

    # 更新 c 中大于0小于4 的区域
    i.ravel()[bool_c04.ravel()] = (
        outer_intensity + (inner_intensity - outer_intensity) * grid
    ).reshape((-1,))
    # i[bool_c04] = (outer_intensity + (inner_intensity - outer_intensity) * grid).reshape((-1,)) # numba不支持多维索引

    return i


@njit(cache=True)
def circle_vertical_window(
    x_window_center: int,  # 子像素边缘中心
    y_window_center: int,  # 子像素边缘中心
    x_center: float,  # 圆的中心 x 坐标
    y_center: float,  # 圆的中心 y 坐标
    radius: float,  # 圆的半径
    inner_intensity: float,  # 两侧强度
    outer_intensity: float,  # 两侧强度
    grid_resolution: int,
) -> np.ndarray:
    # compute pixels completely outside or inside
    r2 = radius**2

    # 生成 xy 索引, x 的每一列值都相同, y 的每一行值都相同 [3, 9]
    # x = np.zeros(
    #     ((np.arange(x_window_center - 1, x_window_center + 1 + 1)).size,
    #      (np.arange(y_window_center - 4, y_window_center + 4 + 1)).size),
    #     dtype=np.float64
    # ).T
    x = np.zeros(
        (
            (x_window_center + 1 + 1) - (x_window_center - 1),
            (y_window_center + 4 + 1) - (y_window_center - 4),
        ),
        dtype=np.float64,
    ).T
    y = x.copy()

    # 每一行都相同
    x[:, :] = np.arange(x_window_center - 1, x_window_center + 1 + 1)

    # 生成一列 y 的值
    y_vect = np.arange(y_window_center - 4, y_window_center + 4 + 1).reshape((-1, 1))
    # 每一列都相同
    y[:, :] = y_vect

    # meshgrid 生成 xy 索引, 和上面代码相同, 但是 meshgrid 不支持 numba [9, 3]
    # x_ = np.arange(x_window_center - 1, x_window_center + 1 + 1)
    # y_ = np.arange(y_window_center - 4, y_window_center + 4 + 1)
    # x, y = np.meshgrid(x_, y_, indexing='xy')

    # 判断x,y在偏移0.5像素的范围是否在圆内部
    c = ((x - x_center - 0.5) ** 2 + (y - y_center - 0.5) ** 2 < r2).astype(
        np.float64
    )  # 左上
    c += (x - x_center - 0.5) ** 2 + (y - y_center + 0.5) ** 2 < r2  # 左下
    c += (x - x_center + 0.5) ** 2 + (y - y_center - 0.5) ** 2 < r2  # 右上
    c += (x - x_center + 0.5) ** 2 + (y - y_center + 0.5) ** 2 < r2  # 右下

    # bool_c0 指的是 BB 的区域, bool_c4 指的是 AA 的区域
    bool_c0 = np.where(c.ravel() == 0)
    bool_c4 = np.where(c.ravel() == 4)
    # bool_c0 = np.where(c == 0) # numba不支持多维索引
    # bool_c4 = np.where(c == 4)

    # c 中大于0小于4的index [9, 3]
    bool_c04 = np.logical_and(c > 0, c < 4)

    # 将 AA 和 BB 的值放入重建图像中
    i = np.copy(c)
    i.ravel()[bool_c0] = outer_intensity
    i.ravel()[bool_c4] = inner_intensity
    # i[bool_c0] = outer_intensity # numba不支持多维索引
    # i[bool_c4] = inner_intensity

    # compute contour pixels
    delta = 1 / (grid_resolution - 1)

    #  -0.5到0.5 的范围,作用是标识在这个范围内的坐标,用在 circle_grid 中确定 c 中大于0小于4的的像素 grid [50, 50]
    dx = np.zeros(
        ((np.arange(-0.5, 0.5, delta)).size, (np.arange(-0.5, 0.5, delta)).size),
        dtype=np.float64,
    ).T
    dy = dx.copy()
    dx[:, :] = np.arange(-0.5, 0.5, delta)
    dy_vect = np.arange(-0.5, 0.5, delta).reshape((-1, 1))
    dy[:, :] = dy_vect

    # 获取 c 中大于0小于4的 [n, 1]
    grid = circle_grid(
        x.ravel()[bool_c04.ravel()],
        # x[bool_c04], # numba不支持多维索引
        y.ravel()[bool_c04.ravel()],
        # y[bool_c04], # numba不支持多维索引
        r2,
        x_center,
        y_center,
        dx,
        dy,
    )

    # 更新 c 中大于0小于4 的区域
    i.ravel()[bool_c04.ravel()] = (
        outer_intensity + (inner_intensity - outer_intensity) * grid
    ).reshape((-1,))
    # i[bool_c04] = (outer_intensity + (inner_intensity - outer_intensity) * grid).reshape((-1,)) # numba不支持多维索引

    return i


def test_circle_grid():
    print("test_circle_grid start")
    x = np.array([10.0, 9.0, 10.0, 8.0, 9.0, 8.0, 8.0])
    y = np.array([7.0, 8.0, 8.0, 9.0, 9.0, 10.0, 11.0])
    radius2 = 248.85607190069615
    x_center = np.array([21.00412171])
    y_center = np.array([18.74682164])
    dx = np.array(
        [
            [
                -0.5,
                -0.47959184,
                -0.45918367,
                -0.43877551,
                -0.41836735,
                -0.39795918,
                -0.3775510,
                -0.35714286,
                -0.33673469,
                -0.31632653,
                -0.29591837,
                -0.2755102,
                -0.2551020,
                -0.23469388,
                -0.21428571,
                -0.19387755,
                -0.17346939,
                -0.15306122,
                -0.1326530,
                -0.1122449,
                -0.09183673,
                -0.07142857,
                -0.05102041,
                -0.03061224,
                -0.0102040,
                0.01020408,
                0.03061224,
                0.05102041,
                0.07142857,
                0.09183673,
                0.1122449,
                0.13265306,
                0.15306122,
                0.17346939,
                0.19387755,
                0.21428571,
                0.2346938,
                0.25510204,
                0.2755102,
                0.29591837,
                0.31632653,
                0.33673469,
                0.3571428,
                0.37755102,
                0.39795918,
                0.41836735,
                0.43877551,
                0.45918367,
                0.4795918,
                0.5,
            ]
        ]
    )
    dy = np.swapaxes(dx, 0, 1)
    dx = np.repeat(dx, 50, axis=0)
    dy = np.repeat(dy, 50, axis=1)
    result = circle_grid(x, y, radius2, x_center, y_center, dx, dy)
    print(result.shape)
    # (7, 1)
    print(result)
    # [[0.1536]
    # [0.1412]
    # [0.8972]
    # [0.056 ]
    # [0.8376]
    # [0.6172]
    # [0.996 ]]
    print("test_circle_grid end\n")


def test_circle_horizontal_window():
    print("test_circle_horizontal_window start")
    result = circle_horizontal_window(
        6,
        13,
        11.41280755,
        14.06383106,
        5.073668424405436,
        29.611111111111107,
        130.0,
        50,
    )
    print(result.shape)
    print(result)
    print("-" * 100)
    result = circle_horizontal_window(
        269,
        171,
        241.04769697,
        167.43933401,
        27.721881801255833,
        46.74452311788913,
        255.0,
        50,
    )
    print(result.shape)
    print(result)
    print("test_circle_horizontal_window end")


def test_circle_vertical_window():
    print("test_circle_vertical_window start")
    result = circle_vertical_window(
        9,
        8,
        21.00412171,
        18.74682164,
        15.775172642500499,
        34.0,
        124.0,
        50,
    )
    print(result.shape)
    print(result)
    print("-" * 100)
    result = circle_vertical_window(
        10,
        23,
        22.03027806,
        7.30056223,
        19.779292893446936,
        33.44444444444444,
        150.5,
        50,
    )
    print(result.shape)
    print(result)
    print("test_circle_vertical_window end\n")


if __name__ == "__main__":
    test_circle_grid()
    print("*" * 100)
    test_circle_horizontal_window()
    print("*" * 100)
    test_circle_vertical_window()
