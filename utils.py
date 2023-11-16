import numpy as np


def get_indexed_matrix(_matrix):
    # 扁平化矩阵，然后获取扁平化矩阵中元素的排序索引
    flat_indices = np.argsort(-_matrix, axis=None)

    # 创建一个与矩阵扁平化版本大小相同的排名数组，值从1开始
    ranks = np.arange(1, _matrix.size + 1)

    # 创建一个空的矩阵来存储排名
    ranked_matrix = np.zeros_like(_matrix, dtype=int)

    # 将排名放置到扁平化矩阵的相应位置
    ranked_matrix.ravel()[flat_indices] = ranks

    return ranked_matrix


# 添加扰动
def noise_it(_m, _indexed_m, _top_k=50):
    m_size = _m.shape
    fig_x = m_size[0]
    fig_y = m_size[1]
    fig_z = m_size[2]
    for z in range(fig_z):
        v = _m[:, :, z]
        for index in range(1, _top_k, 1):
            x, y = np.where(_indexed_m == index)
            r = np.random.randint(1, v[x, y]).astype(int) if v[x, y] > 1 else 1
            v[x, y] = v[x, y] + np.random.randint(-r, r) / index
    return _m


def random_noise_it(_m, _top_k=50):
    """添加随机扰动

    Args:
        _m (numpy.array): 被扰动的图片矩阵
        _top_k (int, optional): _description_. Defaults to 50.
    """
    m_size = _m.shape
    fig_z = m_size[2]
    for z in range(fig_z):
        v = _m[:, :, z]
        # 随机选择top_k个点
        for index in range(1, _top_k, 1):
            x = np.random.randint(0, v.shape[0])
            y = np.random.randint(0, v.shape[1])
            v[x, y] = np.random.randint(0, 255)
    return _m


def noise_lz(_m, _indexed_m, _top_k=50):
    """nju方法, 交换_indexed_m里的前k个点

    Args:
        _m (_type_): _description_
        _indexed_m (_type_): _description_
        _top_k (int, optional): _description_. Defaults to 50.
    """
    m_size = _m.shape
    fig_x = m_size[0]
    fig_y = m_size[1]
    fig_z = m_size[2]
    for z in range(fig_z):
        v = _m[:, :, z]
        for index in range(1, _top_k, 1):
            x_top, y_top = np.where(_indexed_m == index)
            # 寻找倒数第k个点
            x_bottom, y_bottom = np.where(_indexed_m == _indexed_m.size - index)
            # 交换
            v[x_top, y_top], v[x_bottom, y_bottom] = (
                v[x_bottom, y_bottom],
                v[x_top, y_top],
            )
    return _m
