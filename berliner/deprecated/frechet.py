import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d


def frechet_distance(x1, x2):
    # 计算距离矩阵
    distance_matrix = cdist(x1, x2)

    # 动态规划计算 Frechet 距离
    dp_matrix = np.zeros_like(distance_matrix)
    dp_matrix[0, 0] = distance_matrix[0, 0]
    for i in range(1, x1.shape[0]):
        dp_matrix[i, 0] = max(dp_matrix[i - 1, 0], distance_matrix[i, 0])
    for j in range(1, x2.shape[0]):
        dp_matrix[0, j] = max(dp_matrix[0, j - 1], distance_matrix[0, j])
    for i in range(1, x1.shape[0]):
        for j in range(1, x2.shape[0]):
            dp_matrix[i, j] = max(
                min(dp_matrix[i - 1, j], dp_matrix[i, j - 1], dp_matrix[i - 1, j - 1]),
                distance_matrix[i, j],
            )

    # 回溯最佳匹配路径
    matching_indices = []
    i, j = x1.shape[0] - 1, x2.shape[0] - 1
    while i > 0 and j > 0:
        matching_indices.append((i, j))
        if (
            dp_matrix[i - 1, j - 1] <= dp_matrix[i, j - 1]
            and dp_matrix[i - 1, j - 1] <= dp_matrix[i - 1, j]
        ):
            i, j = i - 1, j - 1
        elif (
            dp_matrix[i, j - 1] <= dp_matrix[i - 1, j]
            and dp_matrix[i, j - 1] <= dp_matrix[i - 1, j - 1]
        ):
            j = j - 1
        else:
            i = i - 1
    matching_indices.append((0, 0))

    # 返回匹配点对
    return matching_indices[::-1]


def interpolate_mapping(x1, x2, matching_indices):
    # 插值计算映射
    x_indices = np.array([idx[0] for idx in matching_indices])
    x2_indices = np.array([idx[1] for idx in matching_indices])
    f = interp1d(x_indices, x2_indices, kind="linear")
    return f(np.arange(x1.shape[0]))


# 获取映射关系
# 计算 Frechet 距离并找到对应点
matching_indices = frechet_distance(x1, x2)

# 根据匹配点对进行插值映射
mapping = interpolate_mapping(x1, x2, matching_indices)

# 打印映射结果
print("Mapping:", mapping)

# mapping: for each point in x1, find the mapping index in x2

fig, ax = plt.subplots(1, 1)
ax.plot(isoc8["logTe"], isoc8["logg"], lw=2, label="8.00")
ax.plot(isoc802["logTe"], isoc802["logg"], lw=2, label="8.02")
logTe_interp = np.interp(mapping, np.arange(len(isoc802)), isoc802["logTe"])
logg_interp = np.interp(mapping, np.arange(len(isoc802)), isoc802["logg"])
for i in range(len(x1)):
    ax.plot(
        [isoc8["logTe"][i], logTe_interp[i]],
        [isoc8["logg"][i], logg_interp[i]],
        "k-",
    )
ax.plot(logTe_interp, logg_interp, "o-")
