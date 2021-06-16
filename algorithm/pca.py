'''
我们有 [公式] 个 [公式] 维的样本数据.

将原始数据转化成为一个 [公式] 行 [公式] 列的数据矩阵 [公式];
每一行减去均值, 进行"中心化";
求出协方差矩阵 [公式];
求出协方差矩阵的特征值和特征向量, 形成特征矩阵;
计算 [公式], [公式] 即为降维过后的数据;
将数据按特征值由大到小排列, 并选取前面 [公式] 维数据输出.

'''
import numpy as np


def pca(X, n_component) -> np.array:
    data = X.T

    # 2. Subtract the mean
    data -= data.mean(axis=1).reshape(-1, 1)

    # 3. Get the covariance matrix: 1/m * XX^T
    cov_matrix = 1 / len(data[0]) * data @ data.T

    # 4. Calculate the eigenvalue and eigenvector
    eigenvalues, eigenmatrix = np.linalg.eig(cov_matrix)

    # 5. Get data after dimensionality reduction and Sort eigenmatrix by eigenvalues
    new_data = eigenmatrix.T @ data

    # 6. Sort data and reduce dimension
    return new_data[sorted(range(len(eigenvalues)),
                           key=lambda v: -eigenvalues[v])][:n_component]
