import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox  # 定义图形box的格式


def plot_embedding(X, X_raw, y, title=None):
    ''' 首先定义函数画出二维空间中的样本点，输入参数：1.降维后的数据；2.图片标题
    '''
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)  # 对每一个维度进行0-1归一化，注意此时X只有两个维度

    plt.figure(figsize=(8, 8))  # 设置整个图形大小
    ax = plt.subplot(111)
    colors = ['#5dbe80', '#2d9ed8', '#a290c4', '#efab40', '#eb4e4f', '#929591',  'black',  'gray',  'yellow',  'brown']

    # 画出样本点
    for i in range(X.shape[0]):  # 每一行代表一个样本
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 # color=plt.cm.Set1(y[i] / 10.),
                 color=colors[y[i]],
                 fontdict={'weight': 'bold', 'size': 9})  # 在样本点所在位置画出样本点的数字标签

    # 在样本点上画出缩略图，并保证缩略图够稀疏不至于相互覆盖
    # 只有matplotlib 1.0版本以上，offsetbox才有'AnnotationBbox'，所以需要先判断是否有这个功能
    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])  # 假设最开始出现的缩略图在(1,1)位置上
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)  # 算出样本点与所有展示过的图片（shown_images）的距离
            if np.min(dist) < 4e-3:  # 若最小的距离小于4e-3，即存在有两个样本点靠的很近的情况，则通过continue跳过展示该数字图片缩略图
                continue
            shown_images = np.r_[shown_images, [X[i]]]  # 展示缩略图的样本点通过纵向拼接加入到shown_images矩阵中

            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X_raw[i].reshape(8, 8), cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)

    plt.xticks([]), plt.yticks([])  # 不显示横纵坐标刻度
    if title is not None:
        plt.title(title)
