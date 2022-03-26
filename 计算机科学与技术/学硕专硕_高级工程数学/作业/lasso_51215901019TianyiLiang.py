"""
reference:https://github.com/PytLab/MLBox
高级工程数学 lasso回归python实现
"""
import itertools
import numpy as np

def get_corrcoef(X, Y):
    # X Y 的协方差
    cov = np.mean(X*Y) - np.mean(X)*np.mean(Y)
    return cov/(np.var(X)*np.var(Y))**0.5

def load_data(filename):
    ''' 加载数据
    '''
    X, Y = [], []
    with open(filename, 'r') as f:
        for line in f:
            splited_line = [float(i) for i in line.split()]
            x, y = splited_line[: -1], splited_line[-1]
            X.append(x)
            Y.append(y)
    X, Y = np.matrix(X), np.matrix(Y).T
    return X, Y

def standarize(X):
    ''' 中心化 & 标准化数据 (零均值, 单位标准差)
    '''
    std_deviation = np.std(X, 0)
    mean = np.mean(X, 0)
    return (X - mean)/std_deviation

def lasso_regression(X, y, lambd=0.2, threshold=0.1):
    ''' 通过坐标下降(coordinate descent)法获取LASSO回归系数
    '''
    # 计算残差平方和
    rss = lambda X, y, w: (y - X*w).T*(y - X*w)
    # 初始化回归系数w.
    m, n = X.shape
    w = np.matrix(np.zeros((n, 1)))
    r = rss(X, y, w)
    # 使用坐标下降法优化回归系数w
    niter = itertools.count(1)
    for it in niter:
        for k in range(n):
            # 计算常量值z_k和p_k
            z_k = (X[:, k].T*X[:, k])[0, 0]
            p_k = 0
            for i in range(m):
                p_k += X[i, k]*(y[i, 0] - sum([X[i, j]*w[j, 0] for j in range(n) if j != k]))
            if p_k < -lambd/2:
                w_k = (p_k + lambd/2)/z_k
            elif p_k > lambd/2:
                w_k = (p_k - lambd/2)/z_k
            else:
                w_k = 0
            w[k, 0] = w_k
        r_prime = rss(X, y, w)
        delta = abs(r_prime - r)[0, 0]
        r = r_prime
        print('Iteration: {}, delta = {}'.format(it, delta))
        if delta < threshold:
            break
    return w

if '__main__' == __name__:
    X, y = load_data('abalone.txt')
    X, y = standarize(X), standarize(y)
    w = lasso_regression(X, y, lambd=10)
    y_prime = X*w
    # 计算相关系数
    corrcoef = get_corrcoef(np.array(y.reshape(1, -1)),
                            np.array(y_prime.reshape(1, -1)))
    print('Correlation coefficient: {}'.format(corrcoef))
    """
result:
>python3 lasso_51215901019TianyiLiang.py
    Iteration: 146, delta = 0.1081124857935265
    Iteration: 147, delta = 0.10565615985365184
    Iteration: 148, delta = 0.10326058648411163
    Iteration: 149, delta = 0.10092418256476776
    Iteration: 150, delta = 0.09864540659987142
    Correlation coefficient: 0.7255254877587117
    """
