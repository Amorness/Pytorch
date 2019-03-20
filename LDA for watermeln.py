# -*- coding: utf-8 -*

import numpy as np
import matplotlib.pyplot as plt

data_file = open('./watermelonData/watermelon_3a.csv')
dataset = np.loadtxt(data_file, delimiter=",")

# 第一列和第二列为样本
X = dataset[:, 1:3]
#第三列为标记
y = dataset[:, 3]

# 可视化样例点
f1 = plt.figure(1)
plt.title('watermelon_3a')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
#X[y==0 ，0]表示选取X中标记y为0的第一列数据
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=100, label='bad')
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=100, label='good')
plt.legend(loc='upper right')
# plt.show()



# 1-st. get the mean vector of each class

u = []
for i in range(2):  # two class
    u.append(np.mean(X[y == i], axis=0))  # column mean

# 2-nd. computing the within-class scatter matrix, refer on book (3.33)
m, n = np.shape(X)
Sw = np.zeros((n, n))
for i in range(m):
    x_tmp = X[i].reshape(n, 1)  # row -> cloumn vector
    if y[i] == 0: u_tmp = u[0].reshape(n, 1)
    if y[i] == 1: u_tmp = u[1].reshape(n, 1)
    Sw += np.dot(x_tmp - u_tmp, (x_tmp - u_tmp).T)

Sw = np.mat(Sw)

#计算w，对Sw做奇异值分解，但python的svd得出的V是已经转置后的，sigma只有对角线元素
U, sigma, V = np.linalg.svd(Sw)

#求Sw^-1,inv是求逆矩阵，diag是根据对角线元素创建对角矩阵
Sw_inv = V.T * np.linalg.inv(np.diag(sigma)) * U.T
# 3-th. computing the parameter w, refer on book (3.39)
w = np.dot(Sw_inv, (u[0] - u[1]).reshape(n, 1))

print(w)

# 4-th draw the LDA line in scatter figure

# f2 = plt.figure(2)
f3 = plt.figure(3)
plt.xlim(-0.2, 1)
plt.ylim(-0.5, 0.7)
#w[1, 0] / w[0, 0] 斜率
p0_x0 = -X[:, 0].max()
p0_x1 = (w[1, 0] / w[0, 0]) * p0_x0
p1_x0 = X[:, 0].max()
p1_x1 = (w[1, 0] / w[0, 0]) * p1_x0

plt.title('watermelon_3a - LDA')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=10, label='bad')
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=10, label='good')
#显示图例在右上角
plt.legend(loc='upper right')

plt.plot([p0_x0, p1_x0], [p0_x1, p1_x1])
print([p0_x0, p1_x0], [p0_x1, p1_x1])

# draw projective point on the line
def GetProjectivePoint_2D(point, line):
    a = point[0]
    b = point[1]
    k = line[0]
    t = line[1]

    if   k == 0:      return [a, t]
    elif k == np.inf: return [0, b]
    x = (a+k*b-k*t) / (k*k+1)
    y = k*x + t
    return [x, y]


m, n = np.shape(X)
for i in range(m):
    x_p = GetProjectivePoint_2D([X[i, 0], X[i, 1]], [w[1, 0] / w[0, 0], 0])
    if y[i] == 0:
        plt.plot(x_p[0], x_p[1], 'ko', markersize=5)
    if y[i] == 1:
        plt.plot(x_p[0], x_p[1], 'go', markersize=5)
    plt.plot([x_p[0], X[i, 0]], [x_p[1], X[i, 1]], 'c--', linewidth=0.3)

# plt.show()



#去除第15个错误的点，重新进行LDA
# 1-st. get the mean vector of each class
X = np.delete(X, 14, 0)
y = np.delete(y, 14, 0)

u = []
for i in range(2):  # two class
    u.append(np.mean(X[y == i], axis=0))  # column mean

# 2-nd. computing the within-class scatter matrix, refer on book (3.33)
m, n = np.shape(X)
Sw = np.zeros((n, n))
for i in range(m):
    x_tmp = X[i].reshape(n, 1)  # row -> cloumn vector
    if y[i] == 0: u_tmp = u[0].reshape(n, 1)
    if y[i] == 1: u_tmp = u[1].reshape(n, 1)
    Sw += np.dot(x_tmp - u_tmp, (x_tmp - u_tmp).T)

Sw = np.mat(Sw)
U, sigma, V = np.linalg.svd(Sw)

Sw_inv = V.T * np.linalg.inv(np.diag(sigma)) * U.T
# 3-th. computing the parameter w, refer on book (3.39)
w = np.dot(Sw_inv, (u[0] - u[1]).reshape(n, 1))  # here we use a**-1 to get the inverse of a ndarray

print(w)

# 4-th draw the LDA line in scatter figure

# f2 = plt.figure(2)
f4 = plt.figure(4)
plt.xlim(-0.2, 1)
plt.ylim(-0.5, 0.7)

p0_x0 = -X[:, 0].max()
p0_x1 = (w[1, 0] / w[0, 0]) * p0_x0
p1_x0 = X[:, 0].max()
p1_x1 = (w[1, 0] / w[0, 0]) * p1_x0

plt.title('watermelon_3a - LDA')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=10, label='bad')
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=10, label='good')
plt.legend(loc='upper right')

plt.plot([p0_x0, p1_x0], [p0_x1, p1_x1])
print([p0_x0, p1_x0], [p0_x1, p1_x1])


m, n = np.shape(X)
for i in range(m):
    x_p = GetProjectivePoint_2D([X[i, 0], X[i, 1]], [w[1, 0] / w[0, 0], 0])
    if y[i] == 0:
        plt.plot(x_p[0], x_p[1], 'ko', markersize=5)
    if y[i] == 1:
        plt.plot(x_p[0], x_p[1], 'go', markersize=5)
    plt.plot([x_p[0], X[i, 0]], [x_p[1], X[i, 1]], 'c--', linewidth=0.3)

plt.show()