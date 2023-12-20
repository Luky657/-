import numpy as np
from scipy import linalg
from copy import deepcopy




#A与b的构成
def main():
    para_n = int(input('请输入n阶矩阵A的n:'))
    para_1 = float(input('请输入矩阵A的上对角线元素:'))
    para_2 = float(input('请输入矩阵A的主对角线元素:'))
    para_3 = float(input('请输入矩阵A的下对角线元素:'))

    A = np.eye(para_n)
    A = A * para_2
    A[0][0] = para_1
    for i in range(1, para_n):
        A[i][i - 1] = para_1
        A[i - 1][i] = para_3

    b_ = np.zeros(para_n)
    for i in range(0, para_n):
        b_[i] = float(input(f'请输入向量b的元素b{i+1}:'))
        i = i + 1


    print("A=\n", A)
    print("b=\n", b_.reshape(para_n, 1))

    check(A, b_)
    fengjie(A, b_)

    return


# 利用linalg库函数求得真实解X，与后续解得的X进行对比，以检验追赶法程序是否正确（观察对比）
def check(A, b):
    really_x = linalg.solve(A, b)
    print("真实解X =\n", really_x.reshape(len(A[0]), 1))
    return really_x


# 求解LY=b的Y
def solve_y(L, b):
    n = len(L[0])
    y = np.zeros(n)
    y[0] = b[0]
    for i in range(1, n):
        y[i] = b[i] - L[i][i-1] * y[i-1]
    print("y=\n", y.reshape(n, 1))
    return y


# 求解UX=y的X
def solve_x(U, y):
    n = len(U[0])
    x = np.zeros(n)
    x[n-1] = y[n-1] / U[n-1][n-1]
    for i in range(1, n):
        j = n - i - 1
        x[j] = ( y[j] - U[j][j+1] * x[j+1] ) / U[j][j]
    print("x=\n", x.reshape(n, 1))
    return x


# 将A=LU的LU分解成立
def fengjie(A, b):
    n = len(A[0])
    for i in range(1, n):
        A[i][i-1] = A[i][i-1] / A[i-1][i-1]
        A[i][i] = A[i][i] - A[i][i-1] * A[i-1][i]
    L = deepcopy(A)
    U = deepcopy(A)
    L[0][0] = 1.
    for i in range(1, n):
        L[i-1][i] = 0.
        U[i][i-1] = 0.
        L[i][i] = 1.
    print("L=\n", L)
    print("U=\n", U)


    y = solve_y(L, b)
    x = solve_x(U, y)

    return x


main()
