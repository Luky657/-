import numpy as np
from scipy import linalg
#A与b的构成 A为三阶矩阵
def main():
    para_n = int(input('请输入n阶矩阵A的n:'))

    A = np.eye(para_n)
    for i in range(0,para_n):

        for j in range(0,para_n):
            A[i][j] = float(input(f"请输入{para_n}阶矩阵A的元素a{i + 1}{j + 1}："))

    b = np.zeros(para_n)
    for i in range(0, para_n):
        b[i] = float(input(f'请输入向量b的元素b{i+1}:'))
        i = i + 1


    print("A=\n", A)
    print("b=\n", b.reshape(para_n, 1))

    check(A, b)
    Jacobi(A, b)
    Gauss(A, b)

    return

# 利用linalg库函数求得真实解X，与后续解得的X进行对比，以检验雅可比和高斯迭代法的收敛性（观察对比）
def check(A, b):
    para_n = len(A)
    really_x = np.linalg.solve(A, b)
    print("真实解X =\n", really_x.reshape(para_n, 1))
    return really_x




#雅可比迭代法
def Jacobi(A , b):
    k = 0
    m = int(input('请输入在雅可比迭代法中，你想要的迭代次数：'))
    para_n = len(A)
    x = np.zeros(para_n)
    for l in range(0, para_n):
        x[l] = float(0)          #初始值X0
        while True:              #迭代程序
            k += 1
            for i in range(0, para_n):
                z = 0
                for j in range(0, para_n):
                    if j != i :
                        z += A[i][j] * x[j]

                x[i] = (b[i] - z)/A[i][i]

            if k == m:
                print(f'用雅可比迭代法迭代了{m}次得到的解X({m})=\n', x.reshape(para_n, 1))
                return

        return


#高斯迭代法
def Gauss(A , b):
    k = 0
    m = int(input('请输入在高斯迭代法中，你想要的迭代次数：'))
    para_n = len(A)
    x = np.zeros(para_n)
    for l in range(0, para_n):
        x[l] = float(0)            # 初始值X0
        while True:                # 迭代程序
            k += 1
            for i in range(0, para_n):
                z = 0
                for j in range(0, para_n):
                    if j != i:
                        z += A[i][j] * x[j]
                        x[i] = (b[i] - z) / A[i][i]

            if k == m:
                print(f'用高斯迭代法迭代了{m}次得到的解X({m})=\n',x.reshape(para_n,1))
                return

        return


main()