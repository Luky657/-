import numpy as np


#矩阵A和向量U0，以及利用库函数求主特征值的真实解，与后续解得的主特征值进行对比，以检验幂法程序是否正确（观察对比）
def main():
    para_n = int(input("请输入n阶矩阵A的n："))

    A = np.eye(para_n)
    for i in range(0,para_n):

        for j in range(0,para_n):
            A[i][j] = float(input(f"请输入{para_n}阶矩阵A的元素a{i + 1}{j + 1}："))

    U0_ = np.zeros(para_n)
    for i in range(0, para_n):
        U0_[i] = float(1)
        i = i + 1
    U0 = U0_.reshape(para_n, 1)

    r,X =  np.linalg.eig(A)
    index = np.argmax(r)
    r1_ = np.real(r[index])


    print("A=\n", A)
    print('U0=\n',U0)
    print(f'主特征值r1*真实解=\n{r1_}')

    MiFa(A, U0)


#幂法算法
def MiFa(A, U0):
    Uk = U0
    k = 0
    M = int(input('请输入你想要的迭代次数：'))
    while True:
        k += 1
        Vk = np.dot(A,Uk)
        maxVk = np.max(Vk)
        Uk = Vk / maxVk


        if k == M:
            print(f'迭代了{M}次后主特征值r1=\n',maxVk)
            print(f'迭代了{M}次后主特征向量X1=\n', Uk)
            return



    return


main()
