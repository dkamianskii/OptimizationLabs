import numpy as np


def f(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + 10*x[0] + 5*x[1] - 3*x[3] - 20


def grad_f(x):
    return np.array([2*x[0] + 10, 2*x[1] + 5, 2*x[2], 2*x[3] - 3])


C_mat = np.array([[1, -1, 1, 0], [1, 0, 1, 1]])
F_mat = np.array([[1, 2, 3, 4], [1, 4, 0, 0]])
g_v = np.array([9, 8])


def f2(x):
    return x[0]**2 + x[1]**2


def grad_f2(x):
    return np.array([2*x[0], 2*x[1]])


C_mat2 = np.array([[1, 2]])
F_mat2 = np.array([[-2, 1]])
g_v2 = np.array([0])


def rozen(func, grad_func, C, F, g, n, alpha_0, x_0, lbd):
    x_k = x_0.copy()
    k = 0
    while True:
        print('----------------------------------')
        print('Step =  ' + str(k))
        print('x_k =  ', x_k)
        print('F(x_k) =  ' + str(func(x_k)))
        print(
            f'({x_k[0]:.3f})*({C[0][0]}) + ({x_k[1]:.3f})*({C[0][1]}) + ({x_k[2]:.3f})*({C[0][2]}) + ({x_k[3]:.3f})*({C[0][3]}) = '
            f' {np.dot(C, np.reshape(x_k, (n, 1)))[0][0]:.3f}, d1 = 2')
        print(
            f'({x_k[0]:.3f})*({C[1][0]}) + ({x_k[1]:.3f})*({C[1][1]}) + ({x_k[2]:.3f})*({C[1][2]}) + ({x_k[3]:.3f})*({C[1][3]}) = '
            f' {np.dot(C, np.reshape(x_k, (n, 1)))[1][0]:.3f}, d2 = 3')
        print(
            f'({x_k[0]:.3f})*({F[0][0]}) + ({x_k[1]:.3f})*({F[0][1]}) + ({x_k[2]:.3f})*({F[0][2]}) + ({x_k[3]:.3f})*({F[0][3]}) = '
            f' {np.dot(F, np.reshape(x_k, (n, 1)))[0][0]:.3f} <=  9')
        print(
            f'({x_k[0]:.3f})*({F[1][0]}) + ({x_k[1]:.3f})*({F[1][1]}) + ({x_k[2]:.3f})*({F[1][2]}) + ({x_k[3]:.3f})*({F[1][3]}) = '
            f' {np.dot(F, np.reshape(x_k, (n, 1)))[1][0]:.3f} <=  8')
        bords = np.dot(F, np.reshape(x_k, (n, 1)))
        F1 = np.empty((0, n))
        F2 = np.empty((0, n))
        g2 = np.array([])
        for i in range(bords.shape[0]):
            if bords[i] == g[i]:
                F1 = np.append(F1, np.array([F[i]]), axis=0)
            else:
                F2 = np.append(F2, np.array([F[i]]), axis=0)
                g2 = np.append(g2, g[i])
        alpha_k = alpha_0
        step1 = True
        print('Matrix F1:')
        print(F1)
        while step1:
            if C.size == 0 and F1.size == 0:
                A = None
                P_k = 1
            else:
                if F1.size != 0:
                    A = np.append(C, F1, axis=0)
                else:
                    A = C
                A_t = np.transpose(A)
                P_k = np.identity(n) - np.dot(np.dot(A_t, np.linalg.inv(np.dot(A, A_t))), A)
            print('Matrix A:')
            print(A)
            s_k = -np.dot(P_k, np.reshape(grad_func(x_k), (n, 1)))
            print('s_k:  ')
            print(s_k)
            step1 = False
            if np.linalg.norm(s_k) <= 1e-13:
                if A is None:
                    return x_k, func(x_k), k
                else:
                    w = - np.dot(np.dot(np.linalg.inv(np.dot(A, A_t)), A), np.reshape(grad_func(x_k), (n, 1)))
                    print('W vector:  ')
                    print(w)
                    for j in range(F1.shape[0]):
                        if w[C.shape[0] + j] < 0:
                            F1 = np.delete(F1, j, axis=0)
                            step1 = True
                            break
                    if not step1:
                        return x_k, func(x_k), k
        F2_flag = False
        s_k = np.reshape(s_k, (1, n))[0]
        F2_from_xk = np.dot(F2, np.reshape((x_k + alpha_k * s_k), (n, 1)))
        for i in range(F2_from_xk.size):
            if F2_from_xk[i] > g2[i]:
                F2_flag = True
                break
        while (func(x_k + alpha_k * s_k) > func(x_k)) or F2_flag:
            alpha_k /= lbd
            F2_flag = False
            F2_from_xk = np.dot(F2, np.reshape((x_k + alpha_k * s_k), (n, 1)))
            for i in range(F2_from_xk.size):
                if F2_from_xk[i] > g2[i]:
                    F2_flag = True
                    break
        print('alpha_k =  ', alpha_k)
        x_k = x_k + alpha_k * s_k
        k += 1


file = open('out.txt', 'w')
file.write(str(rozen(f, grad_f, C_mat, F_mat, g_v, 4, 0.1, np.array([2, 1, 1, 0]), 2)) + '\n')
# file.write(str(rozen(f, grad_f, C_mat, F_mat, g_v, 4, 0.1, np.array([2, 1, 1, 0]), 2)) + '\n')
# file.write(str(rozen(f, grad_f, C_mat, F_mat, g_v, 4, 0.1, np.array([10, -4, -12, 5]), 2)) + '\n')