import numpy as np

def f(x):
    # return 2 * x[0] ** 2 + 3 * x[1] ** 2 - 2 * np.sin((x[0] - x[1])/2) + x[1]
    # return x[0] ** 2 + 3 * x[1] ** 2 + np.sin(2 * x[0] + 7 * x[1]) + 3 * x[0] + 2 * x[1]
    return 2*x[0]**2 + 3*x[1]**2 + np.sin(2*x[0] + 7*x[1])/49 + 3*x[0] + 2*x[1]


def grad_f(x):
    # return np.array([4 * x[0] - np.cos((x[0] - x[1])/2), 6 * x[1] + np.cos((x[0] - x[1])/2) + 1])
    # return np.array([2 * x[0] + 2 * np.cos(2 * x[0] + 7 * x[1]) + 3, 6 * x[1] + 7 * np.cos(2 * x[0] + 7 * x[1]) + 2])
    return np.array([4*x[0] + 2*np.cos(2*x[0] + 7*x[1])/49 + 3, 6*x[1] + 7*np.cos(2*x[0] + 7*x[1])/49 + 2])


def matrix_hessa(x):
    # return np.array([[4 + np.sin((x[0] - x[1])/2)/2, -np.sin((x[0] - x[1])/2)/2],
    #                  [-np.sin((x[0] - x[1])/2)/2, 6 + np.sin((x[0] - x[1])/2)/2]])
    # return np.array([[2 - 4 * np.sin(2 * x[0] + 7 * x[1]), -14 * np.sin(2 * x[0] + 7 * x[1])],
    #                  [-14 * np.sin(2 * x[0] + 7 * x[1]), 6 - 49 * np.sin(2 * x[0] + 7 * x[1])]])
    return np.array([[4 - 4*np.sin(2*x[0] + 7*x[1])/49, -14*np.sin(2*x[0] + 7*x[1])/49],
                     [-14*np.sin(2*x[0] + 7*x[1])/49, 6 - 49*np.sin(2*x[0] + 7*x[1])/49]])


# m = 3
# M = 7

m = 2
M = 6


def grad_const_step(grad_func, alpha, eps, x_0):
    count = 0
    x_k = x_0.copy()
    grad = grad_func(x_k)
    grad_norm = np.linalg.norm(grad)
    while grad_norm**2 > eps:
        x_k = x_k - alpha*grad_func(x_k)
        grad = grad_func(x_k)
        grad_norm = np.linalg.norm(grad)
        count += 1
    return x_k, count


def Newton_pshenichii(func, grad_func, hess_func, alpha_0, eps, e, x_0):
    x_k = x_0.copy()
    alpha_k = alpha_0
    grad = grad_func(x_k)
    grad_norm = np.linalg.norm(grad)
    d_k = -np.dot(np.linalg.inv(hess_func(x_k)), (grad.reshape(2, 1)))
    d_k_str = d_k.flatten()
    count = 0
    while grad_norm**2 > eps:
        while func(x_k + alpha_k*d_k_str) - func(x_k) > e * alpha_k * (np.dot(grad, d_k)[0]):
            alpha_k /= 2
        x_k += alpha_k*d_k_str
        count += 1
        grad = grad_func(x_k)
        grad_norm = np.linalg.norm(grad)
        d_k = -np.dot(np.linalg.inv(hess_func(x_k)), (grad.reshape(2, 1)))
        d_k_str = d_k.flatten()
    return x_k, count


def Newton_pshenichii_count_beta(func, grad_func, hess_func, alpha_0, eps, e, x_0, x_toch, m_e):
    x_k = x_0.copy()
    alpha_k = alpha_0
    grad = grad_func(x_k)
    grad_norm = np.linalg.norm(grad)
    d_k = -np.dot(np.linalg.inv(hess_func(x_k)), (grad.reshape(2, 1)))
    d_k_str = d_k.flatten()
    count = 0
    beta = []
    while grad_norm**2 > eps:
        while func(x_k + alpha_k*d_k_str) - func(x_k) > e * alpha_k * (np.dot(grad, d_k)[0]):
            alpha_k /= 2
        x_prev = x_k.copy()
        x_k += alpha_k*d_k_str
        log_k = np.log(np.linalg.norm(x_k - x_toch))
        log_prev = np.log(np.linalg.norm(x_prev - x_toch))
        beta.append(log_k/log_prev)
        count += 1
        grad = grad_func(x_k)
        grad_norm = np.linalg.norm(grad)
        d_k = -np.dot(np.linalg.inv(hess_func(x_k)), (grad.reshape(2, 1)))
        d_k_str = d_k.flatten()
    return x_k, count, beta, grad_norm/m


accuracy = [0.1, 0.001, 0.0001]


file = open('out2.txt', 'w')

x_start = np.array([-100., 100.])
file.write('starting point = ' + str(x_start) + '\n')
for acc in accuracy:
    file.write('accuracy =' + str(acc) + ':  \n')
    file.write('1st grade ----------------\n')
    x_acc, k = grad_const_step(grad_f, 2/(M+m), acc, x_start)
    file.write('x_acc=' + str(x_acc) + '  count=' + str(k) + '  f(x_acc)=' + str(f(x_acc)) + '\n')
    file.write('Newton ----------------\n')
    x_acc, k = Newton_pshenichii(f, grad_f, matrix_hessa, 1, acc, 0.2, x_start)
    file.write('x_acc=' + str(x_acc) + '  count=' + str(k) + '  f(x_acc)=' + str(f(x_acc)) + '\n')


file.write('\n\n')
file.write('Research of convergence rate \n')
high_acc = 10e-16
acc_for_beta = 10e-5
x_high_acc, count = Newton_pshenichii(f, grad_f, matrix_hessa, 1, high_acc, 0.2, x_start)
x_acc, k, beta, acc_t = Newton_pshenichii_count_beta(f, grad_f, matrix_hessa, 1, acc_for_beta, 0.2, x_start, x_high_acc, m)
file.write('x_acc=' + str(x_acc) + '  count=' + str(k) + '  f(x_acc)=' + str(f(x_acc)) + '  betas for interation = ' + str(beta)+ '  accuracy = ' + str(acc_t))