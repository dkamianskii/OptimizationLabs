import numpy as np


def f(x):
    return 2 * x[0] ** 2 + 3 * x[1] ** 2 + np.sin(2 * x[0] + 7 * x[1]) / 49 + 3 * x[0] + 2 * x[1]


def grad_f(x):
    return np.array(
        [4 * x[0] + 2 * np.cos(2 * x[0] + 7 * x[1]) / 49 + 3, 6 * x[1] + 7 * np.cos(2 * x[0] + 7 * x[1]) / 49 + 2])


def golden_slice(func, segment, eps=0.01):
    tay = 0.618
    right_k = segment[1]
    left_k = segment[0]
    lbda_k = left_k + (1 - tay)*(right_k - left_k)
    mu_k = left_k + tay*(right_k - left_k)
    f_lbda = func(lbda_k)
    f_mu = func(mu_k)
    func_count = 2
    arithmetic_count = 5
    while right_k - left_k > eps:
        if f_lbda > f_mu:
            left_k = lbda_k
            lbda_k = mu_k
            mu_k = left_k + tay*(right_k - left_k)
            f_lbda = f_mu
            f_mu = func(mu_k)
        else:
            right_k = mu_k
            mu_k = lbda_k
            lbda_k = left_k + (1 - tay)*(right_k - left_k)
            f_mu = f_lbda
            f_lbda = func(lbda_k)
        func_count += 1
        arithmetic_count += 3
    return left_k, func_count, arithmetic_count


def gen_fib():
    a, b = 1, 1
    yield a
    yield b
    while True:
        a, b = b, a+b
        yield b


def phibonacci(func, segment, eps=0.001):
    g = gen_fib()
    F = []
    needed_F = (segment[1] - segment[0])/eps
    f_n = 1
    n = -1
    right_k = segment[1]
    left_k = segment[0]
    while f_n <= needed_F:
        f_n = next(g)
        n += 1
        F.append(f_n)
    k = 1
    lbda_k = left_k + (F[n - k - 1]/F[n - k + 1])*(right_k - left_k)
    mu_k = left_k + (F[n - k]/F[n - k + 1])*(right_k - left_k)
    f_lbda = func(lbda_k)
    f_mu = func(mu_k)
    func_count = 2
    arithmetic_count = 7
    k += 1
    while k != n - 1:
        if f_lbda > f_mu:
            left_k = lbda_k
            lbda_k = mu_k
            mu_k = left_k + (F[n - k - 1]/F[n - k + 1])*(right_k - left_k)
            f_lbda = f_mu
            f_mu = func(mu_k)
        else:
            right_k = mu_k
            mu_k = lbda_k
            lbda_k = left_k + (F[n - k - 2]/F[n - k + 1])*(right_k - left_k)
            f_mu = f_lbda
            f_lbda = func(lbda_k)
        func_count += 1
        arithmetic_count += 4
        k += 1
    return lbda_k, func_count, arithmetic_count


def fastest_descent(func, grad_func, min_func, eps, x_0):
    descent_count = 0
    x_k = x_0.copy()
    grad = grad_func(x_k)
    grad_norm = np.linalg.norm(grad)
    func_count = 0
    arithmetic_count = 0
    while grad_norm ** 2 > eps:
        alpha, min_count, min_arithmetic_count = min_func(lambda alpha_k: func(x_k - alpha_k * grad), (0, 1))
        func_count += min_count
        arithmetic_count += min_arithmetic_count
        x_k = x_k - alpha * grad
        grad = grad_func(x_k)
        grad_norm = np.linalg.norm(grad)
        descent_count += 1
    return x_k, func(x_k), descent_count, func_count, arithmetic_count


accuracy = [1/(10**e) for e in range(1, 11)]

file = open('out.txt', 'w')

x_start = np.array([-100., 100.])
file.write('starting point = ' + str(x_start) + '\n')
for acc in accuracy:
    file.write('accuracy =' + str(acc) + ':  \n')
    file.write('golden ratio ----------------\n')
    x_acc, f_acc, iter_main, f_count, arithmetic_count = fastest_descent(f, grad_f, golden_slice, acc, x_start)
    file.write('x_acc=' + str(x_acc) + ' main iter count=' + str(iter_main) + ' func count=' + str(f_count)
               + ' arithmetic count=' + str(arithmetic_count) + '  f(x_acc)=' + str(f_acc) + '\n')
    file.write('Phibonacci ----------------\n')
    x_acc, f_acc, iter_main, f_count, arithmetic_count = fastest_descent(f, grad_f, phibonacci, acc, x_start)
    file.write('x_acc=' + str(x_acc) + ' main iter count=' + str(iter_main) + ' func count=' + str(f_count)
               + ' arithmetic count=' + str(arithmetic_count) + '  f(x_acc)=' + str(f_acc) + '\n')
