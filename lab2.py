import sys

import numpy as np
import matplotlib.pyplot as plt

# y'' = e^y
# y'' = -e^y

def precise1(x, C1, C2):
    x = x + 0j
    y = np.log(1/2*C1*(np.power(np.tanh(1/2*np.sqrt(C1 * (C2 + x) * (C2 + x))), 2) - 1))
    return np.real(y)

def precise1_grad(x, C1, C2, h=1e-3):
    return np.float64([
        (precise1(x, C1 + h, C2) - precise1(x, C1 - h, C2)) / (2 * h),
        (precise1(x, C1, C2 + h) - precise1(x, C1, C2 - h)) / (2 * h),
    ])

def precise2(x, C1, C2):
    x = x + 0j
    y = np.log(1/2*(C1 - C1 * np.power(np.tanh(1/2*np.sqrt(C1 * (C2 + x) * (C2 + x))), 2)))
    return np.real(y)

def precise2_grad(x, C1, C2, h=1e-3):
    return np.float64([
        (precise2(x, C1 + h, C2) - precise2(x, C1 - h, C2)) / (2 * h),
        (precise2(x, C1, C2 + h) - precise2(x, C1, C2 - h)) / (2 * h),
    ])

a = 1
b1 = 1
b2 = 2


def calc_C1_C2(func, grad, b, C10, C20, p=0.2, p1=1e-2, eps=1e-3):
    C1 = C10
    C2 = C20
    i = 0
    while True:
        f0 = func(0, C1, C2)
        f1 = func(1, C1, C2)
        f = (f0 - a) * (f0 - a)
        if np.abs(f) < eps*eps:
            break
        g = 2 * (f0 - a) * grad(0, C1, C2)
        step = -p * f / g
        C1 += step[0]
        C2 += step[1]
        i += 1

    while True:
        f0 = func(0, C1, C2)
        f1 = func(1, C1, C2)
        f = (f0 - a) * (f0 - a) + (f1 - b) * (f1 - b)
        if np.abs(f) < eps*eps:
            break
        g = 2 * (f0 - a) * grad(0, C1, C2) + 2 * (f1 - b) * grad(1, C1, C2)
        step = -p1 * g
        C1 += step[0]
        C2 += step[1]
        i += 1

    return C1, C2


def shoot(func_yxx, yx0, h=1e-3):
    N = int(1 / h)
    y = np.zeros((N,), dtype=np.float64)
    y[0] = a
    yx = yx0
    for i in range(1, N):
        yxx = func_yxx(y[i-1])
        yx += h * yxx
        y[i] = y[i-1] + yx * h

    return y

def calc_shooting(func_yxx, b, h=1e-3, eps=1e-3, p=0.1, plot=False):
    yx0 = 0
    while True:
        y = shoot(func_yxx, yx0, h)
        if plot:
            plt.plot(x, y)
            # plt.show()
        f = y[-1] - b
        dy1_dyx0 = (shoot(func_yxx, yx0 + h, h)[-1] - shoot(func_yxx, yx0 - h, h)[-1]) / 2 / h
        if np.abs(f) < eps:
            return y
        yx0 -= p * f / dy1_dyx0

def shoot_Numerov(func_yxx, grad, y1, h=1e-3, eps=1e-3, p=1, max_step=1):
    N = int(1 / h)
    y = np.zeros((N,), dtype=np.float64)
    f = np.zeros((N,), dtype=np.float64)
    y[0] = a
    y[1] = y1
    f[0] = func_yxx(y[0])
    f[1] = func_yxx(y[1])
    for i in range(2, N):
        # y_n+1 - 2 * y_n + y_n-1
        y[i] = y[i-1] + (y[i-1] - y[i-2])
        while True:
            f[i] = func_yxx(y[i])
            err = (y[i] - 2 * y[i-1] + y[i-2]) / (h*h) - (f[i-1] + 1/12*(f[i] -2*f[i-1] + f[i-2]))
            if np.abs(err) < eps:
                break
            g = 1/(h*h) - 1/12 * grad(y[i])
            step = -p * err / g
            if np.abs(step) >  max_step:
                step = step / np.abs(step) * max_step
            y[i] += step

    return y

def calc_Numerov(func_yxx, grad, b, h=1e-3, eps=1e-3, p=1):
    y1 = a - h
    while True:
        y = shoot_Numerov(func_yxx, grad, y1, h)
        f = y[-1] - b
        dy1_dyx0 = (shoot_Numerov(func_yxx, grad, y1 + h, h)[-1] - shoot_Numerov(func_yxx, grad, y1 - h, h)[-1]) / 2 / h
        if np.abs(f) < eps:
            return y
        y1 -= p * f / dy1_dyx0



h = 1e-3
x = np.linspace(0, 1, int(1/h))

C11b1, C21b1 = calc_C1_C2(precise1, precise1_grad, b1, -1, 1)
C11b2, C21b2 = calc_C1_C2(precise1, precise1_grad, b2, -1, 1)
C12b1, C22b1 = calc_C1_C2(precise2, precise2_grad, b1, 1, 1)
# # C12b2, C22b2 = calc_C1_C2(precise2, precise2_grad, b2, -1, -2)

y1b1_calc = calc_shooting(lambda y: np.exp(y), b1)
y1b2_calc = calc_shooting(lambda y: np.exp(y), b2)
y1b1_precise = precise1(x, C11b1, C21b1)
y1b2_precise = precise1(x, C11b2, C21b2)
y1b1_num = calc_Numerov(lambda y: np.exp(y), lambda y: np.exp(y), b1)
y1b2_num = calc_Numerov(lambda y: np.exp(y), lambda y: np.exp(y), b2)
plt.plot(x, y1b1_calc)
plt.plot(x, y1b2_calc)
plt.plot(x, y1b1_num)
plt.plot(x, y1b2_num)
plt.plot(x, y1b1_precise)
plt.plot(x, y1b2_precise)
plt.show()
# print(np.average(np.abs(y1b1_calc - y1b1_precise)))
# print(np.average(np.abs(y1b2_calc - y1b2_precise)))

y2b1_calc = calc_shooting(lambda y: -np.exp(y), b1)
# y2b2_calc = calc_shooting(lambda y: -np.exp(y), b2)
# для метода Ньютона f != 0 никогда (упирается в лок.минимум)
y2b1_num = calc_Numerov(lambda y: -np.exp(y), lambda y: -np.exp(y), b1)
y2b1_precise = precise2(x, C12b1, C22b1)
plt.plot(x, y2b1_calc)
plt.plot(x, y2b1_num)
plt.plot(x, y2b1_precise)
plt.show()
# print(np.average(np.abs(y2b1_calc - y2b1_precise)))