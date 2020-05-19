# X.9.5
import sys

import numpy as np
import matplotlib.pyplot as plt


y0 = [2, 0]
T0 = 0
Tk = 20

a1 = 1e3
a2 = 1e6
b = 0.1
c = 0.5


def F(x, y: np.ndarray, a) -> np.ndarray:
    return np.float64([
        a * (y[1] - (y[0] * y[0] * y[0] / 3 - y[0])),
        c - y[0] - b * y[1]
    ])
    # для данного ОДУ y' = f(y) матрица Якоби функции f
    # | -a(y1^2 - 1)    a |
    # |      -1        -b |
    # собств. числа в т. y1=2 при больших a (a^2 >> a)
    # l1 ~ -3a
    # l2 ~ -b

def J(x, y, a):
    return np.float64([
        [-a * (y[0]*y[0] - 1), a],
        [-1, -b]
    ])



# явный метод Рунге-Кутты
# A = np.float64([
#     [0, 0, 0],
#     [1/2, 0, 0],
#     [-1, 2, 0]
# ])
# B = np.float64([1/6, 2/3, 1/6])
# C = np.float64([0, 1/2, 1])
# s = len(B)
# n = len(y0)
# E = np.eye(n, n)
# # для данной таблицы и действительного z < 0 область устойчивости метода:
# # z=l*h > -2
# def scheme_rk(x, y, h, a):
#     k = np.zeros((n, s), dtype=np.float64)
#     for i in range(s):
#         dy = h*(np.sum(A[i] * k, axis=-1))
#         k[:,i] = F(x + C[i]*h, y + dy, a)
#
#     y_next = y + h * np.sum(B * k, axis=-1)
#     return y_next



# Однократно диагонально-неявный метод Рунге-Кутты
# Неявные вычисления с помощью метода Ньютона
# L - устойчив, если выполняется за конечное время. Подберем h и eps, чтобы можно было дождаться.
A = np.float64([
    [1/4, 0, 0, 0, 0],
    [1/2, 1/4, 0, 0, 0],
    [17/50, -1/25, 1/4, 0, 0],
    [371/1360, -137/2720, 15/544, 1/4, 0],
    [25/24, -49/48, 125/16, -85/12, 1/4]
])
A_lower = np.float64([
    [0, 0, 0, 0, 0],
    [1/2, 0, 0, 0, 0],
    [17/50, -1/25, 0, 0, 0],
    [371/1360, -137/2720, 15/544, 0, 0],
    [25/24, -49/48, 125/16, -85/12, 0]
])
a_d = 1/4
B = np.float64([
    25/24, -49/48, 125/16, -85/12, 1/4
])
C = np.float64([
    1/4, 3/4, 11/20, 1/2, 1
])
s = len(B)
n = len(y0)
En = np.eye(n, n)
def impl_runge_kutt(x, y, h, a, p=1, eps=1e-1):
    k = np.zeros((n, s), dtype=np.float64)
    for i in range(s):
        # Вычисляем ki через неявное уравнение, считая предыдущие k известными

        # Начнем приближение методом Ньютона с ki = (E - h*a_d*J(y))^-1 * F(y + h * A_l[i] * K)
        y_ = y + h * np.sum(A_lower[i] * k, axis=-1)
        J_ = En - h * a_d * J(x, y_, a)
        J_1 = np.linalg.inv(J_)
        k[:,i] = np.matmul(J_1, F(x, y_, a))

        # Невязка
        y_ = y + h * np.sum(A[i] * k, axis=-1)
        f = k[:,i] - F(x, y_, a) # -> 0
        while np.linalg.norm(f) > eps:
            # Шаг метода Ньютона
            # J_ = (df / dki)
            J_ = En - h * a_d * J(x, y_, a)
            J_1 = np.linalg.inv(J_)
            step = -p * np.matmul(J_1, f)

            # if np.linalg.norm(step) > step_clamp:
            #     step *= step_clamp / np.linalg.norm(step)

            k[:,i] += step
            # Вычисляем невязку
            y_ = y + h * np.sum(A[i] * k, axis=-1)
            f = k[:,i] - F(x, y_, a)
    # Все коэффициенты ki вычислены

    # Собираем y_n+1
    y_next = y + h * np.sum(B * k, axis=-1)
    return y_next


def euler(x, y, h, a):
    return y + h * F(x, y, a)

def impl_euler_wtf(x, y, h, a):
    # y_n+1 = y_n + h * (E - h * J(y))^-1 * f(y)
    return y + h * np.matmul(np.linalg.inv(En - h * J(x, y, a)), F(x, y, a))
    # читерский метод Эйлера (почти неявный) (используем знание матрицы Якоби)
    # на модельном уравнении устойчив при |z - 1| > 1
    # на практике получается обл.устойчивости, как у явного метода Эйлера,
    # но точность выше
    # просто оставлю это здесь >_<


def solve_a1():
    # Для отрицательных z явный метод Эйлера устойчив при z > -2
    h = 1.99999999 / max(3*a1, b)
    N = int((Tk - T0) / h + 1)
    t = np.linspace(T0, Tk, N)
    y = np.zeros((N, 2), dtype=np.float64)
    y[0] = y0

    for i in range(1, len(t)):
        # y[i] = euler(t[i-1], y[i-1], h, a1)

        # Лучше приближает решение, и быстрее явного МРК,
        # но с доказательством устойчивости не очень понятно
        y[i] = impl_euler_wtf(t[i-1], y[i-1], h, a1)

        if i % 1000 == 0:
            sys.stdout.write("\r{} / {}".format(i, len(t)))
    print()

    return t, y

def solve_a2():
    N = 100001
    h = (Tk - T0) / (N - 1)
    t = np.linspace(T0, Tk, N)
    y = np.zeros((N, 2), dtype=np.float64)
    y[0] = y0

    for i in range(1, len(t)):
        y[i] = impl_runge_kutt(t[i - 1], y[i - 1], h, a2)

        # if i % 1000 == 0:
        sys.stdout.write("\r{} / {}".format(i, len(t)))
    print()

    return t, y


t_a1, y_a1 = solve_a1()

plt.plot(t_a1, y_a1[:,0])
plt.plot(t_a1, y_a1[:,1])
plt.show()


t_a2, y_a2 = solve_a2()

plt.plot(t_a2, y_a2[:,0])
plt.plot(t_a2, y_a2[:,1])
plt.show()