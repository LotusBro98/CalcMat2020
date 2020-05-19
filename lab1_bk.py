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
    return np.float32([
        a * (y[1] - (y[0] * y[0] * y[0] - y[0])),
        c - y[0] - b * y[1]
    ])
    # для данного ОДУ y' = f(y) матрица Якоби функции f
    # | -a(y1^2 - 1)    a |
    # |      -1        -b |
    # собств. числа в т. y1=2 при больших a (a^2 >> a)
    # l1 ~ -3a
    # l2 ~ -b

def J(x, y, a):
    return np.float32([
        [-a * (y[0]*y[0] - 1), a],
        [-1, -b]
    ])

# явный метод Рунге-Кутты с таблицей Бутчера
# C | A
# --+---
#   | B
A = np.float32([
    [0, 0, 0],
    [1/2, 0, 0],
    [-1, 2, 0]
])
B = np.float32([1/6, 2/3, 1/6])
C = np.float32([0, 1/2, 1])
s = len(B)
n = len(y0)
# для данной таблицы и действительного z < 0 область устойчивости метода:
# z=l*h > -2
# будем выбирать h так, что h = -1/l
def scheme_rk(x, y, h, a):
    k = np.zeros((n, s), dtype=np.float32)
    for i in range(s):
        dy = h*(np.sum(A[i] * k, axis=-1))
        k[:,i] = F(x + C[i]*h, y + dy, a)

    y_next = y + h * np.sum(B * k, axis=-1)
    return y_next

# A = np.float32([
#     [1/4, 0, 0, 0, 0],
#     [1/2, 1/4, 0, 0, 0],
#     [17/50, -1/25, 1/4, 0, 0],
#     [371/1360, -137/2720, 15/544, 1/4, 0],
#     [25/24, -49/48, 125/16, -85/12, 1/4]
# ])
# B = np.float32([
#     25/24, -49/48, 125/16, -85/12, 1/4
# ])
# C = np.float32([
#     1/4, 3/4, 11/20, 1/2, 1
# ])
# s = len(B)
# n = len(y0)



def euler(x, y, h, a):
    return y + h * F(x, y, a)

E = np.eye(n, n)
def impl_euler_wtf(x, y, h, a):
    # y_n+1 = y_n + h * (E - h * J(y))^-1 * f(y)
    return y + h * np.matmul(np.linalg.inv(E - h * J(x, y, a)), F(x, y, a))
    # область устойчивости модельного уравнения - вне круга |z - 1| > 1


def solve_a1():
    h = 1 / max(3*a1, b)
    N = int((Tk - T0) / h + 1)
    t = np.linspace(T0, Tk, N)
    y = np.zeros((N, 2), dtype=np.float32)
    y[0] = y0

    for i in range(1, len(t)):
        y[i] = impl_euler_wtf(t[i-1], y[i-1], h, a1)

        if i % 1000 == 0:
            sys.stdout.write("\r{} / {}".format(i, len(t)))
    print()

    return t, y

def solve_a2():
    N = 60001
    h = (Tk - T0) / (N - 1)
    t = np.linspace(T0, Tk, N)
    y = np.zeros((N, 2), dtype=np.float32)
    y[0] = y0

    for i in range(1, len(t)):
        y[i] = impl_euler_wtf(t[i - 1], y[i - 1], h, a2)

        if i % 1000 == 0:
            sys.stdout.write("\r{} / {}".format(i, len(t)))
    print()

    return t, y

plt.ylim((-2, 2))

t_a1, y_a1 = solve_a1()

plt.plot(t_a1, y_a1[:,0], scaley=False)
plt.plot(t_a1, y_a1[:,1], scaley=False)
plt.show()

plt.ylim((-2, 2))
t_a2, y_a2 = solve_a2()
plt.plot(t_a2, y_a2[:,0], scaley=False)
plt.plot(t_a2, y_a2[:,1], scaley=False)
plt.show()