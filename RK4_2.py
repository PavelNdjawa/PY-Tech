import matplotlib.pyplot as plt
import numpy as np


def feval(funcName, *args):
    return eval(funcName)(*args)


def RK4thOrder(func, yinit, x_range, h):
    m = len(yinit)
    n = int((x_range[-1] - x_range[0])/h)
    
    x = x_range[0]
    y = yinit
    
    # Containers for solutions
    xsol = np.empty(0)
    xsol = np.append(xsol, x)

    ysol = np.empty(0)
    ysol = np.append(ysol, y)

    for i in range(n):
        k1 = feval(func, x, y)

        yp2 = y + k1*(h/2)

        k2 = feval(func, x+h/2, yp2)

        yp3 = y + k2*(h/2)

        k3 = feval(func, x+h/2, yp3)

        yp4 = y + k3*h

        k4 = feval(func, x+h, yp4)

        for j in range(m):
            y[j] = y[j] + (h/6)*(k1[j] + 2*k2[j] + 2*k3[j] + k4[j])

        x = x + h
        xsol = np.append(xsol, x)

        for r in range(len(y)):
            ysol = np.append(ysol, y[r])  

    return [xsol, ysol]


def myFunc(x, y):
    # Van der Pol oscillator
    a = 1.0
    dy = np.zeros((len(y)))
    dy[0] = y[1]
    dy[1] = a*(1 - y[0]**2)*y[1] - y[0]
    return dy

# -----------------------

h = 0.01
x = np.array([0.0, 30.0])
yinit = np.array([2.0, 0.0])


[ts, ys] = RK4thOrder('myFunc', yinit, x, h)

node = len(yinit)
ys1 = ys[0::node]
ys2 = ys[1::node]



plt.plot(ts, ys1, 'r')
plt.plot(ts, ys2, 'b')
plt.xlim(x[0], x[1])
plt.legend(["y(1)", "y(2)"], loc=2)
plt.xlabel('x', fontsize=17)
plt.ylabel('y', fontsize=17)
plt.tight_layout()
plt.show()

# Uncomment the following to print the figure:
#plt.savefig('Fig_ex7_RK4th.png', dpi=600)

