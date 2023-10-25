
import matplotlib.pyplot as plt
import numpy as np


def feval(funcName, *args):
    return eval(funcName)(*args)


def RKF45(func, yinit, x_range, h):
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

        yp2 = y + k1*(h/5)

        k2 = feval(func, x+h/5, yp2)

        yp3 = y + k1*(3*h/40) + k2*(9*h/40)

        k3 = feval(func, x+(3*h/10), yp3)

        yp4 = y + k1*(3*h/10) - k2*(9*h/10) + k3*(6*h/5)

        k4 = feval(func, x+(3*h/5), yp4)

        yp5 = y - k1*(11*h/54) + k2*(5*h/2) - k3*(70*h/27) + k4*(35*h/27)

        k5 = feval(func, x+h, yp5)

        yp6 = y + k1*(1631*h/55296) + k2*(175*h/512) + k3*(575*h/13824) + k4*(44275*h/110592) + k5*(253*h/4096)

        k6 = feval(func, x+(7*h/8), yp6)

        for j in range(m):
            y[j] = y[j] + h*(37*k1[j]/378 + 250*k3[j]/621 + 125*k4[j]/594 + 512*k6[j]/1771)

        x = x + h
        xsol = np.append(xsol, x)

        for r in range(len(y)):
            ysol = np.append(ysol, y[r])  

    return [xsol, ysol]


def myFunc(x, y):
    dy = np.zeros((len(y)))
    dy[0] = np.exp(-2*x) - 2*y[0]
    return dy

# -----------------------

h = 0.2
x = np.array([0.0, 2.0])
yinit = np.array([1.0/10])


[ts, ys] = RKF45('myFunc', yinit, x, h)


dt = int((x[-1]-x[0])/h)
t = [x[0]+i*h for i in range(dt+1)] 
yexact = []
for i in range(dt+1):
    ye = (1.0 / 10) * np.exp(-2 * t[i]) + t[i] * np.exp(-2 * t[i])
    yexact.append(ye)


y_diff = ys - yexact
print("max diff =", np.max(abs(y_diff)))


plt.plot(ts, ys, 'rs')
plt.plot(t, yexact, 'b')
plt.xlim(x[0], x[1])
plt.legend(["RKF45", "Exact solution"], loc=1)
plt.xlabel('x', fontsize=17)
plt.ylabel('y', fontsize=17)
plt.tight_layout()
plt.show()

# Uncomment the following to print the figure:
#plt.savefig('Fig_ex2_RK4_h0p1.png', dpi=600)



