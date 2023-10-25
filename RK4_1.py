
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
    dy = np.zeros((len(y)))
    dy[0] = np.exp(-2*x) - 2*y[0]
    return dy

# -----------------------

h = 0.2
x = np.array([0.0, 2.0])
yinit = np.array([1.0/10])


[ts, ys] = RK4thOrder('myFunc', yinit, x, h)


dt = int((x[-1]-x[0])/h)
t = [x[0]+i*h for i in range(dt+1)] 
yexact = []
for i in range(dt+1):
    ye = (1.0/10)*np.exp(-2*t[i]) + t[i]*np.exp(-2*t[i])
    yexact.append(ye)


diff = ys - yexact
print("Maximum difference =", np.max(abs(diff)))

plt.plot(ts, ys, 'r*')
plt.plot(t, yexact, 'b-')
plt.xlim(x[0], x[1])
plt.legend(["4th Order RK", "Exact solution"], loc=1)
plt.xlabel('x', fontsize=17)
plt.ylabel('y', fontsize=17)
plt.tight_layout()
plt.show()

# Uncomment the following to print the figure:
#plt.savefig('Fig_ex2_RK4_h0p1.png', dpi=600)



