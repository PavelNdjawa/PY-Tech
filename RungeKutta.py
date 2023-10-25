import numpy as np
import scipy as sp
import pylab as pl
import math
########initialisaion#########################################################################


#############################################################################################
def f(x,y,p):
	return p

def g(x,y,p):
	s=((1-x-2*p)/x)
	return s

x0=1    #initialisation of x
xf=4
h=0.01
n=int((xf-x0)/h+1)

x=np.array([x0+i*h for i in range (n)])
#x=np.linspace(1,4,n)
y0=2
p0=1

y=[]
p=[]

y.append(y0)
p.append(p0)

for i in range(n-1):
	L1=h*f(x[i],y[i],p[i])
	M1=h*g(x[i],y[i],p[i])

	L2=h*f(x[i]+h/2,y[i]+L1/2,p[i]+M1/2)
	M2=h*g(x[i]+h/2,y[i]+M1/2,p[i]+M1/2)

	L3=h*f(x[i]+h/2,y[i]+L2/2,p[i]+M2/2)
	M3=h*g(x[i]+h/2,y[i]+M2/2,p[i]+M2/2)

	L4=h*f(x[i]+h,y[i]+L3,p[i]+M3)
	M4=h*g(x[i]+h,y[i]+M3,p[i]+M3)

	L=(L1+2*L2+2*L3+L4)/6
	M=(M1+2*M2+2*M3+M4)/6

	y.append(y[i] + L)
	p.append(p[i] + M)

y=sp.array(y)
p=sp.array(p)

print(y)
print(p)

pl.plot(x,y)
pl.show()
