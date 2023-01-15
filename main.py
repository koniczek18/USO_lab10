import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy
import scipy.interpolate

l = 1
u = 0
m = 9
J = 1
d = 0.5
g = 9.81

showU=True

def model(x, t, u, l, m, J, d, g):
    x1 = x[0]
    x2 = x[1]
    x1p = x2
    x2p = u / J - d * x2 / J - m * g * l * np.sin(x1)  #a tutaj bez /J ?
    return np.array([x1p, x2p])

t=np.linspace(0,10,1001)
sim1=odeint(model,t=t,y0=[np.pi/4,0],args=(u, l, m, J, d, g))
plt.figure('Base model')
plt.plot(t,sim1[:,0],label='x1')
plt.legend()

def A_model(x):
    x1 = x[0,0]
    return np.array([[0,1],[m*g*l*np.sin(x1)/x1,-d/J]])

def B_model(x):
    return np.array([[0],[1/J]])

def model_Inf1(x, t, l, m, J, d, g):
    x1 = x[0]
    x2 = x[1]
    X=np.array([[x1],[x2]])
    A=A_model(X)
    B=B_model(X)
    Q = np.array([[1, 0], [0, 1]])
    R=1
    P=scipy.linalg.solve_continuous_are(A,B,Q,R)
    K=R*B.T@P
    u=-K@X
    XP=A@X+B*u[0,0]
    return np.array([XP[0,0],XP[1,0],u[0,0]])

sim2=odeint(model_Inf1, t=t, y0=[2 * np.pi, 0, 0], args=(l, m, J, d, g))
plt.figure('Inf model - static Q')
plt.plot(t,sim2[:,0],label='x1')
if showU:
    plt.plot(t,sim2[:,2],label='u')
plt.legend()

def model_Inf2(x, t, l, m, J, d, g):
    x1 = x[0]
    x2 = x[1]
    X=np.array([[x1],[x2]])
    A=A_model(X)
    B=B_model(X)
    Q = np.array([[x1*x1, 0], [0, x2*x2]])
    R=1
    P=scipy.linalg.solve_continuous_are(A,B,Q,R)
    K=R*B.T@P
    u=-K@X
    XP=A@X+B*u[0,0]
    return np.array([XP[0,0],XP[1,0],u[0,0]])

sim3=odeint(model_Inf2, t=t, y0=[2 * np.pi, 0, 0], args=(l, m, J, d, g))
plt.figure('Inf model - dyn Q')
plt.plot(t,sim3[:,0],label='x1')
if showU:
    plt.plot(t,sim3[:,2],label='u')
plt.legend()

def riccati(P,A,B,R,q):
    Pp=(P@A-P@B*R@B.T@P+A.T@P+q)
    return Pp

def model_Fin1(x, t, l, m, J, d, g):
    x1 = x[0]
    x2 = x[1]
    X=np.array([[x1],[x2]])
    P = np.array([[x[2], x[3]], [x[4], x[5]]])
    A=A_model(X)
    B=B_model(X)
    Q = np.array([[1, 0], [0, 1]])
    R=1
    PP=riccati(P,A,B,R,Q)
    K=R*B.T@P
    u=-K@X
    XP=A@X+B*u[0,0]
    return np.array([XP[0,0],XP[1,0],PP[0,0],PP[0,1],PP[1,0],PP[1,1],u[0,0]])

sim4=odeint(model_Fin1, t=t, y0=[2 * np.pi, 0, 0,0,0,0,0], args=(l, m, J, d, g))
plt.figure('Fin model - static Q')
plt.plot(t,sim4[:,0],label='x1')
if showU:
    plt.plot(t,sim4[:,6],label='u')
plt.legend()

def model_Fin2(x, t, l, m, J, d, g):
    x1 = x[0]
    x2 = x[1]
    X=np.array([[x1],[x2]])
    P = np.array([[x[2], x[3]], [x[4], x[5]]])
    A=A_model(X)
    B=B_model(X)
    Q = np.array([[x1*x1, 0], [0, x2*x2]])
    R=1
    PP=riccati(P,A,B,R,Q)
    K=R*B.T@P
    u=-K@X
    XP=A@X+B*u[0,0]
    return np.array([XP[0,0],XP[1,0],PP[0,0],PP[0,1],PP[1,0],PP[1,1],u[0,0]])

sim5=odeint(model_Fin2, t=t, y0=[2 * np.pi, 0, 0,0,0,0,0], args=(l, m, J, d, g))
plt.figure('Fin model - dyn Q')
plt.plot(t,sim5[:,0],label='x1')
if showU:
    plt.plot(t,sim5[:,6],label='u')
plt.legend()

#
plt.show()
#