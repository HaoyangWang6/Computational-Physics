import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def pendulum(y, t, k, nu, Q, Omega):
    x, v = y
    dxdt = v
    dvdt = -k*np.sin(x) - nu*v + Q*np.sin(Omega*t)
    return [dxdt, dvdt]

# 参数设置
k, nu, Q, Omega = 1.0, 0.1, 0, 0  # k:固有频率；nu:阻尼系数；Q:驱动力强度；Omega:驱动力频率
y0 = [1, 0]  # 初始条件 [x, v]
t = np.linspace(0, 100, 10000)

# 数值求解
sol = odeint(pendulum, y0, t, args=(k, nu, Q, Omega))

# 绘制相空间图
plt.plot(sol[:, 0], sol[:, 1], lw=0.5)
plt.xlabel('Angle (x)'); plt.ylabel('Velocity (v)')
plt.title('Phase Space of Damped Driven Pendulum')
plt.show()