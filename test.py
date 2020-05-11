import numpy as np
from numpy import sin, cos
from numpy.linalg import inv
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML

gg = 9.8  # gravitational acceleration (m/s^2)
L1 = 1.0  # length of pendulum-1 (meter)
L2 = 1.0  # length of pendulum-2 (meter)
L3 = 1.0  # length of pendulum-3 (meter)
m1 = 1.0  # mass of pendulum-1 (kg)
m2 = 1.0  # mass of pendulum-2 (kg)
m3 = 1.0  # mass of pendulum-3 (kg)

th1 = 200.0  # initial angle of pendulum-1 (degrees)
w1 = 0.0  # initial angular velocity of pendulum-1 (degrees/second)
th2 = -10.0  # initial angle of pendulum-2 (degrees)
w2 = 0.0  # initial angular velocity of pendulum-2 (degrees/second)
th3 = 60.0  # initial angle of pendulum-3 (degrees)
w3 = 0.0  # initial angular velocity of pendulum-3 (degrees/second)


def func(intt, t):
    dydx = np.zeros_like(intt)
    dydx[0] = intt[1]

    dif12 = intt[0] - intt[2]
    dif13 = intt[0] - intt[4]
    dif23 = intt[2] - intt[4]

    a1 = (m1 + m2 + m3) * L1 * L1
    a2 = (m2 + m3) * L1 * L2 * cos(dif12)
    a3 = m3 * L1 * L3 * cos(dif13)

    b1 = a2
    b2 = (m2 + m3) * L2 * L2
    b3 = m3 * L2 * L3 * cos(dif23)

    c1 = m3 * L1 * L3 * cos(dif13)
    c2 = b3
    c3 = m3 * L3 * L3

    d1 = -(m1 + m2 + m3) * gg * L1 * sin(intt[0]) - (m2 + m3) * L1 * L2 * sin(dif12) * intt[3] * intt[
        3] - m3 * L1 * L3 * sin(dif13) * intt[5] * intt[5]
    d2 = -(m2 + m3) * gg * L2 * sin(intt[2]) + (m2 + m3) * L1 * L2 * sin(dif12) * intt[1] * intt[
        1] - m3 * L2 * L3 * sin(dif23) * intt[5] * intt[5]
    d3 = -m3 * gg * L3 * sin(intt[4]) + m3 * L1 * L3 * sin(dif13) * intt[1] * intt[1] + m3 * L2 * L3 * sin(dif23) * \
         intt[3] * intt[3]

    aaa = np.matrix([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])
    bbb = np.matrix([[d1], [d2], [d3]])

    xxx = np.matmul(inv(aaa), bbb)

    dydx[1] = xxx[0]

    dydx[2] = intt[3]

    dydx[3] = xxx[1]

    dydx[4] = intt[5]

    dydx[5] = xxx[2]

    return dydx


dt = 0.05  # time between every frame, every position
t = np.arange(0.0, 60., dt)  # middle number in "()" is the time interval

intt = np.radians([th1, w1, th2, w2, th3, w3])  # list of initial values in radian

y = integrate.odeint(func, intt, t)  # integrate the function and make an array of every position data

x1 = L1 * sin(y[:, 0])  # List of position and velocity of mass-1 in x coordinate
y1 = -L1 * cos(y[:, 0])  # List of position and velocity of mass-1 in y coordinate

x2 = L2 * sin(y[:, 2]) + x1  # List of position and velocity of mass-2 in x coordinate
y2 = -L2 * cos(y[:, 2]) + y1  # List of position and velocity of mass-2 in y coordinate

x3 = L3 * sin(y[:, 4]) + x2  # List of position and velocity of mass-3 in x coordinate
y3 = -L3 * cos(y[:, 4]) + y2  # List of position and velocity of mass-3 in y coordinate

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.2 * (L1 + L2 + L3), 1.2 * (L1 + L2 + L3)),
                     ylim=(-1.2 * (L1 + L2 + L3), 1.2 * (L1 + L2 + L3)))
# ax.grid(linewidth=1)
ax.set_aspect('equal', 'box')
for i in range(1, int(L1 + L2 + L3) + 1):
    ax.axhline(i, linestyle='--', color='k', alpha=0.7)
    ax.axvline(i, linestyle='--', color='k', alpha=0.7)
    ax.axhline(-i, linestyle='--', color='k', alpha=0.7)
    ax.axvline(-i, linestyle='--', color='k', alpha=0.7)

ax.axhline(0, linestyle='--', color='k')
ax.axvline(0, linestyle='--', color='k')

line, = ax.plot([], [], 'ro-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.1, 1.05, '', transform=ax.transAxes)
theta_temp1 = 'Theta1 = %.1f'
theta_text1 = ax.text(1.05, 0.9, '', transform=ax.transAxes)
theta_temp2 = 'Theta2 = %.1f'
theta_text2 = ax.text(1.05, 0.75, '', transform=ax.transAxes)
theta_temp3 = 'Theta3 = %.1f'
theta_text3 = ax.text(1.05, 0.6, '', transform=ax.transAxes)
omega_temp1 = 'omega1 = %.1f'
omega_text1 = ax.text(1.05, 0.85, '', transform=ax.transAxes)
omega_temp2 = 'omega2 = %.1f'
omega_text2 = ax.text(1.05, 0.7, '', transform=ax.transAxes)
omega_temp3 = 'omega3 = %.1f'
omega_text3 = ax.text(1.05, 0.55, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    theta_text1.set_text('')
    theta_text2.set_text('')
    theta_text3.set_text('')
    omega_text2.set_text('')
    omega_text2.set_text('')
    omega_text3.set_text('')
    return line, time_text, theta_text1, theta_text2, theta_text3, omega_text1, omega_text2, omega_text3


def animate(i):
    thisx = [0, x1[i], x2[i], x3[i]]
    thisy = [0, y1[i], y2[i], y3[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i * dt))
    theta_text1.set_text(theta_temp1 % ((y[i][0] * (180 / np.pi)) % 360))
    theta_text2.set_text(theta_temp2 % ((y[i][2] * (180 / np.pi)) % 360))
    theta_text3.set_text(theta_temp3 % ((y[i][4] * (180 / np.pi)) % 360))
    omega_text1.set_text(omega_temp1 % ((y[i][1] * (180 / np.pi))))
    omega_text2.set_text(omega_temp2 % ((y[i][3] * (180 / np.pi))))
    omega_text3.set_text(omega_temp3 % ((y[i][5] * (180 / np.pi))))
    return line, time_text, theta_text1, theta_text2, theta_text3, omega_text1, omega_text2, omega_text3


anim = animation.FuncAnimation(fig, animate, np.arange(1, len(y)), interval=25, blit=True, init_func=init)

rc('animation', html='jshtml')
plt.show()
anim