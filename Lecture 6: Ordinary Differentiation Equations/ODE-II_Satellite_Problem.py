import math
import numpy as np

# Constants (equivalent to Fortran module systemparam)
pi = 3.141592653589793
gm = 3.987e14  # Earth's mass times constant of gravity
arocket = 5.0  # deceleration due to rocket motor
dragc = 8e-4   # Air drag coefficient
re = 6.378e6   # Earth's radius

# Global variables
dt = 0.0       # time step
dt2 = 0.0      # half of the time step
tbrake = 0.0   # run-time of rocket motor

def airdens(r):
    """Air density as a function of altitude"""
    k1 = 1.2e4
    k2 = 2.2e4
    
    if r > re:
        return 1.225 * math.exp(-((r - re)/k1 + ((r - re)/k2)**1.5))
    else:
        return 0.0

def accel(x, y, vx, vy, t):
    """Calculates acceleration components due to gravity, air drag and rocket thrust"""
    r = math.sqrt(x**2 + y**2)
    v2 = vx**2 + vy**2
    v1 = math.sqrt(v2)
    
    # Acceleration due to gravitation
    r3 = 1.0 / r**3
    ax = -gm * x * r3
    ay = -gm * y * r3
    
    # Acceleration due to air drag
    if v1 > 1e-12:
        ad = dragc * airdens(r) * v2
        ax -= ad * vx / v1
        ay -= ad * vy / v1
    
    # Acceleration due to rocket motor thrust
    if t < tbrake and v1 > 1e-12:
        ax -= arocket * vx / v1
        ay -= arocket * vy / v1
    
    return ax, ay

def polar_position(x, y):
    """Converts Cartesian coordinates to polar coordinates"""
    r = math.sqrt(x**2 + y**2)
    if y >= 0.0:
        a = math.acos(x / r) / (2 * pi)
    else:
        a = 1.0 - math.acos(x / r) / (2 * pi)
    return r, a

def rk_step(t0, x0, y0, vx0, vy0):
    """Runge-Kutta integration step for the equations of motion"""
    global dt, dt2
    
    t1 = t0 + dt
    th = t0 + dt2
    
    # First RK step
    ax, ay = accel(x0, y0, vx0, vy0, t0)
    kx1 = dt2 * ax
    ky1 = dt2 * ay
    lx1 = dt2 * vx0
    ly1 = dt2 * vy0
    
    # Second RK step
    ax, ay = accel(x0 + lx1, y0 + ly1, vx0 + kx1, vy0 + ky1, th)
    kx2 = dt2 * ax
    ky2 = dt2 * ay
    lx2 = dt2 * (vx0 + kx1)
    ly2 = dt2 * (vy0 + ky1)
    
    # Third RK step
    ax, ay = accel(x0 + lx2, y0 + ly2, vx0 + kx2, vy0 + ky2, th)
    kx3 = dt * ax
    ky3 = dt * ay
    lx3 = dt * (vx0 + kx2)
    ly3 = dt * (vy0 + ky2)
    
    # Fourth RK step
    ax, ay = accel(x0 + lx3, y0 + ly3, vx0 + kx3, vy0 + ky3, t1)
    kx4 = dt2 * ax
    ky4 = dt2 * ay
    lx4 = dt2 * (vx0 + kx3)
    ly4 = dt2 * (vy0 + ky3)
    
    # Combine results
    x1 = x0 + (lx1 + 2.0 * lx2 + lx3 + lx4) / 3.0
    y1 = y0 + (ly1 + 2.0 * ly2 + ly3 + ly4) / 3.0
    vx1 = vx0 + (kx1 + 2.0 * kx2 + kx3 + kx4) / 3.0
    vy1 = vy0 + (ky1 + 2.0 * ky2 + ky3 + ky4) / 3.0
    
    return x1, y1, vx1, vy1

def satellite_crash():
    """Main simulation function"""
    global dt, dt2, tbrake
    
    # Get user input
    r0 = float(input("Initial altitude of satellite (km): ")) * 1e3 + re
    tbrake = float(input("Rocket motor run-time (seconds): "))
    dt = float(input("Time step delta-t for RK integration (seconds): "))
    dt2 = dt / 2.0
    wstp = int(input("Writing results every Nth step; give N: "))
    tmax = float(input("Maximum integration time (hours): ")) * 3600.0
    
    # Initial conditions
    x = r0
    y = 0.0
    vx = 0.0
    vy = math.sqrt(gm / r0)
    nstp = int(tmax / dt)
    
    # Open output file
    with open('sat.dat', 'w') as f:
        for i in range(nstp + 1):
            r, a = polar_position(x, y)
            if r > re:
                t = float(i) * dt
                if i % wstp == 0:
                    speed = math.sqrt(vx**2 + vy**2)
                    f.write(f"{t:12.3f}  {a:12.8f}  {(r - re)/1e3:14.6f}  {speed:12.4f}\n")
                
                # Perform RK integration step
                x, y, vx, vy = rk_step(t, x, y, vx, vy)
            else:
                print("The satellite has successfully crashed!")
                return
    
    print("The satellite did not crash within the specified time.")

if __name__ == "__main__":
    satellite_crash()