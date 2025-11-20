#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# grav_acc  

def grav_acc(s_x, s_y,  Mp):
    """
    Calculate the gravitational acceleration at a point (x, y) due to a mass M located at the origin.

    Parameters:
    x (float): x-coordinate of the point in m
    y (float): y-coordinate of the point in m
    M (float): mass located at the origin kg

    Returns:
    tuple: gravitational acceleration components (ax, ay) units in m/s^2
    """
    G = 6.67e-11  # gravitational constant in m^3 kg^-1 s^-2
    r = (s_x**2 + s_y**2)**0.5
    a = G * Mp / r**2
    
    a_x = -a * (s_x / r)
    a_y = -a * (s_y / r)

    return a_x, a_y



#  checkinit
def checkinit(s_x0, s_y0, v_x0, v_y0, Rp):
    """
    Check that the initial spacecraft conditions are sensible.

    Conditions:
    1) Starting position must be above the planetary surface: |s| > Rp.
    2) Starting velocity must be in the +y direction: v_y0 > 0.

    Parameters
    ----------
    s_x0, s_y0 : float
        Initial spacecraft position components (m).
    v_x0, v_y0 : float
        Initial spacecraft velocity components (m/s).
    Rp : float
        Planet radius (m).

    Raises
    ------
    ValueError
        If either condition is not met.
    """
    s0 = np.sqrt(s_x0**2 + s_y0**2)

    if s0 <= Rp:
        raise ValueError("Initial position is below or at the planetary surface (|s0| <= Rp).")

    if v_y0 <= 0:
        raise ValueError("Initial velocity must be in the +y direction (v_y0 > 0).")



# sc_vel_pos_change

def sc_vel_pos_change(a_x, a_y, v_x, v_y, time_step):
    """
    Compute the change in spacecraft velocity and position over one time step
    using Equations (10)-(13) and (16)-(17) from the reading.

    Parameters
    ----------
    a_x, a_y : float
        Instantaneous acceleration components (m/s^2).
    v_x, v_y : float
        Current velocity components (m/s).
    time_step : float
        Time increment Δt (s).

    Returns
    -------
    ds_x, ds_y : float
        Change in position components (m).
    dv_x, dv_y : float
        Change in velocity components (m/s).
    """
    dt = time_step

    # change in velocity: Δv = a Δt
    dv_x = a_x * dt
    dv_y = a_y * dt

    # change in position: Δs = v Δt + 0.5 a (Δt)^2
    ds_x = v_x * dt + 0.5 * a_x * dt**2
    ds_y = v_y * dt + 0.5 * a_y * dt**2

    return ds_x, ds_y, dv_x, dv_y



# get_traj

def get_traj(s_x0, s_y0, v_x0, v_y0, time_step, total_time, Mp, Rp):
    """
    Compute a full spacecraft trajectory during a flyby.

    This implements the Part 1 pseudo-code:
    - checks initial conditions,
    - sets up arrays for time, acceleration, velocity, and position,
    - advances the solution using sc_vel_pos_change and grav_acc.

    Parameters
    ----------
    s_x0, s_y0 : float
        Initial spacecraft position components (m).
    v_x0, v_y0 : float
        Initial spacecraft velocity components (m/s).
    time_step : float
        Time increment Δt (s).
    total_time : float
        Total integration time (s).
    Mp : float
        Planet mass (kg).
    Rp : float
        Planet radius (m).

    Returns
    -------
    time : 1D ndarray, shape (nt+1,)
        Times from 0 to total_time.
    acc : 2D ndarray, shape (nt+1, 2)
        Acceleration components [a_x, a_y] at each time.
    vel : 2D ndarray, shape (nt+1, 2)
        Velocity components [v_x, v_y] at each time.
    pos : 2D ndarray, shape (nt+1, 2)
        Position components [s_x, s_y] at each time.
    """
    # --- check initial conditions ---
    checkinit(s_x0, s_y0, v_x0, v_y0, Rp)

    # number of time steps
    nt = int(total_time / time_step)

    # allocate arrays (nt+1 points: from t=0 to t=total_time)
    time = np.zeros(nt + 1)
    acc  = np.zeros((nt + 1, 2))
    vel  = np.zeros((nt + 1, 2))
    pos  = np.zeros((nt + 1, 2))

    # set initial conditions at index 0
    time[0] = 0.0
    pos[0, 0] = s_x0
    pos[0, 1] = s_y0
    vel[0, 0] = v_x0
    vel[0, 1] = v_y0

    # initial acceleration at t=0
    a_x0, a_y0 = grav_acc(s_x0, s_y0, Mp)
    acc[0, 0] = a_x0
    acc[0, 1] = a_y0

    # --- time-stepping loop ---
    for i in range(nt):
        # current state
        s_x_i, s_y_i = pos[i, :]
        v_x_i, v_y_i = vel[i, :]
        a_x_i, a_y_i = acc[i, :]

        # changes over one time step
        ds_x, ds_y, dv_x, dv_y = sc_vel_pos_change(a_x_i, a_y_i, v_x_i, v_y_i, time_step)

        # update to new state (i+1)
        s_x_new = s_x_i + ds_x
        s_y_new = s_y_i + ds_y
        v_x_new = v_x_i + dv_x
        v_y_new = v_y_i + dv_y
        t_new   = time[i] + time_step

        # store new time, position, velocity
        time[i + 1]   = t_new
        pos[i + 1, :] = [s_x_new, s_y_new]
        vel[i + 1, :] = [v_x_new, v_y_new]

        # compute new acceleration at new position
        a_x_new, a_y_new = grav_acc(s_x_new, s_y_new, Mp)
        acc[i + 1, :] = [a_x_new, a_y_new]

    return time, acc, vel, pos


# In[ ]:




