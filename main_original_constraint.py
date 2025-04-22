import opengen as og
import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.ticker as ticker

import tikzplotlib

# A quick fix
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
Line2D._us_dashSeq    = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
Legend._ncol = property(lambda self: self._ncols)

plt.rcParams['font.family'] = 'Times New Roman'

sampler_period = 0.05
sampling_time_sim = 0.001
simulation_steps = int(20 / sampling_time_sim)
T = 20
sampler_interval = int(sampler_period / sampling_time_sim)

tcp_server_name1 = "OpEn/python_cartpole_original_constraint" + \
                "/cartpole_original_constraint"
mng1 = og.tcp.OptimizerTcpManager(tcp_server_name1.replace(".", "_"), port=8333)
tcp_server_name2 = "OpEn/python_cartpole_original_lifted_constraint" + \
                "/cartpole_original_lifted_constraint"
mng2 = og.tcp.OptimizerTcpManager(tcp_server_name2.replace(".", "_"), port=8334)

mng1.start()
mng2.start()

x_state_0 = [0,np.pi,0,0] # initial condition

Q = [2.5, 10, 0.01, 0.01]
Qt = [3.0, 10, 0.02, 0.02]
R = [0.1]

# model parameters
gravity_acceleration = 9.8
length = 1.0
mass_cart = 1.0
mass_pole = 0.2
dim = 4

def dynamics_ct(x, u):
    dx1 = x[2]
    dx2 = x[3]
    dx3 = (-mass_pole*length*cs.sin(x[1])*x[3]**2 \
           + mass_pole*gravity_acceleration*cs.sin(x[1])*cs.cos(x[1])
           + u) / (mass_cart+mass_pole*cs.sin(x[1])**2)
    dx4 = (-mass_pole*length*cs.sin(x[1])*cs.cos(x[1])*x[3]**2 \
           + (mass_cart+mass_pole)*gravity_acceleration*cs.sin(x[1]) \
           + u*cs.cos(x[1])) / (length*(mass_cart+mass_pole*cs.sin(x[1])**2))
    return [dx1, dx2, dx3, dx4]

def dynamics_dt(x, u, sampling_time_sim):
    k1 = dynamics_ct(x, u)

    x_k2 = [x[i] + 0.5 * sampling_time_sim * k1[i] for i in range(dim)]
    k2 = dynamics_ct(x_k2, u)

    x_k3 = [x[i] + 0.5 * sampling_time_sim * k2[i] for i in range(dim)]
    k3 = dynamics_ct(x_k3, u)

    x_k4 = [x[i] + sampling_time_sim * k3[i] for i in range(dim)]
    k4 = dynamics_ct(x_k4, u)

    x_next = [x[i] + (sampling_time_sim / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(dim)]
    return x_next

bounds = [-2.5, 5.0, -7.5, 7.5] # [x_min, x_max, x_dot_min, x_dot_max]

x1 = x_state_0
state_sequence_1 = []
solve_time_1 = []
us = [0]*T  # Initial control input
u_sequence_1 = []
solve_time_one_step = 0.0
for k in range(simulation_steps):
    if k % sampler_interval == 0:
        solver_status = mng1.call(x1+Q+Qt+R+bounds, initial_guess=us)
        us = solver_status['solution']
        u1 = us[0]
        solve_time_one_step = solver_status['solve_time_ms']

    x1_next = dynamics_dt(x1, u1, sampling_time_sim=sampling_time_sim)

    state_sequence_1.append(x1_next)
    solve_time_1.append(solve_time_one_step)
    u_sequence_1.append(u1)

    x1 = x1_next

x2 = x_state_0
state_sequence_2 = []
solve_time_2 = []
us = [0]*T  # Initial control input
u_sequence_2 = []
solve_time_one_step = 0.0
for k in range(simulation_steps):
    if k % sampler_interval == 0:
        solver_status = mng2.call(x2+Q+Qt+R+bounds, initial_guess=us)
        us = solver_status['solution']
        u2 = us[0]
        solve_time_one_step = solver_status['solve_time_ms']

    x2_next = dynamics_dt(x2, u2, sampling_time_sim=sampling_time_sim)

    state_sequence_2.append(x2_next)
    solve_time_2.append(solve_time_one_step)
    u_sequence_2.append(u2)

    x2 = x2_next

# Stop the TCP servers
mng1.kill()
mng2.kill()

# Convert state_sequence into a flattened list for plotting
state_sequence_1_flat = [item for sublist in state_sequence_1 for item in sublist]
state_sequence_2_flat = [item for sublist in state_sequence_2 for item in sublist]

# Generate time vector for plotting
time = np.arange(0, sampling_time_sim*simulation_steps, sampling_time_sim)

right_limit = time[-1]

# plot x1
plt.plot(time, state_sequence_1_flat[0::4], 'b-', label="NMPC")
plt.plot(time, state_sequence_2_flat[0::4], 'r--', label="Lifted NMPC")
plt.ylabel(r'$x_1$ (m)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
plt.xlim(left=0, right=right_limit)
plt.axhline(bounds[0], color='k', linestyle='-.', label="Bounds")  # Lower bound for x1
plt.axhline(bounds[1], color='k', linestyle='-.')  # Upper bound for x1
#tikzplotlib.save("icp1.tex")

plt.figure()
# plot x1_dot
plt.plot(time, state_sequence_1_flat[2::4], 'b-', label="NMPC")
plt.plot(time, state_sequence_2_flat[2::4], 'r--', label="Lifted NMPC")
plt.ylabel(r'$x_3$ (m/s)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
plt.xlim(left=0, right=right_limit)
plt.axhline(bounds[2], color='k', linestyle='-.', label="Bounds")  # Lower bound for x1_dot
plt.axhline(bounds[3], color='k', linestyle='-.')  # Upper bound for x1_dot
#tikzplotlib.save("icp2.tex")

# Plot Control Input (u)
plt.figure()
plt.plot(time, u_sequence_1, 'b-', label="NMPC")
plt.plot(time, u_sequence_2, 'r--', label="Lifted NMPC")
plt.ylabel(r'Control Input: $u$')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
plt.xlim(left=0, right=right_limit)  # Set both left and right limits for the x-axis
plt.tight_layout() 
#plt.title("Sampling Period: " + str(sampler_period) + "s" + " Control Period: " + str(sampling_time_control) + "s")
#tikzplotlib.save("icp4.tex")

state_norms_1 = [
    np.sqrt(
        state_sequence_1_flat[i]**2 +
        state_sequence_1_flat[i + 1]**2 +
        state_sequence_1_flat[i + 2]**2 +
        state_sequence_1_flat[i + 3]**2
    )
    for i in range(0, len(state_sequence_1_flat), 4)
]
state_norms_2 = [
    np.sqrt(
        state_sequence_2_flat[i]**2 +
        state_sequence_2_flat[i + 1]**2 +
        state_sequence_2_flat[i + 2]**2 +
        state_sequence_2_flat[i + 3]**2
    )
    for i in range(0, len(state_sequence_2_flat), 4)
]

# State NORM
plt.figure()
plt.plot(time, state_norms_1, 'b-', label="NMPC")
plt.plot(time, state_norms_2, 'r--', label="Lifted NMPC")
plt.ylabel(r'$||x||_{2}$') 
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
plt.xlim(left=0, right=right_limit)  # Set both left and right limits for the x-axis
#plt.title('Samping time = ' + str(sampler_period) + 's')
plt.tight_layout()
#tikzplotlib.save("icp3.tex")

plt.figure()
plt.plot(time, solve_time_1, 'b-', label="NMPC")
plt.plot(time, solve_time_2, 'r--', label="Lifted NMPC")
plt.ylabel('Solve Time (ms)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
plt.xlim(left=0, right=right_limit)  # Set both left and right limits for the x-axis
plt.tight_layout()
#tikzplotlib.save("icp_solve_time.tex")

plt.show()