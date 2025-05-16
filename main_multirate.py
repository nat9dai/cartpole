import opengen as og
import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.ticker as ticker

# import tikzplotlib

# A quick fix
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
Line2D._us_dashSeq    = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
Legend._ncol = property(lambda self: self._ncols)

sampler_period = 0.1
sampling_time_sim = 0.001
simulation_steps = int(10 / sampling_time_sim)
sampling_time_control = 0.05
N_P = int(sampler_period/sampling_time_control)
horizon_time = 1.0
T = int(horizon_time / sampler_period)
control_interval = int(sampling_time_control / sampling_time_sim) # 50
sampler_interval = int(sampler_period / sampling_time_sim) # 100 250 500\

tcp_server_name1 = "OpEn/multirate/python_cartpole_" + str(sampler_period) + \
                "/cartpole_" + str(sampler_period)
mng1 = og.tcp.OptimizerTcpManager(tcp_server_name1.replace(".", "_"), port=8331)
tcp_server_name2 = "OpEn/multirate/python_cartpole_lifting_" + str(sampler_period) + \
                "/cartpole_lifting_" + str(sampler_period)
mng2 = og.tcp.OptimizerTcpManager(tcp_server_name2.replace(".", "_"), port=8332)
tcp_server_name3 = "OpEn/multirate/python_cartpole_lifting_multirate_" + str(sampler_period) + \
                "/cartpole_lifting_multirate_" + str(sampler_period)
mng3 = og.tcp.OptimizerTcpManager(tcp_server_name3.replace(".", "_"), port=8333)

mng1.start()
mng2.start()
mng3.start()

bounds = [-15, 15, -15, 15] # [x_min, x_max, x_dot_min, x_dot_max]

x_state_0 = [0,np.pi,0,0]
Q = [2.5, 10, 0.01, 0.01]
Qt = [3.0, 30, 0.1, 0.02]
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

x1 = x_state_0
state_sequence_1 = []
us = [0]*T  # Initial control input
u_sequence_1 = []
solve_time_1 = []
solve_time_one_step = 0.0
for k in range(simulation_steps):
    if k % sampler_interval == 0:
        solver_status = mng1.call(x1+Q+Qt+R+bounds, initial_guess=us)
        us = solver_status['solution']
        u1 = us[0]
        solve_time_one_step = solver_status['solve_time_ms']

    x1_next = dynamics_dt(x1, u1, sampling_time_sim=sampling_time_sim)

    state_sequence_1.append(x1_next)
    u_sequence_1.append(u1)
    solve_time_1.append(solve_time_one_step)

    x1 = x1_next

x2 = x_state_0
state_sequence_2 = []
us = [0]*T  # Initial control input
u_sequence_2 = []
solve_time_2 = []
solve_time_one_step = 0.0
for k in range(simulation_steps):
    if k % sampler_interval == 0:
        solver_status = mng2.call(x2+Q+Qt+R+bounds, initial_guess=us)
        us = solver_status['solution']
        u2 = us[0]
        solve_time_one_step = solver_status['solve_time_ms']

    x2_next = dynamics_dt(x2, u2, sampling_time_sim=sampling_time_sim)

    state_sequence_2.append(x2_next)
    u_sequence_2.append(u2)
    solve_time_2.append(solve_time_one_step)

    x2 = x2_next

def shift_left(L, N):
    shifted_list = L[N:]
    shifted_list.extend([shifted_list[-1]] * (N))
    return shifted_list

x3 = x_state_0
state_sequence_3 = []
us = [0]*N_P*T  # Initial control input
u3 = 0.0
u_sequence_3 = []
solve_time_3 = []
solve_time_one_step = 0.0

count = 0
control_sequence = [0]*N_P
cost_list = []
for k in range(simulation_steps):
    if k % sampler_interval == 0:
        solver_status = mng3.call(x3+Q+Qt+R+bounds, initial_guess=us)
        us = solver_status['solution']
        solve_time_one_step = solver_status['solve_time_ms']

        control_sequence = us[0:N_P]
        count = 0
    if k % control_interval == 0:
        u3 = control_sequence[count]
        count += 1
         
    x3_next = dynamics_dt(x3, u3, sampling_time_sim=sampling_time_sim)

    state_sequence_3.append(x3_next)
    u_sequence_3.append(u3)
    solve_time_3.append(solve_time_one_step)

    x3 = x3_next

# Stop the TCP servers
mng1.kill()
mng2.kill()
mng3.kill()

################### Change this parameter ###################
sampler_period = 0.25 # 0.1, 0.25, 0.5
#############################################################
sampling_time_sim = 0.001
simulation_steps = int(10 / sampling_time_sim)
sampling_time_control = 0.05
N_P = int(sampler_period/sampling_time_control)
horizon_time = 1.0
T = int(horizon_time / sampler_period)
control_interval = int(sampling_time_control / sampling_time_sim) # 50
sampler_interval = int(sampler_period / sampling_time_sim) # 100 250 500\

tcp_server_name1 = "OpEn/multirate/python_cartpole_" + str(sampler_period) + \
                "/cartpole_" + str(sampler_period)
mng4 = og.tcp.OptimizerTcpManager(tcp_server_name1.replace(".", "_"), port=8334)
tcp_server_name2 = "OpEn/multirate/python_cartpole_lifting_" + str(sampler_period) + \
                "/cartpole_lifting_" + str(sampler_period)
mng5 = og.tcp.OptimizerTcpManager(tcp_server_name2.replace(".", "_"), port=8335)
tcp_server_name3 = "OpEn/multirate/python_cartpole_lifting_multirate_" + str(sampler_period) + \
                "/cartpole_lifting_multirate_" + str(sampler_period)
mng6 = og.tcp.OptimizerTcpManager(tcp_server_name3.replace(".", "_"), port=8336)

mng4.start()
mng5.start()
mng6.start()

x1 = x_state_0
state_sequence_4 = []
us = [0]*T  # Initial control input
u_sequence_4 = []
solve_time_4 = []
solve_time_one_step = 0.0
for k in range(simulation_steps):
    if k % sampler_interval == 0:
        solver_status = mng4.call(x1+Q+Qt+R+bounds, initial_guess=us)
        #solver_status = mng4.call(x1+Q+Qt+R)
        us = solver_status['solution']
        u1 = us[0]
        solve_time_one_step = solver_status['solve_time_ms']

    x1_next = dynamics_dt(x1, u1, sampling_time_sim=sampling_time_sim)

    state_sequence_4.append(x1_next)
    u_sequence_4.append(u1)
    solve_time_4.append(solve_time_one_step)

    x1 = x1_next

x2 = x_state_0
state_sequence_5 = []
us = [0]*T  # Initial control input
u_sequence_5 = []
solve_time_5 = []
solve_time_one_step = 0.0
for k in range(simulation_steps):
    if k % sampler_interval == 0:
        solver_status = mng5.call(x2+Q+Qt+R+bounds, initial_guess=us)
        #solver_status = mng5.call(x2+Q+Qt+R)
        us = solver_status['solution']
        u2 = us[0]
        solve_time_one_step = solver_status['solve_time_ms']

    x2_next = dynamics_dt(x2, u2, sampling_time_sim=sampling_time_sim)

    state_sequence_5.append(x2_next)
    u_sequence_5.append(u2)
    solve_time_5.append(solve_time_one_step)

    x2 = x2_next

x3 = x_state_0
state_sequence_6 = []
us = [0]*N_P*T  # Initial control input
u3 = 0.0
u_sequence_6 = []
solve_time_6 = []
solve_time_one_step = 0.0

count = 0
control_sequence = [0]*N_P
cost_list = []
for k in range(simulation_steps):
    if k % sampler_interval == 0:
        solver_status = mng6.call(x3+Q+Qt+R+bounds, initial_guess=us)
        #solver_status = mng6.call(x3+Q+Qt+R)
        us = solver_status['solution']
        solve_time_one_step = solver_status['solve_time_ms']

        control_sequence = us[0:N_P]
        count = 0
    if k % control_interval == 0:
        u3 = control_sequence[count]
        count += 1
         
    x3_next = dynamics_dt(x3, u3, sampling_time_sim=sampling_time_sim)

    state_sequence_6.append(x3_next)
    u_sequence_6.append(u3)
    solve_time_6.append(solve_time_one_step)

    x3 = x3_next

# Stop the TCP servers
mng4.kill()
mng5.kill()
mng6.kill()

################### Change this parameter ###################
sampler_period = 0.5 # 0.1, 0.25, 0.5
#############################################################
sampling_time_sim = 0.001
simulation_steps = int(10 / sampling_time_sim)
sampling_time_control = 0.05
N_P = int(sampler_period/sampling_time_control)
horizon_time = 1.0
T = int(horizon_time / sampler_period)
control_interval = int(sampling_time_control / sampling_time_sim) # 50
sampler_interval = int(sampler_period / sampling_time_sim) # 100 250 500\

tcp_server_name1 = "OpEn/multirate/python_cartpole_" + str(sampler_period) + \
                "/cartpole_" + str(sampler_period)
mng7 = og.tcp.OptimizerTcpManager(tcp_server_name1.replace(".", "_"), port=8337)
tcp_server_name2 = "OpEn/multirate/python_cartpole_lifting_" + str(sampler_period) + \
                "/cartpole_lifting_" + str(sampler_period)
mng8 = og.tcp.OptimizerTcpManager(tcp_server_name2.replace(".", "_"), port=8338)
tcp_server_name3 = "OpEn/multirate/python_cartpole_lifting_multirate_" + str(sampler_period) + \
                "/cartpole_lifting_multirate_" + str(sampler_period)
mng9 = og.tcp.OptimizerTcpManager(tcp_server_name3.replace(".", "_"), port=8339)

mng7.start()
mng8.start()
mng9.start()

x1 = x_state_0
state_sequence_7 = []
us = [0]*T  # Initial control input
u_sequence_7 = []
solve_time_7 = []
solve_time_one_step = 0.0
for k in range(simulation_steps):
    if k % sampler_interval == 0:
        solver_status = mng7.call(x1+Q+Qt+R+bounds, initial_guess=us)
        #solver_status = mng7.call(x1+Q+Qt+R)
        us = solver_status['solution']
        u1 = us[0]
        solve_time_one_step = solver_status['solve_time_ms']

    x1_next = dynamics_dt(x1, u1, sampling_time_sim=sampling_time_sim)

    state_sequence_7.append(x1_next)
    u_sequence_7.append(u1)
    solve_time_7.append(solve_time_one_step)

    x1 = x1_next

x2 = x_state_0
state_sequence_8 = []
us = [0]*T  # Initial control input
u_sequence_8 = []
solve_time_8 = []
solve_time_one_step = 0.0
for k in range(simulation_steps):
    if k % sampler_interval == 0:
        solver_status = mng8.call(x2+Q+Qt+R+bounds, initial_guess=us)
        #solver_status = mng8.call(x2+Q+Qt+R)
        us = solver_status['solution']
        u2 = us[0]
        solve_time_one_step = solver_status['solve_time_ms']

    x2_next = dynamics_dt(x2, u2, sampling_time_sim=sampling_time_sim)

    state_sequence_8.append(x2_next)
    u_sequence_8.append(u2)
    solve_time_8.append(solve_time_one_step)

    x2 = x2_next

x3 = x_state_0
state_sequence_9 = []
us = [0]*N_P*T  # Initial control input
u3 = 0.0
u_sequence_9 = []
solve_time_9 = []
solve_time_one_step = 0.0

count = 0
control_sequence = [0]*N_P
cost_list = []
for k in range(simulation_steps):
    if k % sampler_interval == 0:
        solver_status = mng9.call(x3+Q+Qt+R+bounds, initial_guess=us)
        #solver_status = mng9.call(x3+Q+Qt+R)
        us = solver_status['solution']
        solve_time_one_step = solver_status['solve_time_ms']

        control_sequence = us[0:N_P]
        count = 0
    if k % control_interval == 0:
        u3 = control_sequence[count]
        count += 1
         
    x3_next = dynamics_dt(x3, u3, sampling_time_sim=sampling_time_sim)

    state_sequence_9.append(x3_next)
    u_sequence_9.append(u3)
    solve_time_9.append(solve_time_one_step)

    x3 = x3_next

# Stop the TCP servers
mng7.kill()
mng8.kill()
mng9.kill()

# Convert state_sequence into a flattened list for plotting
state_sequence_1_flat = [item for sublist in state_sequence_1 for item in sublist]
state_sequence_2_flat = [item for sublist in state_sequence_2 for item in sublist]
state_sequence_3_flat = [item for sublist in state_sequence_3 for item in sublist]
state_sequence_4_flat = [item for sublist in state_sequence_4 for item in sublist]
state_sequence_5_flat = [item for sublist in state_sequence_5 for item in sublist]
state_sequence_6_flat = [item for sublist in state_sequence_6 for item in sublist]
state_sequence_7_flat = [item for sublist in state_sequence_7 for item in sublist]
state_sequence_8_flat = [item for sublist in state_sequence_8 for item in sublist]
state_sequence_9_flat = [item for sublist in state_sequence_9 for item in sublist]

# Generate time vector for plotting
time = np.arange(0, sampling_time_sim*simulation_steps, sampling_time_sim)
control_time = np.arange(0, sampling_time_control*len(u_sequence_1) , sampling_time_control)

right_limit = time[-1]
control_right_limit = control_time[-1]

# Set font to Times New Roman for all plots
plt.rcParams["font.family"] = "Times New Roman"

# Plot Control Input (u) as subplots
plt.figure()

# Sub-plot for (1, 2, 3)
plt.subplot(3, 1, 1)
plt.plot(time, u_sequence_1, 'b-', label="NMPC")
plt.plot(time, u_sequence_2, 'r--', label="Lifted NMPC")
plt.plot(time, u_sequence_3, 'g-', label="Lifted NMPC (multi-rate)")
plt.ylabel('T = 0.1 (s)')
plt.grid(True)
plt.legend(loc='lower right')
plt.xlim(left=0, right=right_limit)

# Sub-plot for (4, 5, 6)
plt.subplot(3, 1, 2)
plt.plot(time, u_sequence_4, 'b-', label="NMPC")
plt.plot(time, u_sequence_5, 'r--', label="Lifted NMPC")
plt.plot(time, u_sequence_6, 'g-', label="Lifted NMPC (multi-rate)")
plt.ylabel('T = 0.25 (s)')
plt.grid(True)
plt.xlim(left=0, right=right_limit)

# Sub-plot for (7, 8, 9)
plt.subplot(3, 1, 3)
plt.plot(time, u_sequence_7, 'b-', label="NMPC")
plt.plot(time, u_sequence_8, 'r--', label="Lifted NMPC")
plt.plot(time, u_sequence_9, 'g-', label="Lifted NMPC (multi-rate)")
plt.ylabel('T = 0.5 (s)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.xlim(left=0, right=right_limit)

plt.tight_layout()
#plt.title("Sampling Period: " + str(sampler_period) + "s" + " Control Period: " + str(sampling_time_control) + "s")
#tikzplotlib.save("icp5.tex")
#tikzplotlib.save("mr_icp_input.tex")

def compute_state_norms(state_sequence_flat):
    return [
        np.linalg.norm(state_sequence_flat[i:i + 4])
        for i in range(0, len(state_sequence_flat), 4)
    ]

state_norms_1 = compute_state_norms(state_sequence_1_flat)
state_norms_2 = compute_state_norms(state_sequence_2_flat)
state_norms_3 = compute_state_norms(state_sequence_3_flat)
state_norms_4 = compute_state_norms(state_sequence_4_flat)
state_norms_5 = compute_state_norms(state_sequence_5_flat)
state_norms_6 = compute_state_norms(state_sequence_6_flat)
state_norms_7 = compute_state_norms(state_sequence_7_flat)
state_norms_8 = compute_state_norms(state_sequence_8_flat)
state_norms_9 = compute_state_norms(state_sequence_9_flat)

# print RMS values of state norm
print("RMS of state norm for NMPC (0.1): ", np.sqrt(np.mean(np.square(state_norms_1))))
print("RMS of state norm for Lifted NMPC (0.1): ", np.sqrt(np.mean(np.square(state_norms_2))))
print("RMS of state norm for Lifted NMPC (multi-rate) (0.1): ", np.sqrt(np.mean(np.square(state_norms_3))))
print("RMS of state norm for NMPC (0.25): ", np.sqrt(np.mean(np.square(state_norms_4))))
print("RMS of state norm for Lifted NMPC (0.25): ", np.sqrt(np.mean(np.square(state_norms_5))))
print("RMS of state norm for Lifted NMPC (multi-rate) (0.25): ", np.sqrt(np.mean(np.square(state_norms_6))))
print("RMS of state norm for NMPC (0.5): ", np.sqrt(np.mean(np.square(state_norms_7))))
print("RMS of state norm for Lifted NMPC (0.5): ", np.sqrt(np.mean(np.square(state_norms_8))))
print("RMS of state norm for Lifted NMPC (multi-rate) (0.5): ", np.sqrt(np.mean(np.square(state_norms_9))))

# print RMS values of control inputs
rms_u_1 = np.sqrt(np.mean(np.square(u_sequence_1)))
rms_u_2 = np.sqrt(np.mean(np.square(u_sequence_2)))
rms_u_3 = np.sqrt(np.mean(np.square(u_sequence_3)))
rms_u_4 = np.sqrt(np.mean(np.square(u_sequence_4)))
rms_u_5 = np.sqrt(np.mean(np.square(u_sequence_5)))
rms_u_6 = np.sqrt(np.mean(np.square(u_sequence_6)))
rms_u_7 = np.sqrt(np.mean(np.square(u_sequence_7)))
rms_u_8 = np.sqrt(np.mean(np.square(u_sequence_8)))
rms_u_9 = np.sqrt(np.mean(np.square(u_sequence_9)))
print("RMS of control inputs for NMPC (0.1): ", rms_u_1)
print("RMS of control inputs for Lifted NMPC (0.1): ", rms_u_2)
print("RMS of control inputs for Lifted NMPC (multi-rate) (0.1): ", rms_u_3)
print("RMS of control inputs for NMPC (0.25): ", rms_u_4)
print("RMS of control inputs for Lifted NMPC (0.25): ", rms_u_5)
print("RMS of control inputs for Lifted NMPC (multi-rate) (0.25): ", rms_u_6)
print("RMS of control inputs for NMPC (0.5): ", rms_u_7)
print("RMS of control inputs for Lifted NMPC (0.5): ", rms_u_8)
print("RMS of control inputs for Lifted NMPC (multi-rate) (0.5): ", rms_u_9)

# print average solve time
avg_solve_time_1 = np.mean(solve_time_1)
avg_solve_time_2 = np.mean(solve_time_2)
avg_solve_time_3 = np.mean(solve_time_3)
avg_solve_time_4 = np.mean(solve_time_4)
avg_solve_time_5 = np.mean(solve_time_5)
avg_solve_time_6 = np.mean(solve_time_6)
avg_solve_time_7 = np.mean(solve_time_7)
avg_solve_time_8 = np.mean(solve_time_8)
avg_solve_time_9 = np.mean(solve_time_9)
print("Average solve time for NMPC (0.1): ", avg_solve_time_1)
print("Average solve time for Lifted NMPC (0.1): ", avg_solve_time_2)
print("Average solve time for Lifted NMPC (multi-rate) (0.1): ", avg_solve_time_3)
print("Average solve time for NMPC (0.25): ", avg_solve_time_4)
print("Average solve time for Lifted NMPC (0.25): ", avg_solve_time_5)
print("Average solve time for Lifted NMPC (multi-rate) (0.25): ", avg_solve_time_6)
print("Average solve time for NMPC (0.5): ", avg_solve_time_7)
print("Average solve time for Lifted NMPC (0.5): ", avg_solve_time_8)
print("Average solve time for Lifted NMPC (multi-rate) (0.5): ", avg_solve_time_9)
# print average solve time

# Cart's position
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(time, [s[0] for s in state_sequence_1], 'b-', label="NMPC")
plt.plot(time, [s[0] for s in state_sequence_2], 'r--', label="Lifted NMPC")
plt.plot(time, [s[0] for s in state_sequence_3], 'g-', label="Lifted NMPC (multi-rate)")
plt.ylabel('T = 0.1 (s)')
plt.grid(True)
plt.legend(loc='lower right')
plt.xlim(left=0, right=right_limit)

# Sub-plot for (4, 5, 6)
plt.subplot(3, 1, 2)
plt.plot(time, [s[0] for s in state_sequence_4], 'b-', label="NMPC")
plt.plot(time, [s[0] for s in state_sequence_5], 'r--', label="Lifted NMPC")
plt.plot(time, [s[0] for s in state_sequence_6], 'g-', label="Lifted NMPC (multi-rate)")
plt.ylabel('T = 0.25 (s)')
plt.grid(True)
plt.xlim(left=0, right=right_limit)

# Sub-plot for (7, 8, 9)
plt.subplot(3, 1, 3)
plt.plot(time, [s[0] for s in state_sequence_7], 'b-', label="NMPC")
plt.plot(time, [s[0] for s in state_sequence_8], 'r--', label="Lifted NMPC")
plt.plot(time, [s[0] for s in state_sequence_9], 'g-', label="Lifted NMPC (multi-rate)")
plt.ylabel('T = 0.5 (s)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.xlim(left=0, right=right_limit)

plt.tight_layout()

# Pole's angle
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(time, [s[1] for s in state_sequence_1], 'b-', label="NMPC")
plt.plot(time, [s[1] for s in state_sequence_2], 'r--', label="Lifted NMPC")
plt.plot(time, [s[1] for s in state_sequence_3], 'g-', label="Lifted NMPC (multi-rate)")
plt.ylabel('T = 0.1 (s)')
plt.grid(True)
plt.legend(loc='lower right')
plt.xlim(left=0, right=right_limit)

# Sub-plot for (4, 5, 6)
plt.subplot(3, 1, 2)
plt.plot(time, [s[1] for s in state_sequence_4], 'b-', label="NMPC")
plt.plot(time, [s[1] for s in state_sequence_5], 'r--', label="Lifted NMPC")
plt.plot(time, [s[1] for s in state_sequence_6], 'g-', label="Lifted NMPC (multi-rate)")
plt.ylabel('T = 0.25 (s)')
plt.grid(True)
plt.xlim(left=0, right=right_limit)

# Sub-plot for (7, 8, 9)
plt.subplot(3, 1, 3)
plt.plot(time, [s[1] for s in state_sequence_7], 'b-', label="NMPC")
plt.plot(time, [s[1] for s in state_sequence_8], 'r--', label="Lifted NMPC")
plt.plot(time, [s[1] for s in state_sequence_9], 'g-', label="Lifted NMPC (multi-rate)")
plt.ylabel('T = 0.5 (s)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.xlim(left=0, right=right_limit)

plt.tight_layout()

# State NORM
plt.figure()

# Sub-plot for (1, 2, 3)
plt.subplot(3, 1, 1)
plt.plot(time, state_norms_1, 'b-', label="NMPC")
plt.plot(time, state_norms_2, 'r--', label="Lifted NMPC")
plt.plot(time, state_norms_3, 'g-', label="Lifted NMPC (multi-rate)")
plt.ylabel('T = 0.1 (s)')
plt.grid(True)
plt.legend(loc='upper right')
plt.xlim(left=0, right=right_limit)

# Sub-plot for (4, 5, 6)
plt.subplot(3, 1, 2)
plt.plot(time, state_norms_4, 'b-', label="NMPC")
plt.plot(time, state_norms_5, 'r--', label="Lifted NMPC")
plt.plot(time, state_norms_6, 'g-', label="Lifted NMPC (multi-rate)")
plt.ylabel('T = 0.25 (s)')
plt.grid(True)
#plt.legend(loc='upper right')
plt.xlim(left=0, right=right_limit)

# Sub-plot for (7, 8, 9)
plt.subplot(3, 1, 3)
plt.plot(time, state_norms_7, 'b-', label="NMPC")
plt.plot(time, state_norms_8, 'r--', label="Lifted NMPC")
plt.plot(time, state_norms_9, 'g-', label="Lifted NMPC (multi-rate)")
plt.ylabel('T = 0.5 (s)')
plt.xlabel('Time (s)')
plt.grid(True)
#plt.legend(loc='upper right')
plt.xlim(left=0, right=right_limit)

plt.tight_layout()
#tikzplotlib.save("mr_icp_state_norm.tex")

plt.figure()

# Sub-plot for (1, 2, 3)
plt.subplot(3, 1, 1)
plt.plot(time, np.minimum(solve_time_1, 45), 'b-', label="NMPC")
plt.plot(time, np.minimum(solve_time_2, 45), 'r--', label="Lifted NMPC")
plt.plot(time, np.minimum(solve_time_3, 45), 'g-', label="Lifted NMPC (multi-rate)")
plt.ylabel('T = 0.1 (s)')
plt.grid(True)
plt.legend(loc='upper right')
plt.xlim(left=0, right=right_limit)

# Sub-plot for (4, 5, 6)
plt.subplot(3, 1, 2)
plt.plot(time, np.minimum(solve_time_4, 45), 'b-', label="NMPC")
plt.plot(time, np.minimum(solve_time_5, 45), 'r--', label="Lifted NMPC")
plt.plot(time, np.minimum(solve_time_6, 45), 'g-', label="Lifted NMPC (multi-rate)")
plt.ylabel('T = 0.25 (s)')
plt.grid(True)
#plt.legend(loc='upper right')
plt.xlim(left=0, right=right_limit)

# Sub-plot for (7, 8, 9)
plt.subplot(3, 1, 3)
plt.plot(time, np.minimum(solve_time_7, 45), 'b-', label="NMPC")
plt.plot(time, np.minimum(solve_time_8, 45), 'r--', label="Lifted NMPC")
plt.plot(time, np.minimum(solve_time_9, 45), 'g-', label="Lifted NMPC (multi-rate)")
plt.ylabel('T = 0.5 (s)')
plt.xlabel('Time (s)')
plt.grid(True)
#plt.legend(loc='upper right')
plt.xlim(left=0, right=right_limit)

plt.tight_layout()
#tikzplotlib.save("mr_icp_solve_time.tex")

plt.show()

#plt.tight_layout()
#plt.savefig('cartpole.png')
#plt.show()