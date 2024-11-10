import opengen as og
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.ticker as ticker

import tikzplotlib

sampling_time_control = 0.05
################### Change this parameter ###################
sampler_period = 0.1 # 0.1, 0.25, 0.5
#############################################################

N_P = int(sampler_period/sampling_time_control)
horizon_time = 1.0
T = int(horizon_time / sampler_period)

tcp_server_name1 = "OpEn/python_cartpole_" + str(sampler_period) + \
                "/cartpole_" + str(sampler_period)
mng1 = og.tcp.OptimizerTcpManager(tcp_server_name1.replace(".", "_"), port=8333)
tcp_server_name2 = "OpEn/python_cartpole_lifting_" + str(sampler_period) + \
                "/cartpole_lifting_n10_" + str(sampler_period)
mng2 = og.tcp.OptimizerTcpManager(tcp_server_name2.replace(".", "_"), port=8334)
tcp_server_name3 = "OpEn/python_cartpole_lifting_2_" + str(sampler_period) + \
                "/cartpole_lifting_2_n10_new_" + str(sampler_period)
mng3 = og.tcp.OptimizerTcpManager(tcp_server_name3.replace(".", "_"), port=8335)

mng1.start()
mng2.start()
mng3.start()

x_state_0 = [0,np.pi,0,0]

state_sequence = []
input_sequence = []

gravity_acceleration = 9.8
length = 1.0
mass_cart = 1.0
mass_pole = 0.2

dim = 4
#sampling_time_sim = sampler_period/10
sampling_time_sim = 0.001

simulation_steps = int(20 / sampling_time_sim)

def dynamics_ct(x, u):
    dx1 = x[2]
    dx2 = x[3]
    dx3 = (-mass_pole*length*np.sin(x[1])*x[3]**2 \
           + mass_pole*gravity_acceleration*np.sin(x[1])*np.cos(x[1])
           + u) / (mass_cart+mass_pole*np.sin(x[1])**2)
    dx4 = (-mass_pole*length*np.sin(x[1])*np.cos(x[1])*x[3]**2 \
           + (mass_cart+mass_pole)*gravity_acceleration*np.sin(x[1]) \
           + u*np.cos(x[1])) / (length*(mass_cart+mass_pole*np.sin(x[1])**2))
    return [dx1, dx2, dx3, dx4]

# RK4
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
u1 = 0  # Initial control input
u_sequence_1 = []
control_interval = int(sampling_time_control / sampling_time_sim)
sampler_interval = int(sampler_period / sampling_time_sim)
sampler_interval_test = int(1.0/sampling_time_sim)

for k in range(simulation_steps):
    if k % sampler_interval == 0:
        solver_status = mng1.call(x1, initial_guess=[u1]*T)
        #solver_status = mng1.call(x1)
        us = solver_status['solution']
        u1 = us[0]

    x1_next = dynamics_dt(x1, u1, sampling_time_sim=sampling_time_sim)

    state_sequence_1.append(x1_next)
    u_sequence_1.append(u1)

    x1 = x1_next

x2 = x_state_0
state_sequence_2 = []
u2 = 0  # Initial control input
u_sequence_2 = []

for k in range(simulation_steps):
    if k % sampler_interval == 0:
        solver_status = mng2.call(x2, initial_guess=[u2]*T)
        #solver_status = mng2.call(x2)
        us = solver_status['solution']
        u2 = us[0]

    x2_next = dynamics_dt(x2, u2, sampling_time_sim=sampling_time_sim)

    state_sequence_2.append(x2_next)
    u_sequence_2.append(u2)

    x2 = x2_next

x3 = x_state_0
state_sequence_3 = []
u3 = 0  # Initial control input
u_sequence_3 = []

def weighted_average_resample(data, new_size):
    old_size = len(data)
    new_indices = np.linspace(0, old_size - 1, new_size)
    new_data = np.interp(new_indices, np.arange(old_size), data)
    return new_data

count = 0
control_sequence = [0]*N_P
u_sequence_3_test = []
u_sequence_3_test_2 = []
cost_list = []
for k in range(simulation_steps):
    if k % sampler_interval == 0:
        solver_status = mng3.call(x3, initial_guess=[u3]*T*N_P)
        #solver_status = mng3.call(x3)
        us = solver_status['solution']
        #print(solver_status.keys())
        #print(solver_status['num_inner_iterations'], solver_status['num_outer_iterations']) 
        #print(solver_status['exit_status'])
        #control_sequence = weighted_average_resample(us[0:N], len(control_sequence))
        #control_sequence = np.append(control_sequence, control_sequence[-1])
        control_sequence = us[0:N_P]
        count = 0
    if k % control_interval == 0:
        u3 = control_sequence[count]
        count += 1
    
    if k == int(4.5/sampling_time_sim):
        for i in range(T*N_P):
            for j in range(int(1000/(N_P*T))):
                u_sequence_3_test.append(us[i])
        #print(total_cost(x3, u_sequence_3_test))

    if k == int(4.6/sampling_time_sim):
        for i in range(T*N_P):
            for j in range(int(1000/(N_P*T))):
                u_sequence_3_test_2.append(us[i])
    
    
    #print(cost)
         
    x3_next = dynamics_dt(x3, u3, sampling_time_sim=sampling_time_sim)

    state_sequence_3.append(x3_next)
    u_sequence_3.append(u3)

    x3 = x3_next

u_test_1 = range(T)
u_test_2 = []
for i in u_test_1:
    for j in range(N_P):
        u_test_2.append(i)
x_test = [0,np.pi,0,0]

# Convert state_sequence into a flattened list for plotting
state_sequence_1_flat = [item for sublist in state_sequence_1 for item in sublist]
state_sequence_2_flat = [item for sublist in state_sequence_2 for item in sublist]
state_sequence_3_flat = [item for sublist in state_sequence_3 for item in sublist]

# Generate time vector for plotting
time = np.arange(0, sampling_time_sim*simulation_steps, sampling_time_sim)
control_time = np.arange(0, sampling_time_control*len(u_sequence_1) , sampling_time_control)

right_limit = time[-1]
control_right_limit = control_time[-1]

#fig, ax = plt.subplots(4, 1, figsize=(10, 12))

# fig.suptitle('Horizon Length: ' + str(sampling_time_control) + '[s]')
#fig.suptitle('Horizon Length: 0.75[s], Controller Sampling Period: 0.05[s]')

plt.rcParams.update({'font.size': 18})

# Plot Position (x[0])
plt.figure(figsize=(8, 6))
plt.plot(time, state_sequence_1_flat[0::4], 'b-', label="NMPC")
plt.plot(time, state_sequence_2_flat[0::4], 'r--', label="Lifted NMPC 1")
plt.plot(time, state_sequence_3_flat[0::4], 'g-', label="Lifted NMPC 2")
plt.ylabel(r'$x_1$ (m)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend(loc='upper right')
plt.xlim(left=0, right=right_limit)  # Set both left and right limits for the x-axis
plt.tight_layout()
#plt.title("Sampling Period: " + str(sampler_period) + "s" + " Control Period: " + str(sampling_time_control) + "s")
#tikzplotlib.save("icp1.tex")

# Plot Angle (x[1])
plt.figure(figsize=(8, 6))
plt.plot(time, state_sequence_1_flat[1::4], 'b-', label="NMPC")
plt.plot(time, state_sequence_2_flat[1::4], 'r--', label="Lifted NMPC 1")
plt.plot(time, state_sequence_3_flat[1::4], 'g-', label="Lifted NMPC 2")
plt.ylabel(r'$x_2$ (rad)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend(loc='upper right')
plt.xlim(left=0, right=right_limit)  # Set both left and right limits for the x-axis
plt.tight_layout()
#plt.title("Sampling Period: " + str(sampler_period) + "s" + " Control Period: " + str(sampling_time_control) + "s")
#tikzplotlib.save("icp2.tex")

# Plot Velocity (x[2])
plt.figure(figsize=(8, 6))
plt.plot(time, state_sequence_1_flat[2::4], 'b-', label="NMPC")
plt.plot(time, state_sequence_2_flat[2::4], 'r--', label="Lifted NMPC 1")
plt.plot(time, state_sequence_3_flat[2::4], 'g-', label="Lifted NMPC 2")
plt.ylabel(r'$x_3$ (m/s)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend(loc='lower right')
plt.xlim(left=0, right=right_limit)  # Set both left and right limits for the x-axis
plt.tight_layout() 
#plt.title("Sampling Period: " + str(sampler_period) + "s" + " Control Period: " + str(sampling_time_control) + "s")
#tikzplotlib.save("icp3.tex")

# Plot Velocity
plt.figure(figsize=(8, 6))
plt.plot(time, state_sequence_1_flat[3::4], 'b-', label="NMPC")
plt.plot(time, state_sequence_2_flat[3::4], 'r--', label="Lifted NMPC 1")
plt.plot(time, state_sequence_3_flat[3::4], 'g-', label="Lifted NMPC 2")
plt.ylabel(r'$x_4$ (rad/s)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend(loc='lower right')
plt.xlim(left=0, right=right_limit)  # Set both left and right limits for the x-axis
plt.tight_layout()
#plt.title("Sampling Period: " + str(sampler_period) + "s" + " Control Period: " + str(sampling_time_control) + "s")
#tikzplotlib.save("icp4.tex")

# Plot Control Input (u)
plt.figure(figsize=(8, 6))
plt.plot(time, u_sequence_1, 'b-', label="NMPC")
plt.plot(time, u_sequence_2, 'r--', label="Lifted NMPC 1")
plt.plot(time, u_sequence_3, 'g-', label="Lifted NMPC 2")
time_test = np.arange(4.5, 5.5, sampling_time_sim)
time_test2 = np.arange(4.6, 5.6, sampling_time_sim)
plt.plot(time_test, u_sequence_3_test, 'k-', label="Lifted NMPC 2 (Test 1)")
plt.plot(time_test2, u_sequence_3_test_2, 'm-', label="Lifted NMPC 2 (Test 2)")
plt.ylabel(r'Control Input: $u$')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend(loc='lower right')
plt.xlim(left=0, right=right_limit)  # Set both left and right limits for the x-axis
plt.tight_layout() 
#plt.title("Sampling Period: " + str(sampler_period) + "s" + " Control Period: " + str(sampling_time_control) + "s")
#tikzplotlib.save("icp5.tex")

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
state_norms_3 = [
    np.sqrt(
        state_sequence_3_flat[i]**2 +
        state_sequence_3_flat[i + 1]**2 +
        state_sequence_3_flat[i + 2]**2 +
        state_sequence_3_flat[i + 3]**2
    )
    for i in range(0, len(state_sequence_3_flat), 4)
]

# State NORM
plt.figure(figsize=(8, 6))
plt.plot(time, state_norms_1, 'b-', label="NMPC")
plt.plot(time, state_norms_2, 'r--', label="Lifted NMPC 1")
plt.plot(time, state_norms_3, 'g-', label="Lifted NMPC 2")
plt.ylabel(r'$||e||_{2}$') 
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend(loc='upper right')
plt.xlim(left=0, right=right_limit)  # Set both left and right limits for the x-axis
plt.tight_layout() 

plt.show()

#plt.tight_layout()
#plt.savefig('cartpole.png')
#plt.show()

# Stop the TCP servers
mng1.kill()
mng2.kill()
mng3.kill()