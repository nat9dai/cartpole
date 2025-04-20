import opengen as og
import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

import matplotlib.ticker as ticker

import tikzplotlib

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

tcp_server_name1 = "OpEn/python_cartpole_" + str(sampler_period) + \
                "/cartpole_" + str(sampler_period)
mng1 = og.tcp.OptimizerTcpManager(tcp_server_name1.replace(".", "_"), port=8333)
tcp_server_name2 = "OpEn/python_cartpole_lifting_" + str(sampler_period) + \
                "/cartpole_lifting_" + str(sampler_period)
mng2 = og.tcp.OptimizerTcpManager(tcp_server_name2.replace(".", "_"), port=8334)
tcp_server_name3 = "OpEn/python_cartpole_lifting_2_" + str(sampler_period) + \
                "/cartpole_lifting_2_" + str(sampler_period)
mng3 = og.tcp.OptimizerTcpManager(tcp_server_name3.replace(".", "_"), port=8335)

mng1.start()
mng2.start()
mng3.start()

x_state_0 = [0,np.pi,0,0]
#Q = [10, 30, 0.4, 3]
#Qt = [20, 40, 0.8, 6]
Q = [5, 20, 0.02, 2]
Qt = [6, 30, 0.04, 4]
R = [0.1]
# parameter for lifted 2
Rtr = [0.0] # default = 0
gamma =[1] # [0,1] default = 1
lambda_ = [1] #[0,1] default = 1

state_sequence = []
input_sequence = []

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
for k in range(simulation_steps):
    if k % sampler_interval == 0:
        solver_status = mng1.call(x1+Q+Qt+R, initial_guess=us)
        #solver_status = mng1.call(x1+Q+Qt+R)
        us = solver_status['solution']
        u1 = us[0]

    x1_next = dynamics_dt(x1, u1, sampling_time_sim=sampling_time_sim)

    state_sequence_1.append(x1_next)
    u_sequence_1.append(u1)

    x1 = x1_next

x2 = x_state_0
state_sequence_2 = []
us = [0]*T  # Initial control input
u_sequence_2 = []
for k in range(simulation_steps):
    if k % sampler_interval == 0:
        solver_status = mng2.call(x2+Q+Qt+R, initial_guess=us)
        #solver_status = mng2.call(x2+Q+Qt+R)
        us = solver_status['solution']
        u2 = us[0]

    x2_next = dynamics_dt(x2, u2, sampling_time_sim=sampling_time_sim)

    state_sequence_2.append(x2_next)
    u_sequence_2.append(u2)

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

count = 0
control_sequence = [0]*N_P
u_sequence_3_test = []
u_sequence_3_test_2 = []
cost_list = []
for k in range(simulation_steps):
    if k % sampler_interval == 0:
        solver_status = mng3.call(x3+Q+Qt+R+[0.0]+[1.0]+us[N_P:2*N_P]+[1.0], initial_guess=shift_left(us, N_P))
        #solver_status = mng3.call(x3+Q+Qt+R)
        us = solver_status['solution']
        #print(solver_status.keys())
        #print(solver_status['num_inner_iterations'], solver_status['num_outer_iterations']) 
        #print(solver_status['exit_status'])
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
         
    x3_next = dynamics_dt(x3, u3, sampling_time_sim=sampling_time_sim)

    state_sequence_3.append(x3_next)
    u_sequence_3.append(u3)

    x3 = x3_next

# Convert state_sequence into a flattened list for plotting
state_sequence_1_flat = [item for sublist in state_sequence_1 for item in sublist]
state_sequence_2_flat = [item for sublist in state_sequence_2 for item in sublist]
state_sequence_3_flat = [item for sublist in state_sequence_3 for item in sublist]
# Generate time vector for plotting
time = np.arange(0, sampling_time_sim*simulation_steps, sampling_time_sim)

x_1  = state_sequence_1_flat[0::4]
theta_1 = state_sequence_1_flat[1::4]

x_2  = state_sequence_2_flat[0::4]
theta_2 = state_sequence_2_flat[1::4]

x_3  = state_sequence_3_flat[0::4]
theta_3 = state_sequence_3_flat[1::4]

l = 2.0

fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2, figsize=(30, 30))

fig.suptitle('Sampling period: 0.5 s', fontsize=18, y=0.95)

fig.text(0.1, 0.77, 'NMPC', va='center', ha='center', rotation='horizontal', fontsize=14)
fig.text(0.1, 0.50, 'Lifted NMPC', va='center', ha='center', rotation='horizontal', fontsize=14)
fig.text(0.1, 0.22, 'Lifted NMPC\n(multi-rate)', va='center', ha='center', rotation='horizontal', fontsize=14)

ax1.set_xlim(-6, 6)
ax1.set_ylim(-2.5, 2.5)
ax2.set_xlim(-6, 6)
ax2.set_ylim(-2.5, 2.5)
ax3.set_xlim(-6, 6)
ax3.set_ylim(-2.5, 2.5)

ax1.set_yticks([])
ax1.set_ylabel('')
ax2.set_yticks([])
ax2.set_ylabel('')
ax3.set_yticks([])
ax3.set_ylabel('')

ax1.set_xticks([])
ax1.set_xlabel('')
ax3.set_xticks([])
ax3.set_xlabel('')

ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax3.set_aspect('equal')

ax4.set_xlim(time[0], time[-1])
ax5.set_xlim(time[0], time[-1])
ax6.set_xlim(time[0], time[-1])
ax4.set_ylim(6.0, -2.0)
ax5.set_ylim(6.0, -2.0)
ax6.set_ylim(6.0, -2.0)
ax4.grid(True)
ax5.grid(True)
ax6.grid(True)
# Add legends to time-series axes
ax4.legend(loc='upper right', fontsize=10)
ax5.legend(loc='upper right', fontsize=10)
ax6.legend(loc='upper right', fontsize=10)

pos_1, = ax4.plot(time, x_1, color='blue', label='Controller 1 - Position')
angle_1, = ax4.plot(time, theta_1, color='red', label='Controller 1 - Angle')

pos_2, = ax5.plot(time, x_2, color='blue', label='Position (m)')
angle_2, = ax5.plot(time, theta_2, color='red', label='Angle (rad)')

pos_3, = ax6.plot(time, x_3, color='blue', label='Controller 3 - Position')
angle_3, = ax6.plot(time, theta_3, color='red', label='Controller 3 - Angle')

ax5.legend(loc='upper right')

cart_width = 0.8
cart_height = 0.4

cart_1 = plt.Rectangle((-cart_width / 2, -cart_height / 2), cart_width, cart_height, fill=True, color="blue")
ax1.add_patch(cart_1)
pendulum_line_1, = ax1.plot([], [], lw=2, color='black')

cart_2 = plt.Rectangle((-cart_width / 2, -cart_height / 2), cart_width, cart_height, fill=True, color="red")
ax2.add_patch(cart_2)
pendulum_line_2, = ax2.plot([], [], lw=2, color='black')

cart_3 = plt.Rectangle((-cart_width / 2, -cart_height / 2), cart_width, cart_height, fill=True, color="green")
ax3.add_patch(cart_3)
pendulum_line_3, = ax3.plot([], [], lw=2, color='black')

time_text_ax1 = ax1.text(0.2, 0.95, '', transform=ax1.transAxes, fontsize=10, ha='right', va='top')
time_text_ax2 = ax2.text(0.2, 0.95, '', transform=ax2.transAxes, fontsize=10, ha='right', va='top')
time_text_ax3 = ax3.text(0.2, 0.95, '', transform=ax2.transAxes, fontsize=10, ha='right', va='top')

def init():
    # Controller 1
    cart_1.set_xy((-cart_width / 2, -cart_height / 2))
    pendulum_line_1.set_data([], [])
    
    # Controller 2
    cart_2.set_xy((-cart_width / 2, -cart_height / 2))
    pendulum_line_2.set_data([], [])

    # Controller 3
    cart_3.set_xy((-cart_width / 2, -cart_height / 2))
    pendulum_line_3.set_data([], [])

    #time_text_ax1.set_text('')
    time_text_ax2.set_text('')
    #time_text_ax3.set_text('')

    pos_1.set_data([], [])
    angle_1.set_data([], [])
    pos_2.set_data([], [])
    angle_2.set_data([], [])
    pos_3.set_data([], [])
    angle_3.set_data([], [])
    
    return cart_1, pendulum_line_1, cart_2, pendulum_line_2, cart_3, pendulum_line_3, time_text_ax2, \
           pos_1, angle_1, pos_2, angle_2, pos_3, angle_3,

def update(frame):
    cart_center_x_1 = x_1[frame]
    cart_1.set_xy((cart_center_x_1 - cart_width / 2, -cart_height / 2))
    pendulum_x_1 = cart_center_x_1 + l * np.sin(theta_1[frame]+np.pi)
    pendulum_y_1 = -l * np.cos(theta_1[frame]+np.pi)
    pendulum_line_1.set_data([cart_center_x_1, pendulum_x_1], [0, pendulum_y_1])
    
    cart_center_x_2 = x_2[frame]
    cart_2.set_xy((cart_center_x_2 - cart_width / 2, -cart_height / 2))
    pendulum_x_2 = cart_center_x_2 + l * np.sin(theta_2[frame]+np.pi)
    pendulum_y_2 = -l * np.cos(theta_2[frame]+np.pi)
    pendulum_line_2.set_data([cart_center_x_2, pendulum_x_2], [0, pendulum_y_2])

    cart_center_x_3 = x_3[frame]
    cart_3.set_xy((cart_center_x_3 - cart_width / 2, -cart_height / 2))
    pendulum_x_3 = cart_center_x_3 + l * np.sin(theta_3[frame]+np.pi)
    pendulum_y_3 = -l * np.cos(theta_3[frame]+np.pi)
    pendulum_line_3.set_data([cart_center_x_3, pendulum_x_3], [0, pendulum_y_3])

    # Update the time shown on each subplot
    real_time = time[frame]  # Real-time based on the current frame
    #time_text_ax1.set_text(f'Time: {real_time:.2f} s')
    time_text_ax2.set_text(f'Time: {real_time:.2f} s')
    #time_text_ax3.set_text(f'Time: {real_time:.2f} s')

    pos_1.set_data(time[:frame], x_1[:frame])
    angle_1.set_data(time[:frame], theta_1[:frame])
    pos_2.set_data(time[:frame], x_2[:frame])
    angle_2.set_data(time[:frame], theta_2[:frame])
    pos_3.set_data(time[:frame], x_3[:frame])
    angle_3.set_data(time[:frame], theta_3[:frame])

    plt.savefig(f'frames_050/frame_{frame:06d}.png')
    
    if frame >= simulation_steps - 1:
        ani.event_source.stop()
        print("yo")
        plt.close(fig)

    return cart_1, pendulum_line_1, cart_2, pendulum_line_2, cart_3, pendulum_line_3, time_text_ax2, \
              pos_1, angle_1, pos_2, angle_2, pos_3, angle_3,

ani = animation.FuncAnimation(fig, update, frames=simulation_steps, init_func=init, blit=True, interval=0.001)

#ani.save('inverted_pendulum_comparison.gif', writer='pillow', fps=120)

plt.show()