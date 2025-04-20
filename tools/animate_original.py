import opengen as og
import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

import matplotlib.ticker as ticker

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'

sampler_period = 0.05
sampling_time_sim = 0.001
simulation_steps = int(20 / sampling_time_sim)
T = 20
sampler_interval = int(sampler_period / sampling_time_sim)

tcp_server_name1 = "/Users/nuthasith/Documents/python_workspace/cartpole/OpEn/python_cartpole_original_constraint" + \
                   "/cartpole_original_constraint"
mng1 = og.tcp.OptimizerTcpManager(tcp_server_name1.replace(".", "_"), port=8333)
tcp_server_name2 = "/Users/nuthasith/Documents/python_workspace/cartpole/OpEn/python_cartpole_original_lifted_constraint" + \
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

x_1  = state_sequence_1_flat[0::4]
theta_1 = state_sequence_1_flat[1::4]
x_dot_1  = state_sequence_1_flat[2::4]
theta_dot_1 = state_sequence_1_flat[3::4]

x_2  = state_sequence_2_flat[0::4]
theta_2 = state_sequence_2_flat[1::4]
x_dot_2  = state_sequence_2_flat[2::4]
theta_dot_2 = state_sequence_2_flat[3::4]

l = 2.0

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(12, 9))

#fig.suptitle('Sampling period: 0.5 s', fontsize=18, y=0.95)

ax1.text(0, 2.7, 'NMPC', va='center', ha='center', rotation='horizontal', fontsize=12)
ax2.text(0, 2.7, 'Lifted NMPC', va='center', ha='center', rotation='horizontal', fontsize=12)

ax1.set_xlim(-6, 6)
ax1.set_ylim(-2.5, 2.5)
ax2.set_xlim(-6, 6)
ax2.set_ylim(-2.5, 2.5)

ax1.set_yticks([])
ax1.set_ylabel('')
ax2.set_yticks([])
ax2.set_ylabel('')

ax1.set_xticks([])
ax1.set_xlabel('')
ax2.set_xticks([])
ax2.set_xlabel('')

ax1.set_aspect('equal')
ax2.set_aspect('equal')

ax3.set_xlim(time[0], time[-1])
ax4.set_xlim(time[0], time[-1])
ax3.set_ylim(-2.5, 5.5)
ax4.set_ylim(-2.5, 5.5)
#ax3.set_aspect(1.0)  # Adjust aspect ratio to make y-axis shorter
#ax4.set_aspect(1.0)  # Adjust aspect ratio to make y-axis shorter
ax3.grid(True)
ax4.grid(True)

ax5.set_xlim(time[0], time[-1])
ax6.set_xlim(time[0], time[-1])
ax5.set_ylim(-7.6, 7.6)
ax6.set_ylim(-7.6, 7.6)
#ax5.set_aspect(1.0)  # Adjust aspect ratio to make y-axis shorter
#ax6.set_aspect(1.0)  # Adjust aspect ratio to make y-axis shorter
ax5.grid(True)
ax6.grid(True)
# Label x-axis for time-series plots
#ax5.set_xlabel('Time (s)')
#ax6.set_xlabel('Time (s)')

ax7.set_xlim(time[0], time[-1])
ax8.set_xlim(time[0], time[-1])
ax7.set_ylim(-15.0, 15.0)
ax8.set_ylim(-15.0, 15.0)
#ax7.set_aspect(1.0)  # Adjust aspect ratio to make y-axis shorter
#ax8.set_aspect(1.0)  # Adjust aspect ratio to make y-axis shorter
ax7.grid(True)
ax8.grid(True)
# Label x-axis for time-series plots
ax7.set_xlabel('Time (s)')
ax8.set_xlabel('Time (s)')

pos_1, = ax3.plot(time, x_1, color='blue', label="Cart's Position (m)")
angle_1, = ax3.plot(time, theta_1, color='red', label="Pole's Angle (rad)")
vel_1, = ax5.plot(time, x_dot_1, color='blue', label="Cart's Velocity (m/s)")
angle_dot_1, = ax5.plot(time, theta_dot_1, color='red', label="Pole's Velocity (rad/s)")
input_1, = ax7.plot(time, u_sequence_1, color='blue', label="Control Input (N)")

pos_2, = ax4.plot(time, x_2, color='blue', label='Position (m)')
angle_2, = ax4.plot(time, theta_2, color='red', label='Angle (rad)')
vel_2, = ax6.plot(time, x_dot_2, color='blue', label='Velocity (m/s)')
angle_dot_2, = ax6.plot(time, theta_dot_2, color='red', label='Velocity (rad/s)')
input_2, = ax8.plot(time, u_sequence_2, color='blue', label='Control Input (N)')

ax3.legend(loc='upper right')
ax5.legend(loc='upper right')
ax7.legend(loc='upper right')

cart_width = 0.8
cart_height = 0.4

cart_1 = plt.Rectangle((-cart_width / 2, -cart_height / 2), cart_width, cart_height, fill=True, color="blue")
ax1.add_patch(cart_1)
pendulum_line_1, = ax1.plot([], [], lw=2, color='black')

cart_2 = plt.Rectangle((-cart_width / 2, -cart_height / 2), cart_width, cart_height, fill=True, color="red")
ax2.add_patch(cart_2)
pendulum_line_2, = ax2.plot([], [], lw=2, color='black')

time_text_ax1 = ax1.text(0.95, 0.95, '', transform=ax1.transAxes, fontsize=10, ha='right', va='top')
#time_text_ax2 = ax2.text(0.2, 0.95, '', transform=ax2.transAxes, fontsize=10, ha='right', va='top')

def init():
    # Controller 1
    cart_1.set_xy((-cart_width / 2, -cart_height / 2))
    pendulum_line_1.set_data([], [])
    
    # Controller 2
    cart_2.set_xy((-cart_width / 2, -cart_height / 2))
    pendulum_line_2.set_data([], [])

    time_text_ax1.set_text('')
    #time_text_ax2.set_text('')
    #time_text_ax3.set_text('')

    pos_1.set_data([], [])
    angle_1.set_data([], [])
    pos_2.set_data([], [])
    angle_2.set_data([], [])
    vel_1.set_data([], [])
    angle_dot_1.set_data([], [])
    vel_2.set_data([], [])
    angle_dot_2.set_data([], [])
    input_1.set_data([], [])
    input_2.set_data([], [])

    
    return cart_1, pendulum_line_1, cart_2, pendulum_line_2, time_text_ax1, \
           pos_1, angle_1, pos_2, angle_2, vel_1, angle_dot_1, vel_2, angle_dot_2, input_1, input_2,

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

    # Update the time shown on each subplot
    real_time = time[frame]  # Real-time based on the current frame
    time_text_ax1.set_text(f'Time: {real_time:.2f} s')
    #time_text_ax2.set_text(f'Time: {real_time:.2f} s')
    #time_text_ax3.set_text(f'Time: {real_time:.2f} s')

    pos_1.set_data(time[:frame], x_1[:frame])
    angle_1.set_data(time[:frame], theta_1[:frame])
    pos_2.set_data(time[:frame], x_2[:frame])
    angle_2.set_data(time[:frame], theta_2[:frame])
    vel_1.set_data(time[:frame], x_dot_1[:frame])
    angle_dot_1.set_data(time[:frame], theta_dot_1[:frame])
    vel_2.set_data(time[:frame], x_dot_2[:frame])
    angle_dot_2.set_data(time[:frame], theta_dot_2[:frame])
    input_1.set_data(time[:frame], u_sequence_1[:frame])
    input_2.set_data(time[:frame], u_sequence_2[:frame])

    #plt.savefig(f'frames_2/frame_{frame:06d}.png')
    
    if frame >= simulation_steps - 1:
        ani.event_source.stop()
        print("yo")
        plt.close(fig)

    return cart_1, pendulum_line_1, cart_2, pendulum_line_2, time_text_ax1, \
              pos_1, angle_1, pos_2, angle_2, vel_1, angle_dot_1, vel_2, angle_dot_2, input_1, input_2,

ani = animation.FuncAnimation(fig, update, frames=simulation_steps, init_func=init, blit=True, interval=10)

#ani.save('inverted_pendulum_comparison.gif', writer='pillow', fps=120)

from tqdm import tqdm

def progress_callback(current_frame, total_frames):
    print(f'Saving frame {current_frame + 1} of {total_frames}', end='\r')

writer = animation.FFMpegWriter(fps=1/sampling_time_sim, metadata=dict(artist='Me'))

with writer.saving(fig, "output.mp4", dpi=100):
    for i in tqdm(range(simulation_steps)):  # or however many frames you have
        update(i)  # your update function for frame i
        writer.grab_frame()

#ani.save('ipc_2.mp4', writer=writer)
plt.close() 

plt.show()