import casadi.casadi as cs
import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
gravity_acceleration = 9.8
length = 1.0
mass_cart = 1.0
mass_pole = 0.2
dim = 4

# Time parameters
horizon_time = 1.0
sampling_time = 0.25   # 0.1, 0.25, 0.5
control_sampling_time = 0.05  # Control sampling time
simulation_period = 0.001  # Simulation time step
T = int(horizon_time / sampling_time)  # Number of steps in the MPC horizon
simulation_time = 10.0  # Total simulation time in seconds
num_steps = int(simulation_time / simulation_period)

N = 10
#N_P = 2 # 2, 5, 10
N_P = int(sampling_time/control_sampling_time)

# Dynamics and cost functions
def dynamics_ct_cs(x, u):
    dx1 = x[2]
    dx2 = x[3]
    dx3 = (-mass_pole * length * cs.sin(x[1]) * x[3]**2 +
           mass_pole * gravity_acceleration * cs.sin(x[1]) * cs.cos(x[1]) +
           u) / (mass_cart + mass_pole * cs.sin(x[1])**2)
    dx4 = (-mass_pole * length * cs.sin(x[1]) * cs.cos(x[1]) * x[3]**2 +
           (mass_cart + mass_pole) * gravity_acceleration * cs.sin(x[1]) +
           u * cs.cos(x[1])) / (length * (mass_cart + mass_pole * cs.sin(x[1])**2))
    return [dx1, dx2, dx3, dx4]

def dynamics_dt_cs(x, u, h):
    k1 = dynamics_ct_cs(x, u)
    x_k2 = [x[i] + 0.5 * h * k1[i] for i in range(dim)]
    k2 = dynamics_ct_cs(x_k2, u)
    x_k3 = [x[i] + 0.5 * h * k2[i] for i in range(dim)]
    k3 = dynamics_ct_cs(x_k3, u)
    x_k4 = [x[i] + h * k3[i] for i in range(dim)]
    k4 = dynamics_ct_cs(x_k4, u)
    x_next = [x[i] + (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(dim)]
    return x_next

def stage_cost(x, u):
    cost = 5 * x[0]**2 + 20 * x[1]**2 + 0.02 * x[2]**2 + 0.02 * x[3]**2 + 0.1 * u**2
    return cost

def stage_cost_2(x):
    cost = 5 * x[0]**2 + 20 * x[1]**2 + 0.02 * x[2]**2 + 0.02 * x[3]**2
    return cost

def terminal_cost(x):
    cost = 6 * x[0]**2 + 30 * x[1]**2 + 0.04 * x[2]**2 + 0.04 * x[3]**2
    return cost

def simulate_mpc(method):
    x0 = cs.MX.sym("x0", dim)  # Initial state

    # some precompute parameters for Simpson's rule
    sampling_div_2N = sampling_time / (2 * N)
    sampling_div_N = sampling_time / N

    if (method == 1) or (method == 2):
        u_seq = cs.MX.sym("u", T)  # Sequence of all control inputs
        # Define state trajectory and cost
        x_traj = [x0]
        cost_total = 0
        if method == 1:
            for t in range(T):
                x_next = dynamics_dt_cs(x_traj[-1], u_seq[t], sampling_time)
                x_traj.append(x_next)
                cost_total += stage_cost(x_traj[-1], u_seq[t])
            # Adding terminal cost
            cost_total += terminal_cost(x_traj[-1])
        elif method == 2:
            for i in range(T):
                for j in range(N):
                    integral = stage_cost_2(x_traj[-1])
                    x_mid = dynamics_dt_cs(x_traj[-1], u_seq[i], sampling_div_2N)
                    integral += 4 * stage_cost_2(x_mid)
                    x_traj.append(dynamics_dt_cs(x_traj[-1], u_seq[i], sampling_div_N))
                    integral += stage_cost_2(x_traj[-1])
                    cost_total += integral / (6 * N)
                cost_total += 0.1 * u_seq[i]**2
            cost_total += terminal_cost(x_traj[-1])

    elif method == 3:
        u_seq = cs.MX.sym("u", T*N_P)  # Sequence of all control inputs
        # Define state trajectory and cost
        x_traj = [x0]
        cost_total = 0
    #########################################################
    # Define state trajectory and cost
    """
    x_traj = [x0]
    cost_total = 0
    for t in range(T):
        x_next = dynamics_dt_cs(x_traj[-1], u_seq[t], sampling_time)
        x_traj.append(x_next)
        cost_total += stage_cost(x_traj[-1], u_seq[t])

    # Adding terminal cost
    cost_total += terminal_cost(x_traj[-1])
    """
    #########################################################
    # Define the optimization problem
    opt_variables = cs.vertcat(u_seq)
    objective = cost_total
    constraints = [x_traj[0] - x0]

    # Set up NLP problem
    nlp = {'x': opt_variables, 'f': objective, 'g': cs.vertcat(*constraints), 'p': x0}
    solver = cs.nlpsol('solver', 'ipopt', nlp)

    # Initial state and simulation setup
    initial_state = cs.DM([0, cs.pi, 0, 0])
    current_state = initial_state
    u = cs.DM(0.0)
    optimal_u_seq = cs.DM.zeros(T) # Initial guess for the control input
    state_trajectory = [current_state]
    input_trajectory = [u]
    time_points = [0.0]

    # Run the simulation
    for k in range(num_steps):
        if k % int(sampling_time / simulation_period) == 0:
            # Solve the MPC optimization problem at the sampling time intervals
            solution = solver(x0=optimal_u_seq, p=current_state, ubg=0,lbg=0, lbx=[-20]*T, ubx=[20]*T)
            optimal_u_seq = solution['x']
            u = optimal_u_seq[0]  # Use the first control input from the sequence
        else:
            u = u  # Keep using the same control input until the next MPC update

        # Propagate the dynamics using the current control input and simulation period
        next_state = dynamics_dt_cs(current_state, u, simulation_period)
        current_state = cs.DM(next_state)
        
        # Record the state and time
        state_trajectory.append(current_state)
        input_trajectory.append(u)
        time_points.append((k + 1) * simulation_period)

    # Convert state trajectory to a NumPy array for plotting
    return np.array(state_trajectory).squeeze(), np.array(input_trajectory).squeeze(), time_points

# Simulate for both cost functions
state_trajectory_1, input_trajectory_1, time_points = simulate_mpc(method=1)
state_trajectory_2, input_trajectory_2, _ = simulate_mpc(method=2)

# Plot the results
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(10, 8))

# Plot cart position
plt.subplot(4, 1, 1)
plt.plot(time_points, state_trajectory_1[:, 0], label='NMPC')
plt.plot(time_points, state_trajectory_2[:, 0], label='Lifted NMPC', linestyle='--')
plt.ylabel('Cart Position')
plt.xlim(left=0, right=simulation_time)
plt.legend()
plt.grid()

# Plot pole angle
plt.subplot(4, 1, 2)
plt.plot(time_points, state_trajectory_1[:, 1], label='NMPC')
plt.plot(time_points, state_trajectory_2[:, 1], label='Lifted NMPC', linestyle='--')
plt.ylabel('Pole Angle')
plt.xlim(left=0, right=simulation_time)
plt.grid()

# Plot cart velocity
plt.subplot(4, 1, 3)
plt.plot(time_points, state_trajectory_1[:, 2], label='NMPC')
plt.plot(time_points, state_trajectory_2[:, 2], label='Lifted NMPC', linestyle='--')
plt.ylabel('Cart Velocity')
plt.xlim(left=0, right=simulation_time)
plt.grid()

# Plot pole angular velocity
plt.subplot(4, 1, 4)
plt.plot(time_points, state_trajectory_1[:, 3], label='NMPC')
plt.plot(time_points, state_trajectory_2[:, 3], label='Lifted NMPC', linestyle='--')
plt.ylabel('Pole Angular Velocity')
plt.xlim(left=0, right=simulation_time)
plt.xlabel('Time (s)')
plt.grid()

plt.tight_layout()

# Plot Control Input (u)
plt.figure(figsize=(8, 6))
plt.plot(time_points, input_trajectory_1, label='Control Input')
plt.plot(time_points, input_trajectory_2, label='Control Input 2', linestyle='--')
plt.ylabel('Control Input (u)')
plt.xlim(left=0, right=simulation_time)
plt.xlabel('Time (s)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
