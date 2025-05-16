import casadi.casadi as cs
import opengen as og

# Physical parameters
gravity_acceleration = 9.8
length = 1.0
mass_cart = 1.0
mass_pole = 0.2
dim = 4

# Time parameters
horizon_time = 1.0
################## Change these parameters ##################
sampling_time = 0.1 # 0.1, 0.25, 0.5
#############################################################
T = int(horizon_time / sampling_time)

def state_constraints_alm(x, u):
    global sampling_time, T
    x_min = -7.0
    x_max = 7.0
    x_dot_min = -15.0
    x_dot_max = 15.0
    c_min = [x_min, x_dot_min]*T
    c_max = [x_max, x_dot_max]*T
    f1 = []
    x_t = x
    
    for i in range(T):
        f1 = cs.vertcat(f1, x_t[0])
        f1 = cs.vertcat(f1, x_t[2])
        x_t = dynamics_dt(x_t, u[i], sampling_time)

    C = og.constraints.Rectangle(c_min, c_max)
    return f1, C

def state_constraints_pm(x, u, bounds):
    global sampling_time, T
    # bounds = [x_min, x_max, x_dot_min, x_dot_max]
    x_min = bounds[0]
    x_max = bounds[1]
    x_dot_min = bounds[2]
    x_dot_max = bounds[3]
    f2 = []
    x_t = x
    for i in range(T):
        f2 = cs.vertcat(f2, cs.fmax(0, x_t[0] - x_max))
        f2 = cs.vertcat(f2, cs.fmax(0, x_min - x_t[0]))
        f2 = cs.vertcat(f2, cs.fmax(0, x_t[2] - x_dot_max))
        f2 = cs.vertcat(f2, cs.fmax(0, x_dot_min - x_t[2]))
        x_t = dynamics_dt(x_t, u[i], sampling_time)

    return f2

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

def dynamics_dt(x, u, h):
    k1 = dynamics_ct(x, u)

    x_k2 = [x[i] + 0.5 * h * k1[i] for i in range(dim)]
    k2 = dynamics_ct(x_k2, u)

    x_k3 = [x[i] + 0.5 * h * k2[i] for i in range(dim)]
    k3 = dynamics_ct(x_k3, u)

    x_k4 = [x[i] + h * k3[i] for i in range(dim)]
    k4 = dynamics_ct(x_k4, u)

    x_next = [x[i] + (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(dim)]
    return x_next

def stage_cost(x, u, Q, R):
    cost = Q[0]*x[0]**2 + Q[1]*x[1]**2 + Q[2]*x[2]**2 + Q[3]*x[3]**2 + R*u**2
    return cost

def terminal_cost(x, Q):
    cost = Q[0]*x[0]**2 + Q[1]*x[1]**2 + Q[2]*x[2]**2 + Q[3]*x[3]**2
    return cost

u_seq = cs.MX.sym("u", T)  # sequence of all u's
x0 = cs.MX.sym("x0", dim)   # initial state
Q = cs.MX.sym("Q", dim)
Qt = cs.MX.sym("Qt", dim)
R = cs.MX.sym("R", 1)
state_bounds = cs.MX.sym("state_bounds", 4)  # [x_min, x_max, x_dot_min, x_dot_max]

x_t = x0
total_cost = 0
for t in range(0, T):
    total_cost += stage_cost(x_t, u_seq[t], Q, R)  # update cost
    x_t = dynamics_dt(x_t, u_seq[t], sampling_time)         # update state

total_cost += terminal_cost(x_t, Qt)  # terminal cost

optimization_variables = []
optimization_parameters = []

optimization_variables += [u_seq]
optimization_parameters += [x0]
optimization_parameters += [Q]
optimization_parameters += [Qt]
optimization_parameters += [R]
optimization_parameters += [state_bounds]

optimization_variables = cs.vertcat(*optimization_variables)
optimization_parameters = cs.vertcat(*optimization_parameters)

umin = [-20] * T
umax = [20] * T

bounds = og.constraints.Rectangle(umin, umax)
#f1, C = state_constraints_alm(x0, u_seq)
f2 = state_constraints_pm(x0, u_seq, state_bounds)

problem = og.builder.Problem(optimization_variables,
                             optimization_parameters,
                             total_cost) \
    .with_constraints(bounds) \
    .with_penalty_constraints(f2) 
    #.with_aug_lagrangian_constraints(f1, C)

build_dir_name = "python_cartpole_"+str(sampling_time)
build_config = og.config.BuildConfiguration()  \
    .with_build_directory(build_dir_name.replace(".", "_"))      \
    .with_tcp_interface_config()

opt_name = "cartpole_"+str(sampling_time)
meta = og.config.OptimizerMeta().with_optimizer_name(opt_name.replace(".", "_"))

solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-6)\
    .with_initial_tolerance(1e-6) \
    .with_max_duration_micros(450000)

builder = og.builder.OpEnOptimizerBuilder(problem, meta,
                                          build_config, solver_config)
builder.build()