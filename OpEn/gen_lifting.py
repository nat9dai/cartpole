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
sampling_time = 0.25 # 0.1, 0.25, 0.5
approach = 1 # 1, 2
#########################################################
N = 10
T = int(horizon_time / sampling_time)

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
    #cost = 5*x[0]**2 + 20*x[1]**2 + 0.02*x[2]**2 + 0.02*x[3]**2 + 0.1*u**2
    cost = Q[0]*x[0]**2 + Q[1]*x[1]**2 + Q[2]*x[2]**2 + Q[3]*x[3]**2 + R*u**2
    return cost

def stage_cost_2(x, Q):
    #cost = 5*x[0]**2 + 20*x[1]**2 + 0.02*x[2]**2 + 0.02*x[3]**2
    cost = Q[0]*x[0]**2 + Q[1]*x[1]**2 + Q[2]*x[2]**2 + Q[3]*x[3]**2
    return cost

def terminal_cost(x, Q):
    #cost = 6*x[0]**2 + 30*x[1]**2 + 0.04*x[2]**2 + 0.04*x[3]**2
    cost = Q[0]*x[0]**2 + Q[1]*x[1]**2 + Q[2]*x[2]**2 + Q[3]*x[3]**2
    return cost

u_seq = cs.MX.sym("u", T)  # sequence of all u's
x0 = cs.MX.sym("x0", dim)   # initial state
Q = cs.MX.sym("Q", dim)
Qt = cs.MX.sym("Qt", dim)
R = cs.MX.sym("R", 1)

x_t = x0
total_cost = 0

# some precompute parameters for Simpson's rule
sampling_div_2N = sampling_time / (2 * N)
sampling_div_N = sampling_time / N

if approach == 1:
    for i in range(0,T):
        for j in range(0,N):
            # use Simpson's rule to approximate the integral
            integral = stage_cost_2(x_t, Q)
            x_t_mid = dynamics_dt(x_t, u_seq[i], sampling_div_2N)
            integral += 4 * stage_cost_2(x_t_mid, Q)
            x_t = dynamics_dt(x_t, u_seq[i], sampling_div_N)
            integral += stage_cost_2(x_t, Q)
            total_cost += integral / (6*N)
        total_cost += R*u_seq[i]**2

    total_cost += terminal_cost(x_t, Qt)

# No significant differece between approach 1 and 2
elif approach == 2:
    for i in range(0,T):
        for j in range(0,N):
            # use Simpson's rule to approximate the integral
            integral = stage_cost(x_t, u_seq[i], Q, R)
            x_t_mid = dynamics_dt(x_t, u_seq[i], sampling_div_2N)
            integral += 4 * stage_cost(x_t_mid, u_seq[i], Q, R)
            x_t = dynamics_dt(x_t, u_seq[i], sampling_div_N)
            integral += stage_cost(x_t, u_seq[i], Q, R)
            total_cost += integral / (6*N)

    total_cost += terminal_cost(x_t, Qt)

optimization_variables = []
optimization_parameters = []

optimization_variables += [u_seq]
optimization_parameters += [x0]
optimization_parameters += [Q]
optimization_parameters += [Qt]
optimization_parameters += [R]

optimization_variables = cs.vertcat(*optimization_variables)
optimization_parameters = cs.vertcat(*optimization_parameters)

umin = [-20] * T
umax = [20] * T

bounds = og.constraints.Rectangle(umin, umax)

problem = og.builder.Problem(optimization_variables,
                             optimization_parameters,
                             total_cost) \
    .with_constraints(bounds)

build_dir_name = "python_cartpole_lifting_"+str(sampling_time)
build_config = og.config.BuildConfiguration()  \
    .with_build_directory(build_dir_name.replace(".", "_"))      \
    .with_tcp_interface_config()

opt_name = "cartpole_lifting_"+str(sampling_time)
meta = og.config.OptimizerMeta().with_optimizer_name(opt_name.replace(".", "_"))

solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-6)\
    .with_initial_tolerance(1e-6)

builder = og.builder.OpEnOptimizerBuilder(problem, meta,
                                          build_config, solver_config)
builder.build()