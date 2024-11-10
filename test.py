import numpy as np

sampling_time_control = 0.05
################### Change this parameter ###################
sampler_period = 0.1 # 0.1, 0.25, 0.5
#############################################################

N_P = int(sampler_period/sampling_time_control) # 2, 5, 10
N_P = 1
horizon_time = 1.0
T = int(horizon_time / sampler_period)

gravity_acceleration = 9.8
length = 1.0
mass_cart = 1.0
mass_pole = 0.2

dim = 4
sampling_time_sim = 0.001

simulation_steps = int(50 / sampling_time_sim)

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

def stage_cost(x):
    cost = 5*x[0]**2 + 20*x[1]**2 + 0.02*x[2]**2 + 0.02*x[3]**2
    return cost

def terminal_cost(x):
    cost = 6*x[0]**2 + 30*x[1]**2 + 0.04*x[2]**2 + 0.04*x[3]**2
    return cost

def total_cost_1(x,u):
    x_t = x
    total_cost = 0
    sampling_time = sampler_period
    N = 10

    # some precompute parameters for Simpson's rule
    sampling_div_2N = sampling_time / (2 * N)
    sampling_div_N = sampling_time / N
    weight = sampling_time / (6 * N)

    u_index = 0
    for i in range(0,T):
        for j in range(0,N):
            # use Simpson's rule to approximate the integral
            integral = stage_cost(x_t)
            x_t_mid = dynamics_dt(x_t, u[i], sampling_div_2N)
            integral += 4 * stage_cost(x_t_mid)
            x_t = dynamics_dt(x_t, u[i], sampling_div_N)
            integral += stage_cost(x_t)
            total_cost += integral / (6*N)
            #print(u[i])
        total_cost += 0.1*u[i]**2
        u_index += 1

    total_cost += terminal_cost(x_t)
    print("u_index_1: ", u_index)
    return total_cost

def total_cost_2(x, u):
    x_t = x
    total_cost = 0
    sampling_time = sampler_period
    N = 10

    u_index = 0

    sampling_div_2N = sampling_time / (2 * N)
    sampling_div_N = sampling_time / N
    weight = sampling_time / (6 * N)

    for i in range(0,T):
        for j in range(0,N):
            # use Simpson's rule to approximate the integral
            integral = stage_cost(x_t)
            x_t_mid = dynamics_dt(x_t, u[u_index], sampling_div_2N)
            integral += 4 * stage_cost(x_t_mid)
            x_t = dynamics_dt(x_t, u[u_index], sampling_div_N)
            integral += stage_cost(x_t)
            total_cost += integral / (6*N)
            print(u[u_index])
            if (j % (N / N_P) == (N / N_P) - 1):
                total_cost += (0.1/N_P)*u[u_index]**2
                u_index += 1
    total_cost += terminal_cost(x_t)
    print("u_index_2: ", u_index) 
    return total_cost

u_test_1 = []
for i in range(T):
    u_test_1.append(i*2)
u_test_2 = []
for i in u_test_1:
    for j in range(N_P):
        u_test_2.append(i)
x_test = [0,np.pi,0,0]

cost_1 = total_cost_1(x_test, u_test_1)
cost_2 = total_cost_2(x_test, u_test_2)

print("Cost 1: ", cost_1)
print("Cost 2: ", cost_2)