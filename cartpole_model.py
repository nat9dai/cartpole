import casadi as cs
import numpy as np


class CartPoleModel:
    """CartPole dynamics model with continuous and discrete-time representations."""

    def __init__(self, gravity_acceleration=9.8, length=1.0, mass_cart=1.0, mass_pole=0.2):
        self.gravity_acceleration = gravity_acceleration
        self.length = length
        self.mass_cart = mass_cart
        self.mass_pole = mass_pole
        self.dim = 4

    def dynamics_ct(self, x, u):
        """Continuous-time dynamics of the CartPole system.

        Args:
            x: State vector [position, angle, velocity, angular_velocity]
            u: Control input (force on cart)

        Returns:
            State derivative [dx1, dx2, dx3, dx4]
        """
        dx1 = x[2]
        dx2 = x[3]
        dx3 = (-self.mass_pole * self.length * cs.sin(x[1]) * x[3]**2
               + self.mass_pole * self.gravity_acceleration * cs.sin(x[1]) * cs.cos(x[1])
               + u) / (self.mass_cart + self.mass_pole * cs.sin(x[1])**2)
        dx4 = (-self.mass_pole * self.length * cs.sin(x[1]) * cs.cos(x[1]) * x[3]**2
               + (self.mass_cart + self.mass_pole) * self.gravity_acceleration * cs.sin(x[1])
               + u * cs.cos(x[1])) / (self.length * (self.mass_cart + self.mass_pole * cs.sin(x[1])**2))
        return [dx1, dx2, dx3, dx4]

    def dynamics_dt(self, x, u, sampling_time):
        """Discrete-time dynamics using RK4 integration.

        Args:
            x: Current state vector
            u: Control input
            sampling_time: Integration time step

        Returns:
            Next state vector
        """
        k1 = self.dynamics_ct(x, u)

        x_k2 = [x[i] + 0.5 * sampling_time * k1[i] for i in range(self.dim)]
        k2 = self.dynamics_ct(x_k2, u)

        x_k3 = [x[i] + 0.5 * sampling_time * k2[i] for i in range(self.dim)]
        k3 = self.dynamics_ct(x_k3, u)

        x_k4 = [x[i] + sampling_time * k3[i] for i in range(self.dim)]
        k4 = self.dynamics_ct(x_k4, u)

        x_next = [x[i] + (sampling_time / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
                  for i in range(self.dim)]
        return x_next
