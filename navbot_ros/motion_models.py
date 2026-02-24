# External Libraries
import math
import random

# This class is an example structure for implementing your motion model.
class MyMotionModel:

    # Constructor, change as you see fit.
    def __init__(self, initial_state = [0, 0, 0], last_encoder_count = 0):
        #State Structure: 
        #state[0] = x
        #state[1] = y
        #state[2] = theta (heading)
        self.state = initial_state
        self.last_encoder_count = last_encoder_count

        #constants
        self.ticks_per_meter = 3481.84
        self.meters_per_tick = 1 / self.ticks_per_meter
        self.distance_variance = 2.535e-05
        
        self.drivetrain_length = 0.140

        manual_correction = 0.8
        self.steering_gain = math.radians(-0.8343) * manual_correction
        self.steering_bias = math.radians(-0.8695)

        self.steering_variance = 0.0123878

    def get_distance_travelled(self, delta_encoder):
        s = self.meters_per_tick * delta_encoder
        return s

    def get_variance_distance_travelled(self):
        return self.distance_variance

    def get_linear_velocity(self, delta_encoder, delta_t):
        delta_s = self.get_distance_travelled(delta_encoder)
        return delta_s / delta_t

    def get_variance_linear_velocity(self, delta_t):
        sigma_s = self.get_variance_distance_travelled()
        sigma_v = ((1 / delta_t)**2) * sigma_s
        return sigma_v
    
    def get_steering_angle(self, steering_command):
        return self.steering_gain * steering_command + self.steering_bias
    
    def get_variance_steering_angle(self):
        return self.steering_variance

    def get_rotational_velocity(self, delta_encoder, steering_command, delta_t):
        steering_angle = self.get_steering_angle(steering_command)
        v = self.get_linear_velocity(delta_encoder, delta_t)
        d_theta = (1 / self.drivetrain_length) * v * math.tan(steering_angle)
        return d_theta
        
    def get_variance_rotational_velocity(self, delta_encoder, steering_command, delta_t):
        v  = self.get_linear_velocity(delta_encoder, delta_t)
        sigma_v = self.get_variance_linear_velocity(delta_t)

        phi = self.get_steering_angle(steering_command)
        sigma_phi = self.get_variance_steering_angle()

        domega_dv = delta_t * (1/self.drivetrain_length)*math.tan(phi)
        domega_dphi = delta_t * (v/self.drivetrain_length) * (1/math.cos(phi))**2
        
        return (domega_dv**2) * sigma_v + (domega_dphi**2) * sigma_phi

    def step_update(self, encoder_counts, steering_command, delta_t):
        # Add student code here
        old_state = self.state.copy()
        delta_encoder = encoder_counts - self.last_encoder_count
        delta_s = self.get_distance_travelled(delta_encoder)

        d_theta = self.get_rotational_velocity(delta_encoder, steering_command, delta_t)

        #theta
        theta = old_state[2] + 0.5 * d_theta * delta_t

        new_state = old_state

        #x
        new_state[0] += delta_s * math.cos(theta)
        #y
        new_state[1] += delta_s * math.sin(theta)
        #theta
        new_state[2] = theta

        self.state = new_state

        self.last_encoder_count = encoder_counts

        return self.state
    
    # This is a great tool to take in data from a trial and iterate over the data to create 
    # a robot trajectory in the global frame, using your motion model.
    def traj_propagation(self, time_list, encoder_count_list, steering_angle_list):
        x_list = [self.state[0]]
        y_list = [self.state[1]]
        theta_list = [self.state[2]]
        self.last_encoder_count = encoder_count_list[0]
        for i in range(1, len(encoder_count_list)):
            delta_t = time_list[i] - time_list[i-1]
            new_state = self.step_update(encoder_count_list[i], steering_angle_list[i], delta_t)
            x_list.append(new_state[0])
            y_list.append(new_state[1])
            theta_list.append(new_state[2])

        return x_list, y_list, theta_list
    

    # Coming soon
    def generate_simulated_traj(self, duration):
        delta_t = 0.1
        t_list = []
        x_list = []
        y_list = []
        theta_list = []
        t = 0
        encoder_counts = 0
        while t < duration:

            t += delta_t 
        return t_list, x_list, y_list, theta_list
            