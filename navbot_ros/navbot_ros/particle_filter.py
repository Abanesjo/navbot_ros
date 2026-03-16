# External libraries
import copy
import os
import math
import numpy as np
import random

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    distance_transform_edt = None

try:
    import yaml
except ImportError:
    yaml = None

try:
    from ament_index_python.packages import (
        PackageNotFoundError,
        get_package_share_directory,
    )
except ImportError:
    PackageNotFoundError = None
    get_package_share_directory = None

# Local libraries
from . import parameters
from .motion_models import MyMotionModel


# Local defaults for the distance-field map.
DEFAULT_MAP_RESOLUTION = 0.025
DEFAULT_MAP_ORIGIN = [-6.1, -3.69]
DEFAULT_OCCUPIED_THRESH = 0.65

# Likelihood-field tuning
LIKELIHOOD_SIGMA = math.sqrt(max(parameters.distance_variance, 1e-9))
MIN_VALID_LIDAR_RANGE = 0.02
MAX_USEFUL_LIDAR_RANGE = 5.0
LIDAR_BEAM_STRIDE = 4
MIN_PARTICLE_WEIGHT = 1e-12
RESAMPLE_EFFECTIVE_RATIO = 0.5
RANDOM_PARTICLE_FRACTION = 0.05
ROUGHEN_X_STD = 0.02
ROUGHEN_Y_STD = 0.02
ROUGHEN_THETA_STD = math.radians(3.0)

PF_MODE_BASELINE = "baseline"
PF_MODE_VARIANT = "variant"

# Helper function to make sure all angles are between -pi and pi
def angle_wrap(angle):
    while angle > math.pi:
        angle -= 2*math.pi
    while angle < -math.pi:
        angle += 2*math.pi
    return angle


def compute_distance_field(obstacle_mask, resolution):
    if distance_transform_edt is not None:
        return distance_transform_edt(~obstacle_mask) * resolution

    if cv2 is not None:
        free_space = (~obstacle_mask).astype(np.uint8)
        return cv2.distanceTransform(free_space, cv2.DIST_L2, 5) * resolution

    obstacle_indices = np.argwhere(obstacle_mask)
    if obstacle_indices.shape[0] == 0:
        return np.full(obstacle_mask.shape, np.inf, dtype=float)

    distance_field = np.zeros(obstacle_mask.shape, dtype=float)
    for row_index in range(obstacle_mask.shape[0]):
        row_cols = np.arange(obstacle_mask.shape[1], dtype=float)
        row_points = np.column_stack((np.full(obstacle_mask.shape[1], row_index, dtype=float), row_cols))
        squared_distances = (
            (row_points[:, None, 0] - obstacle_indices[None, :, 0]) ** 2
            + (row_points[:, None, 1] - obstacle_indices[None, :, 1]) ** 2
        )
        distance_field[row_index, :] = np.sqrt(np.min(squared_distances, axis=1)) * resolution
    return distance_field

# Helper class to store and manipulate your states.
class State:

    # Constructor
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    # Get the euclidean distance between 2 states
    def distance_to(self, other_state):
        return math.sqrt(math.pow(self.x - other_state.x, 2) + math.pow(self.y - other_state.y, 2))

    # Get the distance squared between two states
    def distance_to_squared(self, other_state):
        return math.pow(self.x - other_state.x, 2) + math.pow(self.y - other_state.y, 2)

    # return a deep copy of the state.
    def deepcopy(self):
        return copy.deepcopy(self)

    # Print the state
    def print(self):
        print("State: ",self.x, self.y, self.theta)


# Class to store walls as objects (specifically when represented as line segments in a 2D map.)
class Wall:

    # Constructor
    def __init__(self, wall_corners):
        self.corner1 = State(wall_corners[0], wall_corners[1], 0)
        self.corner2 = State(wall_corners[2], wall_corners[3], 0)
        self.corner1_mm = State(wall_corners[0] * 1000, wall_corners[1] * 1000, 0)
        self.corner2_mm = State(wall_corners[2] * 1000, wall_corners[3] * 1000, 0)

        self.m = (wall_corners[3] - wall_corners[1])/(0.0001 + wall_corners[2] -  wall_corners[0])
        self.b = wall_corners[3] - self.m * wall_corners[2]
        self.b_mm =  wall_corners[3] * 1000 - self.m * wall_corners[2] * 1000
        self.length = self.corner1.distance_to(self.corner2)
        self.length_mm_squared = self.corner1_mm.distance_to_squared(self.corner2_mm)

        if self.m > 1000:
            self.vertical = True
        else:
            self.vertical = False
        if abs(self.m) < 0.1:
            self.horizontal = True
        else:
            self.horizontal = False


# A class to store 2D maps
class Map:
    def __init__(self, wall_corner_list):
        self.wall_list = []
        for wall_corners in wall_corner_list:
            if len(wall_corners) == 4:
                self.wall_list.append(Wall(wall_corners))

        self._load_distance_field_map()

        if len(self.wall_list) > 0:
            min_x = 999999
            max_x = -99999
            min_y = 999999
            max_y = -99999
            for wall in self.wall_list:
                min_x = min(min_x, min(wall.corner1.x, wall.corner2.x))
                max_x = max(max_x, max(wall.corner1.x, wall.corner2.x))
                min_y = min(min_y, min(wall.corner1.y, wall.corner2.y))
                max_y = max(max_y, max(wall.corner1.y, wall.corner2.y))
            border = 0.5
            self.plot_range = [min_x - border, max_x + border, min_y - border, max_y + border]
            self.particle_range = [min_x, max_x, min_y, max_y]
        else:
            border = 0.5
            self.plot_range = [
                self.map_min_x - border,
                self.map_max_x + border,
                self.map_min_y - border,
                self.map_max_y + border,
            ]
            self.particle_range = [
                self.map_min_x,
                self.map_max_x,
                self.map_min_y,
                self.map_max_y,
            ]

    def _load_distance_field_map(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_map_dirs = [os.path.normpath(os.path.join(file_dir, "..", "map"))]
        if get_package_share_directory is not None:
            try:
                package_share_dir = get_package_share_directory("navbot_ros")
                candidate_map_dirs.append(os.path.join(package_share_dir, "map"))
            except (PackageNotFoundError, LookupError):
                pass

        map_dir = candidate_map_dirs[0]
        map_yaml_path = os.path.join(map_dir, "map_processed.yaml")
        processed_map_path = os.path.join(map_dir, "map_processed.png")
        for candidate_dir in candidate_map_dirs:
            candidate_processed_map_path = os.path.join(candidate_dir, "map_processed.png")
            if os.path.exists(candidate_processed_map_path):
                map_dir = candidate_dir
                map_yaml_path = os.path.join(map_dir, "map_processed.yaml")
                processed_map_path = candidate_processed_map_path
                break

        self.resolution = DEFAULT_MAP_RESOLUTION
        self.origin = list(DEFAULT_MAP_ORIGIN)
        occupied_thresh = DEFAULT_OCCUPIED_THRESH

        if yaml is not None and os.path.exists(map_yaml_path):
            with open(map_yaml_path, "r", encoding="utf-8") as map_yaml_file:
                map_metadata = yaml.safe_load(map_yaml_file) or {}
            self.resolution = float(map_metadata.get("resolution", self.resolution))
            yaml_origin = map_metadata.get("origin", self.origin)
            if isinstance(yaml_origin, list) and len(yaml_origin) >= 2:
                self.origin = [float(yaml_origin[0]), float(yaml_origin[1])]
            occupied_thresh = float(map_metadata.get("occupied_thresh", occupied_thresh))

        if cv2 is not None:
            self.map_arr = cv2.imread(processed_map_path, cv2.IMREAD_GRAYSCALE)
        else:
            try:
                from matplotlib import pyplot as plt
                self.map_arr = plt.imread(processed_map_path)
                if self.map_arr.ndim == 3:
                    self.map_arr = self.map_arr[..., 0]
                if self.map_arr.dtype != np.uint8:
                    self.map_arr = np.clip(self.map_arr * 255.0, 0, 255).astype(np.uint8)
            except Exception:
                self.map_arr = None

        if self.map_arr is None:
            searched_paths = [
                os.path.join(path, "map_processed.png")
                for path in candidate_map_dirs
            ]
            raise FileNotFoundError(
                "Could not load processed map image. Searched: "
                + ", ".join(searched_paths)
            )

        self.height, self.width = self.map_arr.shape
        occ_pixel_thresh = int((1.0 - occupied_thresh) * 255)
        obstacle_mask = self.map_arr <= occ_pixel_thresh
        self.distance_field = compute_distance_field(obstacle_mask, self.resolution)

        self.map_min_x = self.origin[0]
        self.map_max_x = self.origin[0] + self.width * self.resolution
        self.map_min_y = self.origin[1]
        self.map_max_y = self.origin[1] + self.height * self.resolution

    def world_to_pixel(self, x, y):
        col = int(math.floor((x - self.origin[0]) / self.resolution))
        row = int(math.floor(self.height - (y - self.origin[1]) / self.resolution - 1))
        in_bounds = (0 <= col < self.width) and (0 <= row < self.height)
        col = min(max(col, 0), self.width - 1)
        row = min(max(row, 0), self.height - 1)
        return col, row, in_bounds

    def distance_to_nearest_obstacle(self, x, y):
        col, row, in_bounds = self.world_to_pixel(x, y)
        if not in_bounds:
            return float("inf")
        return float(self.distance_field[row, col])

    # Function to calculate the distance between any state and its closest wall, accounting for direction of the state.
    def closest_distance_to_walls(self, state):
        closest_distance = 999999999999
        for wall in self.wall_list:
            closest_distance = self.get_distance_to_wall(state, wall, closest_distance)

        return closest_distance

    # Function to get distance to a wall from a state, in the direction of the state's theta angle.
    def get_distance_to_wall(self, state, wall, closest_distance):
        ray_dx = math.cos(state.theta)
        ray_dy = math.sin(state.theta)

        seg_dx = wall.corner2.x - wall.corner1.x
        seg_dy = wall.corner2.y - wall.corner1.y
        det = ray_dx * seg_dy - ray_dy * seg_dx
        if abs(det) < 1e-9:
            return closest_distance

        rel_x = wall.corner1.x - state.x
        rel_y = wall.corner1.y - state.y
        distance_along_ray = (rel_x * seg_dy - rel_y * seg_dx) / det
        segment_fraction = (rel_x * ray_dy - rel_y * ray_dx) / det

        if distance_along_ray < 0:
            return closest_distance
        if segment_fraction < 0 or segment_fraction > 1:
            return closest_distance

        return min(closest_distance, distance_along_ray)


# Class to hold a particle
class Particle:

    def __init__(self, rng=None):
        self.state = State(0, 0, 0)
        self.weight = 1
        self.motion_model = MyMotionModel()
        self.rng = rng if rng is not None else random.Random()

    # Function to create a new random particle state within a range
    def randomize_uniformly(self, xy_range):
        x = self.rng.uniform(xy_range[0], xy_range[1])
        y = self.rng.uniform(xy_range[2], xy_range[3])
        theta = self.rng.uniform(-math.pi, math.pi)
        self.state = State(x, y, theta)
        self.weight = 1

    # Function to create a new random particle state with a normal distribution
    def randomize_around_initial_state(self, initial_state, state_stdev):
        x = self.rng.gauss(initial_state.x, state_stdev.x)
        y = self.rng.gauss(initial_state.y, state_stdev.y)
        theta = angle_wrap(self.rng.gauss(initial_state.theta, state_stdev.theta))
        self.state = State(x, y, theta)
        self.weight = 1

    # Function to take a particle and "randomly" propagate it forward according to a motion model.
    def propagate_state(self, last_state, delta_encoder_counts, steering, delta_t):
        delta_s_nominal = self.motion_model.get_distance_travelled(delta_encoder_counts)
        sigma_s = math.sqrt(max(self.motion_model.get_variance_distance_travelled(), 1e-12))
        phi_nominal = self.motion_model.get_steering_angle(steering)
        sigma_phi = math.sqrt(max(self.motion_model.get_variance_steering_angle(), 1e-12))

        delta_s_sampled = self.rng.gauss(delta_s_nominal, sigma_s)
        phi_sampled = self.rng.gauss(phi_nominal, sigma_phi)
        delta_theta = (delta_s_sampled / self.motion_model.drivetrain_length) * math.tan(phi_sampled)
        theta_mid = last_state.theta + 0.5 * delta_theta

        x = last_state.x + delta_s_sampled * math.cos(theta_mid)
        y = last_state.y + delta_s_sampled * math.sin(theta_mid)
        theta = angle_wrap(last_state.theta + delta_theta)
        self.state = State(x, y, theta)

    # Function to determine a particle's weight based on how well the lidar measurement matches up with the map.
    def calculate_weight(self, lidar_signal, map):
        if not (map.map_min_x <= self.state.x <= map.map_max_x and map.map_min_y <= self.state.y <= map.map_max_y):
            self.weight = MIN_PARTICLE_WEIGHT
            return

        log_weight = 0.0
        valid_beam_count = 0

        for i in range(0, len(lidar_signal.angles), LIDAR_BEAM_STRIDE):
            distance = lidar_signal.convert_hardware_distance(lidar_signal.distances[i])
            if not np.isfinite(distance):
                continue
            if distance < MIN_VALID_LIDAR_RANGE or distance > MAX_USEFUL_LIDAR_RANGE:
                continue

            beam_angle = lidar_signal.convert_hardware_angle(lidar_signal.angles[i])
            beam_theta = angle_wrap(self.state.theta + beam_angle)
            x_hit = self.state.x + distance * math.cos(beam_theta)
            y_hit = self.state.y + distance * math.sin(beam_theta)

            obstacle_distance = map.distance_to_nearest_obstacle(x_hit, y_hit)
            if not np.isfinite(obstacle_distance):
                log_weight += math.log(MIN_PARTICLE_WEIGHT)
            else:
                log_weight += -(obstacle_distance ** 2) / (2.0 * LIKELIHOOD_SIGMA ** 2)
            valid_beam_count += 1

        if valid_beam_count == 0:
            self.weight = MIN_PARTICLE_WEIGHT
            return

        avg_log_weight = log_weight / valid_beam_count
        self.weight = max(MIN_PARTICLE_WEIGHT, math.exp(avg_log_weight))

    # Return the normal distribution function output.
    def gaussian(self, expected_distance, distance):
        return math.exp(-math.pow(expected_distance - distance, 2)/ 2 / parameters.distance_variance)

    # Deep copy the particle
    def deepcopy(self):
        particle_copy = Particle(rng=self.rng)
        particle_copy.state = self.state.deepcopy()
        particle_copy.weight = self.weight
        return particle_copy

    # Print the particle
    def print(self):
        print("Particle: ", self.state.x, self.state.y, self.state.theta, " w: ", self.weight)


# This class holds the collection of particles.
class ParticleSet:

    # Constructor, which calls the known start or unknown start initialization.
    def __init__(
        self,
        num_particles,
        xy_range,
        initial_state,
        state_stdev,
        known_start_state,
        rng=None,
    ):
        self.num_particles = num_particles
        self.rng = rng if rng is not None else random.Random()
        self.particle_list = []
        if known_start_state:
            self.generate_initial_state_particles(initial_state, state_stdev)
        else:
            self.generate_uniform_random_particles(xy_range)
        self.mean_state = State(0, 0, 0)
        self.update_mean_state()

    # Function to reset particles at random locations in the workspace.
    def generate_uniform_random_particles(self, xy_range):
        for i in range(self.num_particles):
            random_particle = Particle(rng=self.rng)
            random_particle.randomize_uniformly(xy_range)
            self.particle_list.append(random_particle)

    # Function to reset particles, normally distributed around the initial state.
    def generate_initial_state_particles(self, initial_state, state_stdev):
        for i in range(self.num_particles):
            random_particle = Particle(rng=self.rng)
            random_particle.randomize_around_initial_state(initial_state, state_stdev)
            self.particle_list.append(random_particle)

    # Effective sample size: lower values indicate stronger weight collapse.
    def effective_sample_size(self):
        if len(self.particle_list) == 0:
            return 0.0

        weights = np.array(
            [max(float(particle.weight), 0.0) for particle in self.particle_list],
            dtype=float,
        )
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            return 0.0
        normalized_weights = weights / weight_sum
        return float(1.0 / np.sum(normalized_weights ** 2))

    def roughen_particles(self, skip_indices=None):
        if skip_indices is None:
            skip_indices = set()
        for particle_index, particle in enumerate(self.particle_list):
            if particle_index in skip_indices:
                continue
            particle.state.x += self.rng.gauss(0.0, ROUGHEN_X_STD)
            particle.state.y += self.rng.gauss(0.0, ROUGHEN_Y_STD)
            particle.state.theta = angle_wrap(
                particle.state.theta + self.rng.gauss(0.0, ROUGHEN_THETA_STD)
            )

    # Function to resample the particle set using low-variance systematic resampling.
    def resample(self, xy_range, enable_random_injection=True, enable_roughening=True):
        if len(self.particle_list) == 0:
            return

        weight_sum = sum(particle.weight for particle in self.particle_list)
        if weight_sum <= 0:
            uniform_weight = 1.0 / len(self.particle_list)
            for particle in self.particle_list:
                particle.weight = uniform_weight
            return

        normalized_weights = [particle.weight / weight_sum for particle in self.particle_list]
        step = 1.0 / self.num_particles
        position = self.rng.uniform(0.0, step)
        cumulative_weight = normalized_weights[0]
        source_index = 0
        resampled_particles = []

        for particle_index in range(self.num_particles):
            target = position + particle_index * step
            while target > cumulative_weight and source_index < self.num_particles - 1:
                source_index += 1
                cumulative_weight += normalized_weights[source_index]

            particle_copy = self.particle_list[source_index].deepcopy()
            particle_copy.weight = step
            resampled_particles.append(particle_copy)

        injected_indices = set()
        num_to_inject = int(round(RANDOM_PARTICLE_FRACTION * self.num_particles))
        num_to_inject = max(0, min(self.num_particles, num_to_inject))
        if enable_random_injection and num_to_inject > 0:
            injected_indices = set(self.rng.sample(range(self.num_particles), num_to_inject))
            for injected_index in injected_indices:
                random_particle = Particle(rng=self.rng)
                random_particle.randomize_uniformly(xy_range)
                resampled_particles[injected_index] = random_particle

        self.particle_list = resampled_particles
        if enable_roughening:
            self.roughen_particles(skip_indices=injected_indices)

        uniform_weight = 1.0 / self.num_particles
        for particle in self.particle_list:
            particle.weight = uniform_weight

    # Calculate the mean state.
    def update_mean_state(self):
        if len(self.particle_list) == 0:
            self.mean_state.x = 0
            self.mean_state.y = 0
            self.mean_state.theta = 0
            return

        weights = np.array(
            [max(float(particle.weight), 0.0) for particle in self.particle_list],
            dtype=float,
        )
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            normalized_weights = np.full(len(self.particle_list), 1.0 / len(self.particle_list))
        else:
            normalized_weights = weights / weight_sum

        x_mean = 0.0
        y_mean = 0.0
        sin_mean = 0.0
        cos_mean = 0.0
        for weight, particle in zip(normalized_weights, self.particle_list):
            x_mean += weight * particle.state.x
            y_mean += weight * particle.state.y
            sin_mean += weight * math.sin(particle.state.theta)
            cos_mean += weight * math.cos(particle.state.theta)

        self.mean_state.x = x_mean
        self.mean_state.y = y_mean
        self.mean_state.theta = math.atan2(sin_mean, cos_mean)

    # Print the particle set. Useful for debugging.
    def print_particles(self):
        for particle in self.particle_list:
            particle.print()
        print()

# Class to hold the particle filter and its functions.
class ParticleFilter:

    # Constructor
    def __init__(
        self,
        num_particles,
        map,
        initial_state,
        state_stdev,
        known_start_state,
        encoder_counts_0,
        algorithm_mode=PF_MODE_VARIANT,
        rng=None,
    ):
        self.map = map
        self.rng = rng if rng is not None else random.Random()
        mode = str(algorithm_mode).strip().lower()
        if mode not in (PF_MODE_BASELINE, PF_MODE_VARIANT):
            mode = PF_MODE_VARIANT
        self.algorithm_mode = mode
        self.particle_set = ParticleSet(
            num_particles,
            map.particle_range,
            initial_state,
            state_stdev,
            known_start_state,
            rng=self.rng,
        )
        self.state_estimate = self.particle_set.mean_state
        self.state_estimate_list = []
        self.last_time = 0
        self.last_encoder_counts = encoder_counts_0
        self.motion_model = MyMotionModel(initial_state=[initial_state.x, initial_state.y, initial_state.theta], last_encoder_count=encoder_counts_0)

    # Update the states given new measurements
    def update(self, odometery_signal, measurement_signal, delta_t):
        self.prediction(odometery_signal, delta_t)
        if len(measurement_signal.angles)>0:
            self.correction(measurement_signal)
        self.particle_set.update_mean_state()
        self.state_estimate = self.particle_set.mean_state
        self.state_estimate_list.append(self.state_estimate.deepcopy())

    # Predict the current state from the last state.
    def prediction(self, odometry_signal, delta_t):
        delta_encoder_counts = odometry_signal[0] - self.last_encoder_counts
        steering = odometry_signal[1]
        for particle in self.particle_set.particle_list:
            previous_state = particle.state.deepcopy()
            particle.propagate_state(previous_state, delta_encoder_counts, steering, delta_t)
        self.last_encoder_counts = odometry_signal[0]
        return

    # Correct the predicted states.
    def correction(self, measurement_signal):
        weight_sum = 0.0
        for particle in self.particle_set.particle_list:
            particle.calculate_weight(measurement_signal, self.map)
            weight_sum += particle.weight

        if weight_sum <= 0:
            uniform_weight = 1.0 / max(len(self.particle_set.particle_list), 1)
            for particle in self.particle_set.particle_list:
                particle.weight = uniform_weight
        else:
            for particle in self.particle_set.particle_list:
                particle.weight = particle.weight / weight_sum

        if self.algorithm_mode == PF_MODE_BASELINE:
            # Baseline: resample every step, no injection/roughening
            self.particle_set.resample(
                self.map.particle_range,
                enable_random_injection=False,
                enable_roughening=False,
            )
        else:
            # Variant: adaptive resampling with diversity preservation
            n_eff = self.particle_set.effective_sample_size()
            resample_threshold = RESAMPLE_EFFECTIVE_RATIO * self.particle_set.num_particles
            if n_eff < resample_threshold:
                self.particle_set.resample(self.map.particle_range)

        self.particle_set.update_mean_state()
        self.state_estimate = self.particle_set.mean_state

    # Output to terminal the mean state.
    def print_state_estimate(self):
        print("Mean state: ", self.particle_set.mean_state.x, self.particle_set.mean_state.y, self.particle_set.mean_state.theta)
