import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import orient
from shapely.affinity import rotate, scale, translate
from shapely.ops import unary_union
from matplotlib.animation import FuncAnimation
import polygenerator
import warnings
import csv
import os
import random
import copy
import pickle

# Ignore the FutureWarning for Shapely affinity comparisons
warnings.filterwarnings("ignore", category=FutureWarning, module="shapely")

# Create a folder to save all files
output_folder = 'output_files'
os.makedirs(output_folder, exist_ok=True)

class PolygonObject:
    def __init__(self, polygon):
        self.polygon = polygon

    def draw(self, ax):
        x, y = self.polygon.exterior.xy
        self.patch = ax.fill(x, y, alpha=0.5, fc='gray', ec='black', label='Polygon')[0]
        
        if self.polygon.interiors:
            for interior in self.polygon.interiors:
                x, y = interior.xy
                ax.fill(x, y, alpha=0.5, fc='white', ec='black')

class SuctionCup:
    def __init__(self, num_chambers, radius=5, center=[0, 0], rotation_angle=0):
        self.num_chambers = num_chambers
        self.radius = radius  # in mm
        self.suction_cup_area = np.pi * (self.radius ** 2)
        self.center = np.array(center)
        self.velocity = np.array([0.0, 0.0])  # Initialize velocity
        self.rotation_angle = rotation_angle
        self.direction_vector = np.array([0, 0])
        self.chambers = self.create_chambers()

    def create_chambers(self):
        angles = np.linspace(0, 2 * np.pi, self.num_chambers + 1)
        chambers = []
        for i in range(self.num_chambers):
            theta = np.linspace(angles[i], angles[i + 1], 1000)
            x = np.append([self.center[0]], self.center[0] + np.cos(theta) * self.radius)
            y = np.append([self.center[1]], self.center[1] + np.sin(theta) * self.radius)
            chamber_points = list(zip(x, y))
            chamber_polygon = Polygon(chamber_points)
            chamber_polygon = orient(chamber_polygon, sign=1.0)
            chamber_polygon = rotate(chamber_polygon, self.rotation_angle, origin=self.center)
            chambers.append(chamber_polygon)
        return chambers

    def draw(self, ax):
        colors = plt.cm.get_cmap('hsv', self.num_chambers+1)
        self.patches = []
        for i, chamber in enumerate(self.chambers):
            x, y = chamber.exterior.xy
            patch = ax.fill(x, y, color=colors(i), alpha=0.5, label=f'Chamber {i + 1}')[0]
            ax.text((x[0] + x[500]) / 2, (y[0] + y[500]) / 2, f'Ch{i + 1}', ha='center', va='center', fontsize=8, color='black')
            self.patches.append(patch)

def generate_random_polygon(num_points=random.randint(5, 10), min_area=300):
    while True:
        polygon = polygenerator.random_polygon(num_points=num_points)
        polygon = Polygon(polygon)
        polygon = scale(polygon, xfact=30, yfact=30, origin=(0, 0))
        if polygon.area >= min_area:
            return polygon

def generate_random_hole(main_polygon, max_num_points=10, circle_probability=0.5):
    if random.random() < circle_probability:
        radius = random.uniform(0.1, 1)
        center = Point(random.uniform(main_polygon.bounds[0], main_polygon.bounds[2]),
                       random.uniform(main_polygon.bounds[1], main_polygon.bounds[3]))
        circle = center.buffer(radius)
        if main_polygon.contains(circle):
            return circle
    else:
        num_points = random.randint(3, max_num_points)
        while True:
            hole = polygenerator.random_polygon(num_points=num_points)
            hole = Polygon(hole)
            hole = scale(hole, xfact=1, yfact=1, origin=(0, 0))
            hole_centroid = Point(random.uniform(main_polygon.bounds[0], main_polygon.bounds[2]),
                                  random.uniform(main_polygon.bounds[1], main_polygon.bounds[3]))
            hole = translate(hole, xoff=hole_centroid.x - hole.centroid.x, yoff=hole_centroid.y - hole.centroid.y)
            if main_polygon.contains(hole):
                return hole

def add_random_holes(polygon, num_holes=random.randint(0, 5)):
    holes = []
    for _ in range(num_holes):
        hole = generate_random_hole(polygon)
        holes.append(hole)
    return polygon.difference(unary_union(holes))

def calculate_covered_area(polygon, chambers):
    covered_areas = [polygon.intersection(chamber).area for chamber in chambers]
    return covered_areas

def calculate_vacuum_pressures(covered_areas, suctioncup):
    a = 4.007e4
    b = 0.6895
    total_area = suctioncup.suction_cup_area * 1e-6  # Convert to m^2
    total_chamber_area = total_area / suctioncup.num_chambers
    
    r_total = np.sum(covered_areas) * 1e-6 / total_area  # Convert covered areas to m^2
    covered_areas = np.array(covered_areas) * 1e-6  # Ensure it's a NumPy array and convert to m^2
    r_chamber = covered_areas / total_chamber_area
    
    alpha = 0.5 * (r_chamber - r_total)
    P_m = a * total_area + 2 * b
    P_avg = 10 ** (a * np.sum(covered_areas) + b)
    
    vacuum_pressures = [10 ** (alpha_i * P_m) * P_avg ** (1 - alpha_i) for alpha_i in alpha]
    
    noise = np.random.normal(0, 6, len(vacuum_pressures))
    vacuum_pressures_with_noise = vacuum_pressures + noise
    
    return vacuum_pressures_with_noise

def calculate_unit_vectors(num_chambers, rotation_angle):
    return [np.array([np.cos(2 * np.pi / (num_chambers * 2) + 2 * np.pi * i / num_chambers + np.radians(rotation_angle)),
                      np.sin(2 * np.pi / (num_chambers * 2) + 2 * np.pi * i / num_chambers + np.radians(rotation_angle))])
            for i in range(num_chambers)]

def calculate_direction_vector(unit_vectors, vacuum_pressures):
    direction_vector = np.sum([vp * uv for vp, uv in zip(vacuum_pressures, unit_vectors)], axis=0)
    return direction_vector / np.linalg.norm(direction_vector)

def unit_direction_vector(polygon_obj, suction_cup):
    covered_areas = calculate_covered_area(polygon_obj.polygon, suction_cup.chambers)
    vacuum_pressures = calculate_vacuum_pressures(covered_areas, suction_cup)
    unit_vectors = calculate_unit_vectors(suction_cup.num_chambers, suction_cup.rotation_angle)
    return calculate_direction_vector(unit_vectors, vacuum_pressures)

def check_suction(polygon_obj, suction_cup):
    total_covered_area = sum([polygon_obj.polygon.intersection(chamber).area for chamber in suction_cup.chambers])
    tolerance = 5e-4
    return abs(total_covered_area - suction_cup.suction_cup_area) < tolerance

def plot_polygon_and_suction_cup(ax, polygon_obj, suction_cup):
    ax.cla()
    polygon_obj.draw(ax)
    suction_cup.draw(ax)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-5, 30)
    ax.set_ylim(-5, 30)

def haptic_search_suction_cup(i, ax, polygon_obj, suction_cup, step_size, haptic_path, controller, animation=False):
    if check_suction(polygon_obj, suction_cup):
        if animation:
            plot_polygon_and_suction_cup(ax, polygon_obj, suction_cup)
        return
    
    damping_factor = 0.9
    suction_cup.direction_vector = unit_direction_vector(polygon_obj, suction_cup)
    
    if controller == 'normal':
        suction_cup.center += step_size * suction_cup.direction_vector
    elif controller == 'yaw':
        suction_cup.center += step_size * suction_cup.direction_vector
        suction_cup.rotation_angle += 1
    elif controller == 'momentum':
        suction_cup.velocity = damping_factor * suction_cup.velocity + step_size * suction_cup.direction_vector
        suction_cup.center += suction_cup.velocity
    elif controller == 'yaw_momentum':
        suction_cup.velocity = damping_factor * suction_cup.velocity + step_size * suction_cup.direction_vector
        suction_cup.center += suction_cup.velocity
        suction_cup.rotation_angle += 1

    suction_cup.chambers = suction_cup.create_chambers()
    haptic_path.append(suction_cup.center.tolist())
    if animation:
        plot_polygon_and_suction_cup(ax, polygon_obj, suction_cup)

def save_animation(fig, axs, polygon_objs, suction_cups, step_size, haptic_paths, filename, output_folder, controller):
    def update(i):
        for ax, polygon_obj, suction_cup, haptic_path in zip(axs.flat, polygon_objs, suction_cups, haptic_paths):
            haptic_search_suction_cup(i, ax, polygon_obj, suction_cup, step_size, haptic_path, controller, animation=True)
    ani = FuncAnimation(fig, update, frames=100, repeat=False)
    ani.save(os.path.join(output_folder, filename), writer='pillow', fps=15)

def calculate_path_length(haptic_path):
    diffs = np.diff(haptic_path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    path_length = np.sum(segment_lengths)
    return path_length

def find_initial_suction_cup_position(polygon, radius):
    minx, miny, maxx, maxy = polygon.bounds
    attempts = 100
    for _ in range(attempts):
        center = [np.random.uniform(minx + radius, maxx - radius), np.random.uniform(miny + radius, maxy - radius)]
        suction_cup = Point(center).buffer(radius)
        if polygon.intersects(suction_cup):
            return center
    raise ValueError("Failed to find a valid initial position for the suction cup")

def save_path_length_data(filename, path_lengths, haptic_iterations, excluded_path_count, output_folder):
    with open(os.path.join(output_folder, filename), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Num Chambers', 'Path Length', 'Iteration'])
        for num_chambers, lengths in path_lengths.items():
           for length, iteration in zip(lengths, haptic_iterations[num_chambers]):
                writer.writerow([num_chambers, length, iteration])

def plot_and_save_haptic_paths(polygon_objs, haptic_paths, path_lengths, filename, num_chambers, row_col, output_folder):
    fig, axs = plt.subplots(row_col, row_col, figsize=(15, 15))
    for ax, polygon_obj, haptic_path, path_length in zip(axs.flat, polygon_objs, haptic_paths[num_chambers], path_lengths[num_chambers]):
        polygon_obj.draw(ax)
        haptic_path = np.array(haptic_path)
        ax.scatter(haptic_path[:, 0], haptic_path[:, 1], color='r', s=10)
        ax.set_title(f'Path Length: {path_length:.2f} mm', fontsize=8)
        ax.axis('off')
        ax.set_xlim(-5, 35)
        ax.set_ylim(-5, 35)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename))
    plt.close(fig)

def save_data(polygons, suction_cups, initial_positions, filename='data.p'):
    with open(os.path.join(output_folder, filename), 'wb') as f:
        pickle.dump((polygons, suction_cups, initial_positions), f)

def load_data(filename='data.p'):
    with open(os.path.join(output_folder, filename), 'rb') as f:
        polygons, suction_cups, initial_positions = pickle.load(f)
    return polygons, suction_cups, initial_positions

def main(args):
    num_chambers_list = args.num_chambers_list
    if args.controller == 'normal':
        step_size = 0.5
    elif args.controller == 'yaw':
        step_size = 0.5
    elif args.controller == 'momentum':
        step_size = 0.1
    elif args.controller == 'yaw_momentum':
        step_size = 0.1

    num_polygons = args.num_polygons
    row_col = int(np.sqrt(num_polygons))
    suction_cups_init = {num_chambers: [] for num_chambers in num_chambers_list}
    suction_cups = {num_chambers: [] for num_chambers in num_chambers_list}
    haptic_paths = {num_chambers: [] for num_chambers in num_chambers_list}
    haptic_iterations = {num_chambers: [] for num_chambers in num_chambers_list}
    haptic_iterations_thresh = {num_chambers: [] for num_chambers in num_chambers_list}
    path_lengths = {num_chambers: [] for num_chambers in num_chambers_list}
    path_lengths_thresh = {num_chambers: [] for num_chambers in num_chambers_list}
    excluded_path_count = {num_chambers: 0 for num_chambers in num_chambers_list}

    fig, axs = plt.subplots(row_col, row_col, figsize=(15, 15))
    
    if args.load_items:
        polygon_objs, suction_cups_init, initial_positions = load_data('init_data.p')
    else:
        polygons = [generate_random_polygon() for _ in range(num_polygons)]
        polygon_with_holes = [add_random_holes(polygon) for polygon in polygons]
        polygon_objs = [PolygonObject(polygon) for polygon in polygon_with_holes]
        initial_positions = [find_initial_suction_cup_position(polygon, 5) for polygon in polygons]

    for num_chambers in num_chambers_list:
        for polygon_obj, initial_position in zip(polygon_objs, initial_positions):
            if not args.load_items:
                suction_cup = SuctionCup(num_chambers, 5, initial_position, rotation_angle=random.randint(0, 360))
                suction_cups_init[num_chambers].append(copy.deepcopy(suction_cup))
            else:
                suction_cups = copy.deepcopy(suction_cups_init)
                suction_cup = suction_cups[num_chambers][polygon_objs.index(polygon_obj)]
            haptic_path = [suction_cup.center.tolist()]
            haptic_paths[num_chambers].append(haptic_path)

            for _ in range(100):
                haptic_search_suction_cup(_, axs.flat[0], polygon_obj, suction_cup, step_size, haptic_path, args.controller)

            path_length = calculate_path_length(np.array(haptic_path))
            haptic_iterations[num_chambers].append(len(haptic_path))
            path_lengths[num_chambers].append(path_length)
            if len(haptic_path) > 100 or len(haptic_path) < 2:
                excluded_path_count[num_chambers] += 1
            else:
                path_lengths_thresh[num_chambers].append(path_length)
                haptic_iterations_thresh[num_chambers].append(len(haptic_path))

        if args.animation:
            suction_cup_animation = copy.deepcopy(suction_cups_init)
            save_animation(fig, axs, polygon_objs, suction_cup_animation[num_chambers], step_size, haptic_paths[num_chambers], f'suction_cup_animation_{num_chambers}_chambers_'+ args.controller +'.gif', output_folder, args.controller)

        plot_and_save_haptic_paths(polygon_objs, haptic_paths, path_lengths, f'haptic_paths_{num_chambers}_chambers_'+ args.controller +'.png', num_chambers, row_col, output_folder)

    plt.tight_layout()
    plt.show()

    for num_chambers, count in excluded_path_count.items():
        print(f'Number of haptic paths with length 50 for {num_chambers} chambers: {count}')

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.boxplot([path_lengths_thresh[num_chambers] for num_chambers in num_chambers_list], patch_artist=True, labels=[str(num_chambers) for num_chambers in num_chambers_list])
    ax2.set_title('Haptic Path Lengths for Different Number of Chambers')
    ax2.set_xlabel('Number of Chambers')
    ax2.set_ylabel('Path Length (mm)')
    plt.savefig(os.path.join(output_folder, 'haptic_path_lengths_'+ args.controller +'.svg'))
    # plt.show()
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.boxplot([haptic_iterations_thresh[num_chambers] for num_chambers in num_chambers_list], patch_artist=True, labels=[str(num_chambers) for num_chambers in num_chambers_list])
    ax3.set_title('Haptic search iterations for Different Number of Chambers')
    ax3.set_xlabel('Number of Chambers')
    ax3.set_ylabel('Haptic iteration (# of iterations)')
    plt.savefig(os.path.join(output_folder, 'haptic_search_iterations_'+ args.controller +'.svg'))
    plt.show()

    save_path_length_data('path_lengths.csv', path_lengths, haptic_iterations, excluded_path_count, output_folder)
    
    if not args.load_items:
        save_data(polygon_objs, suction_cups_init, initial_positions, 'init_data.p')
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulate haptic search using a suction cup on random polygons")
    parser.add_argument('--load_items', type=bool, help='load polygons from file', default=False)
    parser.add_argument('--num_polygons', type=int, help='number of polygons to generate', default=100)
    parser.add_argument('--num_chambers_list', type=int, nargs='+', help='list of number of chambers for the suction cup', default=[3, 4, 5, 6])
    parser.add_argument('--animation', type=bool, help='save animation as GIF', default=True)
    parser.add_argument('--controller', type=str, help='controller type: normal, yaw, momentum, yaw_momentum', default='normal')
    
    args = parser.parse_args()
    main(args)
