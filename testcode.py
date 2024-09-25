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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from scipy.io import savemat

# Ignore the FutureWarning for Shapely affinity comparisons
warnings.filterwarnings("ignore", category=FutureWarning, module="shapely")

# Create a folder to save all files
# Get the current date and time
date_str = datetime.now().strftime("%y%m%d")
time_str = datetime.now().strftime("%H%M%S")

# Create the output folder path
output_folder = os.path.join('output_files', date_str, time_str)

# Ensure the folder exists
os.makedirs(output_folder, exist_ok=True)

print(f'Output folder created: {output_folder}')

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

def generate_random_polygon(num_points=random.randint(5, 10), min_area=400):
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
    return [polygon.intersection(chamber).area for chamber in chambers]

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
    angles = np.linspace(0, 2 * np.pi, num_chambers, endpoint=False) + np.pi/num_chambers + np.radians(rotation_angle)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def calculate_direction_vector(unit_vectors, vacuum_pressures):
    direction_vector = np.sum(vacuum_pressures[:, None] * unit_vectors, axis=0)
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

def haptic_search_suction_cup(i, ax, polygon_obj, suction_cup, step_size, center_yaw_history, controller, animation=False):
    if check_suction(polygon_obj, suction_cup):
        if animation:
            plot_polygon_and_suction_cup(ax, polygon_obj, suction_cup)
        return True
    
    damping_factor = 0.9  # Define the damping factor
    max_velocity_x = 0.7  # Define the maximum velocity limit for the x direction
    max_velocity_y = 0.7  # Define the maximum velocity limit for the y direction

    suction_cup.direction_vector = unit_direction_vector(polygon_obj, suction_cup)
    
    if controller == 'normal' or controller == 'yaw':
        suction_cup.center += step_size * suction_cup.direction_vector
        if controller == 'yaw':
            suction_cup.rotation_angle += 1
    elif controller == 'momentum' or controller == 'yaw_momentum':
        suction_cup.velocity = damping_factor * suction_cup.velocity + step_size * suction_cup.direction_vector

        # Apply velocity limit to each component
        suction_cup.velocity[0] = np.clip(suction_cup.velocity[0], -max_velocity_x, max_velocity_x)
        suction_cup.velocity[1] = np.clip(suction_cup.velocity[1], -max_velocity_y, max_velocity_y)

        suction_cup.center += suction_cup.velocity
        if controller == 'yaw_momentum':
            suction_cup.rotation_angle += 1

    suction_cup.chambers = suction_cup.create_chambers()
    center_yaw_history.append((suction_cup.center.tolist(), suction_cup.rotation_angle))
    if animation:
        plot_polygon_and_suction_cup(ax, polygon_obj, suction_cup)
        
    return False


def save_animation(fig, axs, polygon_objs, suction_cups, center_yaw_histories, filename, output_folder):
    def update(i):
        for ax, polygon_obj, suction_cup, center_yaw_history in zip(axs.flat, polygon_objs, suction_cups, center_yaw_histories):
            # haptic_search_suction_cup(i, ax, polygon_obj, suction_cup, step_size, haptic_path, center_yaw_history, controller, animation=True)
            if i < len(center_yaw_history):
                center, yaw = center_yaw_history[i]
                suction_cup.center = np.array(center)
                suction_cup.rotation_angle = yaw
                suction_cup.chambers = suction_cup.create_chambers()
                plot_polygon_and_suction_cup(ax, polygon_obj, suction_cup)
    ani = FuncAnimation(fig, update, frames=100, repeat=False)
    ani.save(os.path.join(output_folder, filename), writer='pillow', fps=15)

def calculate_path_length(haptic_path):
    diffs = np.diff(haptic_path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return np.sum(segment_lengths)

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

def plot_and_save_haptic_paths(polygon_objs, center_yaw_histories, path_lengths, filename, num_chambers, row_col, output_folder):
    fig, axs = plt.subplots(row_col, row_col, figsize=(15, 15))
    for ax, polygon_obj, center_yaw_history, path_length in zip(axs.flat, polygon_objs, center_yaw_histories[num_chambers], path_lengths[num_chambers]):
        polygon_obj.draw(ax)
        haptic_path = np.array([cyh[0] for cyh in center_yaw_history])
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

# for parallel processing
def haptic_search_task(args):
    i, ax, polygon_obj, suction_cup, step_size, center_yaw_history, controller = args
    haptic_search_suction_cup(i, ax, polygon_obj, suction_cup, step_size, center_yaw_history, controller, animation=False)

# Save figures for haptic path length and haptic iterations for different controllers
def plot_and_save_controller_comparison(path_lengths, haptic_iterations, controllers, num_chambers_list, output_folder):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([path_lengths[controller][num_chambers] for controller in controllers for num_chambers in num_chambers_list],
               patch_artist=True, labels=[f'{controller}_{num_chambers}' for controller in controllers for num_chambers in num_chambers_list])
    ax.set_title('Haptic Path Lengths for Different Controllers and Number of Chambers')
    ax.set_xlabel('Controller and Number of Chambers')
    ax.set_ylabel('Path Length (mm)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'haptic_path_lengths_controllers.png'))
    plt.savefig(os.path.join(output_folder, 'haptic_path_lengths_controllers.svg'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([haptic_iterations[controller][num_chambers] for controller in controllers for num_chambers in num_chambers_list],
               patch_artist=True, labels=[f'{controller}_{num_chambers}' for controller in controllers for num_chambers in num_chambers_list])
    ax.set_title('Haptic Iterations for Different Controllers and Number of Chambers')
    ax.set_xlabel('Controller and Number of Chambers')
    ax.set_ylabel('Haptic Iterations (# of iterations)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'haptic_iterations_controllers.png'))
    plt.savefig(os.path.join(output_folder, 'haptic_iterations_controllers.svg'))
    plt.close(fig)
    

def plot_and_save_specific_chambers(path_lengths, haptic_iterations, controllers, num_chambers, output_folder, iteration_threshold = 100):
    fig, ax = plt.subplots(figsize=(10, 5))
    filtered_path_lengths = [
        [
            path_length for path_length, iteration in zip(path_lengths[controller][num_chambers], haptic_iterations[controller][num_chambers])
            if iteration <= iteration_threshold and iteration > 1
        ]
        for controller in controllers
    ]
    ax.boxplot(filtered_path_lengths, patch_artist=True, labels=controllers)
    ax.set_title(f'Haptic Path Lengths for {num_chambers} Chambers (thresholded)')
    ax.set_xlabel('Controller')
    ax.set_ylabel('Path Length (mm)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'haptic_path_lengths_{num_chambers}_chambers_thresholded.png'))
    plt.savefig(os.path.join(output_folder, f'haptic_path_lengths_{num_chambers}_chambers_thresholded.svg'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    filtered_haptic_iterations = [
        [
            iteration for iteration in haptic_iterations[controller][num_chambers]
            if iteration <= iteration_threshold and iteration > 1
        ]
        for controller in controllers
    ]
    ax.boxplot(filtered_haptic_iterations, patch_artist=True, labels=controllers)
    ax.set_title(f'Haptic Iterations for {num_chambers} Chambers (thresholded)')
    ax.set_xlabel('Controller')
    ax.set_ylabel('Haptic Iterations (# of iterations)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'haptic_iterations_{num_chambers}_chambers_thresholded.png'))
    plt.savefig(os.path.join(output_folder, f'haptic_iterations_{num_chambers}_chambers_thresholded.svg'))
    plt.close(fig)
    

def main(args):
    num_chambers_list = args.num_chambers_list
    step_size = 0.1 if 'momentum' in args.controller else 0.5

    num_polygons = args.num_polygons
    row_col = int(np.sqrt(num_polygons))
    suction_cups_init = {num_chambers: [] for num_chambers in num_chambers_list}
    suction_cups = {num_chambers: [] for num_chambers in num_chambers_list}
    center_yaw_histories = {num_chambers: [] for num_chambers in num_chambers_list}
    haptic_iterations = {num_chambers: [] for num_chambers in num_chambers_list}
    haptic_iterations_thresh = {num_chambers: [] for num_chambers in num_chambers_list}
    path_lengths = {num_chambers: [] for num_chambers in num_chambers_list}
    path_lengths_thresh = {num_chambers: [] for num_chambers in num_chambers_list}
    excluded_path_count = {num_chambers: 0 for num_chambers in num_chambers_list}

    fig, axs = plt.subplots(row_col, row_col, figsize=(15, 15))
    
    if args.load_items:
        polygon_objs, suction_cups_init, initial_positions = load_data('init_data.p')
    else:
        with ThreadPoolExecutor() as executor:
            polygons = list(executor.map(generate_random_polygon, [random.randint(5, 10) for _ in range(num_polygons)]))
        polygon_with_holes = [add_random_holes(polygon) for polygon in polygons]
        polygon_objs = [PolygonObject(polygon) for polygon in polygon_with_holes]
        initial_positions = [find_initial_suction_cup_position(polygon, 5) for polygon in polygons]

    for num_chambers in num_chambers_list:
        for polygon_obj, initial_position in zip(polygon_objs, initial_positions):
            if not args.load_items:
                # suction_cup = SuctionCup(num_chambers, 5, initial_position, rotation_angle=0)
                suction_cup = SuctionCup(num_chambers, 5, initial_position, rotation_angle=random.randint(0, 360))
                suction_cups_init[num_chambers].append(copy.deepcopy(suction_cup))
            else:
                suction_cups = copy.deepcopy(suction_cups_init)
                suction_cup = suction_cups[num_chambers][polygon_objs.index(polygon_obj)]
            center_yaw_history = [(suction_cup.center.tolist(), suction_cup.rotation_angle)]
            center_yaw_histories[num_chambers].append(center_yaw_history)

            # with ThreadPoolExecutor() as executor:
            #     tasks = [(i, axs.flat[0], polygon_obj, suction_cup, step_size, haptic_path, center_yaw_history, args.controller) for i in range(100)]
            #     executor.map(haptic_search_task, tasks)
            
            for i in range(100):
                if haptic_search_suction_cup(i, axs.flat[0], polygon_obj, suction_cup, step_size, center_yaw_history, args.controller):
                    break # Stop the loop if the suction cup covers the entire polygon

            path_length = calculate_path_length(np.array([cyh[0] for cyh in center_yaw_history]))
            haptic_iterations[num_chambers].append(len(center_yaw_history))
            path_lengths[num_chambers].append(path_length)
            if len(center_yaw_history) > 100 or len(center_yaw_history) < 2:
                excluded_path_count[num_chambers] += 1
            else:
                path_lengths_thresh[num_chambers].append(path_length)
                haptic_iterations_thresh[num_chambers].append(len(center_yaw_history))

        if args.animation:
            suction_cup_animation = copy.deepcopy(suction_cups_init)
            save_animation(fig, axs, polygon_objs, suction_cup_animation[num_chambers], center_yaw_histories[num_chambers], f'suction_cup_animation_{num_chambers}_chambers_'+ args.controller +'.gif', output_folder)

        plot_and_save_haptic_paths(polygon_objs, center_yaw_histories, path_lengths, f'haptic_paths_{num_chambers}_chambers_'+ args.controller +'.png', num_chambers, row_col, output_folder)

    plt.tight_layout()
    # plt.show()

    # for num_chambers, count in excluded_path_count.items():
    #     print(f'Number of haptic paths with length 50 for {num_chambers} chambers: {count}')

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.boxplot([path_lengths_thresh[num_chambers] for num_chambers in num_chambers_list], patch_artist=True, labels=[str(num_chambers) for num_chambers in num_chambers_list])
    ax2.set_title('Haptic Path Lengths for Different Number of Chambers')
    ax2.set_xlabel('Number of Chambers')
    ax2.set_ylabel('Path Length (mm)')
    plt.savefig(os.path.join(output_folder,'haptic_path_lengths_'+ args.controller +'.svg'))
    plt.savefig(os.path.join(output_folder,'haptic_path_lengths_'+ args.controller +'.png'))
    # plt.show()
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.boxplot([haptic_iterations_thresh[num_chambers] for num_chambers in num_chambers_list], patch_artist=True, labels=[str(num_chambers) for num_chambers in num_chambers_list])
    ax3.set_title('Haptic search iterations for Different Number of Chambers')
    ax3.set_xlabel('Number of Chambers')
    ax3.set_ylabel('Haptic iteration (# of iterations)')
    plt.savefig(os.path.join(output_folder,'haptic_search_iterations_'+ args.controller +'.svg'))
    plt.savefig(os.path.join(output_folder,'haptic_search_iterations_'+ args.controller +'.png'))
    # plt.show()
    
    plt.close('all')

    save_path_length_data('path_lengths_'+ args.controller +'.csv', path_lengths, haptic_iterations, excluded_path_count, output_folder)
    
    with open(os.path.join(output_folder, 'center_yaw_histories_'+ args.controller +'.p'), 'wb') as f:
        pickle.dump(center_yaw_histories, f)
        
    # Convert center_yaw_histories to a format suitable for saving as a .mat file
    center_yaw_histories_mat = {}
    for num_chambers, history in center_yaw_histories.items():
        formatted_history = []
        for entry in history:
            center_list = [h[0] for h in entry]
            yaw_list = [h[1] for h in entry]
            formatted_history.append({'center': center_list, 'yaw': yaw_list})
        center_yaw_histories_mat[f'chambers_{num_chambers}'] = formatted_history
    
    savemat(os.path.join(output_folder, 'center_yaw_histories_'+ args.controller +'.mat'), center_yaw_histories_mat)
    
    if not args.load_items:
        save_data(polygon_objs, suction_cups_init, initial_positions, 'init_data.p')
    
    # Print done with controller type
    print(f'Done with controller type: {args.controller}')
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulate haptic search using a suction cup on random polygons")
    parser.add_argument('--load_items', type=bool, help='load polygons from file', default=True)
    parser.add_argument('--num_polygons', type=int, help='number of polygons to generate', default=400)
    parser.add_argument('--num_chambers_list', type=int, nargs='+', help='list of number of chambers for the suction cup', default=[3, 4, 5, 6])
    parser.add_argument('--animation', type=bool, help='save animation as GIF', default=True)
    parser.add_argument('--controller', type=str, help='controller type: normal, yaw, momentum, yaw_momentum', default='yaw_momentum')
    controllers = ['normal', 'yaw', 'momentum', 'yaw_momentum']
    # controllers = ['normal', 'momentum']
    
    args = parser.parse_args()
    
    # Initialize dictionaries to store results for each controller
    all_path_lengths = {controller: {num_chambers: [] for num_chambers in args.num_chambers_list} for controller in controllers}
    all_haptic_iterations = {controller: {num_chambers: [] for num_chambers in args.num_chambers_list} for controller in controllers}
   
    for controller in controllers:
        args.controller = controller
        args.load_items = False if controller == 'normal' else True
        main(args)
        plt.close('all')
        
         # Load and store the results
        with open(os.path.join(output_folder, 'path_lengths_' + args.controller + '.csv'), 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                all_path_lengths[controller][int(row['Num Chambers'])].append(float(row['Path Length']))
                all_haptic_iterations[controller][int(row['Num Chambers'])].append(int(row['Iteration']))
                
                
    # Plot and save the comparison figures
    for nChamber in args.num_chambers_list:
         plot_and_save_specific_chambers(all_path_lengths, all_haptic_iterations, controllers, nChamber, output_folder)
    print('Done')
