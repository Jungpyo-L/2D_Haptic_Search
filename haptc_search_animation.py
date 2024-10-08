import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.affinity import rotate
from matplotlib.animation import FuncAnimation

class PolygonObject:
    def __init__(self, shape, width=None, height=None, position=(0, 0), concave_angle=180, custom_coords=None):
        self.shape = shape
        self.width = width
        self.height = height
        self.position = position
        self.concave_angle = concave_angle
        self.custom_coords = custom_coords
        self.polygon = self.create_polygon()

    def create_polygon(self):
        if self.custom_coords:
            return Polygon(self.custom_coords)
        
        x, y = self.position
        if self.shape == "Rectangle":
            coords = [(x, y), (x + self.width, y), (x + self.width, y + self.height), (x, y + self.height)]
        elif self.shape == "Circle":
            coords = [(x + self.width/2 * np.cos(t), y + self.width/2 * np.sin(t)) for t in np.linspace(0, 2*np.pi, 100)]
        elif self.shape == "Ellipse":
            coords = [(x + self.width/2 * np.cos(t), y + self.height/2 * np.sin(t)) for t in np.linspace(0, 2*np.pi, 100)]
        elif self.shape == "Concave":
            coords = [(x, y), (x + self.width, y), (x + self.width, y + self.height), (x, y + self.height)]
            concave_point = ((x + self.height / 2 / np.tan(np.radians(self.concave_angle / 2))), (y + self.height / 2 ))
            coords.insert(4, concave_point)
        else:
            raise ValueError("Unsupported shape")

        return Polygon(coords)
    
    def draw(self, ax):
        x, y = self.polygon.exterior.xy
        self.patch = ax.fill(x, y, alpha=0.5, fc='gray', ec='black', label='Polygon')[0]

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
    
    # Add noise
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
    tolerance = 1e-4  # Define a tolerance value for floating-point comparison
    return abs(total_covered_area - suction_cup.suction_cup_area) < tolerance

def plot_polygon_and_suction_cup(ax, polygon_obj, suction_cup):
    ax.cla()
    polygon_obj.draw(ax)
    suction_cup.draw(ax)
    ax.set_aspect('equal')
    ax.axis('off')
    # nozzle shape
    # ax.set_xlim(-15, 15)  # Fix x limits
    # ax.set_ylim(-5, 15)   # Fix y limits
    # dumbbell shape
    ax.set_xlim(-5, 35)  # Fix x limits
    ax.set_ylim(-5, 20)   # Fix y limits

def animate_suction_cup(i, ax, polygon_obj, suction_cup, step_size, haptic_path):
    if check_suction(polygon_obj, suction_cup):
        plot_polygon_and_suction_cup(ax, polygon_obj, suction_cup)
        return
    
    damping_factor = 0.9
    suction_cup.direction_vector = unit_direction_vector(polygon_obj, suction_cup)
    # suction_cup.velocity = damping_factor * suction_cup.velocity + step_size * suction_cup.direction_vector
    # suction_cup.center += suction_cup.velocity
    suction_cup.center += step_size * suction_cup.direction_vector
    # suction_cup.rotation_angle += 1
    suction_cup.chambers = suction_cup.create_chambers()
    
    # Update haptic_path
    haptic_path.append(suction_cup.center.tolist())
    
    plot_polygon_and_suction_cup(ax, polygon_obj, suction_cup)

def plot_haptic_path(ax, polygon_obj, haptic_path, label):
    polygon_obj.draw(ax)
    ax.plot(haptic_path[:, 0], haptic_path[:, 1], 'r-', linewidth=2, label=label)
    
    # Set the aspect, limits, and axis off
    ax.set_aspect('equal')
    ax.axis('off')
    # nozzle shape
    # ax.set_xlim(-15, 15)  # Fix x limits
    # ax.set_ylim(-5, 15)   # Fix y limits
    # dumbbell shape
    ax.set_xlim(-5, 35)  # Fix x limits
    ax.set_ylim(-5, 20)   # Fix y limits
    
    # Display path length
    path_length = calculate_path_length(haptic_path)
    ax.text(0, 15.5, f'Path Length: {path_length:.2f} mm', fontsize=12, color='black')
    
    # Display haptic path size
    ax.text(0, 13.5, f'Haptic Path Size: {len(haptic_path)}', fontsize=12, color='black')
    
    # Display legend
    ax.legend()

def calculate_path_length(haptic_path):
    diffs = np.diff(haptic_path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    path_length = np.sum(segment_lengths)
    return path_length

if __name__ == "__main__":
    # Define custom coordinates for a more complicated polygon
    # Nozzle shape
    # custom_coords = [
    #     (0, 0), (5, -3), (14, 4), (4, 13), (1, 8), (-1, 12), (-4, 3), (-3, 0), (-2, -3), (1, -4)
    # ]
    
    # Dumbbell shape
    custom_coords = [
        (0, 0), (11, 0), (11, 5), (23, 5), (23, 0), (34, 0), (34, 15), (23, 15), (23, 10), (11, 10), (11, 15), (0, 15)
    ]
        
    
    # Initialize the complicated polygon object with custom coordinates
    polygon_obj = PolygonObject(shape="Custom", custom_coords=custom_coords)
    
    xx, yy = polygon_obj.polygon.exterior.coords.xy
    
    step_size = 0.5
    
    # Create figure and axis for the haptic path plot
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Set up the suction cup and animation
    haptic_paths = []
    num_chambers = 4
    label = f'{num_chambers} Chambers'
    
    suction_cup_radius = 5
    initial_yaw_angle = 15
    # suction_cup_center = [xx[4] - 2.5, yy[4] + 1] # nozzle shape
    suction_cup_center = [17, 7.5] # dumbbell shape
    suction_cup = SuctionCup(num_chambers, suction_cup_radius, suction_cup_center, initial_yaw_angle)
    
    haptic_path = [suction_cup.center.tolist()]
    
    fig_anim, ax_anim = plt.subplots(figsize=(5, 4))
    ani = FuncAnimation(fig_anim, animate_suction_cup, fargs=(ax_anim, polygon_obj, suction_cup, step_size, haptic_path), interval=10, repeat=False)
    
    ani.save(f'suction_cup_animation_{num_chambers}_chambers.gif', writer='pillow', fps=15)
    
    haptic_paths.append(haptic_path)
    
    path_length = calculate_path_length(np.array(haptic_path))
    print(f"Haptic path length for {num_chambers} chambers: {path_length:.2f} mm")
    
    # Plot the haptic path
    plot_haptic_path(ax, polygon_obj, np.array(haptic_path), label)
    
    # Save the haptic path plot as a PNG file
    fig.savefig(f"haptic_path_{num_chambers}.png", format='png', bbox_inches='tight')
    
    plt.show()