import matplotlib.pyplot as plt
import random
import polygenerator
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate, scale, translate
from shapely.ops import unary_union

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

def generate_random_polygon(num_points=10, min_area=100):
    polygon = polygenerator.random_polygon(num_points=num_points)
    polygon = Polygon(polygon)
    polygon = scale(polygon, xfact=30, yfact=30, origin=(0, 0))
    return polygon

def generate_random_hole(main_polygon, max_num_points=10, circle_probability=0.8):
    if random.random() < circle_probability:
        # Generate a circular hole
        radius = random.uniform(0.1, 1)
        center = Point(random.uniform(main_polygon.bounds[0], main_polygon.bounds[2]),
                       random.uniform(main_polygon.bounds[1], main_polygon.bounds[3]))
        circle = center.buffer(radius)
        if main_polygon.contains(circle):
            return circle
    else:
        # Generate a polygonal hole
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

def add_random_holes(polygon, num_holes=3):
    holes = []
    for _ in range(num_holes):
        hole = generate_random_hole(polygon)
        holes.append(hole)
    return polygon.difference(unary_union(holes))

def plot_polygon(polygon, out_file_name):
    fig, ax = plt.subplots()
    plt.gca().set_aspect("equal")
    polygon.draw(ax)
    ax.axis('off')
    ax.set_xlim(-5, 35)
    ax.set_ylim(-5, 35)

    plt.show()

polygon = generate_random_polygon(num_points=random.randint(5, 15))
polygon_with_holes = add_random_holes(polygon, num_holes=random.randint(1, 10))

plot_polygon(PolygonObject(polygon_with_holes), "random_polygon.png")
