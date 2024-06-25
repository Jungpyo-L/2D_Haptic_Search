import svgwrite
from shapely.geometry import Polygon

# Define the custom polygon coordinates
custom_coords = [
    (0, 0), (5, -3), (14, 4), (4, 13), (1, 8), (-1, 12), (-4, 3), (-3, 0), (-2, -3), (1, -4)
]

# Initialize the svgwrite drawing with units in mm
dwg = svgwrite.Drawing('custom_polygon.svg', profile='tiny', size=('100mm', '100mm'))
dwg.viewbox(minx=-10, miny=-10, width=30, height=30)  # Adjust viewbox to fit the polygon

# Add the polygon to the drawing
dwg.add(dwg.polygon(points=custom_coords, fill='none', stroke='black', stroke_width='0.5mm'))

# Save the SVG file
dwg.save()