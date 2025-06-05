
import numpy as np
import chroma.geometry as geometry
import os

transmission_data_3wires = np.loadtxt(os.path.join(os.path.dirname(__file__), 'transmission_data_3wires.csv'), delimiter=',')

# wall is stainless steel!
R = 0.8
wire_wall = geometry.Surface(name='wall_3wires', model=4) # model 4 is cuda angular surface
wire_wall.angular_props = geometry.AngularProps(
    angles=transmission_data_3wires[:, 0],
    transmit=transmission_data_3wires[:, 1],
    reflect_specular=np.ones_like(transmission_data_3wires[:, 1]) * R,
    reflect_diffuse=np.ones_like(transmission_data_3wires[:, 1]) * 0.0, # no diffuse reflection
)