import math
import os
import sys
import random
import string

from panda3d.core import Filename
from direct.showbase.ShowBase import ShowBase
from panda3d.core import GeomVertexReader

import panda3d.core as p3d

import simplepbr

from engine.api import Engine

p3d.load_prc_file_data(
    '',
    'window-size 1024 768\n'
    'texture-minfilter mipmap\n'
    'texture-anisotropic-degree 16\n'
)

def take_screenshot(filepath, cam_pos=None, cam_hpr=None):
    """Capture screenshot with specific camera position/rotation"""
    if cam_pos is not None or cam_hpr is not None:
        # Store original camera transform
        original_pos = base.cam.get_pos()
        original_hpr = base.cam.get_hpr()

        # Apply new transform
        if cam_pos: base.cam.set_pos(cam_pos)
        if cam_hpr: base.cam.set_hpr(cam_hpr)

        # Force render updates
        base.graphicsEngine.renderFrame()
        base.graphicsEngine.renderFrame()

    # Capture screenshot
    base.win.saveScreenshot(Filename(filepath))

    # Restore original transform
    if cam_pos is not None or cam_hpr is not None:
        base.cam.set_pos(original_pos)
        base.cam.set_hpr(original_hpr)
        base.graphicsEngine.renderFrame()

class App(ShowBase):
    def __init__(self):
        if len(sys.argv) < 2:
            print("Missing input file")
            sys.exit(1)

        super().__init__()

        self.pipeline = simplepbr.init()

        # load mesh
        infile = p3d.Filename.from_os_specific(os.path.abspath(sys.argv[1]))
        p3d.get_model_path().prepend_directory(infile.get_dirname())
        self.model_root = self.loader.load_model(infile, noCache=True)

        # shortcuts
        self.accept('escape', sys.exit)
        self.accept('q', sys.exit)
        self.accept('w', self.toggle_wireframe)
        self.accept('t', self.toggle_texture)
        self.accept('n', self.toggle_normal_maps)
        self.accept('e', self.toggle_emission_maps)
        self.accept('o', self.toggle_occlusion_maps)
        self.accept('y', self.toggle_ambient)
        self.accept('a', self.toggle_ambient_light)
        self.accept('shift-l', self.model_root.ls)
        self.accept('shift-a', self.model_root.analyze)
        self.accept("f1", take_screenshot, ["current_camera.png"])
        self.accept("f2", take_screenshot, ["custom_camera.png", (20, -20, 3), (45, 0, 0)])

        self.model_root.reparent_to(self.render)

        bounds = self.model_root.getBounds()
        center = bounds.get_center()
        if bounds.is_empty():
            radius = 1
        else:
            radius = bounds.get_radius()

        fov = self.camLens.get_fov()
        distance = radius / math.tan(math.radians(min(fov[0], fov[1]) / 2.0))
        self.camLens.set_near(min(self.camLens.get_default_near(), radius / 2))
        self.camLens.set_far(max(self.camLens.get_default_far(), distance + radius * 2))
        trackball = self.trackball.node()
        trackball.set_origin(center)
        trackball.set_pos(0, distance, 0)
        trackball.setForwardScale(distance * 0.006)

        # Create a light if the model does not have one
        if not self.model_root.find('**/+Light'):
            self.light = self.render.attach_new_node(p3d.PointLight('light'))
            self.light.set_pos(0, -distance, distance)
            self.render.set_light(self.light)

        # Move lights to render
        self.model_root.clear_light()
        for light in self.model_root.find_all_matches('**/+Light'):
            light.parent.wrt_reparent_to(self.render)
            self.render.set_light(light)

        # Add some ambient light
        self.ambient = self.render.attach_new_node(p3d.AmbientLight('ambient'))
        self.ambient.node().set_color((.2, .2, .2,.2))
        self.render.set_light(self.ambient)

        if self.model_root.find('**/+Character'):
            self.anims = p3d.AnimControlCollection()
            p3d.autoBind(self.model_root.node(), self.anims, ~0)
            if self.anims.get_num_anims() > 0:
                self.anims.get_anim(0).loop(True)

        # Add this new section to print vertices
        self.print_first_vertices(100)

    def print_first_vertices(self, count=100):
        """Print first N vertices from all geometries"""
        print(f"\nFirst {count} vertices:")
        found = 0
        
        # Traverse all geometries in the model
        for geom_node in self.model_root.find_all_matches('**/+GeomNode'):
            geom_node = geom_node.node()
            for i in range(geom_node.get_num_geoms()):
                geom = geom_node.get_geom(i)
                vdata = geom.get_vertex_data()
                
                # Read vertex positions
                vertex = GeomVertexReader(vdata, 'vertex')
                while not vertex.isAtEnd() and found < count:
                    x, y, z = vertex.getData3()
                    print(f"Vertex {found + 1}: ({x:.3f}, {y:.3f}, {z:.3f})")
                    found += 1
                
                if found >= count:
                    return

    def toggle_normal_maps(self):
        self.pipeline.use_normal_maps = not self.pipeline.use_normal_maps

    def toggle_emission_maps(self):
        self.pipeline.use_emission_maps = not self.pipeline.use_emission_maps

    def toggle_occlusion_maps(self):
        self.pipeline.use_occlusion_maps = not self.pipeline.use_occlusion_maps

    def toggle_ambient_light(self):
        if self.render.has_light(self.ambient):
            self.render.clear_light(self.ambient)
        else:
            self.render.set_light(self.ambient)

    def toggle_ambient(self):
        e = self.return_engine(5)
        for i, (pos, rot) in enumerate(zip(e.node_positions, e.node_rotations)):
            print(f"Node {i}:")
            print(f"  Position: {pos}")
            print(f"  Rotation: {rot}")
            
    def random_string(self, length=6):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


    def return_engine(self, sample_size: int = 1):
        node_positions = []
        node_rotations = []

        bounds = self.model_root.getBounds()
        center = bounds.get_center()
        radius = bounds.get_radius()

        for _ in range(sample_size):
            # Random angle around model
            angle_deg = random.uniform(35, 155)
            angle = math.radians(angle_deg)
            distance = random.uniform(radius * 0.8, radius * 1.2)
            
            # Position in a circle above the model
            x = center.x + distance * math.cos(angle)
            y = center.y + distance * math.sin(angle)
            z = center.z + radius * 1.5  # Elevated above model

            pos = (x, y, z)
            node_positions.append(pos)

            # Point down toward model center
            dx = center.x - x
            dy = center.y - y
            dz = center.z - z
            heading = math.degrees(math.atan2(dy, dx))
            pitch = -math.degrees(math.atan2(dz, math.sqrt(dx**2 + dy**2)))
            roll = 0  # Assume no roll for a stable top-down view
            rot = (heading, pitch, roll)
            node_rotations.append(rot)

            take_screenshot(f"img_{self.random_string()}.png", pos, rot)

        return Engine(node_positions=node_positions, node_rotations=node_rotations)

def main():
    App().run()

if __name__ == '__main__':
    main()
